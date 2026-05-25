from common.params import ParticleRopeParams
from common import (
    polar_coord_3d,
    sample_bezier_and_two_derivative,
    rot_log_error,
    HumanDemo,
    apply_transform_to_piecewise_pose
)
from pydrake.all import (
    MultibodyForces_,
    MultibodyForces,
    AutoDiffXd,
    RigidTransform,
    PiecewisePose,
    BezierCurve,
    PositionConstraint,
    OrientationConstraint,
    PiecewisePolynomial,
    MathematicalProgram,
    ClarabelSolver,
    jacobian,
    eq
)

from xarm7.kinematics import xarm_forward_kinematics
import numpy as np


def calc_ee_pvRw_states(U_bar, params: ParticleRopeParams, cmd_total_time, idxs, ts,
                        des_traj: PiecewisePose):
    """
    Stacks [p(3), v(3), eR(3), w(3)] for each i in idxs.
    eR = log(R_des(t)^T R_current).
    Output shape: (12 * len(idxs),)
    """
    if U_bar.dtype == float:
        plant = params.plant
        context = params.plant_context
    else:
        plant = params.plant_ad
        context = params.plant_context_ad

    N_cmd = int(cmd_total_time / params.dt) + 1
    q_v_vd = sample_bezier_and_two_derivative(
        U_bar, params.nu // 2, N_cmd, cmd_total_time
    )

    y = []
    for i in idxs:
        plant.SetPositionsAndVelocities(context, q_v_vd[i, :params.nu])
        X_WT, V_WT = xarm_forward_kinematics(plant, context, "tip_frame", "tip_body")

        p = X_WT.translation()        # (3,)

        w = V_WT.rotational()         # (3,)
        v = V_WT.translational()      # (3,)

        # Desired pose from demo at this time
        t = float(ts[i])
        X_des = des_traj.GetPose(t)

        R_cur = X_WT.rotation()
        R_des = X_des.rotation()
        eR = rot_log_error(R_cur, R_des)   # (3,)

        y.append(np.hstack([p, v, eR, w]))

    return np.concatenate(y, axis=0)

def calc_joint_poses(U_bar, params: ParticleRopeParams, cmd_total_time):
    if U_bar.dtype == float:
        plant = params.plant
        context = params.plant_context
        forces = MultibodyForces(plant)
    else:
        # Assume AutoDiff.
        plant = params.plant_ad
        context = params.plant_context_ad
        forces = MultibodyForces_[AutoDiffXd](plant)

    N_cmd = int(cmd_total_time/params.dt) + 1
    q_v_vd = sample_bezier_and_two_derivative(U_bar, params.nu//2, N_cmd, cmd_total_time)
    return q_v_vd[:, 4:11].flatten()


def calc_joint_vels(U_bar, params: ParticleRopeParams, cmd_total_time):
    if U_bar.dtype == float:
        plant = params.plant
        context = params.plant_context
        forces = MultibodyForces(plant)
    else:
        # Assume AutoDiff.
        plant = params.plant_ad
        context = params.plant_context_ad
        forces = MultibodyForces_[AutoDiffXd](plant)

    N_cmd = int(cmd_total_time/params.dt) + 1
    q_v_vd = sample_bezier_and_two_derivative(U_bar, params.nu//2, N_cmd, cmd_total_time)
    return q_v_vd[:, 14:21].flatten()


def calc_joint_accels(U_bar, params: ParticleRopeParams, cmd_total_time):
    if U_bar.dtype == float:
        plant = params.plant
        context = params.plant_context
        forces = MultibodyForces(plant)
    else:
        # Assume AutoDiff.
        plant = params.plant_ad
        context = params.plant_context_ad
        forces = MultibodyForces_[AutoDiffXd](plant)

    N_cmd = int(cmd_total_time/params.dt) + 1
    q_v_vd = sample_bezier_and_two_derivative(U_bar, params.nu//2, N_cmd, cmd_total_time)
    return q_v_vd[:, -7:].flatten()


def calc_joint_torques(U_bar, params: ParticleRopeParams, cmd_total_time):
    if U_bar.dtype == float:
        plant = params.plant
        context = params.plant_context
        forces = MultibodyForces(plant)
    else:
        # Assume AutoDiff.
        plant = params.plant_ad
        context = params.plant_context_ad
        forces = MultibodyForces_[AutoDiffXd](plant)

    N_cmd = int(cmd_total_time/params.dt) + 1
    q_v_vd = sample_bezier_and_two_derivative(U_bar, params.nu//2, N_cmd, cmd_total_time)
    taus = []
    for i in range(N_cmd):
        plant.SetPositionsAndVelocities(context, q_v_vd[i, :params.nu])
        taus.append(
            plant.CalcInverseDynamics(context, q_v_vd[i, params.nu :], forces)[4:]
        )

    return np.array(taus).flatten()


def inverse_particle_model_constraints(
    command : BezierCurve,
    particle_sim_output,
    Z_trial,
    demo_data: HumanDemo,
    params: ParticleRopeParams,
    config,
):
    U_knot_pts = command.control_points()
    nq, nc, N = params.nq, params.nc, params.N

    F = particle_sim_output["F"]

    rope_trajectory_goal : PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(
        demo_data.frame_times[:N], demo_data.unlabeled_markers[:N].T
    )

    Z_goal = np.c_[
        rope_trajectory_goal.vector_values(particle_sim_output["T"]).T,
        rope_trajectory_goal.MakeDerivative(1).vector_values(particle_sim_output["T"]).T,
        np.zeros((N, nc)),
    ][1:]
    # plt.plot(Z_goal[:, params.nq:params.nq+1])
    # plt.plot(Z_trial[:, params.nq:params.nq+1])
    # plt.show()
    # breakpoint()
    critical_point_costs = config.get("critical_point_costs", {})
    num_rope_markers = nq // 3
    marker_spacing = params.rope_length / num_rope_markers
    T = particle_sim_output["T"]

    per_frame_diag = np.zeros((N - 1, 2 * nq + nc))
    for cp in demo_data.critical_points.values():
        if cp.name not in critical_point_costs:
            continue
        costs = critical_point_costs[cp.name]
        frame_idx = int(np.argmin(np.abs(T[1:] - cp.time_offset)))

        start_p = max(0, int(round(cp.space_range[0] / marker_spacing)))
        end_p = min(num_rope_markers, int(round(cp.space_range[1] / marker_spacing)))

        per_frame_diag[frame_idx, start_p * 3 : end_p * 3] = costs["position_cost"]
        per_frame_diag[frame_idx, nq + start_p * 3 : nq + end_p * 3] = costs[
            "velocity_cost"
        ]

    Q = np.diag(per_frame_diag.flatten())

    R = np.kron(
        np.eye(U_knot_pts.shape[1]),
        np.diag(np.r_[np.zeros(4), np.ones(7)]) * config["control_cost"],
    )
    if Z_trial[1:].shape != Z_goal.shape:
        return None

    J = (
        (Z_trial[1:].reshape((-1)) - Z_goal.reshape((-1))).T
        @ Q
        @ (Z_trial[1:].reshape((-1)) - Z_goal.reshape((-1)))
    )

    delta_u, delta_z, J_solve = tracking_ilc_update_constraints(
        Z_trial[1:].reshape((-1)),
        U_knot_pts.reshape((-1), order="F"),
        Z_goal.reshape((-1)),
        F,
        Q,
        R,
        params,
        config,
        demo_data
    )
    return (
        delta_u.reshape(U_knot_pts.shape, order="F"),
        delta_z.reshape(Z_goal.shape),
        J,J_solve
    )


def tracking_ilc_update_constraints(
    Z_vec, U_bar_vec, Z_goal_vec, F, Q, R, params: ParticleRopeParams, config, demo_data : HumanDemo
):
    cmd_total_time = demo_data.rope_trajectory.end_time()
    nu2 = params.nu // 2

    prog = MathematicalProgram()

    delta_z = prog.NewContinuousVariables(len(Z_vec), 1)
    delta_u = prog.NewContinuousVariables(len(U_bar_vec), 1)

    if "trust_region" in config:
        tr = config["trust_region"]
        prog.AddBoundingBoxConstraint(-tr, tr, delta_u)

    # Costs
    prog.AddQuadraticCost(R, np.zeros(R.shape[0]), delta_u)

    # Same as prog.AddQuadraticErrorCost(Q, Z_goal_vec - Z_vec, delta_z)
    d = Z_goal_vec - Z_vec
    prog.AddQuadraticCost(
        2 * Q,
        -2 * Q @ d,
        delta_z,
        is_convex=True
    )

    N_cmd = int(cmd_total_time / params.dt) + 1
    if "follow_through_tracking" in config:
        target_frame_traj = demo_data.hand_trajectory
        ts = np.linspace(0.0, cmd_total_time, N_cmd)
        last_cp_time = max(
            (cp.time_offset for cp in demo_data.critical_points.values()),
            default=0.0,
        )
        K = int(np.argmin(np.abs(ts - last_cp_time)))
        follow_through_times = ts[K:]
        follow_through_idxs = np.arange(K, N_cmd, 1, dtype=int)
        y_des_list = []
        for t in follow_through_times:
            Xd = target_frame_traj.GetPose(float(t))
            # If this exists and returns a 6-vector [w; v] in world:
            Vd = target_frame_traj.GetVelocity(float(t))
            p_des = Xd.translation()
            v_des = Vd[3:]           # translational
            w_des = Vd[:3]           # rotational
            eR_des = np.zeros(3)     # because eR is error vs desired

            y_des_list.append(np.r_[p_des, v_des, eR_des, w_des])

        y_des = np.array(y_des_list).reshape((-1,))

        # Current at U_bar
        y0 = calc_ee_pvRw_states(
            U_bar_vec, params, cmd_total_time, follow_through_idxs, ts,
            target_frame_traj,
        )

        # Jacobian dy/dU at U_bar
        Ju = jacobian(
            lambda U_: calc_ee_pvRw_states(
                U_, params, cmd_total_time, follow_through_idxs, ts,
                target_frame_traj
            ),
            U_bar_vec
        )

        # Weights per frame
        w_pos = config["follow_through_tracking"]["position_tracking_cost"]
        w_vel = config["follow_through_tracking"]["velocity_translation_tracking_cost"]
        w_rot = config["follow_through_tracking"]["orientation_tracking_cost"]
        w_w   = config["follow_through_tracking"]["velocity_orientation_tracking_cost"]

        W_frame = np.diag([w_pos]*3 + [w_vel]*3 + [w_rot]*3 + [w_w]*3)  # 12x12
        W = np.kron(np.eye(len(follow_through_times)), W_frame)

        if (
            "at_last_critical_point_cost" in config["follow_through_tracking"]
            and demo_data.critical_points
        ):
            w_pos_cont = config["follow_through_tracking"].get("at_last_critical_point_cost", w_pos)
            w_vel_cont = config["follow_through_tracking"].get("at_last_critical_point_vel_cost", w_vel)
            w_ori_cont = config["follow_through_tracking"].get("at_last_critical_point_ori_cost", w_rot)
            W_contact = np.diag([w_pos_cont]*3 + [w_vel_cont]*3 + [w_ori_cont]*3 + [w_w]*3)
            W[:12, :12] = W_contact

        # Cost: ||Ju * du - (y_des - y0)||_W^2
        e = (y_des - y0).reshape((-1, 1))
        Qft = 2.0 * (Ju.T @ W @ Ju)
        bft = (-2.0 * (Ju.T @ W @ e)).reshape((-1,))
        prog.AddQuadraticCost(Qft, bft, delta_u, is_convex=True)
        # state_current = calc_ee_states(U_bar_vec, params, cmd_total_time, follow_through=True)
        # prog.AddQuadraticErrorCost(Q, Z_goal_vec - Z_vec, delta_z)
        # prog.AddQuadraticErrorCost()
        # (p_t - p_d).T Q (p_t - p_d)
        # p_t approx = p + J(dp)
        # (Jp - p_d).T Q (Jp - p_d)
        # (Jp - p_d).T (QJp - Qp_d)
        # p'J'QJp - p_d'QJp - p'J'Qp_d + p_d'Qp_d
        # p'J'QJp - 2p_d'QJp + p_d'Qp_d
        # A = J'QJ  b = -2p_d'QJ c = p_d'Qp_d
        # p'Ap + bp + c
        

    # Constraints
    prog.AddConstraint(eq(delta_z, F @ delta_u)) # type: ignore

    base_vars = delta_u.reshape((nu2, -1), order="F")[:4, :].T
    if "fixed_base" in config and config["fixed_base"]:
        prog.AddBoundingBoxConstraint(np.zeros_like(base_vars), np.zeros_like(base_vars), base_vars)
    else:
        for i in range(base_vars.shape[0]-1):
            prog.AddConstraint(eq(base_vars[i], base_vars[i+1])) # type: ignore

    # Linear Torque Constraint
    if "joint_tau_limit" in config:
        tau_current = calc_joint_torques(U_bar_vec, params, cmd_total_time)
        tau_constraint = jacobian(lambda U_: calc_joint_torques(U_, params, cmd_total_time), U_bar_vec)
        tau_limit = np.repeat(
            [config["joint_tau_limit"]], N_cmd, axis=0
        ).flatten()
        prog.AddLinearConstraint(
            tau_constraint, -tau_limit - tau_current, tau_limit - tau_current, delta_u
        )

    # Linear Velocity Limits
    if "joint_vel_limit" in config:
        vel_current = calc_joint_vels(U_bar_vec, params, cmd_total_time)
        vel_constraint = jacobian(lambda U_: calc_joint_vels(U_, params, cmd_total_time), U_bar_vec)
        vel_limit = np.repeat(
            [config["joint_vel_limit"]], N_cmd, axis=0
        ).flatten()
        prog.AddLinearConstraint(
            vel_constraint, -vel_limit - vel_current, vel_limit - vel_current, delta_u
        )

    # Linear Accel Limits
    if "joint_accel_limit" in config:
        accel_current = calc_joint_accels(U_bar_vec, params, cmd_total_time)
        accel_constraint = jacobian(lambda U_: calc_joint_accels(U_, params, cmd_total_time), U_bar_vec)
        accel_limit = np.repeat(
            [config["joint_accel_limit"]], N_cmd, axis=0
        ).flatten()
        prog.AddLinearConstraint(
            accel_constraint,
            -accel_limit - accel_current,
            accel_limit - accel_current,
            delta_u,
        )
    if "joint_pos_limit" in config:
        pos_current = calc_joint_poses(U_bar_vec, params, cmd_total_time)
        pos_constraint = jacobian(lambda U_: calc_joint_poses(U_, params, cmd_total_time), U_bar_vec)
        pos_lower_limit = np.repeat(
            [config["joint_pos_limit"]["lb"]], N_cmd, axis=0
        ).flatten()
        pos_upper_limit = np.repeat(
            [config["joint_pos_limit"]["ub"]], N_cmd, axis=0
        ).flatten()
        prog.AddLinearConstraint(
            pos_constraint,
            pos_lower_limit - pos_current,
            pos_upper_limit - pos_current,
            delta_u,
        )

    # Constraint on the first knot point
    if "zero_initial_du" in config and config["zero_initial_du"]:
        prog.AddBoundingBoxConstraint(np.zeros(nu2), np.zeros(nu2), delta_u[:nu2])

    if "zero_initial_joint_vel" in config and config["zero_initial_joint_vel"]:
        vel_constraint_0 = jacobian(lambda U_: calc_joint_vels(U_, params, cmd_total_time)[:7], U_bar_vec)
        prog.AddLinearConstraint(
            vel_constraint_0, np.zeros(7), np.zeros(7), delta_u
        )
    if "zero_final_joint_vel" in config and config["zero_final_joint_vel"]:
        vel_constraint_end = jacobian(lambda U_: calc_joint_vels(U_, params, cmd_total_time)[-7:], U_bar_vec)
        prog.AddLinearConstraint(
            vel_constraint_end, np.zeros(7), np.zeros(7), delta_u
        )

    if "start_finger_tip_constraint" in config:
        start_cfg = config["start_finger_tip_constraint"]
        frame_name = start_cfg.get("frame", "finger_tip")
        min_dist = start_cfg["min_dist_to_floor"]

        def finger_tip_distance_to_floor(U_):
            if U_.dtype == float:
                plant_ = params.plant
                context_ = params.plant_context
            else:
                plant_ = params.plant_ad
                context_ = params.plant_context_ad

            N_cmd = int(cmd_total_time / params.dt) + 1
            q_v_vd = sample_bezier_and_two_derivative(
                U_, params.nu // 2, N_cmd, cmd_total_time
            )
            plant_.SetPositionsAndVelocities(context_, q_v_vd[0, :params.nu])
            dist = (
                plant_.CalcRelativeTransform(
                    context_,
                    plant_.GetFrameByName("floor"),
                    plant_.GetFrameByName(frame_name),
                ).translation()[2]
                - min_dist
            )
            return np.array([dist])

        dist_current = finger_tip_distance_to_floor(U_bar_vec)
        dist_constraint = jacobian(finger_tip_distance_to_floor, U_bar_vec)
        prog.AddLinearConstraint(
            dist_constraint, -dist_current, np.array([np.inf]), delta_u
        )

    if "tip_constraint_to_fixed_end" in config:
        def finger_tip_distance_to_end_point(U_):
            X_WE = params.fixed_end_pose
            if U_.dtype == float:
                plant_ = params.plant
                context_ = params.plant_context
            else:
                plant_ = params.plant_ad
                context_ = params.plant_context_ad
                X_WE = X_WE.cast[AutoDiffXd]()

            N_cmd = int(cmd_total_time / params.dt) + 1
            q_v_vd = sample_bezier_and_two_derivative(
                U_, params.nu // 2, N_cmd, cmd_total_time
            )
            dists = []
            for q_v in q_v_vd[:, :params.nu]:
                plant_.SetPositionsAndVelocities(context_, q_v)
                # X_TW @ X_WE
                dist = (plant_.CalcRelativeTransform(
                        context_,
                        plant_.GetFrameByName("tip_frame"),
                        plant_.world_frame(),
                    )@ X_WE).translation()
                dists.append(np.linalg.norm(dist))
            return np.array(dists)

        dists_current = finger_tip_distance_to_end_point(U_bar_vec)
        J = jacobian(finger_tip_distance_to_end_point, U_bar_vec)
        # -dc < J*du < maxdist - dc
        prog.AddLinearConstraint(
            J, -dists_current, params.rope_length-0.1-dists_current, delta_u
        )


    print("Starting solve...")
    # solver = OsqpSolver()
    solver = ClarabelSolver()
    result = solver.Solve(prog)
    print(result.get_solution_result())
    
    return result.GetSolution(delta_u), result.GetSolution(delta_z), result.get_optimal_cost()
