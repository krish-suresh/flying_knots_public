from pydrake.all import (
        MultibodyPlant,
        AutoDiffXd,
    Parser,
    PiecewisePose,
    MathematicalProgram,
    Context,
    PositionCost,
    OrientationCost,
    SnoptSolver,
    RigidTransform,
    RotationMatrix,
    FixedOffsetFrame,
    SpatialInertia,
    WeldJoint,
    UnitInertia,
    MultibodyForces,
    MultibodyForces_,
)
import numpy as np
from common.math import make_bezier
from pydrake.math import eq
# from common.spline import *
from functools import partial

from pydrake.polynomial import *


def xarm_plant_2d(model_name="xarm7_3_link_float", finalize=True):
    plant = MultibodyPlant(0.0)
    parser = Parser(plant)
    package_xml = "models/xarm_description/package.xml"
    parser.package_map().AddPackageXml(filename=package_xml)
    xarm7 = parser.AddModelsFromUrl(f"package://xarm_description/{model_name}.sdf")[0]

    if finalize:
        plant.Finalize()
    return plant


def xarm_plant_3d(model_name="xarm7_no_hand_float", finalize=True, handle_frame_body=None):
    plant = MultibodyPlant(0.0)
    parser = Parser(plant)
    package_xml = "models/xarm_description/package.xml"
    parser.package_map().AddPackageXml(filename=package_xml)
    xarm7 = parser.AddModelsFromUrl(f"package://xarm_description/{model_name}.sdf")[0]

    if handle_frame_body is not None:
        plant.AddFrame(
            FixedOffsetFrame(
                "tip_frame",plant.GetFrameByName("handle_attachment"),
                handle_frame_body))
        tip_body = plant.AddRigidBody(
            "tip_body",
            plant.GetModelInstanceByName("xarm7"),
        )
        plant.WeldFrames(plant.GetFrameByName("handle_attachment"), tip_body.body_frame(), handle_frame_body)
    if finalize:
        plant.Finalize()

    return plant


def add_xarm_to_plant(plant: MultibodyPlant, model_name="xarm7_no_hand_float"):
    parser = Parser(plant)
    package_xml = "models/xarm_description/package.xml"
    parser.package_map().AddPackageXml(filename=package_xml)
    return parser.AddModelsFromUrl(f"package://xarm_description/{model_name}.sdf")[0]


def xarm_forward_kinematics(
    plant: MultibodyPlant, plant_context: Context, frame_name="link7", body_name="link7", pose_only=False
):
    X_WF = plant.CalcRelativeTransform(
        plant_context, plant.world_frame(), plant.GetFrameByName(frame_name)
    )
    if pose_only:
        return X_WF

    V_WF = plant.EvalBodySpatialVelocityInWorld(
        plant_context, plant.GetBodyByName(body_name)
    )
    return X_WF, V_WF
    


def xarm_ik_3d_bezier(hand_path: PiecewisePose, plant: MultibodyPlant, config, end_point:RigidTransform=None):
    print("Starting IK...")
    num_control_points = config["num_control_points"]

    plant_ad: MultibodyPlant = plant.ToAutoDiffXd()
    plant_ad_context = plant_ad.CreateDefaultContext()

    prog = MathematicalProgram()
    num_floating_base_dim = 4

    base_var = prog.NewContinuousVariables(num_floating_base_dim, "base")

    control_points = prog.NewContinuousVariables(7, num_control_points)

    traj_var = make_bezier(control_points, hand_path.end_time())
    traj_der_var = traj_var.MakeDerivative(1)
    traj_der_der_var = traj_var.MakeDerivative(2)
    traj_der_der_der_var = traj_var.MakeDerivative(3)

    prog.AddConstraint(eq(traj_der_var.value(0).flatten(), np.zeros(7)))
    # prog.AddConstraint(eq(traj_der_der_var.value(0).flatten(), np.zeros(7)))

    prog.AddConstraint(eq(traj_der_var.value(traj_var.end_time()).flatten(), np.zeros(7)))
    # prog.AddConstraint(eq(traj_der_der_var.value(traj_var.end_time()).flatten(), np.zeros(7)))
    
    if "base_loc_bound" in config:
        prog.AddBoundingBoxConstraint(np.array(config["base_loc_bound"]["lb"]), np.array(config["base_loc_bound"]["ub"]), base_var)

    if "base_orientation_locked" in config and config["base_orientation_locked"] and end_point is not None:
        end_point_rot = end_point.rotation().matrix()
        print(f"end_point z axis rotation: {np.arctan2(end_point_rot[1, 0], end_point_rot[0, 0])}")
        # TODO figure this out
        # aligned_ori = -np.arctan2(end_point_rot[1, 0], end_point_rot[0, 0])
        aligned_ori = np.pi
        prog.AddBoundingBoxConstraint(np.array([-np.inf, -np.inf,-np.inf,aligned_ori]), np.array([np.inf, np.inf,np.inf,aligned_ori]), base_var)

    for t in np.linspace(0, hand_path.end_time(), config["num_bound_constraint_points"]):
        for i in range(7):
            prog.AddLinearConstraint(
                traj_var.value(t).flatten()[i],
                np.array(config["joint_pos_bound"]["lb"])[i],
                np.array(config["joint_pos_bound"]["ub"][i]),
            )
            if "joint_vel_bound" in config:
                prog.AddLinearConstraint(
                    traj_der_var.value(t).flatten()[i],
                    -np.array(config["joint_vel_bound"])[i],
                    np.array(config["joint_vel_bound"][i]),
                )
            if "joint_accel_bound" in config:
                prog.AddLinearConstraint(
                    traj_der_der_var.value(t).flatten()[i],
                    -np.array(config["joint_accel_bound"])[i],
                    np.array(config["joint_accel_bound"][i]),
                )

    if "q0_cost" in config:
        q0_cost = np.eye(7) * config["q0_cost"]
        q0_error = traj_var.value(0).flatten() - np.array(config["q0"])
        prog.AddQuadraticCost((q0_error @ q0_cost @ q0_error))

    if "qend_cost" in config:
        qend_cost = np.eye(7) * config["qend_cost"]
        qend_error = traj_var.value(traj_var.end_time()).flatten() - np.array(config["qend"])
        prog.AddQuadraticCost((qend_error @ qend_cost @ qend_error))

    if "vel_cost" in config:
        num_sample_points = config["vel_cost"]["num_points"]
        vel_cost = (np.eye(7) / num_sample_points) * config["vel_cost"]["cost"]
        for t in np.linspace(0, hand_path.end_time(), num_sample_points):
            prog.AddQuadraticCost(
                (traj_der_var.value(t).flatten() @ vel_cost @ traj_der_var.value(t).flatten())
            )

    if "accel_cost" in config:
        num_sample_points = config["accel_cost"]["num_points"]
        accel_cost = (np.eye(7) / num_sample_points) * config["accel_cost"][
            "cost"
        ]
        for t in np.linspace(0, hand_path.end_time(), num_sample_points):
            prog.AddQuadraticCost(
                (traj_der_der_var.value(t).flatten() @ accel_cost @ traj_der_der_var.value(t).flatten())
            )

    if "jerk_cost" in config:
        num_sample_points = config["jerk_cost"]["num_points"]
        jerk_cost = (np.eye(7) / num_sample_points) * config["jerk_cost"][
            "cost"
        ]
        for t in np.linspace(0, hand_path.end_time(), num_sample_points):
            prog.AddQuadraticCost(
                (traj_der_der_der_var.value(t).flatten() @ jerk_cost @ traj_der_der_der_var.value(t).flatten())
            )

    def pos_cost(
        v_: np.ndarray, t, cost=config["position_tracking_cost"]
    ):
        X_WH = hand_path.GetPose(t) # Fingertip pose
        base_v = v_[:num_floating_base_dim]
        joint_v = v_[num_floating_base_dim:]
        traj_var_ = make_bezier(joint_v.reshape((7, -1)), hand_path.end_time())
        return PositionCost(
            plant_ad,
            plant_ad.world_frame(),
            X_WH.translation(),
            plant_ad.GetFrameByName("tip_frame"),
            np.zeros(3),
            np.eye(3) * cost,
            plant_ad_context,
        ).Eval(np.r_[base_v, traj_var_.value(t).flatten()])[0]

    def ori_cost(
        v_: np.ndarray, t, cost=config["orientation_tracking_cost"]
    ):
        X_WH = hand_path.GetPose(t) # Fingertip pose
        base_v = v_[:num_floating_base_dim]
        joint_v = v_[num_floating_base_dim:]
        traj_var_ = make_bezier(joint_v.reshape((7, -1)), hand_path.end_time())
        return OrientationCost(
            plant_ad,
            plant_ad.world_frame(),
            X_WH.rotation(),
            plant_ad.GetFrameByName("tip_frame"),
            RotationMatrix(),
            cost,
            plant_ad_context,
        ).Eval(np.r_[base_v, traj_var_.value(t).flatten()])[0]

    def vel_track_cost(
        v_: np.ndarray,
        t,
        translation_cost=config["velocity_translation_tracking_cost"],
        orientation_cost=config["velocity_orientation_tracking_cost"],
    ):
        V_WH = hand_path.GetVelocity(t)
        base_v = v_[:num_floating_base_dim]
        joint_v = v_[num_floating_base_dim:]
        traj_var_ = make_bezier(joint_v.reshape((7, -1)), hand_path.end_time())
        traj_var_dot_ = traj_var_.MakeDerivative(1)
        q = np.r_[base_v, traj_var_.value(t).flatten()]
        qd = np.r_[np.zeros(num_floating_base_dim), traj_var_dot_.value(t).flatten()]
        plant_ad.SetPositionsAndVelocities(plant_ad_context, np.r_[q, qd])
        V_WT_var = plant_ad.EvalBodySpatialVelocityInWorld(plant_ad_context, plant_ad.GetBodyByName("tip_body")) 
        Q_trans = np.eye(3) * translation_cost
        vel_trans_error = V_WT_var.translational() - V_WH[3:]
        Q_ori = np.eye(3) * orientation_cost
        vel_ori_error = V_WT_var.rotational() - V_WH[:3]
        return vel_trans_error @ Q_trans @ vel_trans_error + vel_ori_error@Q_ori@vel_ori_error

    for t in np.linspace(0, hand_path.end_time(), config["num_tracking_match_points"]):
        prog.AddCost(
            partial(pos_cost, t=t),
            np.r_[base_var, control_points.flatten()],
        )
        prog.AddCost(
            partial(ori_cost, t=t),
            np.r_[base_var, control_points.flatten()],
        )
        prog.AddCost(
            partial(vel_track_cost, t=t),
            np.r_[base_var, control_points.flatten()],
        )
    
    if "joint_tau_bound" in config:
        tau_bound = np.array(config["joint_tau_bound"])
        num_tau_pts = config.get(
            "num_tau_constraint_points", config["num_bound_constraint_points"]
        )

        def joint_torques_at(v_, t):
            if v_.dtype == float:
                plant_ = plant
                context_ = plant.CreateDefaultContext()
                forces_ = MultibodyForces(plant_)
            else:
                plant_ = plant_ad
                context_ = plant_ad_context
                forces_ = MultibodyForces_[AutoDiffXd](plant_)

            base_v = v_[:num_floating_base_dim]
            joint_v = v_[num_floating_base_dim:]
            traj_var_ = make_bezier(joint_v.reshape((7, -1)), hand_path.end_time())
            traj_var_dot_ = traj_var_.MakeDerivative(1)
            traj_var_ddot_ = traj_var_.MakeDerivative(2)

            q = np.r_[base_v, traj_var_.value(t).flatten()]
            qd = np.r_[
                np.zeros(num_floating_base_dim),
                traj_var_dot_.value(t).flatten(),
            ]
            qdd = np.r_[
                np.zeros(num_floating_base_dim),
                traj_var_ddot_.value(t).flatten(),
            ]

            plant_.SetPositionsAndVelocities(context_, np.r_[q, qd])
            return plant_.CalcInverseDynamics(context_, qdd, forces_)[
                num_floating_base_dim:
            ]

        for t in np.linspace(0, hand_path.end_time(), num_tau_pts):
            prog.AddConstraint(
                partial(joint_torques_at, t=t),
                -tau_bound,
                tau_bound,
                np.r_[base_var, control_points.flatten()],
            )

    if "start_finger_tip_constraint" in config:
        def finger_tip_distance_to_floor(v_):
            if v_.dtype == float:
                plant_ = plant
                context_ = plant.CreateDefaultContext()
            else:
                plant_ = plant_ad
                context_ = plant_ad_context
            v = v_[4:]
            traj_var_ = make_bezier(v.reshape((7, -1)), hand_path.end_time())
            q = np.r_[v_[:4], traj_var_.value(0).flatten()]
            plant_.SetPositions(context_, q)
            X_ = plant_.CalcRelativeTransform(context_, plant_.GetFrameByName("floor"), plant_.GetFrameByName("tip_frame"))
            dist = X_.translation()[2] - config["start_finger_tip_constraint"]["min_dist_to_floor"]
            return np.array([dist])
            
        prog.AddConstraint(finger_tip_distance_to_floor, np.zeros(1), np.array([5]), np.r_[base_var, control_points.flatten()])


    if "tip_constraint_to_fixed_end" in config:
        pass

    solver = SnoptSolver()
    result = solver.Solve(prog)
    control_points_soln = result.GetSolution(control_points)

    base_location = result.GetSolution(base_var)
    
    
    print("Computed IK...")
    output_traj =  make_bezier(np.r_[np.repeat(base_location.reshape((-1, 1)), num_control_points, axis=1), control_points_soln], hand_path.end_time())
    
    return output_traj
