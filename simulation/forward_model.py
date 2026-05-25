import argparse
from pydrake.all import *
from common import *
import matplotlib.pyplot as plt
from simulation.particle_dynamics import *
from simulation.spring_mass_damper import *
from xarm7.kinematics import xarm_forward_kinematics, xarm_plant_3d
from xarm7.visualize import *

import viser


def backward_euler_residual(z_k, z_k1, params: ParticleRopeParams):
    nq = params.nq
    dt = params.dt
    v_next = z_k1[nq : 2 * nq]
    return z_k1[:nq] - (z_k[:nq] + dt * v_next)


def serial_distance_constraint(ee_position, x, params: ParticleRopeParams):
    l = params.l
    num_particles = params.num_particles
    nd = params.particle_dim

    r_c = [np.dot(ee_position - x[:nd], ee_position - x[:nd]) - l**2]
    for i in range(num_particles - 1):
        x_a = x[i * nd : (1 + i) * nd]
        x_b = x[(1 + i) * nd : (2 + i) * nd]
        r_c.append(np.dot(x_a - x_b, x_a - x_b) - l**2)
    
    if params.fixed_end:
        x_a = x[-nd :]
        x_b = params.fixed_end_pose.translation()
        r_c.append(np.dot(x_a - x_b, x_a - x_b) - l**2)

    return r_c


def serial_distance_constraint_jac(ee_position, x, params: ParticleRopeParams):
    num_particles = params.num_particles
    nq, nc = params.nq, params.nc
    nd = params.particle_dim

    if x.dtype == object:
        con_jac = np.zeros((nc, nq), dtype=x.dtype)
    elif ee_position.dtype == object:
        con_jac = np.zeros((nc, nq), dtype=ee_position.dtype)
    else:
        con_jac = np.zeros((nc, nq), dtype=np.float64)

    con_jac[0, :nd] = -2 * (ee_position - x[:nd])

    for i in range(0, num_particles - 1):
        x_a = x[i * nd : (1 + i) * nd]
        x_b = x[(1 + i) * nd : (2 + i) * nd]
        G = 2 * (x_a - x_b)
        J = np.concatenate([G, -G])
        con_jac[i + 1, i * nd : (i + 2) * nd] = J

    if params.fixed_end:
        con_jac[-1, -nd:] = -2 * (params.fixed_end_pose.translation() - x[-nd:])

    return con_jac


def dynamics_residual(z, s, u, params: ParticleRopeParams):
    nq, num_particles = params.nq, params.num_particles
    dt, g, M = params.dt, params.g, params.M

    if type(u.flatten()[0]) == Variable or type(u.flatten()[0]) == Expression:
        plant, plant_context = params.plant_sym, params.plant_context_sym
    elif u.dtype == object:
        plant, plant_context = params.plant_ad, params.plant_context_ad
    else:
        plant, plant_context = params.plant, params.plant_context

    plant.SetPositionsAndVelocities(plant_context, u)
    X_WF, V_WF = xarm_forward_kinematics(
        plant,
        plant_context,
        frame_name="tip_frame",
        body_name="tip_body"
    )
    u_hand = np.zeros(12, dtype=u.dtype)
    u_hand[:3] = X_WF.translation()
    u_hand[3:6] = X_WF.rotation().matrix()[:, 0]
    u_hand[6:9] = V_WF.translational()
    u_hand[9:] = V_WF.rotational()

    finger_tip_position = u_hand[:3]

    x_k = z[:nq]
    v_k = z[nq : nq * 2]

    v_next = s[:nq]
    lambda_k = s[nq:]

    x_next = x_k + dt * v_next
    J = serial_distance_constraint_jac(finger_tip_position, x_k, params)

    g_vec = np.array([0, 0, 1] * num_particles) * g
    F_spring = stiffness_force_3d(u_hand, x_next, params)
    F_damping = damping_force_3d(u_hand, x_next, v_next, params)
    F_collision = np.zeros_like(F_damping)
    if params.finger_collision is not None:
        F_collision = cylinder_collision_force(u_hand, x_next, X_WF, params.finger_collision, params)
        

    r_f = (
        M @ ((v_next - v_k) / dt + g_vec)
        - F_spring
        - F_damping
        - F_collision
        - J.T @ lambda_k
    )  # Force Balance
    r_c = serial_distance_constraint(
        finger_tip_position, x_next, params
    )  # Particle distance

    return np.hstack((r_f, r_c))


def dynamics_step(
    z_k, u_k, params: ParticleRopeParams, verbose=False, newton_iters=200
):
    nq, dt = params.nq, params.dt
    s = newton(
        lambda s_: dynamics_residual(z_k, s_, u_k, params),
        z_k[nq:],
        verbose=verbose,
        max_iters=newton_iters,
    )
    v_next, lambda_k = s[:nq], s[nq:]

    z_k1 = np.copy(z_k)
    # Euler Update
    z_k1[:nq] = z_k[:nq] + dt * v_next
    z_k1[nq : nq * 2] = v_next
    z_k1[nq * 2 :] = lambda_k

    return z_k1


def dynamics_simulate(U, params: ParticleRopeParams, verbose=False, newton_iters=200):
    nq, nc = params.nq, params.nc
    z0 = np.concatenate([params.x0, params.v0, [0] * nc])
    dt = params.dt
    N = params.N
    T = np.linspace(0, (N-1) * dt, N)
    Z = np.zeros((N, len(z0)))
    Z[0] = z0
    for k in range(N - 1):
        Z[k + 1] = dynamics_step(Z[k], U[k], params, verbose, newton_iters)

    return T, Z


def dynamics_traj_kkt(z0, Z_vec, U_vec, params: ParticleRopeParams):
    N, nq = params.N, params.nq

    Z = Z_vec.reshape((N - 1, -1))
    U = U_vec.reshape((N - 1, -1))
    kkts = []
    kkts.append(dynamics_residual(z0, Z[0, nq:], U[0], params))
    kkts.append(backward_euler_residual(z0, Z[0], params))
    for u, z_last, z in zip(U[1:], Z[:-1], Z[1:]):
        kkts.append(dynamics_residual(z_last, z[nq:], u, params))
        kkts.append(backward_euler_residual(z_last, z, params))

    kkts = np.concatenate(kkts)
    if type(U_vec.flatten()[0]) == Expression:
        raise RuntimeError("Symbolic not supported")
    if U_vec.dtype == object:
        for i, kkt in enumerate(kkts):
            if not isinstance(kkt, AutoDiffXd):
                kkts[i] = AutoDiffXd(kkt)
    return kkts


def _resolve_per_index_array(config_value, length: int) -> np.ndarray:
    if isinstance(config_value, dict):
        start = config_value["start"]
        end = config_value["end"]
        interp = config_value.get("interpolate", "linear")
        if interp == "linear":
            return np.linspace(start, end, length)
        raise ValueError(f"Unsupported interpolation method: {interp}")
    return np.ones(length) * config_value


def make_particle_model_params(
    config: dict,
    model_total_time,
    handle_mocap_object: MocapHandleObject,
):
    dt = config["dt"]
    N = int(model_total_time / dt) + 1

    plant = xarm_plant_3d(
        "xarm7_no_hand_float",
        handle_frame_body=handle_mocap_object.handle_attachment_frame_to_tip_frame,
    )
    nu = plant.num_continuous_states()
    num_particles = config["num_particles"]


    fixed_end_pose_config = config.get(
        "fixed_end_pose", config.get("fixed_end_position")
    )
    if isinstance(fixed_end_pose_config, RigidTransform):
        fixed_end_pose = fixed_end_pose_config
    elif fixed_end_pose_config is None:
        fixed_end_pose = RigidTransform()
    elif isinstance(fixed_end_pose_config, dict):
        if "xyz" not in fixed_end_pose_config or "wxyz" not in fixed_end_pose_config:
            raise ValueError(
                "fixed_end_pose dict must contain 'xyz' and 'wxyz' keys"
            )
        fixed_end_pose = RigidTransform(
            Quaternion(np.asarray(fixed_end_pose_config["wxyz"])),
            np.asarray(fixed_end_pose_config["xyz"]),
        )
    else:
        fixed_end_pose = RigidTransform(np.asarray(fixed_end_pose_config))

    params = ParticleRopeParams(
        M=np.eye(1),
        x0=None,
        v0=None,
        plant=plant,
        num_particles=num_particles,
        nu=nu,
        l=config["l"],
        N=N,
        dt=dt,
        fixed_end=config.get("fixed_end", False),
        fixed_end_pose=fixed_end_pose,
    )
    
    if "finger_collision" in config:
        params.finger_collision = Cylinder(config["finger_collision"]["radius"], config["finger_collision"]["length"])

    params.stiffness = _resolve_per_index_array(config["stiffness"], params.nc)
    params.damping = _resolve_per_index_array(config["damping"], params.nc)

    if "mass" in config:
        mass_per_particle = _resolve_per_index_array(config["mass"], num_particles)
        params.M = np.diag(np.repeat(mass_per_particle, params.particle_dim))
    else:
        params.M = np.eye(params.nq) * 1
        if "lead_mass" in config:
            params.M[-3:, -3:] = np.eye(3) * config["lead_mass"]

    if "g" in config:
        params.g = config["g"]
    return params


def forward_particle_model(command : BezierCurve, params: ParticleRopeParams, x0=None, actual_arm_traj : Trajectory = None):
    N = params.N
    U_bar = command.control_points()
    command_end_time = command.end_time()

    if actual_arm_traj is None:
        N_cmd = int(command_end_time/params.dt)+1
        U_all = sample_bezier_and_derivative(U_bar, params.nu // 2, N_cmd, end_time=command_end_time)
        u0 = U_all[0]
        U = U_all[1:params.N]
    else:
        N_cmd = int(command_end_time/params.dt)+1
        t_u = np.linspace(0, (N-1)*params.dt, N)
        U_all = actual_arm_traj.vector_values(t_u).T
        U_der = actual_arm_traj.MakeDerivative(1).vector_values(t_u).T
        U_all = np.c_[U_all, U_der]
        u0 = U_all[0]
        U = U_all[1:params.N]
    
    


    # if x0 is None:
    #     params.plant.SetPositionsAndVelocities(params.plant_context, u0)
    #     finger_tip_pos_u0 = params.plant.CalcRelativeTransform(
    #         params.plant_context, params.plant.world_frame(), params.rope_attachment_frame
    #     ).translation()
    #     params.x0 = np.array(
    #         [
    #             finger_tip_pos_u0 + np.array([0, 0, -y])
    #             for y in np.linspace(
    #                 params.l, params.l * params.nc, params.num_particles
    #             )
    #         ]
    #     ).flatten()
    # else:
    params.x0 = x0

    params.v0 = np.zeros(params.nq)

    # if "xarm_dynamics" in particle_model_config:
    #     U_dyn = spring_mass_simulate(u0, U, np.array(particle_model_config["xarm_dynamics"]["xarm_joint_stiffness"]), np.array(particle_model_config["xarm_dynamics"]["xarm_joint_damping"]), params.dt, params.N)
    # else:
    #     U_dyn = U

    print("Starting particle sim...")
    T, Z = dynamics_simulate(U, params)
    print("Starting diff...")
    F = implicit_function_kkt_gradient(
        lambda Z_, U_: dynamics_traj_kkt(
            params.z0,
            Z_,
            sample_bezier_and_derivative(U_, params.nu // 2, N_cmd, end_time=command_end_time)[1:params.N],
            params,
        ),
        Z[1:].reshape((-1)),
        U_bar.reshape((-1), order="F"),
    )

    forward_model_output = {
        "T": T,
        "Z": Z,
        "U": U_all[:params.N],
        "F": F,
        "params": params,
    }

    return forward_model_output

def animate_particles_3d(meshcat: Meshcat, animation: MeshcatAnimation, U, Z, params: ParticleRopeParams, prefix="", rgba: Rgba=Rgba(1,0,0,1), hold_frames=0):
    l, hand_len = params.l, .14
    num_particles = params.num_particles

    rope_name = f"rope_{prefix}"
    meshcat.Delete(rope_name)
    meshcat.SetObject(f"{rope_name}/hand", Box(hand_len, 0.04*l, 0.04*l), rgba=rgba)
    meshcat.SetObject(f"{rope_name}/hand_dot", Sphere(0.05*l), rgba=rgba)

    for i in range(num_particles):
        meshcat.SetObject(f"{rope_name}/{i}", Sphere(0.05*l), rgba=rgba)
        meshcat.SetObject(f"{rope_name}/l{i}", Cylinder(0.02*l, l), rgba)

    if U is None:
        U = np.zeros((params.N, params.nu))
        meshcat.Delete(f"{rope_name}/hand")
        meshcat.Delete(f"{rope_name}/hand_dot")
        meshcat.Delete(f"{rope_name}/0")

    for f, (u, z) in enumerate(zip(U, Z)):
        ee_position = u[:3] + polar_coord_3d(u[3], u[4], hand_len)
        q_last = ee_position
        hand_position = (u[:3] + ee_position)/2
        animation.SetTransform(f, f"{rope_name}/hand", RigidTransform(RollPitchYaw(0, -u[4], u[3]), hand_position))
        animation.SetTransform(f, f"{rope_name}/hand_dot", RigidTransform(ee_position))

        for i in range(num_particles):
            q = z[3*i:3*(i+1)]

            animation.SetTransform(f, f"{rope_name}/{i}", RigidTransform(q))
            v = q-q_last
            phi = np.arctan2(v[1], v[0])
            h = np.sqrt(v[0]**2+ v[1]**2)
            theta = np.pi/2 - np.arctan2(v[2], h) 
            m = (q_last+q)/2
            animation.SetTransform(f, f"{rope_name}/l{i}", RigidTransform(RollPitchYaw(0, theta, phi), m))
            q_last = q


def animate_particles_3d_xarm(meshcat: Meshcat, animation: MeshcatAnimation, U, Z, params: ParticleRopeParams, prefix="a", rgba: Rgba=Rgba(1,0,0,1), opacity=1):
    l, hand_len = params.l, .14
    num_particles = params.num_particles
    plant, plant_context = params.plant, params.plant_context
    assert plant is not None and plant_context is not None

    rope_name = f"{prefix}/rope"
    meshcat.Delete(rope_name)
    meshcat.SetObject(f"{rope_name}/hand", Box(hand_len, 0.04*l, 0.04*l), rgba=rgba)
    meshcat.SetObject(f"{rope_name}/hand_dot", Sphere(0.05*l), rgba=rgba)

    for i in range(num_particles):
        meshcat.SetObject(f"{rope_name}/{i}", Sphere(0.05*l), rgba=rgba)
        meshcat.SetObject(f"{rope_name}/l{i}", Cylinder(0.02*l, l), rgba)

    xarm_name = f"{prefix}/xarm"
    meshcat.Delete(xarm_name)
    meshcat.SetObject(f"{xarm_name}/link_base", Mesh("models/xarm_description/meshes/xarm7/visual_flat/link_base.obj"))
    meshcat.SetProperty(f"{xarm_name}/link_base", "opacity", opacity)
    for i in range(7):
        link_name = f"link{i+1}"
        meshcat.SetObject(f"{xarm_name}/{link_name}", Mesh(f"models/xarm_description/meshes/xarm7/visual_flat/{link_name}.obj"))
        meshcat.SetProperty(f"{xarm_name}/{link_name}", "opacity", opacity)

    for f, (u, z) in enumerate(zip(U, Z)):
        plant.SetPositionsAndVelocities(plant_context, u)
        animation.SetTransform(f, f"{xarm_name}/link_base", plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName("link_base")))
        for i in range(7):
            link_name = f"link{i+1}"
            animation.SetTransform(f, f"{xarm_name}/{link_name}", plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName(link_name)))
        
        X_WF = xarm_forward_kinematics(plant, plant_context, frame_name="tip_frame", pose_only=True)
        ee_position = X_WF.translation()
        q_last = ee_position
        hand_position = (X_WF.translation() + ee_position)/2
        animation.SetTransform(f, f"{rope_name}/hand", RigidTransform(X_WF.rotation(), hand_position))
        animation.SetTransform(f, f"{rope_name}/hand_dot", RigidTransform(ee_position))

        for i in range(num_particles):
            q = z[3*i:3*(i+1)]

            animation.SetTransform(f, f"{rope_name}/{i}", RigidTransform(q))
            v = q-q_last
            phi = np.arctan2(v[1], v[0])
            h = np.sqrt(v[0]**2+ v[1]**2)
            theta = np.pi/2 - np.arctan2(v[2], h) 
            m = (q_last+q)/2
            animation.SetTransform(f, f"{rope_name}/l{i}", RigidTransform(RollPitchYaw(0, theta, phi), m))
            q_last = q

def animate_xarm_states(meshcat : Meshcat, animation : MeshcatAnimation, plant:MultibodyPlant, joint_poses, opacity = 1, prefix="xarm", dt=1/250):
    plant_context = plant.CreateDefaultContext()

    meshcat.Delete(prefix)
    meshcat.SetObject(f"{prefix}/link_base", Mesh("models/xarm_description/meshes/xarm7/visual_flat/link_base.obj"))
    meshcat.SetProperty(f"{prefix}/link_base", "opacity", opacity)
    for i in range(7):
        link_name = f"link{i+1}"
        meshcat.SetObject(f"{prefix}/{link_name}", Mesh(f"models/xarm_description/meshes/xarm7/visual_flat/{link_name}.obj"))
        meshcat.SetProperty(f"{prefix}/{link_name}", "opacity", opacity)

    for f, q in enumerate(joint_poses):
        plant.SetPositions(plant_context, q)
        animation.SetTransform(f, f"{prefix}/link_base", plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName("link_base")))
        for i in range(7):
            link_name = f"link{i+1}"
            animation.SetTransform(f, f"{prefix}/{link_name}", plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName(link_name)))

def create_xarm_particle_visuals(
    server: viser.ViserServer,
    handle_mocap_object: MocapHandleObject,
    params: ParticleRopeParams,
    xarm_name: str = "xarm",
    rope_name: str = "particle_model_zero_control",
    xarm_color=(120, 255, 120),
    rope_color=(0, 200, 255),
    xarm_visible=True
):
    xarm_visual = add_xarm_visual(
        server,
        name=xarm_name,
        handle=handle_mocap_object,
        color=xarm_color,
        visible=xarm_visible
    )

    num_rope_links = params.num_particles + int(params.fixed_end)
    rope_visual = add_rope_visual(
        server,
        rope_length=float(params.l * num_rope_links),
        rope_radius=0.0045,
        num_links=max(num_rope_links, 1),
        name=rope_name,
        color=rope_color,
    )

    if params.finger_collision is not None:
        # Viser cylinder primitives are aligned with local +Y; rotate so +Y aligns
        # with fingertip -X, matching collision_cylinder axis convention.
        X_L7T = params.plant.CalcRelativeTransform(
            params.plant_context,
            params.plant.GetFrameByName("link7"),
            params.plant.GetFrameByName("tip_frame"),
        )
        collision_len = params.finger_collision.length()
        X_TC = RigidTransform(
            RollPitchYaw(0.0, np.pi / 2, 0.0),
            np.array([-0.5 * collision_len, 0.0, 0.0]),
        )
        X_L7C = X_L7T @ X_TC
        server.scene.add_cylinder(
            f"{xarm_name}/link7/finger_collision",
            params.finger_collision.radius(),
            collision_len,
            color=(255, 140, 0),
            opacity=0.45,
            position=X_L7C.translation(),
            wxyz=X_L7C.rotation().ToQuaternion().wxyz(),
            visible=xarm_visible,
        )

    if params.fixed_end:
        fixed_end_sphere = server.scene.add_icosphere(
            f"{rope_name}/fixed_end",
            0.01,
            (255, 0, 0),
        )
        fixed_end_sphere.position = params.fixed_end_pose.translation()

    return xarm_visual, rope_visual

def set_xarm_particle_animation(
    animation: ViserAnimationRealtime,
    xarm_visual,
    rope_visual,
    params: ParticleRopeParams,
    T: np.ndarray,
    Z: np.ndarray,
    U: np.ndarray,
):
    if U.shape[0] == 0:
        joint_positions = np.zeros((len(T), params.nu // 2))
    else:
        joint_positions = np.vstack([U, U[-1]])[: len(T), : params.nu // 2]
    arm_traj = PiecewisePolynomial.FirstOrderHold(T, joint_positions.T)

    rope_points = Z[:, : params.nq].reshape(len(T), params.num_particles, params.particle_dim)
    fingertip_positions = np.empty((joint_positions.shape[0], 3))
    for i, q in enumerate(joint_positions):
        params.plant.SetPositions(params.plant_context, q)
        X_WT = xarm_forward_kinematics(
            params.plant,
            params.plant_context,
            frame_name="tip_frame",
            pose_only=True,
        )
        fingertip_positions[i] = X_WT.translation()
    rope_points_with_tip = np.concatenate(
        [fingertip_positions[:, np.newaxis, :], rope_points], axis=1
    )

    if params.fixed_end:
        fixed_end_poses = np.repeat(
            params.fixed_end_pose.translation()[np.newaxis, np.newaxis, :],
            len(T),
            axis=0,
        )
        rope_points_with_tip = np.concatenate(
            [rope_points_with_tip, fixed_end_poses], axis=1
        )
    rope_traj = RopeTrajectory(T, rope_points_with_tip)

    for mesh_handle, mesh_traj in get_xarm_visual_traj(xarm_visual, arm_traj):
        animation.add_animated_object(mesh_handle, mesh_traj)

    for mesh_handle, mesh_traj in get_rope_visual_traj(rope_visual, rope_traj):
        animation.add_animated_object(mesh_handle, mesh_traj)


def _make_zero_control_initial_state(params: ParticleRopeParams) -> np.ndarray:
    q0 = np.zeros(params.nu // 2)
    params.plant.SetPositions(params.plant_context, q0)
    X_WT_0 = xarm_forward_kinematics(
        params.plant,
        params.plant_context,
        frame_name="tip_frame",
        pose_only=True,
    )
    fingertip_position = X_WT_0.translation()
    fingertip_x_axis = X_WT_0.rotation().matrix()[:, 0]
    ref_x = np.array(
        [
            fingertip_position + (i + 1) * params.l * fingertip_x_axis
            for i in range(params.num_particles)
        ]
    ).reshape(-1)

    rope_end_points = (
        params.fixed_end_pose.translation() if params.fixed_end else None
    )
    try:
        return solve_for_initial_state(
            fingertip_position, ref_x, params.l, rope_end_points=rope_end_points
        ).reshape(-1)
    except Exception as exc:
        print(
            "WARNING: Failed to solve constrained x0, "
            f"falling back to straight-line init. Error: {exc}"
        )
        return ref_x


def _get_trial_command_end_time(trial_data: dict) -> float:
    learning_state = trial_data.get("learning_state")
    if learning_state is not None:
        return float(learning_state.bezier_command.end_time())

    arm_trajectory = trial_data.get("arm_trajectory")
    if arm_trajectory is not None:
        return float(arm_trajectory.traj.end_time())

    raise ValueError(
        "Trial data does not contain `learning_state` or `arm_trajectory`."
    )


def _extract_trial_initial_state(
    trial_data: dict, params: ParticleRopeParams
) -> tuple[np.ndarray, np.ndarray]:
    model_data = trial_data.get("model_data")
    if not isinstance(model_data, dict) or "Z" not in model_data:
        raise ValueError("Trial data missing `model_data['Z']` for initial state.")

    Z_trial = np.asarray(model_data["Z"])
    if Z_trial.ndim != 2 or Z_trial.shape[1] < 2 * params.nq:
        raise ValueError(
            "Trial model state has unexpected shape; expected at least "
            f"(N, {2 * params.nq}), got {Z_trial.shape}."
        )

    z0_trial = Z_trial[0]
    x0 = np.array(z0_trial[: params.nq], dtype=float, copy=True)
    v0 = np.array(z0_trial[params.nq : 2 * params.nq], dtype=float, copy=True)
    return x0, v0


def _build_trial_control_sequence(
    trial_data: dict, params: ParticleRopeParams
) -> np.ndarray:
    learning_state = trial_data.get("learning_state")
    if learning_state is not None:
        command = learning_state.bezier_command
        command_end_time = float(command.end_time())
        N_cmd = int(command_end_time / params.dt) + 1
        U_all = sample_bezier_and_derivative(
            command.control_points(),
            params.nu // 2,
            N_cmd,
            end_time=command_end_time,
        )
    else:
        arm_trajectory = trial_data.get("arm_trajectory")
        if arm_trajectory is None:
            raise ValueError(
                "Trial data does not contain a command source (`learning_state` "
                "or `arm_trajectory`)."
            )
        arm_traj = arm_trajectory.traj
        command_end_time = float(arm_traj.end_time())
        t_u = np.linspace(0, command_end_time, int(command_end_time / params.dt) + 1)
        q = arm_traj.vector_values(t_u).T
        qd = arm_traj.MakeDerivative(1).vector_values(t_u).T
        U_all = np.c_[q, qd]

    if U_all.shape[1] != params.nu:
        raise ValueError(
            f"Expected control dimension {params.nu}, got {U_all.shape[1]}."
        )
    return U_all