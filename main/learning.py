import argparse
import logging
import os
import shutil
import subprocess
import matplotlib.pyplot as plt
import glob

from common import (
    check_if_frame_drops,
    get_latest_trial_name,
    HumanDemo,
    generate_trial_name,
    load_pickle,
    make_bezier,
    parse_yaml,
    RopeTrajectory,
    ViserAnimation,
    add_rope_visual,
    set_rope_visual,
    solve_for_initial_state,
    LearningState,
    save_pickle,
    XarmTrajectory,
    dict_to_yaml,
    XarmRopeMocapData,
    print_frame_drops,
    get_flying_knot_data_dir,
    transform_3d_points,
    translation_zrot_to_transform,
    resample_rope_markers,
)
from pydrake.all import BezierCurve, RigidTransform, MeshcatAnimation, Meshcat, Rgba, PiecewisePolynomial
from dataclasses import dataclass
import numpy as np
from xarm7 import (
    add_xarm_visual,
    set_xarm_visual,
    xarm_plant_3d,
    xarm_forward_kinematics,
)

from simulation import (
    forward_particle_model,
    make_particle_model_params,
    dynamics_simulate,
    animate_particles_3d,
    animate_particles_3d_xarm,
    simulate_drake_rope_trial,
    simulate_rope_elastica,
    inverse_particle_model_constraints,
    animate_xarm_states,
)
from functools import cached_property


def main(args):
    # Load config
    learning_config = parse_yaml(
        args.config_file
    )
    data_root = get_flying_knot_data_dir()
    commands_dir = os.path.join(data_root, "commands")
    human_dir = os.path.join(data_root, "human")
    learning_dir = os.path.join(data_root, "learning")

    if args.start_from_run is not None:
        run_folder = os.path.join(learning_dir, args.start_from_run)
        trials = glob.glob(os.path.join(run_folder, "trial_*.pickle"))
        latest_trial_path = max(
            trials,
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
        )
        starting_trial = load_pickle(latest_trial_path)
        initial_command_name = os.path.join(commands_dir, starting_trial["command_name"], f"{starting_trial['command_name']}.pickle")

    else:
        if args.initial_command == "latest":
            latest_cmd_name = get_latest_trial_name(commands_dir)
            logging.info(f"Loading latest cmd: {latest_cmd_name}")
            initial_command_name = os.path.join(
                get_flying_knot_data_dir(),
                "commands",
                latest_cmd_name,
                f"{latest_cmd_name}.pickle",
            )
        else:
            initial_command_name = os.path.join(
                get_flying_knot_data_dir(),
                "commands",
                args.initial_command,
                f"{args.initial_command}.pickle",
            )
    if args.demo == "latest":
        learning_config["demo"] = get_latest_trial_name(human_dir)
        logging.info(f"Loading latest demo: {learning_config["demo"]}")
    else:
        learning_config["demo"] = args.demo

    learning_config["initial_command"] = initial_command_name
    env_config_file = learning_config["env_name"]
    if learning_config["env"] == "particle":
        env_config_path = f"config/simulation/{env_config_file}"
    elif learning_config["env"] == "real":
        env_config_path = f"config/hardware/{env_config_file}"
    env_config = parse_yaml(env_config_path)

    model_config_file = learning_config["model_config"]
    model_config_path = f"config/simulation/{model_config_file}"
    particle_model_config = parse_yaml(
        model_config_path   
    )

    # Setup data storage
    run_name = generate_trial_name()
    run_folder_path = os.path.join(args.folder_path, run_name)
    os.makedirs(run_folder_path, exist_ok=True)
    dict_to_yaml(os.path.join(run_folder_path, args.config_file.split("/")[-1]), learning_config)
    shutil.copy(env_config_path, run_folder_path)
    # shutil.copy(model_config_path, run_folder_path)

    # Load initial guess
    initial_command = load_pickle(initial_command_name)
    U_init = np.array(initial_command["control_points"])

    learning_state = LearningState(0, U_init, initial_command["end_time"])

    demo_data: HumanDemo = HumanDemo.load(
        human_dir, learning_config["demo"]
    )
    assert learning_state.total_time == demo_data.hand_trajectory.end_time()
    assert demo_data.demo_config["task"]["name"] == learning_config["task"]
    task = learning_config["task"]

    if demo_data.fixed_end_point is not None:
        particle_model_config["fixed_end_pose"] = demo_data.fixed_end_point

    particle_model_config_to_save = dict(particle_model_config)
    if isinstance(particle_model_config_to_save.get("fixed_end_pose"), RigidTransform):
        X_WF = particle_model_config_to_save["fixed_end_pose"]
        particle_model_config_to_save["fixed_end_pose"] = {
            "xyz": X_WF.translation().tolist(),
            "wxyz": X_WF.rotation().ToQuaternion().wxyz().tolist(),
        }

    dict_to_yaml(os.path.join(run_folder_path, model_config_file), particle_model_config_to_save)

    last_cp_time = max(cp.time_offset for cp in demo_data.critical_points.values())
    demo_frame_dt = (
        float(np.mean(np.diff(demo_data.frame_times)))
        if len(demo_data.frame_times) > 1
        else 0.0
    )
    trial_end_track_time = last_cp_time + 10 * demo_frame_dt

    model_params = make_particle_model_params(
        particle_model_config,
        last_cp_time,
        demo_data.handle_mocap_object,
    )

    rope_resample_interp = learning_config.get("rope_resample_interp", "linear")
    if rope_resample_interp not in ("linear", "cubic"):
        raise ValueError(
            f"rope_resample_interp must be 'linear' or 'cubic', got {rope_resample_interp!r}"
        )
    if demo_data.num_rope_markers != model_params.num_particles:
        logging.info(
            f"Resampling demo rope from {demo_data.num_rope_markers} to "
            f"{model_params.num_particles} markers ({rope_resample_interp})"
        )
        demo_data.unlabeled_markers = resample_rope_markers(
            demo_data.unlabeled_markers,
            model_params.num_particles,
            interpolation_type=rope_resample_interp,
            fingertip_positions=demo_data.get_goal_fingertip_points(),
            fixed_end_position=(
                model_params.fixed_end_pose.translation()
                if model_params.fixed_end
                else None
            ),
        )
        demo_data.num_rope_markers = model_params.num_particles
        demo_data.rope_trajectory = PiecewisePolynomial.FirstOrderHold(
            demo_data.frame_times, demo_data.unlabeled_markers.T
        )

    goal_rope_trajectory = demo_data.get_goal_rope_trajectory(
        model_params.fixed_end_pose if model_params.fixed_end else None
    )
        
    meshcat = Meshcat()

    # model_params.plant.SetPositions(
    #     model_params.plant_context, learning_state.bezier_command.value(0)
    # )
    # X_WT_0 = xarm_forward_kinematics(
    #     model_params.plant, model_params.plant_context, "tip_frame", pose_only=True
    # )
    # finger_tip_start_pos = X_WT_0.translation()
    # x0_ref = solve_for_initial_state(
    #     finger_tip_start_pos, demo_data.rope_trajectory.value(0), particle_model_config["l"], rope_end_points=model_params.fixed_end_pose
    # )

    while args.trials is None or learning_state.trial_num < args.trials:
        print(f"Trial {learning_state.trial_num}:")
        raw_trial_data_name = None
        # Save command
        command_name = generate_trial_name()
        print("Saving Command: ", command_name)
        command_save_data = {
            "end_time" : learning_state.total_time,
            "control_points" : learning_state.U_bar,
        }
        command_info = {
            "initial_command": args.initial_command,
            "trial_num":learning_state.trial_num,
        }

        os.makedirs(os.path.join(commands_dir, command_name), exist_ok=True)
        dict_to_yaml(
            os.path.join(commands_dir, command_name, "learning_info.yaml"),
            command_info,
        )
        save_pickle(
            command_save_data,
            os.path.join(commands_dir, command_name, f"{command_name}.pickle"),
        )

        raw_rope_marker_trajectory = None

        # Simulate or execute motion
        if learning_config["env"] == "particle":
            model_params.plant.SetPositions(
                model_params.plant_context, learning_state.bezier_command.value(0)
            )
            X_WT = xarm_forward_kinematics(
                model_params.plant, model_params.plant_context, "tip_frame", pose_only=True
            )

            x0 = solve_for_initial_state(
                X_WT.translation(),
                demo_data.rope_trajectory.value(0),
                particle_model_config["l"],
                rope_end_points=(
                    model_params.fixed_end_pose.translation()
                    if model_params.fixed_end
                    else None
                ),
            ).flatten()

            # HACKY SETTLE TIME THING
            settle_time_s = 1.5
            settle_steps = int(settle_time_s / model_params.dt) + 1
            q_settle = learning_state.bezier_command.value(0).flatten()
            v_settle = np.zeros(model_params.plant.num_velocities())
            u_settle = np.concatenate([q_settle, v_settle])
            if u_settle.shape[0] != model_params.nu:
                raise ValueError(
                    f"Expected settle command size {model_params.nu}, got {u_settle.shape[0]}"
                )

            settle_u = np.repeat(u_settle[None, :], settle_steps - 1, axis=0)
            original_num_steps = model_params.N
            model_params.N = settle_steps
            model_params.x0 = x0
            model_params.v0 = np.zeros(model_params.nq)
            _, settle_Z = dynamics_simulate(settle_u, model_params)
            model_params.N = original_num_steps
            x0 = settle_Z[-1, :model_params.nq].copy()

            particle_model_output = forward_particle_model(
                learning_state.bezier_command, model_params, x0
            )
            Z_trial = particle_model_output["Z"]

            # Make xarm and rope trajectories
            N_cmd = int(learning_state.bezier_command.end_time()/model_params.dt)+1
            t_u = np.linspace(0, learning_state.bezier_command.end_time(), N_cmd)
            angles = learning_state.bezier_command.vector_values(t_u).T
            # angles = particle_model_output["U"][:, :10]
            xarm_trajectory = XarmTrajectory(t_u, angles)
            finger_tip_pos = []
            for a in angles[:model_params.N]:
                model_params.plant.SetPositions(model_params.plant_context, a)
                finger_tip_pos.append(
                    (xarm_forward_kinematics(
                        model_params.plant,
                        model_params.plant_context,
                        "tip_frame",
                        pose_only=True,
                    )).translation()
                )
            finger_tip_pos = np.array(finger_tip_pos)
            rope_positions = np.c_[finger_tip_pos, particle_model_output["Z"][:, :model_params.nq]]
            if model_params.fixed_end:
                fixed_end = np.repeat(
                    model_params.fixed_end_pose.translation()[None, :],
                    rope_positions.shape[0],
                    axis=0,
                )
                rope_positions = np.c_[
                    rope_positions,
                    fixed_end,
                ]
            rope_trajectory = RopeTrajectory(particle_model_output["T"], rope_positions)
        elif learning_config["env"] == "real":
            while True:
                subprocess.run(
                    ["python", "main/run_trajectory.py", "-c", env_config_path],
                    check=True
                )
                raw_trial_data_name = get_latest_trial_name(
                    os.path.join(data_root, "hardware")
                )
                if not check_if_frame_drops(raw_trial_data_name): # TODO load start and end times from hardware config
                    logging.warning("Frame dropped during motion, repeating trial")
                    continue
                logging.info("Raw trial data: %s", raw_trial_data_name)
                trial_mocap_data : XarmRopeMocapData = XarmRopeMocapData.load_file(raw_trial_data_name, trial_end_track_time, env_config)
                if trial_mocap_data is None:
                    logging.warning("Parsing failed, repeating trial")
                    continue
                break

            rope_points = trial_mocap_data.rope_trajectory.rope_points # in base frame
            if "cleat" in task:
                X_xarm_to_base = trial_mocap_data.X_xarm_to_base
                rot = X_xarm_to_base.rotation().matrix()
                base_loc = np.r_[
                    X_xarm_to_base.translation(),
                    np.arctan2(rot[1, 0], rot[0, 0]),
                ]
                learning_state.U_bar[:4, :] = base_loc[:, None]
                X_offset = RigidTransform()
            else:
                base_loc = learning_state.U_bar[:4, 0]
                X_offset = translation_zrot_to_transform(learning_state.U_bar[:3, 0], learning_state.U_bar[3, 0])
            
            rope_points_transformed = transform_3d_points(rope_points, X_offset)

            angles = trial_mocap_data.xarm_trajectory.positions
            angles = np.c_[np.repeat(base_loc[None, :], angles.shape[0], axis=0), angles]
            xarm_trajectory = XarmTrajectory(trial_mocap_data.xarm_trajectory.times, angles)

            rope_times = trial_mocap_data.rope_trajectory.times
            xarm_times = np.clip(
                rope_times,
                xarm_trajectory.traj.start_time(),
                xarm_trajectory.traj.end_time(),
            )
            xarm_positions = xarm_trajectory.traj.vector_values(xarm_times).T
            finger_tip_pos = []
            for a in xarm_positions:
                model_params.plant.SetPositions(model_params.plant_context, a)
                finger_tip_pos.append(
                    (xarm_forward_kinematics(
                        model_params.plant,
                        model_params.plant_context,
                        "tip_frame",
                        pose_only=True,
                    )).translation()
                )
            finger_tip_pos = np.array(finger_tip_pos)

            raw_rope_marker_trajectory = RopeTrajectory(
                rope_times, rope_points_transformed.copy()
            )

            rope_points_transformed = resample_rope_markers(
                rope_points_transformed,
                model_params.num_particles,
                interpolation_type=rope_resample_interp,
                fingertip_positions=finger_tip_pos,
                fixed_end_position=(
                    model_params.fixed_end_pose.translation()
                    if model_params.fixed_end
                    else None
                ),
            )

            rope_positions = np.concatenate(
                [finger_tip_pos[:, None, :], rope_points_transformed], axis=1
            )
            if model_params.fixed_end:
                fixed_end = np.repeat(
                    model_params.fixed_end_pose.translation()[None, None, :],
                    rope_positions.shape[0],
                    axis=0,
                )
                rope_positions = np.concatenate([rope_positions, fixed_end], axis=1)
            rope_trajectory = RopeTrajectory(rope_times, rope_positions)

            rope_points_flat = rope_points_transformed.reshape((rope_points.shape[0], -1))
            marker_traj: PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(
                rope_times,
                rope_points_flat.T,
            )

            sim_times = np.linspace(0, (model_params.N - 1) * model_params.dt, model_params.N)
            sim_times_clipped = np.clip(sim_times, marker_traj.start_time(), marker_traj.end_time())
            rope_points_flat_sim = marker_traj.vector_values(sim_times_clipped).T
            rope_vels_sim = marker_traj.MakeDerivative(1).vector_values(sim_times_clipped).T

            Z_trial = np.c_[rope_points_flat_sim, rope_vels_sim, np.zeros((model_params.N, model_params.nc))]

            model_params.plant.SetPositions(model_params.plant_context, xarm_trajectory.positions[0])
            finger_tip_position_0 = (xarm_forward_kinematics(
                    model_params.plant,
                    model_params.plant_context,
                    "tip_frame",
                    pose_only=True,
                )).translation()

        # Forward Model
        if learning_config["env"] != "particle":
            x0 = solve_for_initial_state(
                finger_tip_position_0,
                Z_trial[0, :model_params.nq],
                particle_model_config["l"],
                rope_end_points=(
                    model_params.fixed_end_pose.translation()
                    if model_params.fixed_end
                    else None
                ),
            )
            particle_model_output = forward_particle_model(
                learning_state.bezier_command, model_params, x0, xarm_trajectory.traj
            )

        #### TEMP VISUALIZATION START
        N_total = int(learning_state.total_time/model_params.dt) + 1
        Z_mocap_temp = np.c_[
            demo_data.rope_trajectory.vector_values(
                np.linspace(0, learning_state.total_time, N_total)
            ).T,
            demo_data.rope_trajectory.MakeDerivative(1).vector_values(
                np.linspace(0, learning_state.total_time, N_total)
            ).T,
            np.zeros((N_total, model_params.nc)),
        ]
        animation = MeshcatAnimation(1 / model_params.dt)
        animate_particles_3d(
            meshcat,
            animation,
            None,
            Z_mocap_temp,
            model_params,
            prefix="b",
            rgba=Rgba(0, 1, 0, 0.5),
        )
        animate_particles_3d(
            meshcat,
            animation,
            None,
            Z_trial,
            model_params,
            prefix="c",
            rgba=Rgba(0, 0, 1, 0.5),
        )
        animate_particles_3d_xarm(
            meshcat,
            animation,
            particle_model_output["U"],
            particle_model_output["Z"],
            model_params,
        )
        animate_xarm_states(
            meshcat,
            animation,
            model_params.plant,
            xarm_trajectory.traj.vector_values(
                np.linspace(0, xarm_trajectory.traj.end_time(), int(xarm_trajectory.traj.end_time()/model_params.dt)+1)
            ).T,
        )

        meshcat.SetAnimation(animation)
        ### TEMP VISUALIZATION END

        # breakpoint()
        # Inverse Model
        print("Inverse...")
        delta_u, delta_z, J, J_solve = inverse_particle_model_constraints(
            learning_state.bezier_command,
            particle_model_output,
            Z_trial,
            demo_data,
            model_params,
            learning_config,
        )
        logging.debug(f"Delta U: {delta_u}")

        # Save data
        trial_save_path = os.path.join(run_folder_path, f"trial_{learning_state.trial_num}.pickle")
        particle_model_output.pop("params")
        trial_save_data = {
            "learning_state": learning_state,
            "rope_trajectory": rope_trajectory,
            "goal_rope_trajectory" : goal_rope_trajectory,
            "raw_rope_marker_trajectory": raw_rope_marker_trajectory,
            "arm_trajectory": xarm_trajectory,
            "model_data": particle_model_output,
            "delta_u" : delta_u,
            "cost" : J,
            "cost_solve": J_solve,
            "command_name" : command_name,
            "initial_command": initial_command_name,
            "raw_trial_data_name": raw_trial_data_name,
            "starting_from_run": args.start_from_run,
        }
        save_pickle(trial_save_data, trial_save_path)

        # Update learning state
        learning_state.U_bar += delta_u
        learning_state.trial_num += 1


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--initial_command",
        default="latest",
    )
    parser.add_argument(
        "-d",
        "--demo",
        default="latest",
    )
    parser.add_argument(
        "-s",
        "--start_from_run",
    )

    parser.add_argument(
        "-c", "--config_file", default="config/learning/particle_rope_learning.yaml"
    )

    parser.add_argument(
        "-f",
        "--folder_path",
        default=os.path.join(get_flying_knot_data_dir(), "learning"),
    )
    
    parser.add_argument("-t", "--trials", type=int, default=None)

    parser.add_argument("-l", "--load_run")

    # try:
    main(parser.parse_args())
    # except Exception as e:
    #     print(e)
    #     breakpoint()
