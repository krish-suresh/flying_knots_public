"""Hardware execution of a learned Bézier trajectory on the xArm7.

Reference only — this script is included to document the hardware-execution
stage of the learning loop. It requires a connected xArm7 + Vicon system
and is therefore not runnable as-is. See ``docs/architecture.md`` for how
it fits into the overall workflow.
"""
from pydrake.all import *
from xarm.wrapper import XArmAPI
import os
from common import *
import pickle
import time
from datetime import datetime
from xarm7 import *
import argparse
import shutil
from pydrake.polynomial import *

def main(args):
    config : dict = parse_yaml(args.config)
    folder_path = args.save_path

    # Load Command
    if args.xarm_command == "latest":
        command_file_name = get_latest_trial_name(args.command_file_path)
    else:
        command_file_name = args.xarm_command
    cmd_data = load_pickle(os.path.join(args.command_file_path, command_file_name, f"{command_file_name}.pickle"))

    # if type(cmd_data["command"]) == tuple:
    #     xarm_trajectory = PiecewisePolynomial.FirstOrderHold(cmd_data["command"][0], cmd_data["command"][1].T)
    # else:
    #     xarm_trajectory : Trajectory = cmd_data["command"]
    
    # xarm_trajectory : PiecewisePolynomial = PiecewisePolynomial(
    #     [np.array([Polynomial(c) for c in coeff]) for coeff in cmd_data["coeffs"]],
    #     cmd_data["breaks"],
    # )
    
    xarm_trajectory : BezierCurve = make_bezier(cmd_data["control_points"][4:, :], cmd_data["end_time"])

    # Run flying knot traj
    # with open(
    #     os.path.join(
    #         get_flying_knot_data_dir(), "8_28_25", "mocap", "20250828-130916.pickle"
    #     ),
    #     "rb",
    # ) as f:
    #     xarm_trajectory : BsplineTrajectory = pickle.load(f)["traj"].CopyBlock(3, 0, 7, 1)
        

    tracker = None
    xarm_raw_data: list[bytes] = []
    arm = None
    tracker_fps = None
    if "vicon" in config:
        tracking_objects = config["vicon"]["tracking_objects"]
        tracker : MultiObjectTracker = setup_vicon(config["vicon"])
        tracker_fps = tracker.get_framerate()
    
    robot_ip = config["xarm"]["xarm_ip"]
    arm = XArmAPI(robot_ip)
    arm.motion_enable()
    arm.set_mode(0)
    arm.set_state(0)
    cmd_start_buffer_time = config["xarm"].get("command_start_buffer_time", 0)
    cmd_end_buffer_time = config["xarm"].get("command_end_buffer_time", 0)

    xarm_raw_data_stop_event = threading.Event()
    xarm_raw_data_thread = threading.Thread(target=capture_xarm_raw_data, args=(xarm_raw_data_stop_event, xarm_raw_data, False, False, robot_ip))


    total_execution_time = xarm_trajectory.end_time() + cmd_start_buffer_time + cmd_end_buffer_time
    xarm_move_to_trajectory_start(arm, xarm_trajectory)
    xarm_send_trajectory(arm, xarm_trajectory, "knot", robot_ip, cmd_start_buffer_time, cmd_end_buffer_time)

    if not config["xarm"].get("autostart", False):
        input("Enter to start capture")
    xarm_raw_data_thread.start()
    arm.playback_trajectory(wait=False)
    print("Starting data capture")

    trial_name = generate_trial_name()
    start_time = time.perf_counter()

    data = []
    last_frame_number = None
    while time.perf_counter()-start_time < total_execution_time:
        if tracker is not None:
            if not (frame := tracker.get_positions()):
                logging.debug("No frame received")
                continue

            latency, frame_number, poses = frame
            # print(poses)
            # if not all(poses):
            #     logging.debug(f"Lost track of object {poses}")
            #     continue
            if last_frame_number and frame_number-last_frame_number > 1:
                logging.warning(f"\t {frame_number-last_frame_number} frames skipped at frame {frame_number}")
            last_frame_number = frame_number
                

            unlabeled_markers = tracker.get_all_unlabeled_marker_positions()
            labeled_markers = tracker.get_all_labeled_marker_positions()

            data.append(
                dict(zip(tracking_objects, poses))
                | {
                    "frame_number": frame_number,
                    "latency": latency,
                    "unlabeled_markers": unlabeled_markers,
                    "labeled_markers": labeled_markers,
                }
            )

    logging.info(f"Ending data capture: {len(data)}")
    for t, d in enumerate(data):
        if len(d["unlabeled_markers"]) < config["rope"]["num_rope_markers"]:
            logging.warning(f"{t}: Only {len(d["unlabeled_markers"])} markers, excepted >= {config["rope"]["num_rope_markers"]}")

    xarm_raw_data_stop_event.set()
    xarm_raw_data_thread.join()
    

    save_data = {
        "fps": tracker_fps,
        "data": data,
        "command_name": command_file_name,
        "xarm_raw_data": xarm_raw_data,
        
    }
    
    if not args.dont_save:
        os.makedirs(os.path.join(folder_path, trial_name), exist_ok=True)
        hardware_config_path = os.path.join(folder_path, trial_name, f"{trial_name}.yaml")
        shutil.copy(args.config, hardware_config_path)
        hardware_config = parse_yaml(hardware_config_path)
        hardware_config["command_name"] = command_file_name
        hardware_config["host_computer"] = get_platform()
        dict_to_yaml(hardware_config_path, hardware_config)

        # shutil.copy(config["xarm"]["gcode_file_path"], os.path.join(folder_path, trial_name, config["xarm"]["gcode_file_path"].split("/")[-1]))
        with open(os.path.join(folder_path, trial_name, f"{trial_name}.pickle"), "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    if args.zero_after:
        arm.set_mode(0)
        arm.set_state(0)
        arm.set_servo_angle(angle=[0, 0, 0, 0, 0, 0, 0], speed=50, mvacc=500, wait=True, is_radian=True)
    else:
        xarm_move_to_trajectory_start(arm, xarm_trajectory)
    arm.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to capture human rope data")
    parser.add_argument(
        "-f",
        "--command_file_path",
        default=os.path.join(get_flying_knot_data_dir(), "commands"),
    )
    parser.add_argument(
        "-x",
        "--xarm_command",
        default="latest",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/hardware/xarm_vicon_cleat.yaml",
        help="Path to the hardware configuration file.",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    parser.add_argument(
        "-s",
        "--save_path",
        default=os.path.join(get_flying_knot_data_dir(), "hardware")
    )
    parser.add_argument(
        "-d",
        "--dont_save",
        action="store_true"
    )
    parser.add_argument(
        "-z",
        "--zero_after",
        action="store_true",
        help="After the trajectory, move the arm to the zero joint configuration instead of the trajectory start.",
    )
    main(parser.parse_args())
