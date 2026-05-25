"""Human-demonstration motion-capture acquisition.

Reference only — this script is included to document the data-collection
pipeline described in the paper. It requires a connected Vicon motion-capture
system and is therefore not runnable as-is. See ``docs/architecture.md`` for
how it fits into the overall workflow.
"""
import argparse
from common.mocap import MultiObjectTracker, setup_vicon
from common.config import parse_yaml, get_flying_knot_data_dir
from common.data import generate_trial_name
import pickle
import os
import yaml
import logging


def main(args):
    config = parse_yaml(args.config)
    folder_path = args.save_path
    trial_name = generate_trial_name()

    tracking_objects = config["vicon"]["tracking_objects"]
    tracker : MultiObjectTracker = setup_vicon(config["vicon"])
    tracker.vicon_client.get_frame()
    assert tracker.get_framerate() == config["vicon"]["vicon_fps"]
    last_frame_number = None

    data = []
    input("Press Enter to Start Capture...")
    try:
        while True:
            if not (frame := tracker.get_positions()):
                logging.debug("No frame received")
                continue

            latency, frame_number, poses = frame
            if not all(poses):
                logging.debug(f"Lost track of object {poses}")
                continue
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

    except KeyboardInterrupt:
        logging.info(f"Ending data capture: {len(data)}")

        save_data = {
            "fps": tracker.get_framerate(),
            "data": data,
        }

        os.makedirs(os.path.join(folder_path, trial_name), exist_ok=True)
        with open(os.path.join(folder_path, trial_name, f"{trial_name}.yaml"), "w") as f:
            yaml.safe_dump(dict(config), f, sort_keys=False)
        with open(os.path.join(folder_path, trial_name, f"{trial_name}.pickle"), "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to capture human rope data")
    parser.add_argument(
        "-c",
        "--config",
        default="config/demo/flying_knot_human.yaml",
        help="Path to the capture configuration file.",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    parser.add_argument(
        "-s",
        "--save_path",
        default=os.path.join(get_flying_knot_data_dir(), "human")
    )
    main(parser.parse_args())
