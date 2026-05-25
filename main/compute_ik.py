import argparse
import logging
import os
import pickle

import numpy as np
import viser
from pydrake.all import PiecewisePose, RigidTransform

from common.config import dict_to_yaml, get_flying_knot_data_dir, parse_yaml
from common.data import HumanDemo, generate_trial_name
from common.visualize import ViserAnimationRealtime
from xarm7 import add_xarm_visual, get_xarm_visual_traj, xarm_ik_3d_bezier, xarm_plant_3d


def _save_command(
    save_cmd_folder: str,
    config_file: str,
    ik_config: dict,
    arm_trajectory,
) -> None:
    trial_name = generate_trial_name()
    print("Saving Command: ", trial_name)
    save_data = {
        "end_time": arm_trajectory.end_time(),
        "control_points": arm_trajectory.control_points(),
    }

    os.makedirs(os.path.join(save_cmd_folder, trial_name), exist_ok=True)
    dict_to_yaml(
        os.path.join(save_cmd_folder, trial_name, os.path.basename(config_file)),
        ik_config,
    )
    with open(os.path.join(save_cmd_folder, trial_name, f"{trial_name}.pickle"), "wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _make_pose_traj(sample_times: np.ndarray, positions: list[np.ndarray]) -> PiecewisePose:
    poses = [RigidTransform(p) for p in positions]
    if sample_times.size <= 1:
        return PiecewisePose.MakeLinear(
            np.array([0.0, 1e-3]),
            [poses[0], poses[0]],
        )
    return PiecewisePose.MakeLinear(sample_times, poses)


def main(args):
    demo_data: HumanDemo = HumanDemo.load(args.folder_path, args.trial_name)
    assert demo_data is not None

    ik_config = parse_yaml(args.config_file)
    arm_type = ik_config["arm_type"]
    if arm_type != "xarm7":
        raise ValueError(
            f"compute_ik.py visualization refactor currently supports arm_type='xarm7' only, got '{arm_type}'"
        )

    target_pose_traj = demo_data.hand_trajectory
    logging.info("Arm type: %s", arm_type)

    plant = xarm_plant_3d(
        handle_frame_body=demo_data.handle_mocap_object.handle_attachment_frame_to_tip_frame
    )
    plant_context = plant.CreateDefaultContext()
    X_L7T = plant.CalcRelativeTransform(
        plant_context,
        plant.GetFrameByName("link7"),
        plant.GetFrameByName("tip_frame"),
    )

    arm_trajectory = xarm_ik_3d_bezier(
        target_pose_traj,
        plant,
        ik_config,
        end_point=demo_data.fixed_end_point,
    )
    _save_command(args.save_cmd_folder, args.config_file, ik_config, arm_trajectory)

    server = viser.ViserServer(port=8081)
    arm_visual = add_xarm_visual(server, handle=demo_data.handle_mocap_object)
    server.scene.add_box(
        "xarm/link_base/floor",
        color=(200, 200, 200),
        dimensions=(10, 10, 0.001),
        position=plant.CalcRelativeTransform(
            plant_context,
            plant.GetFrameByName("link_base"),
            plant.GetFrameByName("floor"),
        ).translation(),
        visible=False,
    )

    target_hand_frame = server.scene.add_frame(
        "target_hand_frame",
        True,
        axes_length=0.1,
        axes_radius=0.01,
        origin_color=(0, 236, 236),
    )

    server.scene.add_frame(
        "xarm/link7/tip",
        True,
        axes_length=0.1,
        axes_radius=0.01,
        position=X_L7T.translation(),
        wxyz=X_L7T.rotation().ToQuaternion().wxyz(),
    )

    labeled_marker_spheres: dict[str, list[viser.MeshHandle]] = {}
    labeled_marker_trajs: list[tuple[viser.MeshHandle, PiecewisePose]] = []
    if demo_data.labeled_markers:
        marker_times = np.asarray(demo_data.frame_times, dtype=float).reshape(-1)
        marker_frames = demo_data.labeled_markers
        if marker_times.size != len(marker_frames):
            marker_times = np.linspace(0.0, target_pose_traj.end_time(), len(marker_frames))

        for object_name, markers in marker_frames[0].items():
            spheres = [
                server.scene.add_icosphere(
                    f"labeled_markers/{object_name}/{i}",
                    0.01,
                    (0, 255, 0),
                )
                for i in range(len(markers))
            ]
            labeled_marker_spheres[object_name] = spheres

            for i, sphere in enumerate(spheres):
                marker_positions: list[np.ndarray] = []
                last_position: np.ndarray | None = None
                for frame in marker_frames:
                    object_markers = frame.get(object_name, [])
                    if i < len(object_markers):
                        last_position = np.asarray(object_markers[i], dtype=float)
                    elif last_position is None:
                        last_position = np.zeros(3)
                    marker_positions.append(last_position)

                marker_traj = _make_pose_traj(marker_times, marker_positions)
                labeled_marker_trajs.append((sphere, marker_traj))

    animation = ViserAnimationRealtime(server, visualization_fps=60)

    gui_show_labeled_markers = server.gui.add_checkbox(
        "Show Labeled Markers",
        True,
        disabled=(len(labeled_marker_spheres) == 0),
    )
    gui_ik_button = server.gui.add_button("Compute IK", icon=viser.Icon.ROBOT)
    gui_cmd_save_button = server.gui.add_button(
        "Save Command", icon=viser.Icon.DEVICE_FLOPPY
    )

    def _set_labeled_markers_visible(visible: bool) -> None:
        for object_markers in labeled_marker_spheres.values():
            for marker in object_markers:
                marker.visible = visible

    def _rebuild_animation() -> None:
        animation.clear()
        animation.add_animated_object(target_hand_frame, target_pose_traj)
        for mesh, traj in get_xarm_visual_traj(arm_visual, arm_trajectory):
            animation.add_animated_object(mesh, traj)
        for marker_handle, marker_traj in labeled_marker_trajs:
            animation.add_animated_object(marker_handle, marker_traj)
        _set_labeled_markers_visible(gui_show_labeled_markers.value)

    @gui_ik_button.on_click
    def _(_) -> None:
        nonlocal arm_trajectory
        nonlocal ik_config
        ik_config = parse_yaml(args.config_file)
        arm_trajectory = xarm_ik_3d_bezier(
            demo_data.hand_trajectory,
            plant,
            ik_config,
            end_point=demo_data.fixed_end_point,
        )
        _rebuild_animation()
        animation.reset()
        animation.play()

    @gui_cmd_save_button.on_click
    def _(_) -> None:
        _save_command(args.save_cmd_folder, args.config_file, ik_config, arm_trajectory)

    @gui_show_labeled_markers.on_update
    def _(_) -> None:
        _set_labeled_markers_visible(gui_show_labeled_markers.value)

    _rebuild_animation()
    animation.reset()
    animation.play()
    server.sleep_forever()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Script to test xarm ik")
    parser.add_argument("-t", "--trial_name", default="latest")

    parser.add_argument(
        "-f",
        "--folder_path",
        default=os.path.join(get_flying_knot_data_dir(), "human"),
    )

    parser.add_argument("-c", "--config_file", default="config/ik/xarm_ik_params.yaml")

    parser.add_argument(
        "-s",
        "--save_cmd_folder",
        default=os.path.join(get_flying_knot_data_dir(), "commands"),
    )
    main(parser.parse_args())
