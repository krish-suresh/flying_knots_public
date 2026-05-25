import viser
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import time
from common.config import (
    get_flying_knot_data_dir,
    get_latest_trial_name,
    load_pickle,
    parse_yaml,
)
from common.data import (
    generate_trial_name,
    MocapHandleObject,
    MocapBaseObject,
    transform_from_mocap_data,
)
from pydrake.all import RigidTransform
import pickle
import os
import shutil
import trimesh
from scipy.interpolate import CubicSpline
from common.tracker import (
    order_initial_frame,
    remove_unlabeled_near_labeled,
    track_markers,
)


def main(args):
    # Load hardware capture trial
    if args.trial_name == "latest":
        trial_name = get_latest_trial_name(args.folder_path)
    else:
        trial_name = args.trial_name

    trial_folder = os.path.join(args.folder_path, trial_name)
    if not os.path.exists(trial_folder):
        logging.error(f"Failed to find trial at {trial_folder}")
        return

    trial_data_path = os.path.join(trial_folder, f"{trial_name}.pickle")
    if not os.path.exists(trial_data_path):
        logging.error(f"Failed to find trial data at {trial_data_path}")
        return

    trial_config_path = os.path.join(trial_folder, f"{trial_name}.yaml")
    if not os.path.exists(trial_config_path):
        logging.error(f"Failed to find trial config at {trial_config_path}")
        return

    trial_data = load_pickle(trial_data_path)
    logging.info(f"Loaded trial {trial_name}")
    trial_config = parse_yaml(trial_config_path)
    annotation_save_path = os.path.join(trial_folder, f"{trial_name}-annotation.pickle")
    loaded_annotation_data = None
    if os.path.exists(annotation_save_path):
        loaded_annotation_data = load_pickle(annotation_save_path)
    logging.debug(trial_config)

    def save_annotation_data(save_path):
        annotation_data = {
            "unlabeled_idxs": unlabeled_idxs,
            "ordered_frames": ordered_frames,
            "start_idx": gui_start_frame_num.value,
            "end_track_idx": gui_end_track_frame_num.value,
            "end_idx": gui_end_frame_num.value,
        }
        with open(save_path, "wb") as f:
            pickle.dump(annotation_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"Saved annotation to {annotation_save_path}")

    num_frames = len(trial_data["data"])
    unlabeled_idxs = [
        np.arange(len(trial_data["data"][i]["unlabeled_markers"]))
        .astype(np.int32)
        .tolist()
        for i in range(num_frames)
    ]

    ordered_frames = [False] * num_frames
    pruned_orderings_per_frame: list[list[list[int]]] = [[] for _ in range(num_frames)]

    frame_numbers = np.array(
        [d.get("frame_number", i) for i, d in enumerate(trial_data["data"])],
        dtype=float,
    )
    vicon_fps = trial_config["vicon"].get("vicon_fps", 200.0)
    vicon_frame_times = (frame_numbers - frame_numbers[0]) / vicon_fps

    vicon_handle_frame_name = trial_config["vicon"]["handle_frame"]
    handle_mocap_object: MocapHandleObject = MocapHandleObject.from_yaml(
        f"config/mocap_objects/{vicon_handle_frame_name}.yaml"
    )
    rope_config = trial_config["rope"]
    num_rope_markers = rope_config["num_rope_markers"]
    rope_length = rope_config["rope_length"]
    marker_spacing_distance = rope_length / num_rope_markers

    # Visualize raw mocap
    server = viser.ViserServer(port=8081)

    refresh = False

    with server.gui.add_folder("Playback"):
        gui_frame_slider = server.gui.add_slider(
            "Current Frame", 0, num_frames - 1, 1, 0
        )
        gui_frame_step_buttons = server.gui.add_button_group(
            "Step Frames", ["<<", "<", ">", ">>"]
        )
        gui_frame_jump_buttons = server.gui.add_button_group(
            "Jump to Frame", ["Start", "End Track", "End"], disabled=False
        )
        gui_play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        gui_pause_button = server.gui.add_button(
            "Pause", icon=viser.Icon.PLAYER_PAUSE_FILLED
        )
        gui_play_speed = server.gui.add_number("Speed", 1.0, step=0.01)

    with server.gui.add_folder("Display", expand_by_default=False):
        gui_show_labeled_markers = server.gui.add_checkbox("Show Labeled Markers", True)
        gui_show_unordered_markers = server.gui.add_checkbox(
            "Show Unordered Markers", True
        )
        gui_show_pruned_branches = server.gui.add_checkbox(
            "Show Pruned Branches", True
        )
        gui_show_spline = server.gui.add_checkbox("Show Spline Fit", False)
        gui_show_handle_mesh = server.gui.add_checkbox("Show Handle Mesh", True)

    with server.gui.add_folder("Tracker"):
        gui_load_button = server.gui.add_button(
            "Load", icon=viser.Icon.FOLDER, disabled=loaded_annotation_data is None
        )
        gui_start_frame_button = server.gui.add_button(
            "Set Start Frame", hint="Click to set initial frame for tracking."
        )
        gui_end_track_frame_button = server.gui.add_button(
            "Set End Tracking Frame",
            hint="Click to set end tracking frame.",
            disabled=False,
        )
        gui_end_frame_button = server.gui.add_button(
            "Set End Frame", hint="Click to set final frame.", disabled=False
        )
        gui_start_frame_num = server.gui.add_number("Start Frame", 0, disabled=True)
        gui_end_track_frame_num = server.gui.add_number(
            "End Tracking Frame", 0, disabled=True
        )
        gui_end_frame_num = server.gui.add_number(
            "End Frame", num_frames - 1, disabled=True
        )
        gui_auto_label_button = server.gui.add_button(
            "Auto Label Markers", disabled=False, icon=viser.Icon.ROBOT
        )
        gui_first_failed_frame_num = server.gui.add_number(
            "First Failed Frame", -1, disabled=True
        )
        gui_jump_to_failure_button = server.gui.add_button(
            "Jump to First Failure",
            hint="Jump to the first frame where tracking failed.",
            disabled=True,
        )
        gui_save_button = server.gui.add_button(
            "Save", icon=viser.Icon.DEVICE_FLOPPY, disabled=False
        )

    @gui_load_button.on_click
    def _(_) -> None:
        nonlocal refresh
        logging.info(f"Loaded data from {annotation_save_path}")
        nonlocal unlabeled_idxs
        unlabeled_idxs = loaded_annotation_data["unlabeled_idxs"]
        ordered_frames[:] = loaded_annotation_data["ordered_frames"]
        gui_start_frame_num.value = loaded_annotation_data["start_idx"]
        gui_end_track_frame_num.value = loaded_annotation_data["end_track_idx"]
        gui_end_frame_num.value = loaded_annotation_data["end_idx"]

        gui_end_track_frame_button.disabled = False
        gui_end_frame_button.disabled = False
        gui_auto_label_button.disabled = False
        gui_save_button.disabled = False

        refresh = True

    @gui_save_button.on_click
    def _(_) -> None:
        save_annotation_data(annotation_save_path)

    @gui_play_button.on_click
    def _(_) -> None:
        gui_play_button.icon = viser.Icon.PLAYER_PLAY_FILLED
        gui_pause_button.icon = viser.Icon.PLAYER_PAUSE

    @gui_pause_button.on_click
    def _(_) -> None:
        gui_play_button.icon = viser.Icon.PLAYER_PLAY
        gui_pause_button.icon = viser.Icon.PLAYER_PAUSE_FILLED

    @gui_start_frame_button.on_click
    def _(_) -> None:
        nonlocal refresh
        logging.info(f"Start set at {gui_frame_slider.value}")
        gui_start_frame_num.value = gui_frame_slider.value
        refresh = True

    @gui_show_labeled_markers.on_update
    def _(_) -> None:
        nonlocal refresh
        refresh = True

    @gui_show_unordered_markers.on_update
    def _(_) -> None:
        nonlocal refresh
        refresh = True

    @gui_show_pruned_branches.on_update
    def _(_) -> None:
        nonlocal refresh
        refresh = True

    @gui_show_spline.on_update
    def _(_) -> None:
        nonlocal refresh
        refresh = True

    @gui_end_track_frame_button.on_click
    def _(_) -> None:
        nonlocal refresh
        logging.info(f"End tracking set at {gui_frame_slider.value}")
        gui_end_track_frame_num.value = gui_frame_slider.value
        refresh = True

        gui_auto_label_button.disabled = False

    @gui_end_frame_button.on_click
    def _(_) -> None:
        nonlocal refresh
        logging.info(f"End set at {gui_frame_slider.value}")
        gui_end_frame_num.value = gui_frame_slider.value
        refresh = True

    @gui_frame_jump_buttons.on_click
    def _(_) -> None:
        match gui_frame_jump_buttons.value:
            case "Start":
                gui_frame_slider.value = gui_start_frame_num.value
            case "End Track":
                gui_frame_slider.value = gui_end_track_frame_num.value
            case "End":
                gui_frame_slider.value = gui_end_frame_num.value

    @gui_frame_step_buttons.on_click
    def _(_) -> None:
        match gui_frame_step_buttons.value:
            case "<<":
                gui_frame_slider.value = max(
                    gui_frame_slider.min, gui_frame_slider.value - 5
                )
            case "<":
                gui_frame_slider.value = max(
                    gui_frame_slider.min, gui_frame_slider.value - 1
                )
            case ">>":
                gui_frame_slider.value = min(
                    gui_frame_slider.max, gui_frame_slider.value + 5
                )
            case ">":
                gui_frame_slider.value = min(
                    gui_frame_slider.max, gui_frame_slider.value + 1
                )

    @gui_jump_to_failure_button.on_click
    def _(_) -> None:
        if gui_first_failed_frame_num.value >= 0:
            gui_frame_slider.value = int(gui_first_failed_frame_num.value)

    @gui_auto_label_button.on_click
    def _(_) -> None:
        logging.info("Autolabeling markers")
        nonlocal refresh
        start_idx = gui_start_frame_num.value
        end_idx = gui_end_track_frame_num.value

        frames_numbers = [d["frame_number"] for d in trial_data["data"]]
        for i in range(start_idx, end_idx):
            if frames_numbers[i] - frames_numbers[i - 1] != 1:
                logging.error(
                    f"mocap: {i} \t Num frame skips: {frames_numbers[i] - frames_numbers[i-1]}"
                )
                return

        for t in range(start_idx, start_idx + 3):
            unlabeled_idxs[t] = remove_unlabeled_near_labeled(
                candidate_idxs=unlabeled_idxs[t],
                frame_data=trial_data["data"][t],
            )
        raw_marker_positions_per_frame = [
            np.array(d["unlabeled_markers"]) / 1000.0 for d in trial_data["data"]
        ]
        for offset in (0, 1):
            t0 = start_idx + offset
            seed = (
                transform_from_mocap_data(trial_data["data"][t0][vicon_handle_frame_name])
                @ handle_mocap_object.mocap_frame_to_tip_frame
            ).translation()
            unlabeled_idxs[t0] = order_initial_frame(
                raw_marker_positions=raw_marker_positions_per_frame[t0],
                candidate_idxs=unlabeled_idxs[t0],
                seed_position=seed,
                num_rope_markers=num_rope_markers,
            )
            ordered_frames[t0] = True

        result = track_markers(
            raw_marker_positions_per_frame=raw_marker_positions_per_frame,
            candidate_idxs_per_frame=unlabeled_idxs,
            start_idx=start_idx,
            end_idx=end_idx,
            num_rope_markers=num_rope_markers,
            marker_spacing_distance=marker_spacing_distance,
        )

        pruned_orderings_per_frame[:] = result.pruned_orderings_per_frame
        for t in range(num_frames):
            if result.ordered_frames[t]:
                unlabeled_idxs[t] = result.candidate_idxs_per_frame[t]
                ordered_frames[t] = True

        first_failed = -1
        for t in range(start_idx, end_idx):
            if not result.ordered_frames[t]:
                first_failed = t
                break
        gui_first_failed_frame_num.value = first_failed
        gui_jump_to_failure_button.disabled = first_failed < 0

        if not result.success:
            logging.warning(
                f"Tracking failed. First failed frame: {first_failed}"
            )
            gui_frame_slider.value = first_failed if first_failed >= 0 else start_idx
            refresh = True
            return

        logging.info("Tracking succeeded across full range.")
        gui_save_button.disabled = False
        gui_frame_slider.value = start_idx
        refresh = True

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        logging.warning("Cam pos loader not setup")

    server.scene.world_axes.visible = True
    labeled_marker_spheres = {}
    labeled_object_frames = {}
    for object_name, markers in trial_data["data"][0]["labeled_markers"].items():
        labeled_marker_spheres[object_name] = [
            server.scene.add_icosphere(f"{object_name}/{i}", 0.01, (0, 255, 0))
            for i in range(len(markers))
        ]
        mocap_object: MocapBaseObject = MocapBaseObject.from_yaml(
            f"config/mocap_objects/{object_name}.yaml"
        )

        labeled_object_frames[object_name] = server.scene.add_frame(
            f"{object_name}_frame", False
        )
        server.scene.add_frame(
            f"{object_name}_frame/frame",
            False,
            axes_length=0.1,
            axes_radius=0.01,
            position=mocap_object.mocap_frame_to_body_frame.translation(),
            wxyz=mocap_object.mocap_frame_to_body_frame.rotation().ToQuaternion().wxyz(),
        )

    X_mocap_cad: RigidTransform = (
        handle_mocap_object.mocap_frame_to_tip_frame
        @ handle_mocap_object.handle_attachment_frame_to_tip_frame.inverse()
        @ handle_mocap_object.cad_frame_to_handle_attachment_frame
    )
    handle_mesh_data = trimesh.load_mesh(
        f"models/handles/meshes/{handle_mocap_object.name}/{handle_mocap_object.name}.obj"
    )
    handle_mesh = server.scene.add_mesh_simple(
        "handle_mesh",
        handle_mesh_data.vertices,
        handle_mesh_data.faces,
        color=(0, 255, 0),
    )

    unlabeled_marker_spheres = [
        server.scene.add_icosphere(
            f"unlabeled_markers/{i}", 0.01, (0, 0, 0), visible=False
        )
        for i in range(50)
    ]

    unordered_marker_spheres = [
        server.scene.add_icosphere(
            f"unordered_markers/{i}",
            0.01,
            (0, 100, 255),
            opacity=0.3,
            visible=False,
        )
        for i in range(50)
    ]

    ordered_marker_lines = server.scene.add_line_segments(
        "ordered_marker_lines",
        points=np.zeros((1, 2, 3)),
        colors=np.zeros((1, 2, 3), dtype=np.uint8),
        line_width=5.0,
        visible=False,
    )

    pruned_branch_lines = server.scene.add_line_segments(
        "pruned_branch_lines",
        points=np.zeros((1, 2, 3)),
        colors=np.zeros((1, 2, 3), dtype=np.uint8),
        line_width=5.0,
        visible=False,
    )

    last_frame_idx = None
    while True:
        frame_idx = gui_frame_slider.value
        playing = gui_play_button.icon == viser.Icon.PLAYER_PLAY_FILLED
        if not playing and frame_idx == last_frame_idx and not refresh:
            continue
        refresh = False
        last_frame_idx = frame_idx
        for object_name, markers in trial_data["data"][frame_idx][
            "labeled_markers"
        ].items():
            labeled_object_frames[object_name].position = (
                np.array(trial_data["data"][frame_idx][object_name][2:5]) / 1000
            )
            labeled_object_frames[object_name].wxyz = np.array(
                trial_data["data"][frame_idx][object_name][5:]
            )
            for i in range(len(markers)):
                labeled_marker_spheres[object_name][
                    i
                ].visible = gui_show_labeled_markers.value
                labeled_marker_spheres[object_name][i].position = (
                    np.array(markers[i]) / 1000
                )

        X_W_tip: RigidTransform = transform_from_mocap_data(
            trial_data["data"][frame_idx][handle_mocap_object.name]
        ) @ handle_mocap_object.mocap_frame_to_tip_frame

        X_W_cad: RigidTransform = transform_from_mocap_data(
            trial_data["data"][frame_idx][handle_mocap_object.name]
        ) @ X_mocap_cad
        handle_mesh.position = X_W_cad.translation()
        handle_mesh.wxyz = X_W_cad.rotation().ToQuaternion().wxyz()
        handle_mesh.visible = gui_show_handle_mesh.value

        all_unlabeled = (
            np.array(trial_data["data"][frame_idx]["unlabeled_markers"]) / 1000.0
        )
        unlabeled_markers = all_unlabeled[unlabeled_idxs[frame_idx]]
        unordered_mask = np.ones(all_unlabeled.shape[0], dtype=bool)
        unordered_mask[unlabeled_idxs[frame_idx]] = False
        unordered_markers = all_unlabeled[unordered_mask]
        cmap = plt.get_cmap("rainbow", unlabeled_markers.shape[0])
        colors = cmap(np.linspace(0, 1, unlabeled_markers.shape[0]))

        for sphere in unlabeled_marker_spheres:
            sphere.visible = False
        for i, marker in enumerate(unlabeled_markers):
            unlabeled_marker_spheres[i].visible = True
            unlabeled_marker_spheres[i].position = marker
            unlabeled_marker_spheres[i].color = tuple(np.array(colors[i])[:3] * 255)

        for sphere in unordered_marker_spheres:
            sphere.visible = False
        if gui_show_unordered_markers.value:
            for i, marker in enumerate(unordered_markers):
                unordered_marker_spheres[i].visible = True
                unordered_marker_spheres[i].position = marker

        if ordered_frames[frame_idx] and unlabeled_markers.shape[0] > 0:
            chain = np.vstack([X_W_tip.translation()[None, :], unlabeled_markers])
            if gui_show_spline.value and chain.shape[0] >= 2:
                chord = np.linalg.norm(np.diff(chain, axis=0), axis=1)
                t = np.concatenate([[0.0], np.cumsum(chord)])
                if t[-1] > 0:
                    spline = CubicSpline(t, chain, axis=0)
                    samples_per_segment = 20
                    t_dense = np.linspace(
                        t[0], t[-1], (chain.shape[0] - 1) * samples_per_segment + 1
                    )
                    curve = spline(t_dense)
                    segments = np.stack([curve[:-1], curve[1:]], axis=1)
                else:
                    segments = np.stack([chain[:-1], chain[1:]], axis=1)
            else:
                segments = np.stack([chain[:-1], chain[1:]], axis=1)
            seg_colors = np.tile(
                np.array([255, 255, 255], dtype=np.uint8), (segments.shape[0], 2, 1)
            )
            ordered_marker_lines.points = segments
            ordered_marker_lines.colors = seg_colors
            ordered_marker_lines.visible = True
        else:
            ordered_marker_lines.visible = False

        pruned_segments = []
        if (
            gui_show_pruned_branches.value
            and ordered_frames[frame_idx]
            and pruned_orderings_per_frame[frame_idx]
        ):
            committed = unlabeled_idxs[frame_idx]
            n_slots = len(committed)
            for pruned_ordering in pruned_orderings_per_frame[frame_idx]:
                if len(pruned_ordering) != n_slots:
                    continue
                diff_slots = [k for k in range(n_slots) if pruned_ordering[k] != committed[k]]
                if not diff_slots:
                    continue
                seg_pairs = set()
                for k in diff_slots:
                    if k > 0:
                        seg_pairs.add((k - 1, k))
                    if k < n_slots - 1:
                        seg_pairs.add((k, k + 1))
                pruned_pos = all_unlabeled[pruned_ordering]
                for (a, b) in seg_pairs:
                    pruned_segments.append([pruned_pos[a], pruned_pos[b]])
        if pruned_segments:
            segs = np.array(pruned_segments)
            colors_arr = np.tile(
                np.array([255, 80, 80], dtype=np.uint8), (segs.shape[0], 2, 1)
            )
            pruned_branch_lines.points = segs
            pruned_branch_lines.colors = colors_arr
            pruned_branch_lines.visible = True
        else:
            pruned_branch_lines.visible = False

        if playing:
            gui_frame_slider.value = min(
                gui_frame_slider.max, gui_frame_slider.value + 1
            )

        play_speed = 1 if gui_play_speed.value == 0 else gui_play_speed.value
        time.sleep(1 / (play_speed * 200.0))


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run the auto marker tracker on hardware trials and debug failures."
    )
    parser.add_argument("-t", "--trial_name", default="latest")

    parser.add_argument(
        "-f",
        "--folder_path",
        default=os.path.join(get_flying_knot_data_dir(), "hardware"),
    )

    main(parser.parse_args())
