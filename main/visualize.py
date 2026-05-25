import logging
import trimesh
import argparse
from common import (
    HumanDemo,
    ViserAnimationRealtime,
    get_latest_trial_name,
    dict_to_yaml,
    parse_yaml,
    load_pickle,
    RopeTrajectory,
    add_rope_visual,
    get_rope_visual_traj,
    get_flying_knot_data_dir,
)
import viser
import numpy as np
import os
import time
from simulation import (
    create_xarm_particle_visuals,
    make_particle_model_params,
    set_xarm_particle_animation,
)
from xarm7.visualize import add_xarm_visual, get_xarm_visual_traj
from xarm7.kinematics import xarm_plant_3d
import plotly.graph_objects as go
from pydrake.all import PiecewisePose, RigidTransform, RollPitchYaw
from common.math import cart_to_polar_3d


def get_trial_paths(run_folder):
    prefix = "trial_"
    suffix = ".pickle"
    trials = []
    with os.scandir(run_folder) as it:
        for entry in it:
            name = entry.name
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            index_str = name[len(prefix) : -len(suffix)]
            try:
                index = int(index_str)
            except ValueError:
                continue
            trials.append((index, entry.path))
    trials.sort(key=lambda x: x[0])
    return [path for _, path in trials]

def _get_particle_model_total_time(trials_data, default_total_time: float) -> float:
    if len(trials_data) > 0:
        t = np.asarray(trials_data[-1]["model_data"]["T"]).reshape(-1)
        return float(t[-1])
    return float(default_total_time)


def _set_mesh_handles_visible(handles: list[viser.MeshHandle], visible: bool) -> None:
    for handle in handles:
        handle.visible = visible


def _set_static_rope_segment(
    handles: list[viser.MeshHandle],
    rope_traj: RopeTrajectory,
    time_offset: float,
    u_low: float,
    u_high: float,
) -> None:
    rope_curve = rope_traj.fit_curve_to_rope(float(time_offset))
    u_samples = np.linspace(u_low, u_high, len(handles) + 1)
    rope_points = rope_curve.vector_values(u_samples).T
    for i, m in enumerate(handles):
        pos = (rope_points[i] + rope_points[i + 1]) / 2
        u, v = cart_to_polar_3d(rope_points[i + 1] - rope_points[i])
        m.position = pos
        m.wxyz = RollPitchYaw(0, np.pi / 2 - v, u).ToQuaternion().wxyz()


def _add_marker_spheres(server, count, name, color, radius=0.008, visible=True):
    handles = []
    for i in range(count):
        handles.append(
            server.scene.add_icosphere(
                f"{name}/sphere_{i}",
                radius=radius,
                color=color,
                visible=visible,
            )
        )
    return handles


def _animate_marker_spheres(handles, marker_positions, times, animation):
    """
    handles: list of MeshHandle
    marker_positions: (N_frames, num_markers, 3) or (N_frames, num_markers*3)
    times: (N_frames,) array of timestamps
    """
    if marker_positions.ndim == 2:
        marker_positions = marker_positions.reshape(marker_positions.shape[0], -1, 3)
    assert marker_positions.shape[1] == len(handles), (
        f"sphere count {len(handles)} != marker count {marker_positions.shape[1]}"
    )
    times = list(np.asarray(times, dtype=float))
    for i, h in enumerate(handles):
        poses = [RigidTransform(marker_positions[k, i]) for k in range(len(times))]
        animation.add_animated_object(h, PiecewisePose.MakeLinear(times, poses))


def _trial_inter_markers(rope_traj: RopeTrajectory, fixed_end: bool):
    """Extract inter markers (between fingertip and optional fixed_end) from a saved rope_trajectory."""
    pts = rope_traj.rope_points
    if pts.ndim == 2:
        pts = pts.reshape(pts.shape[0], -1, 3)
    return pts[:, 1:-1, :] if fixed_end else pts[:, 1:, :]


def update_animation_with_trial(
    animation: ViserAnimationRealtime,
    trial_data,
    objects,
    demo_data: HumanDemo,
):

    arm_traj = trial_data["arm_trajectory"].traj
    for m, t in get_xarm_visual_traj(objects["xarm_visual"], arm_traj):
        animation.add_animated_object(m, t)

    arm_traj_cmd = trial_data["learning_state"].bezier_command
    for m, t in get_xarm_visual_traj(objects["xarm_visual_cmd"], arm_traj_cmd):
        animation.add_animated_object(m, t)

    rope_traj: RopeTrajectory = trial_data["rope_trajectory"]

    for m, t in get_rope_visual_traj(objects["rope_visual"], rope_traj):
        animation.add_animated_object(m, t)

    goal_rope_traj: RopeTrajectory = trial_data["goal_rope_trajectory"]
    for m, t in get_rope_visual_traj(
        objects["goal_rope_visual"], goal_rope_traj
    ):
        animation.add_animated_object(m, t)

    fixed_end = objects["fixed_end"]
    _animate_marker_spheres(
        objects["goal_marker_spheres"],
        demo_data.unlabeled_markers,
        demo_data.frame_times,
        animation,
    )

    raw_rope_marker_traj: RopeTrajectory | None = trial_data.get(
        "raw_rope_marker_trajectory"
    )
    if raw_rope_marker_traj is not None:
        trial_marker_positions = raw_rope_marker_traj.rope_points
        trial_marker_times = raw_rope_marker_traj.times
    else:
        trial_marker_positions = _trial_inter_markers(rope_traj, fixed_end)
        trial_marker_times = rope_traj.times
    _animate_marker_spheres(
        objects["trial_marker_spheres"],
        trial_marker_positions,
        trial_marker_times,
        animation,
    )

    critical_rope_visuals: dict[str, list[viser.MeshHandle]] = objects[
        "critical_rope_visuals"
    ]
    demo_rope_length = float(demo_data.demo_config["task"]["rope_length"])
    for cp_name, cp in demo_data.critical_points.items():
        handles = critical_rope_visuals.get(cp_name)
        if not handles:
            continue
        s_low, s_high = cp.space_range
        u_low = float(np.clip(s_low / demo_rope_length, 0.0, 1.0))
        u_high = float(np.clip(s_high / demo_rope_length, 0.0, 1.0))
        if u_high <= u_low:
            _set_mesh_handles_visible(handles, False)
            continue
        _set_static_rope_segment(handles, goal_rope_traj, cp.time_offset, u_low, u_high)

    model_data = trial_data["model_data"]
    set_xarm_particle_animation(
        animation,
        objects["particle_model_xarm_visual"],
        objects["particle_model_rope_visual"],
        objects["particle_model_params"],
        np.asarray(model_data["T"]).reshape(-1),
        np.asarray(model_data["Z"]),
        np.asarray(model_data["U"]),
    )


def create_costs_figure(
    costs: list[float],
    cost_solve: list[float],
    selected_index: int | None = None,
) -> go.Figure:
    if not costs and not cost_solve:
        return go.Figure()
    max_len = max(len(costs), len(cost_solve))
    x_data = list(range(max_len))
    fig = go.Figure()
    if costs:
        fig.add_trace(
            go.Scatter(
                x=x_data[: len(costs)],
                y=costs,
                mode="lines",
                name="Current trial cost",
            )
        )
    if cost_solve:
        fig.add_trace(
            go.Scatter(
                x=x_data[: len(cost_solve)],
                y=cost_solve,
                mode="lines",
                name="Predicted cost",
            )
        )
    if selected_index is not None and 0 <= selected_index < max_len:
        if selected_index < len(costs) and np.isfinite(costs[selected_index]):
            fig.add_trace(
                go.Scatter(
                    x=[selected_index],
                    y=[costs[selected_index]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    showlegend=False,
                )
            )
        if selected_index < len(cost_solve) and np.isfinite(cost_solve[selected_index]):
            fig.add_trace(
                go.Scatter(
                    x=[selected_index],
                    y=[cost_solve[selected_index]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    showlegend=False,
                )
            )
    fig.layout.title.automargin = True  # type: ignore
    fig.update_layout(
        title="Cost per trial",
        xaxis_title="Trial",
        yaxis_title="Cost",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def main(args):
    if args.run == "latest":
        run_name = get_latest_trial_name(args.folder_path)
    else:
        run_name = args.run

    run_folder_path = os.path.join(args.folder_path, run_name)

    learning_config_path = next(
        (
            os.path.join(run_folder_path, name)
            for name in os.listdir(run_folder_path)
            if name.endswith(("learning.yaml", "learning.yml"))
        ),
        None,
    )
    learning_config = parse_yaml(learning_config_path)

    demo_data: HumanDemo = HumanDemo.load(
        os.path.join(get_flying_knot_data_dir(), "human"), learning_config["demo"]
    )
    env_config = parse_yaml(os.path.join(run_folder_path, learning_config["env_name"]))
    model_config = parse_yaml(
        os.path.join(run_folder_path, learning_config["model_config"])
    )

    trials_data = [load_pickle(p) for p in get_trial_paths(run_folder_path)]
    num_trials = len(trials_data)
    costs = [t.get("cost", np.nan) for t in trials_data]
    cost_solve = [-t.get("cost_solve", np.nan) for t in trials_data]
    knots_path = os.path.join(run_folder_path, "knots.yaml")
    trials_with_knots = [False] * num_trials
    if os.path.exists(knots_path):
        knot_data = parse_yaml(knots_path)
        if isinstance(knot_data, dict):
            trials_with_knots = knot_data.get(
                "trials_with_knots", trials_with_knots
            )
        else:
            knot_data = {"trials_with_knots": trials_with_knots}
    else:
        knot_data = {"trials_with_knots": trials_with_knots}
        if num_trials != 0:
            dict_to_yaml(knots_path, knot_data)

    if not isinstance(trials_with_knots, list):
        trials_with_knots = [False] * num_trials
    if len(trials_with_knots) < num_trials:
        trials_with_knots.extend([False] * (num_trials - len(trials_with_knots)))
    elif len(trials_with_knots) > num_trials:
        trials_with_knots = trials_with_knots[:num_trials]

    if num_trials == 0:
        trial_names = ["empty"]
    else:
        trial_names = [str(i) for i in range(num_trials)]

    server = viser.ViserServer(port=8081)
    logging.info(f"Starting visualization for run {run_name}")

    plant = xarm_plant_3d()
    xarm_visual = add_xarm_visual(server, handle=demo_data.handle_mocap_object)
    xarm_visual_cmd = add_xarm_visual(server, "xarm_cmd", opacity=0.5)
    logging.info(f"Env Type: {learning_config['env']}")
    if learning_config["env"] == "particle":
        # Lol hacky gpt code
        visualize_links = env_config["num_particles"] + int(
            model_config.get("fixed_end", False)
        )
        rope_len = visualize_links * env_config["l"]
    elif learning_config["env"] == "drake":
        rope_len = env_config["rope"]["num_links"] * env_config["rope"]["link_length"]
        visualize_links = 105
    elif learning_config["env"] == "elastica":
        rope_len = env_config["total_length"]
        visualize_links = env_config["num_links"]
    elif learning_config["env"] == "real":
        rope_len = env_config["rope"]["rope_length"]
        visualize_links = model_config["num_particles"] + int(
            model_config.get("fixed_end", False)
        )
    rope_visual = add_rope_visual(
        server, rope_len, 0.0045, visualize_links, color=(0, 255, 0)
    )  # TODO load radius from envconfig

    goal_rope_visual = add_rope_visual(
        server, rope_len, 0.0045, visualize_links, "goal_rope", (255, 0, 0)
    )  # TODO load radius from envconfig

    particle_model_total_time = _get_particle_model_total_time(
        trials_data,
        demo_data.end_track_time,
    )
    particle_model_params = make_particle_model_params(
        model_config,
        particle_model_total_time,
        demo_data.handle_mocap_object,
    )
    particle_model_xarm_visual, particle_model_rope_visual = (
        create_xarm_particle_visuals(
            server,
            demo_data.handle_mocap_object,
            particle_model_params,
            xarm_name="particle_model_xarm",
            rope_name="particle_model_rope",
            xarm_color=(120, 255, 120),
            rope_color=(0, 120, 255),
            xarm_visible=False,
        )
    )

    fixed_end = bool(model_config.get("fixed_end", False))
    if learning_config["env"] == "real":
        trial_marker_count = env_config["rope"]["num_rope_markers"]
    else:
        trial_marker_count = model_config["num_particles"]
    goal_marker_spheres = _add_marker_spheres(
        server,
        demo_data.num_rope_markers,
        "goal_markers",
        color=(255, 80, 80),
    )
    trial_marker_spheres = _add_marker_spheres(
        server,
        trial_marker_count,
        "trial_markers",
        color=(80, 200, 80),
    )

    critical_rope_visuals: dict[str, list[viser.MeshHandle]] = {}
    for cp_name, cp in demo_data.critical_points.items():
        s_low, s_high = cp.space_range
        segment_length = max(float(s_high) - float(s_low), 1e-6)
        critical_rope_visuals[cp_name] = add_rope_visual(
            server,
            segment_length,
            0.012,
            num_links=24,
            name=f"critical_rope/{cp_name}",
            color=(255, 200, 0),
            visible=False,
            opacity=0.5,
        )

    floor_dist = plant.CalcRelativeTransform(
        plant.CreateDefaultContext(),
        plant.GetFrameByName("link_base"),
        plant.GetFrameByName("floor"),
    ).translation()[2]
    xarm_base_visual = server.scene.add_mesh_trimesh("xarm/link_base/base", trimesh.load_mesh("models/xarm_description/meshes/base/xarm_base.obj"), position=(0, 0.4, floor_dist))
    floor_visual = server.scene.add_box(
        "xarm/link_base/floor",
        color=(200, 200, 200),
        dimensions=(10, 10, 0.001),
        position=(
            0,
            0,
            floor_dist,
        ),  # TODO get from sdf
    )

    if demo_data.base_mocap_object is not None:
        X_WB = demo_data.base_frame
        server.scene.add_mesh_trimesh(
            demo_data.base_mocap_object.name,
            trimesh.load_mesh(
                f"models/handles/meshes/{demo_data.base_mocap_object.name}/{demo_data.base_mocap_object.name}.obj"
            ),
            position=X_WB.translation(),
            wxyz=X_WB.rotation().ToQuaternion().wxyz(),
        )

    objects = {}
    objects["xarm_visual"] = xarm_visual
    objects["xarm_visual_cmd"] = xarm_visual_cmd
    objects["rope_visual"] = rope_visual
    objects["goal_rope_visual"] = goal_rope_visual
    objects["particle_model_xarm_visual"] = particle_model_xarm_visual
    objects["particle_model_rope_visual"] = particle_model_rope_visual
    objects["particle_model_params"] = particle_model_params
    objects["critical_rope_visuals"] = critical_rope_visuals
    objects["goal_marker_spheres"] = goal_marker_spheres
    objects["trial_marker_spheres"] = trial_marker_spheres
    objects["fixed_end"] = fixed_end

    animation = ViserAnimationRealtime(server, default_play_speed=0.25)

    critical_point_names = list(demo_data.critical_points.keys())
    critical_point_dropdown = None
    jump_to_critical_button = None
    if critical_point_names:
        critical_point_dropdown = server.gui.add_dropdown(
            "Critical Point",
            options=critical_point_names,
            initial_value=critical_point_names[0],
        )
        jump_to_critical_button = server.gui.add_button("Jump to Critical")
    trial_dropdown = server.gui.add_dropdown(
        "Trial", trial_names, initial_value=trial_names[-1]
    )
    trial_step_buttons = server.gui.add_button_group(
        "Trial Step", ["Prev", "Next"]
    )

    trial_knot_cb = server.gui.add_checkbox("Success?", False)

    autoplay_checkbox = server.gui.add_checkbox("Autoplay", True)
    show_critical_points_checkbox = server.gui.add_checkbox(
        "Show Critical Points", True
    )
    show_rope_markers_checkbox = server.gui.add_checkbox("Show Rope Markers", True)

    @show_rope_markers_checkbox.on_update
    def _(_) -> None:
        v = show_rope_markers_checkbox.value
        _set_mesh_handles_visible(goal_marker_spheres, v)
        _set_mesh_handles_visible(trial_marker_spheres, v)

    with server.gui.add_folder("Metrics"):
        selected_index = num_trials - 1 if num_trials != 0 else None
        cost_plot = server.gui.add_plotly(
            create_costs_figure(costs, cost_solve, selected_index=selected_index),
            aspect=2.0,
        )

    if num_trials != 0:
        update_animation_with_trial(
            animation,
            trials_data[int(trial_names[-1])],
            objects,
            demo_data,
        )
        trial_knot_cb.value = trials_with_knots[-1]
        animation.play()

    @trial_knot_cb.on_update
    def _(_) -> None:
        try:
            trial_idx = int(trial_dropdown.value)
        except ValueError:
            return
        if trial_idx >= len(trials_with_knots):
            trials_with_knots.extend(
                [False] * (trial_idx + 1 - len(trials_with_knots))
            )
        trials_with_knots[trial_idx] = trial_knot_cb.value
        dict_to_yaml(knots_path, {"trials_with_knots": trials_with_knots})

    if jump_to_critical_button is not None:
        @jump_to_critical_button.on_click
        def _(_) -> None:
            name = critical_point_dropdown.value
            if name not in demo_data.critical_points:
                return
            animation.pause()
            animation.gui_time_slider.value = demo_data.critical_points[name].time_offset

    @trial_step_buttons.on_click
    def _(_) -> None:
        nonlocal num_trials
        if num_trials == 0:
            return
        try:
            current_idx = int(trial_dropdown.value)
        except ValueError:
            return
        if trial_step_buttons.value == "Prev":
            next_idx = max(0, current_idx - 1)
        else:
            next_idx = min(num_trials - 1, current_idx + 1)
        if next_idx == current_idx:
            return
        trial_dropdown.value = str(next_idx)

    @trial_dropdown.on_update
    def _(_) -> None:
        animation.clear()
        trial_idx = int(trial_dropdown.value)
        update_animation_with_trial(
            animation,
            trials_data[trial_idx],
            objects,
            demo_data,
        )
        cost_plot.figure = create_costs_figure(
            costs, cost_solve, selected_index=trial_idx
        )
        trial_knot_cb.value = trials_with_knots[trial_idx]
        if autoplay_checkbox.value:
            animation.reset()
            animation.play()

    cp_visibility_window = 1.0 / 30

    def _refresh_critical_rope_visibility() -> None:
        t = animation.gui_time_slider.value
        enabled = show_critical_points_checkbox.value
        for cp_name, cp in demo_data.critical_points.items():
            handles = critical_rope_visuals.get(cp_name)
            if not handles:
                continue
            visible = enabled and abs(t - cp.time_offset) < cp_visibility_window
            _set_mesh_handles_visible(handles, visible)

    @animation.gui_time_slider.on_update
    def _(_) -> None:
        _refresh_critical_rope_visibility()

    @show_critical_points_checkbox.on_update
    def _(_) -> None:
        _refresh_critical_rope_visibility()

    while True:
        trials_paths = get_trial_paths(run_folder_path)
        if len(trials_paths) > num_trials:
            if num_trials == 0:
                trial_names = []
            trials_data = [load_pickle(p) for p in get_trial_paths(run_folder_path)]
            num_trials = len(trials_data)
            trial_names = [str(i) for i in range(num_trials)]
            if len(trials_with_knots) < num_trials:
                trials_with_knots.extend(
                    [False] * (num_trials - len(trials_with_knots))
                )
            elif len(trials_with_knots) > num_trials:
                trials_with_knots = trials_with_knots[:num_trials]
            trial_dropdown.options = trial_names
            trial_dropdown.value = trial_names[-1]
            logging.info(f"New trial added")
            costs = [t.get("cost", np.nan) for t in trials_data]
            cost_solve = [t.get("cost_solve", np.nan) for t in trials_data]
            cost_plot.figure = create_costs_figure(
                costs, cost_solve, selected_index=int(trial_dropdown.value)
            )
            trial_knot_cb.value = trials_with_knots[int(trial_dropdown.value)]
            dict_to_yaml(knots_path, {"trials_with_knots": trials_with_knots})

        time.sleep(0.01)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run", default="latest")

    parser.add_argument(
        "-f",
        "--folder_path",
        default=os.path.join(get_flying_knot_data_dir(), "learning"),
    )

    main(parser.parse_args())
