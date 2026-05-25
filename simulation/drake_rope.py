from models.rope.rope_model_generator import (
    generate_rope_sdf,
    et_to_string,
    RopeParams,
    save_et_to_file,
)
from pydrake.all import *
import builtins
import cv2
import viser
import trimesh
import matplotlib.pyplot as plt
import logging
import os
from common import (
    hash_dict,
    load_pickle,
    make_bezier,
    parse_yaml,
    get_flying_knot_data_dir,
    get_latest_trial_name,
    MocapRopeTrackPoints,
    config_to_camerainfo,
    project_world_points_to_image,
    binary_mask_to_rgb,
    trace_paths,
    generate_point_orderings,
    order_marker_points_along_path,
)
from common.visualize import ViserAnimationRealtime
from xarm7.kinematics import add_xarm_to_plant
from xarm7.visualize import add_xarm_visual
import pickle


logging.getLogger("drake").setLevel(logging.ERROR)


def add_drake_rope_to_plant(
    plant: MultibodyPlant,
    rope_params: RopeParams,
    rope_attachment_frame: Frame,
    prefix="",
):
    parser = Parser(plant)
    package_xml = "models/package.xml"
    parser.package_map().AddPackageXml(filename=package_xml)
    tree = generate_rope_sdf(
        rope_params, template_path="models/rope/sdf/rope_template.sdf", prefix=prefix
    )
    rope_sdf = et_to_string(tree)
    save_et_to_file(tree)

    rope = parser.AddModelsFromString(rope_sdf, "sdf")[0]

    plant.AddJoint(
        WeldJoint(
            f"{prefix}rope_attach_fixed",
            rope_attachment_frame,
            plant.GetFrameByName(f"{prefix}link_0_A"),
            RigidTransform(),
        )
    )

    return rope


def get_static_rope_state(
    diagram_: Diagram, simulation_config, static_arm_q0, static_run_time=10
):
    rope_config = simulation_config["rope"]
    config_hash = hash_dict(rope_config)
    if os.path.exists(f"/tmp/static_rope_state_{config_hash}.pickle"):
        logging.info("Loading cached static rope state...")
        return load_pickle(f"/tmp/static_rope_state_{config_hash}.pickle")
    
    logging.info("No cached static rope state found, computing...")
    # TODO this is a hack to get the static rope state
    diagram: Diagram = create_arm_rope_system(simulation_config)
    plant: MultibodyPlant = diagram.GetSubsystemByName("plant")

    sim = Simulator(diagram)
    plant.SetPositions(
        plant.GetMyMutableContextFromRoot(sim.get_mutable_context()), plant.GetModelInstanceByName("xarm7"), static_arm_q0
    )
    for i in range(7):
        plant.GetJointByName(f"joint{i+1}").Lock(
            plant.GetMyMutableContextFromRoot(sim.get_mutable_context())
        )

    for i in range(rope_config["num_links"] - 1):
        plant.GetJointByName(f"joint_{i}_{i+1}").SetDampingVector(
            plant.GetMyMutableContextFromRoot(sim.get_mutable_context()), np.ones(3) * 0.5
        )

    sim.Initialize()
    sim.AdvanceTo(static_run_time)

    q0_init = plant.GetPositions(
        plant.GetMyContextFromRoot(sim.get_context()),
        plant.GetModelInstanceByName("rope"),
    )
    
    with open(f"/tmp/static_rope_state_{config_hash}.pickle", "wb") as f:
        pickle.dump(q0_init, f, protocol=pickle.HIGHEST_PROTOCOL)

    return q0_init


def create_arm_rope_system(simulation_config):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, simulation_config["dt"])
    plant: MultibodyPlant

    xarm7 = add_xarm_to_plant(plant, "xarm7_no_hand_rot")
    rope = add_drake_rope_to_plant(
        plant,
        RopeParams.from_dict(simulation_config["rope"]),
        plant.GetFrameByName("finger_tip"),
    )

    plant.Finalize()

    return builder.Build()


def _get_model_render_labels(
    plant: MultibodyPlant, scene_graph: SceneGraph, model_instance: ModelInstanceIndex
) -> set[int]:
    inspector = scene_graph.model_inspector()
    labels: set[int] = set()
    for body_index in plant.GetBodyIndices(model_instance):
        body = plant.get_body(body_index)
        for geometry_id in plant.GetVisualGeometriesForBody(body):
            perception = inspector.GetPerceptionProperties(geometry_id)
            if perception is not None and perception.HasProperty("label", "id"):
                labels.add(int(perception.GetProperty("label", "id")))
    return labels


def simulate_drake_rope_trial(command_trajectory: BezierCurve, simulation_config):
    # Commands may contain floating-base components; simulate only the 7 joint commands.
    command_trajectory_slice = make_bezier(
        command_trajectory.control_points()[-7:, :], command_trajectory.end_time()
    )
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, simulation_config["dt"])
    plant: MultibodyPlant

    xarm7 = add_xarm_to_plant(plant, "xarm7_no_hand_rot")

    # TODO from config
    for i in range(7):
        joint_actuator = plant.GetJointActuatorByName(f"joint{i+1}")
        joint_actuator.set_default_gear_ratio(100)
        joint_actuator.set_default_rotor_inertia(0.00001)
        gains = PdControllerGains()
        gains.p = 1000
        gains.d = 10
        joint_actuator.set_controller_gains(gains)

    rope = add_drake_rope_to_plant(
        plant,
        RopeParams.from_dict(simulation_config["rope"]),
        plant.GetFrameByName("finger_tip"),
    )


    plant.Finalize()
    rope_render_labels = _get_model_render_labels(plant, scene_graph, rope)
    rope_render_labels_tuple = tuple(rope_render_labels)
    camera_log_data = {}
    if "cameras" in simulation_config:
        renderer_name = "vtk_renderer"
        scene_graph.AddRenderer(
            renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams())
        )
        for camera_name, camera_config in simulation_config["cameras"].items():
            sensor_info = add_rgbd_sensor_from_config(
                builder=builder,
                scene_graph=scene_graph,
                renderer_name=renderer_name,
                camera_name=camera_name,
                camera_config=camera_config,
                simulation_config=simulation_config,
            )
            camera_log_data[camera_name] = {
                "period": sensor_info["period"],
                "label_enabled": sensor_info["label_enabled"],
                "next_capture_time": 0.0,
                "sample_times": [],
                "rgb": [],
                "depth": [],
                "label": [],
                "rope_segmentation": [],
                "sensor": sensor_info["sensor"],
            }

    traj_source: TrajectorySource = builder.AddSystem(
        TrajectorySource(command_trajectory_slice)
    )
    traj_dot_source: TrajectorySource = builder.AddSystem(
        TrajectorySource(command_trajectory_slice.MakeDerivative(1))
    )

    plex = builder.AddSystem(Multiplexer([7, 7]))

    builder.Connect(traj_source.get_output_port(), plex.get_input_port(0))
    builder.Connect(traj_dot_source.get_output_port(), plex.get_input_port(1))
    builder.Connect(plex.get_output_port(), plant.get_desired_state_input_port(xarm7))

    rope_state_logger = LogVectorOutput(plant.get_state_output_port(rope), builder)
    xarm_state_logger = LogVectorOutput(plant.get_state_output_port(xarm7), builder)
    
    diagram = builder.Build()

    rope_q0 = get_static_rope_state(
        diagram, simulation_config, command_trajectory_slice.value(0)
    )
    plant.SetDefaultPositions(rope, rope_q0)
    plant.SetDefaultPositions(xarm7, command_trajectory_slice.value(0))

    print("Starting simulation...")
    sim = Simulator(diagram)
    if camera_log_data:
        def _camera_monitor(root_context: Context):
            t = float(root_context.get_time())
            for camera_name, data in camera_log_data.items():
                if t + 1e-12 < data["next_capture_time"]:
                    continue

                sensor = data["sensor"]
                sensor_context = sensor.GetMyContextFromRoot(root_context)
                period = float(data["period"])
                # RgbdSensorDiscrete output is the most recent discrete sample.
                # When the monitor runs at the next capture threshold, that output
                # still corresponds to the previous camera period.
                if period > 0:
                    image_time = builtins.max(0.0, data["next_capture_time"] - period)
                else:
                    image_time = t
                data["sample_times"].append(image_time)
                data["rgb"].append(
                    np.asarray(
                        sensor.color_image_output_port().Eval(sensor_context).data
                    ).copy()
                )
                data["depth"].append(
                    np.asarray(
                        sensor.depth_image_32F_output_port().Eval(sensor_context).data
                    ).copy()
                )
                if data["label_enabled"]:
                    label_frame = np.asarray(
                        sensor.label_image_output_port().Eval(sensor_context).data
                    ).copy()
                    data["label"].append(label_frame)
                    label_values = (
                        label_frame[:, :, 0] if label_frame.ndim == 3 else label_frame
                    )
                    rope_mask = np.isin(label_values, rope_render_labels_tuple).astype(
                        np.uint8
                    )
                    data["rope_segmentation"].append(rope_mask)
                else:
                    data["label"].append(None)
                    data["rope_segmentation"].append(None)

                if period <= 0:
                    data["next_capture_time"] = t + 1e-9
                else:
                    while data["next_capture_time"] <= t + 1e-12:
                        data["next_capture_time"] += period

            return EventStatus.Succeeded()

        sim.set_monitor(_camera_monitor)
    sim.Initialize()
    sim.AdvanceTo(command_trajectory.end_time())
    print("Starting data processing...")
    sample_times = rope_state_logger.FindLog(sim.get_context()).sample_times()
    rope_states = rope_state_logger.FindLog(sim.get_context()).data()
    xarm_states = xarm_state_logger.FindLog(sim.get_context()).data()
    if sample_times.size > 0:
        keep = np.r_[True, np.diff(sample_times) > 1e-12]
        sample_times = sample_times[keep]
        rope_states = rope_states[:, keep]
        xarm_states = xarm_states[:, keep]

    plant_context = plant.CreateDefaultContext()
    rope_points = []
    for rope_state, xarm_state in zip(rope_states.T, xarm_states.T):
        plant.SetPositionsAndVelocities(plant_context, rope, rope_state)
        plant.SetPositionsAndVelocities(plant_context, xarm7, xarm_state)
        frames = []
        for i in range(simulation_config["rope"]["num_links"]):
            frames.append(plant.GetFrameByName(f"link_{i}_B"))
        
        rope_points_time = []
        for f in frames:
            rope_points_time.append(plant.CalcRelativeTransform(plant_context, plant.world_frame(), f).translation())
        rope_points.append(rope_points_time)
    

    rope_points = np.array(rope_points)
    rope_points_arc_lens = np.linspace(0, (simulation_config["rope"]["num_links"]-1)*simulation_config["rope"]["link_length"], simulation_config["rope"]["num_links"])
    marker_points_sample_dists = np.linspace(0, rope_points_arc_lens[-1], simulation_config["rope"]["markers"])
    marker_points = []
    for p in rope_points:
        marker_points.append(PiecewisePolynomial.FirstOrderHold(rope_points_arc_lens, p.T).vector_values(marker_points_sample_dists).T)
        

    camera_output = {}
    for camera_name, data in camera_log_data.items():
        camera_output[camera_name] = {
            "sample_times": np.array(data["sample_times"]),
            "rgb": data["rgb"],
            "depth": data["depth"],
            "label": data["label"],
            "rope_segmentation": data["rope_segmentation"],
        }

    return {
        "sample_times": sample_times,
        "rope_states": rope_states.T,
        "xarm_states": xarm_states.T,
        "marker_points_sample_dists": marker_points_sample_dists,
        "marker_points": np.array(marker_points),
        "camera_images": camera_output,
    }


def add_drake_rope_visual(
    server: viser.ViserServer, rope_config, name: str = "rope", color=(0, 255, 0)
):
    rope_link_mesh = trimesh.creation.capsule(
        rope_config["link_length"], rope_config["rope_radius"]
    )

    rope_mesh_handles = []
    for i in range(rope_config["num_links"]):
        rope_mesh_handles.append(
            server.scene.add_mesh_trimesh(f"{name}/link_{i}", rope_link_mesh)
        )

    return rope_mesh_handles


def sample_rope_points_equal_spacing(
    rope_points: np.ndarray, num_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    rope_points = np.asarray(rope_points, dtype=float)
    num_samples = int(num_samples)
    if rope_points.ndim != 2 or rope_points.shape[1] != 3:
        raise ValueError(
            f"rope_points must have shape (N, 3), got {rope_points.shape}"
        )
    if num_samples < 2:
        raise ValueError("num_samples must be >= 2.")

    segment_lengths = np.linalg.norm(np.diff(rope_points, axis=0), axis=1)
    path_dists = np.r_[0.0, np.cumsum(segment_lengths)]
    total_length = float(path_dists[-1])
    if total_length <= 1e-12:
        sampled_points = np.repeat(rope_points[:1], num_samples, axis=0)
        return np.zeros(num_samples), sampled_points

    # Remove duplicate path distances caused by zero-length segments for interpolation.
    keep = np.r_[True, np.diff(path_dists) > 1e-12]
    path_dists_unique = path_dists[keep]
    rope_points_unique = rope_points[keep]

    sample_dists = np.linspace(0.0, total_length, num_samples)
    sampled_points = np.column_stack(
        [
            np.interp(sample_dists, path_dists_unique, rope_points_unique[:, i])
            for i in range(3)
        ]
    )
    # Guarantee endpoints are included exactly.
    sampled_points[0] = rope_points[0]
    sampled_points[-1] = rope_points[-1]
    return sample_dists, sampled_points


def _sample_bezier_control_points(
    control_points: np.ndarray, num_samples: int = 100
) -> np.ndarray:
    control_points = np.asarray(control_points, dtype=float)
    curve = BezierCurve(0.0, 1.0, control_points.T)
    sample_points = np.linspace(0.0, 1.0, int(builtins.max(2, num_samples)))
    return curve.vector_values(sample_points).T


def _control_points_to_line_segments(
    control_points: np.ndarray, num_samples: int = 100
) -> np.ndarray:
    curve_points = _sample_bezier_control_points(control_points, num_samples)
    return np.stack([curve_points[:-1], curve_points[1:]], axis=1)


def _tracked_rope_control_points_at_time(
    tracking_data: dict[str, np.ndarray], t: float
) -> np.ndarray:
    times = np.asarray(tracking_data["times"], dtype=float).reshape(-1)
    control_points = np.asarray(tracking_data["control_points"], dtype=float)
    idx = int(np.searchsorted(times, float(t), side="right") - 1)
    idx = int(np.clip(idx, 0, control_points.shape[0] - 1))
    return control_points[idx]


def set_tracked_rope_line_segments(
    line_segments_handle,
    tracking_data: dict[str, np.ndarray],
    t: float,
    num_samples: int = 100,
):
    line_segments_handle.points = _control_points_to_line_segments(
        _tracked_rope_control_points_at_time(tracking_data, t), num_samples
    )


def set_tracked_rope_sample_spheres(
    sphere_handles: list[viser.MeshHandle],
    tracking_data: dict[str, np.ndarray],
    t: float,
):
    if len(sphere_handles) == 0:
        return
    sample_points = _sample_bezier_control_points(
        _tracked_rope_control_points_at_time(tracking_data, t),
        len(sphere_handles),
    )
    for sphere_handle, sample_point in zip(sphere_handles, sample_points):
        sphere_handle.position = sample_point


def _make_piecewise_pose(sample_times: np.ndarray, poses: list[RigidTransform]) -> PiecewisePose:
    if sample_times.size <= 1:
        return PiecewisePose.MakeLinear(
            np.array([0.0, 1e-3]),
            [poses[0], poses[0]],
        )
    return PiecewisePose.MakeLinear(sample_times, poses)


def _draw_circles_on_rgb_image(
    image: np.ndarray,
    pixels: np.ndarray,
    valid: np.ndarray | None = None,
    radius: int = 5,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    image_rgb = np.asarray(image)
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError(f"image must have shape (H, W, C>=3), got {image_rgb.shape}")

    output = image_rgb[:, :, :3].copy()
    height, width = output.shape[:2]
    pixels = np.asarray(pixels, dtype=float)
    if valid is None:
        valid = np.ones(pixels.shape[0], dtype=bool)
    valid = np.asarray(valid, dtype=bool)

    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    circle_mask = xx**2 + yy**2 <= radius**2
    draw_color = np.array(color, dtype=output.dtype)

    for pixel, is_valid in zip(pixels, valid):
        if not is_valid or not np.all(np.isfinite(pixel)):
            continue
        u, v = np.rint(pixel).astype(int)
        if u < 0 or u >= width or v < 0 or v >= height:
            continue

        x0 = builtins.max(0, u - radius)
        x1 = builtins.min(width, u + radius + 1)
        y0 = builtins.max(0, v - radius)
        y1 = builtins.min(height, v + radius + 1)
        mask_x0 = x0 - (u - radius)
        mask_x1 = mask_x0 + (x1 - x0)
        mask_y0 = y0 - (v - radius)
        mask_y1 = mask_y0 + (y1 - y0)

        output[y0:y1, x0:x1][
            circle_mask[mask_y0:mask_y1, mask_x0:mask_x1]
        ] = draw_color

    return output


def _draw_path_on_rgb_image(
    image: np.ndarray,
    path: np.ndarray,
    *,
    start_color: tuple[int, int, int] = (0, 255, 0),
    end_color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    output = np.ascontiguousarray(np.asarray(image)[:, :, :3].copy())
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) == 0:
        return output
    start_color_array = np.array(start_color, dtype=float)
    end_color_array = np.array(end_color, dtype=float)
    if len(path) == 1:
        cv2.circle(
            output,
            tuple(np.rint(path[0]).astype(int)),
            thickness + 2,
            start_color,
            -1,
            cv2.LINE_AA,
        )
        return output

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative_lengths = np.r_[0.0, np.cumsum(segment_lengths)]
    total_length = float(cumulative_lengths[-1])
    for idx, (point_a, point_b) in enumerate(zip(path[:-1], path[1:])):
        if total_length <= 1e-12:
            fraction = 0.0
        else:
            fraction = float(
                (cumulative_lengths[idx] + cumulative_lengths[idx + 1])
                / (2.0 * total_length)
            )
        color = np.rint(
            (1.0 - fraction) * start_color_array + fraction * end_color_array
        ).astype(int)
        cv2.line(
            output,
            tuple(np.rint(point_a).astype(int)),
            tuple(np.rint(point_b).astype(int)),
            tuple(int(value) for value in color),
            thickness,
            cv2.LINE_AA,
        )

    cv2.circle(
        output,
        tuple(np.rint(path[0]).astype(int)),
        thickness + 2,
        start_color,
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        output,
        tuple(np.rint(path[-1]).astype(int)),
        thickness + 2,
        end_color,
        -1,
        cv2.LINE_AA,
    )
    return output


def _draw_ordered_points_on_rgb_image(
    image: np.ndarray,
    pixels: np.ndarray,
    ordering: np.ndarray,
    *,
    radius: int = 5,
    circle_color: tuple[int, int, int] = (255, 255, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    output = np.ascontiguousarray(np.asarray(image)[:, :, :3].copy())
    height, width = output.shape[:2]
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    ordering = np.asarray(ordering, dtype=int).reshape(-1)

    for order_idx, point_idx in enumerate(ordering):
        if point_idx < 0 or point_idx >= len(pixels):
            continue
        pixel = pixels[point_idx]
        if not np.all(np.isfinite(pixel)):
            continue
        u, v = np.rint(pixel).astype(int)
        if u < 0 or u >= width or v < 0 or v >= height:
            continue

        cv2.circle(output, (u, v), radius, circle_color, -1, cv2.LINE_AA)
        cv2.circle(output, (u, v), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
        label = str(order_idx + 1)
        cv2.putText(
            output,
            label,
            (u + radius + 3, v - radius - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (u + radius + 3, v - radius - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return output


def visualize_drake_rope_trial(
    sim_data: dict,
    simulation_config: dict,
    server: viser.ViserServer,
    visualization_fps: int = 60,
    blocking: bool = True,
):
    sample_times = np.asarray(sim_data["sample_times"], dtype=float).reshape(-1)
    rope_states = np.asarray(sim_data["rope_states"])
    xarm_states = np.asarray(sim_data["xarm_states"])
    marker_points = np.asarray(sim_data["marker_points"], dtype=float)

    xarm_visual = add_xarm_visual(server, name="sim/xarm")
    rope_visual = add_drake_rope_visual(server, simulation_config["rope"], name="sim/rope")
    marker_colors = plt.get_cmap("rainbow", marker_points.shape[1])(
        np.linspace(0, 1, marker_points.shape[1])
    )
    marker_visual = [
        server.scene.add_icosphere(
            f"sim/marker_points/{i}",
            0.01,
            color=tuple(np.array(marker_colors[i])[:3] * 255),
            position=marker_points[0, i],
        )
        for i in range(marker_points.shape[1])
    ]

    diagram = create_arm_rope_system(simulation_config)
    plant: MultibodyPlant = diagram.GetSubsystemByName("plant")
    plant_context = plant.CreateDefaultContext()
    xarm7 = plant.GetModelInstanceByName("xarm7")
    rope = plant.GetModelInstanceByName("rope")

    xarm_poses = {name: [] for name in xarm_visual.keys()}
    rope_poses = [[] for _ in range(len(rope_visual))]
    rope_visual_offset = RigidTransform(RollPitchYaw(0, np.pi / 2, 0), np.zeros(3))

    for rope_state, xarm_state in zip(rope_states, xarm_states):
        plant.SetPositionsAndVelocities(plant_context, rope, rope_state)
        plant.SetPositionsAndVelocities(plant_context, xarm7, xarm_state)

        for name in xarm_visual.keys():
            xarm_poses[name].append(
                plant.CalcRelativeTransform(
                    plant_context, plant.world_frame(), plant.GetFrameByName(name)
                )
            )

        for i in range(len(rope_visual)):
            X_WL = plant.CalcRelativeTransform(
                plant_context, plant.world_frame(), plant.GetFrameByName(f"link_{i}")
            ) @ rope_visual_offset
            rope_poses[i].append(X_WL)

    animation = ViserAnimationRealtime(server, visualization_fps=visualization_fps)
    for name, handle in xarm_visual.items():
        animation.add_animated_object(
            handle, _make_piecewise_pose(sample_times, xarm_poses[name])
        )
    for handle, poses in zip(rope_visual, rope_poses):
        animation.add_animated_object(handle, _make_piecewise_pose(sample_times, poses))
    for marker_idx, handle in enumerate(marker_visual):
        marker_poses = [
            RigidTransform(position)
            for position in marker_points[:, marker_idx, :]
        ]
        animation.add_animated_object(
            handle, _make_piecewise_pose(sample_times, marker_poses)
        )

    camera_images = sim_data.get("camera_images", {})
    camera_frustum_streams = {}
    for camera_name, camera_config in simulation_config.get("cameras", {}).items():
        camera_data = camera_images[camera_name]
        pose_config = camera_config["X_WC"]
        wxyz = np.array(pose_config["wxyz"], dtype=float).reshape(4)
        translation = np.array(
            pose_config.get("translation", [0.0, 0.0, 0.0]), dtype=float
        )
        width = float(camera_config.get("width", 640))
        height = float(camera_config.get("height", 480))
        fov_y = float(camera_config.get("fov_y", np.pi / 4.0))
        stream_times = np.asarray(camera_data["sample_times"], dtype=float).reshape(-1)
        stream_segmented = [
            binary_mask_to_rgb(frame)
            for frame in camera_data["rope_segmentation"]
        ]
        frustum = server.scene.add_camera_frustum(
            f"sim/cameras/{camera_name}",
            fov=fov_y,
            aspect=width / height,
            scale=float(camera_config.get("frustum_scale", 0.2)),
            wxyz=wxyz,
            position=translation,
            color=(160, 160, 160),
            image=stream_segmented[0],
        )
        camera_frustum_streams[camera_name] = {
            "frustum": frustum,
            "times": stream_times,
            "segmented": stream_segmented,
        }

    def _update_camera_frustum_images(t: float):
        for stream in camera_frustum_streams.values():
            times = stream["times"]
            idx = int(np.searchsorted(times, t, side="right") - 1)
            idx = int(np.clip(idx, 0, len(stream["segmented"]) - 1))
            stream["frustum"].image = stream["segmented"][idx]

    _update_camera_frustum_images(animation.gui_time_slider.value)

    @animation.gui_time_slider.on_update
    def _(_event):
        _update_camera_frustum_images(animation.gui_time_slider.value)

    animation.reset()
    animation.play()

    blocking and server.sleep_forever()
    return server, animation


def set_drake_rope_visual(
    rope_mesh_handles: list[viser.MeshHandle],
    plant: MultibodyPlant,
    plant_context: Context,
):
    for i, m in enumerate(rope_mesh_handles):
        X_WL: RigidTransform = plant.CalcRelativeTransform(
            plant_context, plant.world_frame(), plant.GetFrameByName(f"link_{i}")
        ) @ RigidTransform(RollPitchYaw(0, np.pi / 2, 0), np.zeros(3))
        m.position = X_WL.translation()
        m.wxyz = X_WL.rotation().ToQuaternion().wxyz()


def add_rgbd_sensor_from_config(
    builder: DiagramBuilder,
    scene_graph: SceneGraph,
    renderer_name: str,
    camera_name: str,
    camera_config: dict,
    simulation_config: dict,
):
    width = int(camera_config.get("width", 640))
    height = int(camera_config.get("height", 480))

    # TODO allow intrinsics directly
    fov_y = float(camera_config.get("fov_y", np.pi / 4.0))

    clipping_near = float(camera_config.get("clipping_near", 0.01))
    clipping_far = float(camera_config.get("clipping_far", 10.0))
    z_near = float(camera_config.get("z_near", 0.1))
    z_far = float(camera_config.get("z_far", 5.0))

    if "camera_period" in camera_config:
        camera_period = float(camera_config["camera_period"])
    elif "fps" in camera_config and float(camera_config["fps"]) > 0:
        camera_period = 1.0 / float(camera_config["fps"])
    else:
        camera_period = float(simulation_config.get("camera_period", 1.0 / 30.0))

    pose_config = camera_config["X_WC"]

    translation = np.array(
        pose_config.get("translation", [1.2, 0.0, 0.6]), dtype=float
    )

    wxyz = np.array(pose_config["wxyz"], dtype=float).reshape(4)
    q_WC = Quaternion(float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3]))

    camera_info = CameraInfo(width=width, height=height, fov_y=fov_y)
    clipping = ClippingRange(clipping_near, clipping_far)
    camera_core = RenderCameraCore(renderer_name, camera_info, clipping, RigidTransform())
    depth_camera = DepthRenderCamera(camera_core, DepthRange(z_near, z_far))
    X_WC = RigidTransform(q_WC, translation)

    label_enabled = bool(camera_config.get("label", True))

    rgbd_sensor = builder.AddSystem(
        RgbdSensorDiscrete(
            RgbdSensor(
                scene_graph.world_frame_id(),
                X_WC,
                depth_camera,
                bool(camera_config.get("show_rgb", False)),
            ),
            camera_period,
            label_enabled,
        )
    )
    builder.Connect(scene_graph.get_query_output_port(), rgbd_sensor.query_object_input_port())
    return {
        "name": camera_name,
        "sensor": rgbd_sensor,
        "period": camera_period,
        "label_enabled": label_enabled,
    }


if __name__ == "__main__":
    server = viser.ViserServer(port=8081)
    drake_sim_config = parse_yaml("config/simulation/drake_rope.yaml")
    tracker_config = parse_yaml("config/estimation/rope_tracker.yaml")
    commands_dir = os.path.join(get_flying_knot_data_dir(), "commands")
    latest_command_name = get_latest_trial_name(commands_dir)
    latest_command_path = os.path.join(
        commands_dir, latest_command_name, f"{latest_command_name}.pickle"
    )
    command_data = load_pickle(latest_command_path)
    command_trajectory = make_bezier(
        command_data["control_points"], command_data["end_time"]
    )
    sim_data = simulate_drake_rope_trial(command_trajectory, drake_sim_config)

    server, animation = visualize_drake_rope_trial(
        sim_data, drake_sim_config, server=server, blocking=False
    )

    num_paths = []
    num_orderings = []
    for test_idx in range(139, len(sim_data["camera_images"]["side"]["sample_times"])):
        vis_time = sim_data["camera_images"]["side"]["sample_times"][test_idx]
        marker_points_sample = PiecewisePolynomial.FirstOrderHold(
            sim_data["sample_times"],
            sim_data["marker_points"].reshape((len(sim_data["sample_times"]), -1)).T,
        ).value(vis_time).reshape((-1, 3))
        camera_config = drake_sim_config["cameras"]["side"]
        K = config_to_camerainfo(camera_config).intrinsic_matrix()
        pose_config = camera_config["X_WC"]
        X_WC = RigidTransform(
            Quaternion(pose_config["wxyz"]), pose_config["translation"]
        )
        seg_img = sim_data["camera_images"]["side"]["rope_segmentation"][test_idx]
        bin_image = binary_mask_to_rgb(seg_img)
        p_img_start = project_world_points_to_image(marker_points_sample[[0]], K, X_WC)
        p_img_points = project_world_points_to_image(marker_points_sample[1:-1], K, X_WC)
        p_img_end = project_world_points_to_image(marker_points_sample[[-1]], K, X_WC)
        traced_paths = trace_paths(seg_img, p_img_start, p_img_end)
        ordering_paths = generate_point_orderings(seg_img, p_img_start, p_img_end, p_img_points)
        num_paths.append(len(traced_paths))
        num_orderings.append(len(ordering_paths))
        print(test_idx)
        # cv2.imshow(
        #     f"bin",
        #     cv2.cvtColor(bin_image, cv2.COLOR_RGB2BGR),
        # )
        # for path_idx, path in enumerate(traced_paths):
        #     path_overlay = _draw_path_on_rgb_image(bin_image, path)
        #     cv2.namedWindow(f"trace_path_{path_idx}", cv2.WINDOW_NORMAL)
        #     cv2.imshow(
        #         f"trace_path_{path_idx}",
        #         cv2.cvtColor(path_overlay, cv2.COLOR_RGB2BGR),
        #     )
        for path_idx, (ordering, path) in enumerate(ordering_paths):
            path_overlay = _draw_path_on_rgb_image(bin_image, path)
            path_overlay = _draw_ordered_points_on_rgb_image(
                path_overlay, p_img_points, ordering
            )
            cv2.namedWindow(f"trace_path_{path_idx}", cv2.WINDOW_NORMAL)
            cv2.imshow(
                f"trace_path_{path_idx}",
                cv2.cvtColor(path_overlay, cv2.COLOR_RGB2BGR),
            )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    plt.plot(num_paths)
    plt.xlabel("frame number")
    plt.ylabel("number of traced paths")
    plt.figure()
    plt.plot(num_orderings)
    plt.xlabel("frame number")
    plt.ylabel("number of ordered")
    plt.show()

    # for path_idx, path in enumerate(traced_paths):
    #     path_overlay = _draw_path_on_rgb_image(bin_image, path)
    #     cv2.namedWindow(f"trace_path_{path_idx}", cv2.WINDOW_NORMAL)
    #     cv2.imshow(
    #         f"trace_path_{path_idx}",
    #         cv2.cvtColor(path_overlay, cv2.COLOR_RGB2BGR),
    #     )
    # if traced_paths:
    #     cv2.waitKey(1)

    # marker_pixels = project_world_points_to_image(marker_points_sample[1:], K, X_WC)
    # marker_overlay = _draw_circles_on_rgb_image(
    #     bin_image,
    #     marker_pixels,
    #     radius=6,
    #     color=(255, 0, 0),
    # )
    # cv2.namedWindow("marker_overlay", cv2.WINDOW_NORMAL)
    # cv2.imshow("marker_overlay", cv2.cvtColor(marker_overlay, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)


    animation.reset()
    animation.play()
    server.sleep_forever()
