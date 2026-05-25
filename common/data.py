from datetime import datetime
import matplotlib.pyplot as plt
import trimesh
import numpy as np
from dataclasses import dataclass, field
from pydrake.all import (
    PiecewisePose,
    PiecewisePolynomial,
    RigidTransform,
    Quaternion,
    RotationMatrix,
    PiecewiseQuaternionSlerp,
    BezierCurve,
    RollPitchYaw,
)
import os
import logging
from common import (
    get_flying_knot_data_dir,
    get_latest_trial_name,
    load_pickle,
    parse_yaml,
)
from common.math import make_bezier, finite_diff_central, low_pass
from numpy.typing import NDArray
from xarm7.socket_data import XarmRawDataFrame
from xarm7.kinematics import xarm_plant_3d, xarm_forward_kinematics
from common.tracker import (
    order_initial_frame,
    remove_unlabeled_near_labeled,
    track_markers,
)
from typing import Optional, Union

from nanoid import generate

@dataclass
class MocapHandleObject:
    mocap_frame_to_tip_frame: RigidTransform
    handle_attachment_frame_to_tip_frame: RigidTransform
    cad_frame_to_handle_attachment_frame: RigidTransform
    name : str

    @classmethod
    def from_yaml(clc, path):
        logging.info(f"Loading mocap object from {path}")
        object_config = parse_yaml(path)
        mocap_frame_to_tip_frame = RigidTransform()
        if "mocap_frame_to_tip_frame" in object_config:
            mocap_frame_to_tip_frame = RigidTransform(
                Quaternion(np.array(object_config["mocap_frame_to_tip_frame"]["wxyz"])),
                np.array(object_config["mocap_frame_to_tip_frame"]["xyz"]),
            )
        else:
            logging.warning("No assigned mocap to tip transform")

        handle_attachment_frame_to_tip_frame = RigidTransform()
        if "handle_attachment_frame_to_tip_frame" in object_config:
            handle_attachment_frame_to_tip_frame = RigidTransform(
                Quaternion(
                    np.array(object_config["handle_attachment_frame_to_tip_frame"]["wxyz"])
                ),
                np.array(object_config["handle_attachment_frame_to_tip_frame"]["xyz"]),
            )
        else:
            logging.warning("No assigned handle to tip transform")

        cad_frame_to_handle_attachment_frame = RigidTransform()
        if "cad_frame_to_handle_attachment_frame" in object_config:
            cad_frame_to_handle_attachment_frame = RigidTransform(
                Quaternion(
                    np.array(object_config["cad_frame_to_handle_attachment_frame"]["wxyz"])
                ),
                np.array(object_config["cad_frame_to_handle_attachment_frame"]["xyz"]),
            )
        else:
            logging.warning("No assigned CAD to handle attachment transform")

        return MocapHandleObject(
            mocap_frame_to_tip_frame=mocap_frame_to_tip_frame,
            handle_attachment_frame_to_tip_frame=handle_attachment_frame_to_tip_frame,
            cad_frame_to_handle_attachment_frame=cad_frame_to_handle_attachment_frame,
            name=object_config["name"]
        )

@dataclass
class MocapBaseObject:
    mocap_frame_to_body_frame: RigidTransform
    mocap_frame_to_cad_frame: RigidTransform
    frames_in_mocap_frame: dict[str, RigidTransform]
    name : str

    @classmethod
    def from_yaml(clc, path):
        logging.info(f"Loading mocap object from {path}")
        object_config = parse_yaml(path)
        mocap_frame_to_body_frame = RigidTransform()
        if "mocap_frame_to_body_frame" in object_config:
            mocap_frame_to_body_frame = RigidTransform(
                Quaternion(np.array(object_config["mocap_frame_to_body_frame"]["wxyz"])),
                np.array(object_config["mocap_frame_to_body_frame"]["xyz"]),
            )
        else:
            logging.warning("No assigned mocap to body transform")

        mocap_frame_to_cad_frame = RigidTransform()
        if "mocap_frame_to_cad_frame" in object_config:
            mocap_frame_to_cad_frame = RigidTransform(
                Quaternion(
                    np.array(object_config["mocap_frame_to_cad_frame"]["wxyz"])
                ),
                np.array(object_config["mocap_frame_to_cad_frame"]["xyz"]),
            )
        else:
            logging.warning("No assigned mocap to CAD transform")
        
        frames_in_mocap_frame = {}
        if "frames_in_mocap_frame" in object_config:
            for name, X in object_config["frames_in_mocap_frame"].items():
                frames_in_mocap_frame[name] =  RigidTransform(
                Quaternion(
                    np.array(object_config["frames_in_mocap_frame"][name]["wxyz"])
                ),
                np.array(object_config["frames_in_mocap_frame"][name]["xyz"]),
            )
                

        return MocapBaseObject(
            mocap_frame_to_body_frame=mocap_frame_to_body_frame,
            mocap_frame_to_cad_frame=mocap_frame_to_cad_frame,
            frames_in_mocap_frame=frames_in_mocap_frame,
            name=object_config["name"]
        )
    

@dataclass
class CriticalPoint:
    name: str
    time_reference: str # Only "trial_start" is supported for now
    time_offset: float # temporal offset from the time_reference where the critical point occurs
    space_reference: str # only "rope_start" is supported for now
    space_range: tuple[float, float] # start and end distance along the rope with respect to the space_reference

@dataclass
class HumanDemo:
    hand_trajectory: PiecewisePose
    rope_trajectory: PiecewisePolynomial
    end_track_time: float
    critical_points: dict[str, CriticalPoint]
    num_rope_markers: int
    unlabeled_markers: np.ndarray
    frame_times: np.ndarray
    labeled_markers: list[dict[str, np.ndarray]]
    demo_config: dict
    handle_mocap_object: MocapHandleObject
    base_mocap_object: Optional[MocapBaseObject] # This is kinda hacky should make more generic
    base_frame: Optional[RigidTransform]
    fixed_end_point: Optional[RigidTransform]

    def get_goal_fingertip_points(self) -> np.ndarray:
        goal_fingertip_points = self.hand_trajectory.get_position_trajectory().vector_values(
            self.frame_times
        ).T
        end_track_idx = np.searchsorted(
            self.frame_times, self.end_track_time, side="right"
        )
        if end_track_idx < goal_fingertip_points.shape[0]:
            goal_fingertip_points[end_track_idx:] = goal_fingertip_points[end_track_idx]
        return goal_fingertip_points

    def get_goal_rope_trajectory(
        self, fixed_end_pose: Optional[RigidTransform] = None
    ) -> "RopeTrajectory":
        goal_rope_points = np.c_[self.get_goal_fingertip_points(), self.unlabeled_markers]
        if fixed_end_pose is not None:
            goal_fixed_end = np.repeat(
                fixed_end_pose.translation()[None, :],
                goal_rope_points.shape[0],
                axis=0,
            )
            goal_rope_points = np.c_[goal_rope_points, goal_fixed_end]
        return RopeTrajectory(self.frame_times, goal_rope_points)

    @classmethod
    def load(cls, folder_path, trial_name="latest"):
        if trial_name == "latest":
            trial_name = get_latest_trial_name(folder_path)
        else:
            trial_name = trial_name

        trial_folder = os.path.join(folder_path, trial_name)
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
        trial_config = parse_yaml(trial_config_path)
        annotation_save_path = os.path.join(
            trial_folder, f"{trial_name}-annotation.pickle"
        )

        if not os.path.exists(annotation_save_path):
            logging.error(f"Failed to find trial annotation at {annotation_save_path}")
            return
        loaded_annotation_data = load_pickle(annotation_save_path)
        start_idx = loaded_annotation_data["start_idx"]
        end_track_idx = loaded_annotation_data["end_track_idx"]
        end_idx = loaded_annotation_data["end_idx"]
        critical_points = loaded_annotation_data.get("critical_points", {})

        frame_nums = np.array(
            [
                d["frame_number"] - trial_data["data"][start_idx]["frame_number"]
                for d in trial_data["data"][start_idx:end_idx]
            ]
        )
        frame_times = frame_nums / trial_data["fps"]

        vicon_global_frame_name = trial_config["vicon"]["global_frame"]
        vicon_handle_frame_name = trial_config["vicon"]["handle_frame"]
        if vicon_global_frame_name != "world":
            X_mocap_global = MocapBaseObject.from_yaml(
                f"config/mocap_objects/{vicon_global_frame_name}.yaml"
            ).mocap_frame_to_body_frame

        handle_mocap_object: MocapHandleObject = MocapHandleObject.from_yaml(
            f"config/mocap_objects/{vicon_handle_frame_name}.yaml"
        )

        # Assume global frame is not moving in demo
        if vicon_global_frame_name == "world":
            X_WG = RigidTransform()
        else:
            X_WG = (
                transform_from_mocap_data(
                    trial_data["data"][start_idx][vicon_global_frame_name]
                )
                @ X_mocap_global
            )

        hand_poses = []
        for i in range(start_idx, end_idx):
            X_WH = (
                transform_from_mocap_data(
                    trial_data["data"][i][vicon_handle_frame_name]
                )
                @ handle_mocap_object.mocap_frame_to_tip_frame
            )

            X_GH = X_WG.InvertAndCompose(X_WH)
            hand_poses.append(X_GH)

        hand_trajectory = PiecewisePose.MakeLinear(frame_times, hand_poses)

        num_rope_markers = len(loaded_annotation_data["unlabeled_idxs"][start_idx])
        effective_end_track_idx = min(end_track_idx, end_idx)
        unlabeled_markers = []
        for i in range(start_idx, effective_end_track_idx):
            p_B = (
                X_WG.inverse()
                @ (np.array(trial_data["data"][i]["unlabeled_markers"]) / 1000.0).T
            ).T
            unlabeled_markers.append(
                (p_B)[
                    loaded_annotation_data["unlabeled_idxs"][i][:num_rope_markers]
                ].flatten()
            )

        # num_rope_markers = trial_config["num_rope_markers"]
        for i in range(effective_end_track_idx, end_idx):
            unlabeled_markers.append(unlabeled_markers[-1])

        unlabeled_markers = np.array(unlabeled_markers)
        unlabeled_markers = low_pass(unlabeled_markers, 0.2)

        rope_trajectory: PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(
            frame_times, unlabeled_markers.T
        )

        labeled_markers = []
        X_GW = X_WG.inverse()
        for i in range(start_idx, end_idx):
            frame_markers = {}
            for object_name, markers in trial_data["data"][i][
                "labeled_markers"
            ].items():
                p_W = np.array(markers) / 1000.0
                if p_W.size == 0:
                    frame_markers[object_name] = p_W
                else:
                    frame_markers[object_name] = (X_GW @ p_W.T).T
            labeled_markers.append(frame_markers)

        end_point = None
        cleat_base = None
        base_frame = RigidTransform()
        if "cleat" in trial_config["task"]["name"]:
            cleat_base = MocapBaseObject.from_yaml("config/mocap_objects/krishna_cleat_base.yaml")
            # base_frame = (
            #     transform_from_mocap_data(
            #         trial_data["data"][start_idx]["krishna_cleat_base"]
            #     )
            #     @ cleat_base.mocap_frame_to_cad_frame
            # )
            base_frame = cleat_base.mocap_frame_to_cad_frame
            if "left" in trial_config["task"]["name"]:
                # end_point = (
                #     transform_from_mocap_data(
                #         trial_data["data"][start_idx]["krishna_cleat_base"]
                #     )
                #     @ cleat_base.frames_in_mocap_frame["left"]
                # )
                end_point = cleat_base.frames_in_mocap_frame["left"]
            elif "right" in trial_config["task"]["name"]:
                # end_point = (
                #     transform_from_mocap_data(
                #         trial_data["data"][start_idx]["krishna_cleat_base"]
                #     )
                #     @ cleat_base.frames_in_mocap_frame["right"]
                # )
                end_point = cleat_base.frames_in_mocap_frame["right"]

        if end_track_idx < end_idx:
            end_track_time = frame_times[end_track_idx - start_idx]
        else:
            end_track_time = frame_times[-1] + 1.0

        return HumanDemo(
            hand_trajectory=hand_trajectory,
            end_track_time=end_track_time,
            rope_trajectory=rope_trajectory,
            critical_points=critical_points,
            num_rope_markers=num_rope_markers,
            unlabeled_markers=unlabeled_markers,
            frame_times=frame_times,
            labeled_markers=labeled_markers,
            demo_config=trial_config,
            handle_mocap_object=handle_mocap_object,
            fixed_end_point = end_point,
            base_mocap_object=cleat_base,
            base_frame = base_frame
        )


def resample_rope_markers(
    rope_points,
    num_new,
    interpolation_type="linear",
    fingertip_positions=None,
    fixed_end_position=None,
):
    """
    Resample inter-rope markers along the rope's parametric ordering to `num_new` markers.

    rope_points: (N_frames, num_old, 3) or (N_frames, num_old*3) — inter-rope markers,
        excluding the fingertip and any fixed end.
    fingertip_positions: optional (N_frames, 3). When provided, the rope's start endpoint
        is the fingertip per frame; new markers are placed between the fingertip and the
        next endpoint, so doubling adds a particle between the fingertip and the first
        original marker.
    fixed_end_position: optional (3,) array. When provided, the rope's end endpoint is
        the fixed end; without it the last marker is the rope's end. Doubling places a
        new particle between the last original marker and the fixed end.

    Returns the resampled inter-rope markers (excluding endpoints) in the same input
    layout (flat 2D or 3D).
    """
    flat_input = rope_points.ndim == 2
    if flat_input:
        rope_points = rope_points.reshape(rope_points.shape[0], -1, 3)

    has_start = fingertip_positions is not None
    has_end = fixed_end_position is not None

    parts = []
    if has_start:
        parts.append(np.asarray(fingertip_positions)[:, None, :])
    parts.append(rope_points)
    if has_end:
        parts.append(
            np.broadcast_to(
                np.asarray(fixed_end_position).reshape(3),
                (rope_points.shape[0], 1, 3),
            )
        )
    full_rope = np.concatenate(parts, axis=1) if len(parts) > 1 else rope_points

    K_old = full_rope.shape[1]
    K_new = num_new + int(has_start) + int(has_end)

    if K_old == K_new:
        out_full = full_rope
    elif interpolation_type == "linear":
        s_old = np.linspace(0, 1, K_old)
        s_new = np.linspace(0, 1, K_new)
        idx = np.clip(np.searchsorted(s_old, s_new, side="right") - 1, 0, K_old - 2)
        s_lo = s_old[idx]
        s_hi = s_old[idx + 1]
        w = ((s_new - s_lo) / (s_hi - s_lo))[None, :, None]
        p_lo = full_rope[:, idx, :]
        p_hi = full_rope[:, idx + 1, :]
        out_full = p_lo * (1 - w) + p_hi * w
    elif interpolation_type == "cubic":
        s_old = np.linspace(0, 1, K_old)
        s_new = np.linspace(0, 1, K_new)
        out_full = np.empty((full_rope.shape[0], K_new, 3))
        for i in range(full_rope.shape[0]):
            curve = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                s_old, full_rope[i].T
            )
            out_full[i] = curve.vector_values(s_new).T
    else:
        raise ValueError(f"Unknown interpolation_type: {interpolation_type}")

    start_idx = 1 if has_start else 0
    end_idx = -1 if has_end else None
    out = out_full[:, start_idx:end_idx, :]

    if flat_input:
        return out.reshape(out.shape[0], -1)
    return out


@dataclass
class RopeTrajectory:
    """
    rope_points: (N, num_rope_points, 3) or (N, num_rope_points*3)
    """

    times: NDArray[np.float64]
    rope_points: NDArray[np.float64]
    interpolation_type: str = "linear"
    rope_start_rotations: list[RotationMatrix] | None = None
    rope_end_rotations: list[RotationMatrix] | None = None
    rope_start_stiffness_scale: float = 1
    rope_end_stiffness_scale: float = 1

    num_rope_points: int = field(init=False)
    rope_spacing: NDArray[np.float64] = field(init=False, repr=False)
    end_time: float = field(init=False)
    rope_points_trajectory: PiecewisePolynomial = field(init=False, repr=False)
    rope_start_rotations_trajectory: PiecewiseQuaternionSlerp | None = field(
        init=False, repr=False
    )
    rope_end_rotations_trajectory: PiecewiseQuaternionSlerp | None = field(
        init=False, repr=False
    )
    curve_cache: dict[float, PiecewisePolynomial] = field(init=False, repr=False)

    def __post_init__(self):
        self._initialize_derived()

    def _initialize_derived(self):
        assert len(self.times) == self.rope_points.shape[0]
        if len(self.rope_points.shape) == 3:
            rope_points_ = self.rope_points.reshape((self.rope_points.shape[0], -1))
        else:
            rope_points_ = self.rope_points

        self.num_rope_points = rope_points_.shape[1] // 3
        self.rope_spacing = np.linspace(0, 1, self.num_rope_points)
        self.end_time = self.times[-1]
        self.rope_points_trajectory = PiecewisePolynomial.FirstOrderHold(
            self.times, rope_points_.T
        )

        self.rope_start_rotations_trajectory = None
        rope_start_rotations = self._coerce_rotations(self.rope_start_rotations)
        if rope_start_rotations is not None:
            assert len(self.times) == len(rope_start_rotations)
            self.rope_start_rotations = rope_start_rotations
            self.rope_start_rotations_trajectory = PiecewiseQuaternionSlerp(
                self.times, rope_start_rotations
            )

        self.rope_end_rotations_trajectory = None
        rope_end_rotations = self._coerce_rotations(self.rope_end_rotations)
        if rope_end_rotations is not None:
            assert len(self.times) == len(rope_end_rotations)
            self.rope_end_rotations = rope_end_rotations
            self.rope_end_rotations_trajectory = PiecewiseQuaternionSlerp(
                self.times, rope_end_rotations
            )

        self.curve_cache = {}

    def _coerce_rotations(
        self, rotations: list[RotationMatrix] | NDArray[np.float64] | None
    ) -> list[RotationMatrix] | None:
        if rotations is None:
            return None
        if isinstance(rotations, np.ndarray):
            if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
                raise ValueError("Rotation array must be shaped (N, 3, 3).")
            return [RotationMatrix(rotations[i]) for i in range(rotations.shape[0])]
        if isinstance(rotations, (list, tuple)):
            if len(rotations) == 0:
                return []
            if isinstance(rotations[0], RotationMatrix):
                return list(rotations)
            if isinstance(rotations[0], np.ndarray):
                return [RotationMatrix(r) for r in rotations]
        raise TypeError("Rotations must be RotationMatrix or (N, 3, 3) arrays.")

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in (
            "num_rope_points",
            "rope_spacing",
            "end_time",
            "rope_points_trajectory",
            "rope_start_rotations_trajectory",
            "rope_end_rotations_trajectory",
            "curve_cache",
        ):
            state.pop(key, None)
        state["rope_start_rotations"] = self._serialize_rotations(
            self.rope_start_rotations
        )
        state["rope_end_rotations"] = self._serialize_rotations(self.rope_end_rotations)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._initialize_derived()

    def _serialize_rotations(
        self, rotations: list[RotationMatrix] | None
    ) -> NDArray[np.float64] | None:
        if rotations is None:
            return None
        if len(rotations) == 0:
            return np.empty((0, 3, 3))
        if isinstance(rotations[0], RotationMatrix):
            return np.array([r.matrix() for r in rotations])
        if isinstance(rotations[0], np.ndarray):
            return np.array(rotations)
        raise TypeError("Rotations must be RotationMatrix or (N, 3, 3) arrays.")

    def sample_rope_point_trajectory(self, t):
        return self.rope_points_trajectory.value(t).T.reshape((self.num_rope_points, 3))

    def equal_resample_rope(self, t, num_samples):
        return self.resample_rope(t, np.linspace(0, 1, num_samples))

    def fit_curve_to_rope(self, t):
        if t in self.curve_cache:
            return self.curve_cache[t]
        rope_points_at_t = self.sample_rope_point_trajectory(t)  # [Nx3]

        if self.interpolation_type == "linear":
            rope_path = PiecewisePolynomial.FirstOrderHold(
                self.rope_spacing, rope_points_at_t.T
            )
        elif self.interpolation_type == "cubic":
            rope_start_dot = np.zeros(3)
            rope_end_dot = np.zeros(3)
            # if self.rope_start_rotations_trajectory is not None:
            #     rope_start_dot = Quaternion(self.rope_start_rotations_trajectory.value(t)).rotation()[:, 0]
            # else:
            #     rope_start_dot = np.zeros(3)

            # rope_path = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            #     self.rope_spacing, rope_points_at_t.T, rope_start_dot, rope_end_dot
            # )
            rope_path = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                self.rope_spacing, rope_points_at_t.T
            )
        else:
            raise RuntimeError("Invalid rope interpolation type")
        rope_path: PiecewisePolynomial

        self.curve_cache[t] = rope_path
        return rope_path

    def resample_rope(self, t, sample_points):
        return self.fit_curve_to_rope(t).vector_values(sample_points).T


@dataclass
class LearningState:
    trial_num: int
    U_bar: NDArray[np.float64]
    total_time: float
    previous_state: str | None = None

    @property
    def bezier_command(self) -> BezierCurve:
        return make_bezier(self.U_bar, self.total_time)


@dataclass
class XarmTrajectory:
    times: NDArray[np.float64]
    positions: NDArray[np.float64]

    @property
    def traj(self) -> PiecewisePolynomial:
        return PiecewisePolynomial.FirstOrderHold(self.times, self.positions.T)


def transform_from_mocap_data(data):
    return RigidTransform(Quaternion(np.array(data[5:])), np.array(data[2:5]) / 1000)


def generate_trial_name(timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uuid = generate(
        size=8, alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    return f"{timestamp}-{uuid}"


def print_frame_drops(file_name):
    main_folder_path = os.path.join(get_flying_knot_data_dir(), "hardware")
    trial_folder = os.path.join(main_folder_path, file_name)
    trial_data_path = os.path.join(trial_folder, f"{file_name}.pickle")
    data = load_pickle(trial_data_path)
    raw_xarm_data = [
        XarmRawDataFrame.parse_frame_bytes(parse_data)
        for parse_data in data["xarm_raw_data"]
    ]

    trial_joint_timing = (
        np.array(
            [d.timestamp_us - raw_xarm_data[0].timestamp_us for d in raw_xarm_data]
        )
        / 1e6
    )

    a = []
    for i in range(1, len(trial_joint_timing)):
        diff = trial_joint_timing[i] - trial_joint_timing[i - 1]
        if diff > 0.01:
            print(f"xarm: {i} \t Num frame skips: {diff}")
        a.append(diff)

    frames_numbers = [d["frame_number"] for d in data["data"]]
    for i in range(1, len(frames_numbers)):
        if frames_numbers[i] - frames_numbers[i - 1] != 1:
            print(
                f"mocap: {i} \t Num frame skips: {frames_numbers[i] - frames_numbers[i-1]}"
            )


def check_if_frame_drops(file_name, start=2.5, end=2.5):
    main_folder_path = os.path.join(get_flying_knot_data_dir(), "hardware")
    trial_folder = os.path.join(main_folder_path, file_name)
    trial_data_path = os.path.join(trial_folder, f"{file_name}.pickle")
    data = load_pickle(trial_data_path)
    raw_xarm_data = [
        XarmRawDataFrame.parse_frame_bytes(parse_data)
        for parse_data in data["xarm_raw_data"]
    ]

    trial_joint_timing = (
        np.array(
            [d.timestamp_us - raw_xarm_data[0].timestamp_us for d in raw_xarm_data]
        )
        / 1e6
    )

    for i in range(1, len(trial_joint_timing)):
        diff = trial_joint_timing[i] - trial_joint_timing[i - 1]
        if int(start * 250) < i and int(end * 250) > i and diff > 0.01:
            print(f"xarm: {i} \t Num frame skips: {diff}")
            return False

    frames_numbers = [d["frame_number"] for d in data["data"]]
    for i in range(1, len(frames_numbers)):
        if (
            int(start * 200) < i
            and int(end * 200) > i
            and frames_numbers[i] - frames_numbers[i - 1] != 1
        ):
            print(
                f"mocap: {i} \t Num frame skips: {frames_numbers[i] - frames_numbers[i-1]}"
            )
            return False

    return True


@dataclass
class XarmRopeMocapData:
    rope_trajectory: RopeTrajectory
    xarm_trajectory: XarmTrajectory
    X_xarm_to_base: RigidTransform
    unlabeled_markers_base: np.ndarray | None = None
    mocap_frame_times: NDArray[np.float64] | None = None

    @classmethod
    def load_file(
        cls,
        file_name,
        end_track_time,
        env_config,
        main_folder_path=os.path.join(get_flying_knot_data_dir(), "hardware"),
        total_time=None,
    ):
        trial_folder = os.path.join(main_folder_path, file_name)
        if not os.path.exists(trial_folder):
            logging.error(f"Failed to find trial at {trial_folder}")
            return None

        trial_data_path = os.path.join(trial_folder, f"{file_name}.pickle")
        if not os.path.exists(trial_data_path):
            logging.error(f"Failed to find trial data at {trial_data_path}")
            return None

        trial_config_path = os.path.join(trial_folder, f"{file_name}.yaml")
        if not os.path.exists(trial_config_path):
            logging.error(f"Failed to find trial config at {trial_config_path}")
            return None

        data = load_pickle(trial_data_path)
        trial_config = parse_yaml(trial_config_path)

        num_frames = len(data["data"])
        if num_frames == 0:
            logging.error(f"No frames found in {trial_data_path}")
            return None

        frame_numbers = np.array([d["frame_number"] for d in data["data"]])
        fps = env_config["vicon"]["vicon_fps"]

        frame_times_all = (frame_numbers - frame_numbers[0]) / fps

        command_name = data["command_name"]
        command_path = os.path.join(
            get_flying_knot_data_dir(),
            "commands",
            command_name,
            f"{command_name}.pickle",
        )
        command_data = load_pickle(command_path)

        raw_xarm_data = [
            XarmRawDataFrame.parse_frame_bytes(parse_data)
            for parse_data in data["xarm_raw_data"]
        ]
        trial_joint_timing = (
            np.array(
                [d.timestamp_us - raw_xarm_data[0].timestamp_us for d in raw_xarm_data]
            )
            / 1e6
        )
        trial_joint_angles = np.array([d.actual_joint_pos for d in raw_xarm_data])
        trial_target_joint_angles = np.array(
            [d.target_joint_pos for d in raw_xarm_data]
        )
        trial_joint_angles_traj: PiecewisePolynomial = (
            PiecewisePolynomial.FirstOrderHold(trial_joint_timing, trial_joint_angles.T)
        )
        trial_target_joint_angles_traj: PiecewisePolynomial = (
            PiecewisePolynomial.FirstOrderHold(
                trial_joint_timing, trial_target_joint_angles.T
            )
        )

        command_joint_traj = make_bezier(
            command_data["control_points"][4:, :], command_data["end_time"]
        )
        dt = np.median(np.diff(trial_joint_timing))
        if dt > 0:
            trial_duration = trial_joint_timing[-1] - trial_joint_timing[0]
            cmd_end = min(command_joint_traj.end_time(), trial_duration)
            cmd_times = np.arange(0.0, cmd_end, dt)
            if cmd_times.size:
                cmd_angles = command_joint_traj.vector_values(cmd_times).T
                target_angles = trial_target_joint_angles_traj.vector_values(
                    trial_joint_timing
                ).T
                max_start_idx = target_angles.shape[0] - cmd_angles.shape[0]
                if max_start_idx >= 0:
                    best_idx = 0
                    best_err = np.inf
                    for idx in range(max_start_idx + 1):
                        segment = target_angles[idx : idx + cmd_angles.shape[0]]
                        err = np.mean((segment - cmd_angles) ** 2)
                        if err < best_err:
                            best_err = err
                            best_idx = idx
                    xarm_data_start_time = float(trial_joint_timing[best_idx])

        xarm_data_start_time += 0.08
        xarm_resample_points = np.linspace(
            0, command_data["end_time"], int(command_data["end_time"] * fps) + 1
        )
        # breakpoint()
        arm_trajectory = XarmTrajectory(
            xarm_resample_points,
            trial_joint_angles_traj.vector_values(
                xarm_resample_points + xarm_data_start_time
            ).T,
        )
        vicon_global_frame_name = env_config["vicon"]["global_frame"]
        base_mocap_obj = MocapBaseObject.from_yaml(
            f"config/mocap_objects/{vicon_global_frame_name}.yaml"
        )
        X_MB = base_mocap_obj.mocap_frame_to_body_frame        
        hand_obj_name = env_config["vicon"]["handle_frame"]
        handle_mocap_object: MocapHandleObject = MocapHandleObject.from_yaml(
            f"config/mocap_objects/{hand_obj_name}.yaml"
        )

        logging.info(f"Hand object name: {hand_obj_name}")

        X_MT = handle_mocap_object.mocap_frame_to_tip_frame

        X_WB: RigidTransform = transform_from_mocap_data(data["data"][0][vicon_global_frame_name]) @ X_MB

        fingertip_marker_z = []
        fingertip_marker_t = []
        ft_poses = []
        for t, d in zip(frame_times_all, data["data"]):
            labeled = d.get("labeled_markers") or {}
            hand_markers = labeled.get(hand_obj_name)
            if hand_markers and hand_obj_name in d:
                if d[hand_obj_name] is None:
                    logging.warning(f"hand not found at {t}")
                    continue
                X_WT: RigidTransform = transform_from_mocap_data(d[hand_obj_name]) @ X_MT

                X_BT = X_WB.InvertAndCompose(X_WT)
                fingertip_marker_t.append(t)
                fingertip_marker_z.append(X_BT.translation()[2])
                ft_poses.append(X_WT)

        plant = xarm_plant_3d(
            handle_frame_body=handle_mocap_object.handle_attachment_frame_to_tip_frame
        )
        plant_context = plant.CreateDefaultContext()
        xarm_fingertip_z = []
        for q in trial_joint_angles:
            q_full = np.r_[np.zeros(4), q] if q.shape[0] == 7 else q
            plant.SetPositions(plant_context, q_full)
            X_BT = xarm_forward_kinematics(
                plant, plant_context, "tip_frame", pose_only=True
            )
            xarm_fingertip_z.append(X_BT.translation()[2])

        xarm_finger_tip_traj = PiecewisePolynomial.FirstOrderHold(
            trial_joint_timing, [xarm_fingertip_z]
        )
        mocap_finger_tip_traj = PiecewisePolynomial.FirstOrderHold(
            fingertip_marker_t, [fingertip_marker_z]
        )
        tip_times = np.arange(0, mocap_finger_tip_traj.end_time(), 1 / fps)

        peak_xarm_idx = int(np.argmax(xarm_finger_tip_traj.vector_values(tip_times)[0]))
        peak_mocap_idx = int(
            np.argmax(mocap_finger_tip_traj.vector_values(tip_times)[0])
        )
        mocap_offset_time = tip_times[peak_mocap_idx] - tip_times[peak_xarm_idx]

        # print(mocap_offset_time)
        # breakpoint()
        # plt.plot(tip_times, mocap_finger_tip_traj.vector_values(tip_times)[0])
        # plt.plot(tip_times, xarm_finger_tip_traj.vector_values(tip_times)[0])
        # plt.savefig("/tmp/fingertip_traj.png")
        # print("/tmp/fingertip_traj.png")
        # breakpoint()
        trial_start_time = xarm_data_start_time + mocap_offset_time
        # breakpoint()
        # Get start and end indexes
        start_idx = int(np.searchsorted(frame_times_all, trial_start_time, side="left"))
        end_track_idx = min(
            num_frames, start_idx + int(end_track_time * fps) + 1
        )
        if total_time is not None:
            end_idx = min(num_frames, start_idx + int(total_time * fps) + 1)
        else:
            end_idx = end_track_idx
        end_idx = max(end_idx, end_track_idx)

        num_rope_markers = trial_config["rope"]["num_rope_markers"]

        unlabeled_idxs = [
            np.arange(len(data["data"][i]["unlabeled_markers"]))
            .astype(np.int32)
            .tolist()
            for i in range(num_frames)
        ]

        # Remove ghost markers near labeled markers for first couple frames
        for t in range(start_idx, min(start_idx + 3, num_frames)):
            unlabeled_idxs[t] = remove_unlabeled_near_labeled(
                candidate_idxs=unlabeled_idxs[t],
                frame_data=data["data"][t],
            )

        raw_marker_positions_per_frame = [
            np.array(d["unlabeled_markers"]) / 1000.0 for d in data["data"]
        ]
        for offset in (0, 1):
            t0 = start_idx + offset
            seed = (
                transform_from_mocap_data(data["data"][t0][hand_obj_name])
                @ handle_mocap_object.mocap_frame_to_tip_frame
            ).translation()
            unlabeled_idxs[t0] = order_initial_frame(
                raw_marker_positions=raw_marker_positions_per_frame[t0],
                candidate_idxs=unlabeled_idxs[t0],
                seed_position=seed,
                num_rope_markers=num_rope_markers,
            )

        logging.info(
            f"Start idx {start_idx}, End_track_idx {end_track_idx}, End_idx {end_idx}"
        )
        rope_length = trial_config["rope"]["rope_length"]
        fixed_end = "cleat" in handle_mocap_object.name
        if fixed_end:
            marker_spacing_distance = rope_length / (num_rope_markers + 1)
        else:
            marker_spacing_distance = rope_length / num_rope_markers
        result = track_markers(
            raw_marker_positions_per_frame=raw_marker_positions_per_frame,
            candidate_idxs_per_frame=unlabeled_idxs,
            start_idx=start_idx,
            end_idx=end_track_idx,
            num_rope_markers=num_rope_markers,
            marker_spacing_distance=marker_spacing_distance,
            pruning_tolerance=1.15
        )
        if not result.success:
            logging.error("Marker tracking failed")
            return None
        unlabeled_idxs = result.candidate_idxs_per_frame


        X_BW = X_WB.inverse()
        unlabeled_markers = []
        for i in range(start_idx, end_track_idx):
            ordered_markers = (np.array(data["data"][i]["unlabeled_markers"]) / 1000.0)[
                unlabeled_idxs[i]
            ][:num_rope_markers]
            if ordered_markers.shape[0] < num_rope_markers:
                logging.error(f"Insufficient markers at frame {i}")
                end_track_idx = i
                end_idx = i
                break
            p_B = (X_BW @ ordered_markers.T).T
            unlabeled_markers.append(p_B)

        if end_track_idx <= start_idx:
            logging.error("No valid frames after auto-labeling.")
            return None

        # Freeze rope at end_track value through end_idx (follow-through window).
        for _ in range(end_track_idx, end_idx):
            unlabeled_markers.append(unlabeled_markers[-1])

        frame_times = frame_times_all[start_idx:end_idx] - frame_times_all[start_idx]
        unlabeled_markers = np.array(unlabeled_markers)
        unlabeled_markers = low_pass(unlabeled_markers, 0.2)
        rope_trajectory = RopeTrajectory(frame_times, unlabeled_markers)

        unlabeled_markers_world = [
            np.array(frame["unlabeled_markers"], dtype=float) / 1000.0
            for frame in data["data"]
        ]
        max_markers = max((m.shape[0] for m in unlabeled_markers_world), default=0)
        unlabeled_markers_base = np.full(
            (num_frames, max_markers, 3), np.nan, dtype=float
        )
        for i, markers in enumerate(unlabeled_markers_world):
            if markers.size == 0:
                continue
            p_B = (X_BW @ markers.T).T
            unlabeled_markers_base[i, : p_B.shape[0]] = p_B
        
        if "cleat" in handle_mocap_object.name:
            xarm_base_obj_name = "krishna_xarm_new"
            xarm_mocap_obj = MocapBaseObject.from_yaml(
                f"config/mocap_objects/{xarm_base_obj_name}.yaml"
            )
            X_world_to_xarm = (
                transform_from_mocap_data(data["data"][start_idx][xarm_base_obj_name])
                @ xarm_mocap_obj.mocap_frame_to_body_frame
            )

            X_world_to_base = X_WB
            X_xarm_to_base = X_world_to_base.inverse() @ X_world_to_xarm
        else:
            # Xarm base is the base frame
            X_xarm_to_base = RigidTransform()

        return XarmRopeMocapData(
            rope_trajectory=rope_trajectory,
            xarm_trajectory=arm_trajectory,
            unlabeled_markers_base=unlabeled_markers_base[start_idx:],
            mocap_frame_times=frame_times_all[start_idx:] - frame_times_all[start_idx],
            X_xarm_to_base = X_xarm_to_base
        )