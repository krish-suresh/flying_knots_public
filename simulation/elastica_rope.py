from pydrake.all import *
from typing import Any
from collections import defaultdict
import numpy as np
import viser
import trimesh
from common.visualize import ViserAnimation
from common import parse_yaml, cart_to_polar_3d
import elastica as ea
from elastica.typing import RodType
from elastica.dissipation import DamperBase
from common.data import *


class SoftRodSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Contact,
):
    pass


class AxialStretchingCallBack(ea.CallBackBaseClass):
    """
    Records the position of the rod
    """

    def __init__(self, callback_params: dict) -> None:
        ea.CallBackBaseClass.__init__(self)
        self.every = 200
        self.callback_params = callback_params

    def make_callback(self, system: RodType, time: float, current_step: int) -> None:
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["orientation"].append(
                system.director_collection.copy()
            )
            return


class TestCallBack(ea.CallBackBaseClass):
    """
    Records the position of the rod
    """

    def __init__(self, callback_params: dict) -> None:
        ea.CallBackBaseClass.__init__(self)
        self.every = 200
        self.callback_params = callback_params

    def make_callback(
        self, system: ea.Cylinder, time: float, current_step: int
    ) -> None:
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(
                system.position_collection.copy().flatten()
            )
            self.callback_params["orientation"].append(
                system.director_collection.copy()[..., 0]
            )


class HandMotionBC(ea.ConstraintBase):
    def __init__(
        self,
        *args: Any,
        hand_traj: PiecewisePose,
        offset_transform: RigidTransform,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hand_traj: PiecewisePose = hand_traj
        self.offset_transform = offset_transform

    def constrain_values(self, system: RodType, time: float) -> None:
        X_WH: RigidTransform = self.hand_traj.GetPose(time) @ self.offset_transform
        system.position_collection[..., 0] = X_WH.translation()
        system.director_collection[..., 0] = (
            X_WH.rotation().matrix()
            @ RollPitchYaw(np.pi / 2, 0, np.pi / 2).ToRotationMatrix().matrix()
        ).T

    def constrain_rates(self, system: RodType, time: float) -> None:
        V_WH = self.hand_traj.GetVelocity(time)
        system.velocity_collection[..., 0] = V_WH[3:]
        system.omega_collection[..., 0] = V_WH[:3]


def simulate_rope_elastica(
    simulation_config, finger_tip_trajectory: PiecewisePose
):
    simulator = SoftRodSimulator()
    recorded_history: dict[str, list[Any]] = defaultdict(list)
    recorded_history_finger: dict[str, list[Any]] = defaultdict(list)
    final_time = finger_tip_trajectory.end_time()
    initial_fingertip_pose: RigidTransform = finger_tip_trajectory.GetPose(0)

    dt = simulation_config["dt"]
    g = 9.80665

    n_elem = simulation_config["num_links"]

    start = initial_fingertip_pose.translation()
    direction = -initial_fingertip_pose.rotation().matrix()[:, 2]
    normal = initial_fingertip_pose.rotation().matrix()[:, 1]
    base_length = simulation_config["total_length"]
    base_radius = simulation_config["rope_radius"]
    density = simulation_config["density"]
    youngs_modulus = simulation_config["youngs_modulus"]
    shear_modulus = youngs_modulus / (2 * (simulation_config["poisson_ratio"] + 1.0))

    # director_dir = initial_ft_pose.rotation().matrix()[:, [1, 2, 0]]
    # save_state = np.load("elastica_state.npz")

    stretchable_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    # stretchable_rod.shear_matrix[2, 2, :] = EA_target            # axial stretch stiffness per element
    # stretchable_rod.bend_matrix[0, 0, :] = EI1_target            # bending about director 0
    # stretchable_rod.bend_matrix[1, 1, :] = EI2_target            # bending about director 1
    # stretchable_rod.bend_matrix[2, 2, :] = GJ_target             # twist
    axial_stretch = 200
    # bend_stiffness = 0.0005
    # twist_stiffness = 0.01

    stretchable_rod.shear_matrix[2, 2, :] = axial_stretch

    # stretchable_rod.bend_matrix[0, 0, :] = bend_stiffness
    # stretchable_rod.bend_matrix[1, 1, :] = bend_stiffness
    # stretchable_rod.bend_matrix[2, 2, :] = twist_stiffness
    print(stretchable_rod.shear_matrix[2, 2, :])
    print(stretchable_rod.bend_matrix[..., 0])

    simulator.append(stretchable_rod)

    simulator.constrain(stretchable_rod).using(
        HandMotionBC,
        constrained_position_idx=(0,),
        hand_traj=finger_tip_trajectory,
        offset_transform=RigidTransform(),
    )

    # simulator.constrain(stretchable_rod).using(
    #     ea.FixedConstraint, constrained_position_idx=(0,)
    # )

    simulator.add_forcing_to(stretchable_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -g])
    )

    if "lead_mass" in simulation_config:
        extra_tip_force = np.array(
            [0.0, 0.0, -simulation_config["lead_mass"] * g]
        )  # downward z

        simulator.add_forcing_to(stretchable_rod).using(
            ea.EndpointForces,
            np.zeros(3),  # no load at node 0
            extra_tip_force,  # weight at the last node
            ramp_up_time=1e-8,  # or smooth it in if you prefer
        )

    simulator.dampen(stretchable_rod).using(
        ea.AnalyticalLinearDamper,
        translational_damping_constant=simulation_config["translational_damping"],
        rotational_damping_constant=simulation_config["rotational_damping"],
        time_step=dt,
    )
    simulator.dampen(stretchable_rod).using(ea.LaplaceDissipationFilter, filter_order=5)

    finger_start = finger_tip_trajectory.GetPose(0).translation()
    finger_direction = initial_fingertip_pose.rotation().matrix()[:, 0]
    finger_normal = initial_fingertip_pose.rotation().matrix()[:, 1]
    finger_height = 0.14
    finger_radius = 0.007
    finger_density = density

    finger = ea.Cylinder(
        finger_start,
        finger_direction,
        finger_normal,
        finger_height,
        finger_radius,
        finger_density,
    )
    simulator.append(finger)

    simulator.constrain(finger).using(
        HandMotionBC,
        constrained_position_idx=(0,),
        hand_traj=finger_tip_trajectory,
        offset_transform=RigidTransform(RollPitchYaw(0, 0, 0), [0.065, 0, 0]),
    )

    simulator.detect_contact_between(stretchable_rod, finger).using(
        ea.RodCylinderContact, k=1e4, nu=10,
        velocity_damping_coefficient = 0.5,
        friction_coefficient = 0.5,
    )
    simulator.detect_contact_between(stretchable_rod, stretchable_rod).using(
        ea.RodSelfContact, k=1e4, nu=3
    )

    simulator.collect_diagnostics(stretchable_rod).using(
        AxialStretchingCallBack, callback_params=recorded_history
    )

    simulator.collect_diagnostics(finger).using(
        TestCallBack, callback_params=recorded_history_finger
    )

    simulator.finalize()
    timestepper = ea.PositionVerlet()
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(timestepper, simulator, final_time, total_steps)

    times = np.array(recorded_history["time"])
    position_over_time = np.array(recorded_history["position"])
    orientation_over_time = np.array(recorded_history["orientation"])

    # np.savez("elastica_state.npz", position = position_over_time[-1], ors=orientation_over_time[-1])

    # link_pos_trajs = []
    # link_dir_trajs = []
    # for i in range(n_elem + 1):
    #     pos_traj = PiecewisePolynomial.FirstOrderHold(
    #         times, position_over_time[..., i].T
    #     )
    #     if i != n_elem:
    #         ori_traj = PiecewiseQuaternionSlerp(
    #             times, orientation_over_time[..., i].transpose(0, 2, 1)
    #         )
    #     link_pos_trajs.append(pos_traj)
    #     link_dir_trajs.append(ori_traj)


    return times, position_over_time.transpose(0, 2, 1)

def simulate_rope_elastica_old(
    simulation_config, hand_trajectory: PiecewisePose, hand_origin_to_ft: RigidTransform
):
    simulator = SoftRodSimulator()
    recorded_history: dict[str, list[Any]] = defaultdict(list)
    recorded_history_finger: dict[str, list[Any]] = defaultdict(list)
    final_time = hand_trajectory.end_time()
    initial_ft_pose: RigidTransform = hand_trajectory.GetPose(0) @ hand_origin_to_ft

    dt = simulation_config["dt"]
    g = 9.80665

    n_elem = simulation_config["num_links"]

    start = initial_ft_pose.translation()
    direction = -initial_ft_pose.rotation().matrix()[:, 2]
    normal = initial_ft_pose.rotation().matrix()[:, 1]
    base_length = simulation_config["total_length"]
    base_radius = simulation_config["rope_radius"]
    density = simulation_config["density"]
    youngs_modulus = simulation_config["youngs_modulus"]
    shear_modulus = youngs_modulus / (2 * (simulation_config["poisson_ratio"] + 1.0))

    # director_dir = initial_ft_pose.rotation().matrix()[:, [1, 2, 0]]
    # save_state = np.load("elastica_state.npz")

    stretchable_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    # stretchable_rod.shear_matrix[2, 2, :] = EA_target            # axial stretch stiffness per element
    # stretchable_rod.bend_matrix[0, 0, :] = EI1_target            # bending about director 0
    # stretchable_rod.bend_matrix[1, 1, :] = EI2_target            # bending about director 1
    # stretchable_rod.bend_matrix[2, 2, :] = GJ_target             # twist
    axial_stretch = 200
    # bend_stiffness = 0.0005
    # twist_stiffness = 0.01

    stretchable_rod.shear_matrix[2, 2, :] = axial_stretch

    # stretchable_rod.bend_matrix[0, 0, :] = bend_stiffness
    # stretchable_rod.bend_matrix[1, 1, :] = bend_stiffness
    # stretchable_rod.bend_matrix[2, 2, :] = twist_stiffness
    print(stretchable_rod.shear_matrix[2, 2, :])
    print(stretchable_rod.bend_matrix[..., 0])

    simulator.append(stretchable_rod)

    simulator.constrain(stretchable_rod).using(
        HandMotionBC,
        constrained_position_idx=(0,),
        hand_traj=hand_trajectory,
        offset_transform=hand_origin_to_ft,
    )

    # simulator.constrain(stretchable_rod).using(
    #     ea.FixedConstraint, constrained_position_idx=(0,)
    # )

    simulator.add_forcing_to(stretchable_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -g])
    )

    if "lead_mass" in simulation_config:
        extra_tip_force = np.array(
            [0.0, 0.0, -simulation_config["lead_mass"] * g]
        )  # downward z

        simulator.add_forcing_to(stretchable_rod).using(
            ea.EndpointForces,
            np.zeros(3),  # no load at node 0
            extra_tip_force,  # weight at the last node
            ramp_up_time=1e-8,  # or smooth it in if you prefer
        )

    simulator.dampen(stretchable_rod).using(
        ea.AnalyticalLinearDamper,
        translational_damping_constant=simulation_config["translational_damping"],
        rotational_damping_constant=simulation_config["rotational_damping"],
        time_step=dt,
    )
    simulator.dampen(stretchable_rod).using(ea.LaplaceDissipationFilter, filter_order=5)

    finger_start = hand_trajectory.GetPose(0).translation()
    finger_direction = initial_ft_pose.rotation().matrix()[:, 0]
    finger_normal = initial_ft_pose.rotation().matrix()[:, 1]
    finger_height = 0.14
    finger_radius = 0.007
    finger_density = density

    finger = ea.Cylinder(
        finger_start,
        finger_direction,
        finger_normal,
        finger_height,
        finger_radius,
        finger_density,
    )
    simulator.append(finger)

    simulator.constrain(finger).using(
        HandMotionBC,
        constrained_position_idx=(0,),
        hand_traj=hand_trajectory,
        offset_transform=RigidTransform(RollPitchYaw(0, 0, 0), [0.065, 0, 0]),
    )

    simulator.detect_contact_between(stretchable_rod, finger).using(
        ea.RodCylinderContact, k=1e4, nu=10,
        velocity_damping_coefficient = 0.5,
        friction_coefficient = 0.5,
    )
    simulator.detect_contact_between(stretchable_rod, stretchable_rod).using(
        ea.RodSelfContact, k=1e4, nu=3
    )

    simulator.collect_diagnostics(stretchable_rod).using(
        AxialStretchingCallBack, callback_params=recorded_history
    )

    simulator.collect_diagnostics(finger).using(
        TestCallBack, callback_params=recorded_history_finger
    )

    simulator.finalize()
    timestepper = ea.PositionVerlet()
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(timestepper, simulator, final_time, total_steps)

    times = np.array(recorded_history["time"])
    position_over_time = np.array(recorded_history["position"])
    orientation_over_time = np.array(recorded_history["orientation"])

    # np.savez("elastica_state.npz", position = position_over_time[-1], ors=orientation_over_time[-1])

    link_pos_trajs = []
    link_der_trajs = []
    for i in range(n_elem + 1):
        pos_traj = PiecewisePolynomial.FirstOrderHold(
            times, position_over_time[..., i].T
        )
        if i != n_elem:
            ori_traj = PiecewiseQuaternionSlerp(
                times, orientation_over_time[..., i].transpose(0, 2, 1)
            )
        link_pos_trajs.append(pos_traj)
        link_der_trajs.append(ori_traj)

    # rope_spline_traj = []
    # for p in position_over_time:
    #     rope_spline_traj.append(PiecewisePolynomial.FirstOrderHold(np.linspace(0, base_length, n_elem+1), p))

    a = PiecewisePose(
        PiecewisePolynomial.FirstOrderHold(
            np.array(recorded_history_finger["time"]),
            np.array(recorded_history_finger["position"]).T,
        ),
        PiecewiseQuaternionSlerp(
            np.array(recorded_history_finger["time"]),
            np.array(recorded_history_finger["orientation"]).transpose(0, 2, 1),
        ),
    )

    return link_pos_trajs, link_der_trajs, a


def add_elastica_rope_visual(
    server: viser.ViserServer,
    rope_config,
    name: str = "rope",
    color=(0, 255, 0),
    show_axes=False,
):
    link_length = rope_config["total_length"] / rope_config["num_links"]
    rope_link_mesh = trimesh.creation.capsule(link_length, rope_config["rope_radius"])

    rope_mesh_handles = []
    for i in range(rope_config["num_links"]):
        rope_mesh_handles.append(
            (
                server.scene.add_mesh_simple(
                    f"{name}/link_{i}/capsule",
                    rope_link_mesh.vertices,
                    rope_link_mesh.faces,
                    color=color,
                ),
                server.scene.add_frame(
                    f"{name}/link_{i}",
                    show_axes,
                    axes_length=0.015,
                    axes_radius=0.001,
                    origin_radius=0.001,
                ),
            )
        )

    return rope_mesh_handles


def set_elastica_rope_visual(
    rope_mesh_handles: list[tuple[viser.MeshHandle, viser.FrameHandle]],
    rope_points: list[RigidTransform],
):
    for i, (m, f) in enumerate(rope_mesh_handles):
        f.position = (rope_points[i] + rope_points[i + 1]) / 2
        u, v = cart_to_polar_3d(rope_points[i + 1] - rope_points[i])
        f.wxyz = RollPitchYaw(0, np.pi / 2 - v, u).ToQuaternion().wxyz()
        # f.wxyz = link_poses[i].rotation().ToQuaternion().wxyz()


def time_rescale_piecewise_pose(
    X: PiecewisePose, T0: float, T1: float
) -> PiecewisePose:
    """
    Returns a new PiecewisePose whose poses match X(t) but remapped so that
    the original domain [t0, tN] becomes [T0, T1].

    Args:
        X: The original PiecewisePose.
        T0, T1: New start and end times.

    Returns:
        PiecewisePose with rescaled time.
    """

    # Original knots
    torig = np.array(X.get_segment_times())
    t0, tN = torig[0], torig[-1]

    # Affine scale factor
    scale = (T1 - T0) / (tN - t0)

    # New knot times
    tnew = T0 + (torig - t0) * scale

    # Sample poses at original knot times
    poses = [X.GetPose(t) for t in torig]

    # Rebuild trajectory (C⁰ continuity)
    return PiecewisePose.MakeLinear(tnew.tolist(), poses)


def concat_piecewise_pose(X1: PiecewisePose, X2: PiecewisePose) -> PiecewisePose:
    """
    Concatenate X2 to the *end* of X1 in time, by shifting X2's time domain
    so that X2.start_time() maps to X1.end_time().

    Works even if X1.end_time() != X2.start_time().
    """

    # Get the knot times for both trajectories
    t1 = np.array(X1.get_segment_times())
    t2 = np.array(X2.get_segment_times())

    # Compute how much to shift X2 so that its start hits X1's end
    offset = t1[-1] - t2[0]
    t2_shift = t2 + offset

    # Combined time knots (don't duplicate the seam point)
    times = np.hstack([t1, t2_shift[1:]])

    # Sample poses using the ORIGINAL time domains
    poses = [X1.GetPose(t) for t in t1]
    poses += [X2.GetPose(t) for t in t2[1:]]

    # Rebuild a new trajectory; this will be C0 at the seam
    X_concat = PiecewisePose.MakeLinear(times.tolist(), poses)
    return X_concat


if __name__ == "__main__":
    import argparse
    import os
    from common.config import get_flying_knot_data_dir

    parser = argparse.ArgumentParser(description="Elastica rope demo")
    parser.add_argument("trial_name", help="HumanDemo trial folder under $FLYING_KNOT_DATA/human")
    args = parser.parse_args()

    demo_data: HumanDemo = HumanDemo.load(
        os.path.join(get_flying_knot_data_dir(), "human"), args.trial_name
    )

    VISUALIZE_FRAMES = True
    n_circle = 100
    hand_times = np.linspace(0, 1, n_circle)
    theta = np.linspace(0, np.pi * 2, n_circle) - np.pi / 2
    hand_rad = 0.25
    hand_pos_traj = PiecewisePolynomial.FirstOrderHold(
        hand_times,
        np.vstack(
            [
                np.zeros_like(theta),
                np.cos(theta) * hand_rad,
                np.sin(theta) * hand_rad + hand_rad,
            ]
        ),
    )
    hand_ori_traj = PiecewiseQuaternionSlerp(
        hand_times, [Quaternion() for _ in hand_times]
    )

    hand_traj = PiecewisePose(hand_pos_traj, hand_ori_traj)


    hang_time = 3
    hand_origin_to_ft = RigidTransform([0.14, 0, 0])
    initial_ft_pose: RigidTransform = demo_data.hand_trajectory.GetPose(0)
    hand_times = np.linspace(0, hang_time, 100)
    hand_pos_traj = PiecewisePolynomial.FirstOrderHold(
        hand_times, np.array([initial_ft_pose.translation() for _ in hand_times]).T
    )
    hand_ori_traj = PiecewiseQuaternionSlerp(
        hand_times, [initial_ft_pose.rotation().ToQuaternion() for _ in hand_times]
    )

    hang_traj = PiecewisePose(hand_pos_traj, hand_ori_traj)

    hang2_time = 1
    end_ft_pose: RigidTransform = demo_data.hand_trajectory.GetPose(demo_data.hand_trajectory.end_time())
    hand_times = np.linspace(0, hang2_time, 100)
    hand_pos_traj = PiecewisePolynomial.FirstOrderHold(
        hand_times, np.array([end_ft_pose.translation() for _ in hand_times]).T
    )
    hand_ori_traj = PiecewiseQuaternionSlerp(
        hand_times, [end_ft_pose.rotation().ToQuaternion() for _ in hand_times]
    )

    hang2_traj = PiecewisePose(hand_pos_traj, hand_ori_traj)

    # hand_traj = concat_piecewise_pose(hang_traj, time_rescale_piecewise_pose(demo_data.hand_trajectory, 0, 10))
    hand_traj = concat_piecewise_pose(hang_traj, demo_data.hand_trajectory)
    hand_traj = concat_piecewise_pose(hand_traj, hang2_traj)
    # hand_traj = hang_traj

    # hand_traj = demo_data.hand_trajectory

    rope_sim_config = parse_yaml("config/simulation/elastica_rope.yaml")
    rope_link_traj, der_traj, a = simulate_rope_elastica_old(
        rope_sim_config, hand_traj, hand_origin_to_ft
    )

    rope_link_traj: list[PiecewisePolynomial]
    der_traj: list[PiecewiseQuaternionSlerp]

    end_time = rope_link_traj[0].end_time()

    server = viser.ViserServer(port=8081)

    # server.scene.world_axes.visible = True

    hand_handle = server.scene.add_frame(
        "hand", axes_length=0.05, axes_radius=0.005, origin_radius=0.005, visible=True
    )
    ft_handle = server.scene.add_frame(
        "ft", axes_length=0.05, axes_radius=0.005, origin_radius=0.005, visible=False
    )
    ft_handle_ = server.scene.add_frame(
        "ft_", axes_length=0.05, axes_radius=0.005, origin_radius=0.005, visible=False
    )

    ders = []
    for i in range(rope_sim_config["num_links"]):
        ders.append(
            server.scene.add_frame(
                f"ders/{i}",
                axes_length=0.015,
                axes_radius=0.001,
                origin_radius=0.001,
                visible=VISUALIZE_FRAMES,
            )
        )

    finger_handle = server.scene.add_mesh_trimesh(
        "finger",
        trimesh.creation.cylinder(0.007, 0.14),
    )
    rope_mesh_handles = add_elastica_rope_visual(
        server, rope_sim_config, show_axes=False
    )
    if "lead_mass" in rope_sim_config:
        lead_handle = server.scene.add_icosphere("lead", 0.015)
        

    rope_marker_handles = []
    for i in range(demo_data.num_rope_markers):
        rope_marker_handles.append(server.scene.add_icosphere(f"markers/{i}", 0.01))
        


    animation = ViserAnimation(end_time, server, 100)

    def animation_callback(time) -> None:
        rope_points = np.array([a.value(time).flatten() for a in rope_link_traj])
        set_elastica_rope_visual(rope_mesh_handles, rope_points)
        if "lead_mass" in rope_sim_config:
            lead_handle.position = rope_points[-1]

        X_WH = hand_traj.GetPose(time)
        hand_handle.position = X_WH.translation()
        hand_handle.wxyz = X_WH.rotation().ToQuaternion().wxyz()

        finger_handle.position = a.GetPose(time).translation()
        finger_handle.wxyz = a.GetPose(time).rotation().ToQuaternion().wxyz()

        X_WH = hand_traj.GetPose(time) @ hand_origin_to_ft
        ft_handle.position = X_WH.translation()
        ft_handle.wxyz = X_WH.rotation().ToQuaternion().wxyz()
        ft_handle.visible = False

        ft_handle_.position = X_WH.translation()
        X_WH = (
            X_WH.rotation() @ RollPitchYaw(np.pi / 2, 0, np.pi / 2).ToRotationMatrix()
        )
        ft_handle_.wxyz = X_WH.ToQuaternion().wxyz()

        for i, d in enumerate(ders):
            d.position = (rope_points[i] + rope_points[i + 1]) / 2
            d.wxyz = der_traj[i].value(time).flatten()

        if time > hang_time:
            for marker_handle, p  in zip(rope_marker_handles, demo_data.rope_trajectory.value(time-hang_time).reshape((demo_data.num_rope_markers, -1))):
                marker_handle.position = p
                

    animation.update_callback.append(animation_callback)

    animation.play()
