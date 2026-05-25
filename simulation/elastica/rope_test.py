from typing import Any, TypeAlias
from numpy.typing import NDArray
from elastica.typing import RodType

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import elastica as ea

from typing import TypeAlias, Callable, cast

import numpy as np
from numba import njit
from numpy.typing import NDArray
from elastica.typing import RodType
from elastica.external_forces import NoForces

Position: TypeAlias = NDArray[np.float64]  # vector (3)
Orientation: TypeAlias = NDArray[np.float64]  # SO3 matrix (3, 3)
Pose: TypeAlias = tuple[Position, Orientation]


class TargetPoseProportionalControl(NoForces):
    """
    This class applies directional forces on the end node towards a sequence of targets.
    """

    def __init__(
        self,
        elem_index: int,
        p_linear_value: float,
        p_angular_value: float,
        target: Pose | Callable[[float, RodType], Pose],
        target_history: list[Pose],
        ramp_up_time: float = 1.0,
    ) -> None:
        """

        Parameters
        ----------
        elem_index: int
            index of the element to apply the force
        p_linear_value: float
            proportional linear gain
        p_angular_value: float
            proportional angular gain
        target: Pose | Callable[[float, RodType], Pose]
            Target position and orientation.
            array (3,) containing data with 'float' type, or a function that returns the target Pose
            given time and rod.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        """
        super().__init__()
        assert ramp_up_time > 0.0
        self.elem_index = elem_index
        self.linear_gain = p_linear_value
        self.angular_gain = p_angular_value
        self.ramp_up_time = ramp_up_time
        self.target_history = target_history
        self.save_counter = 0
        self.save_every = 200

        self.target: Callable[[float, RodType], Pose]
        if callable(target):
            self.target = target
        else:
            self.target = cast(Callable[[float, RodType], Pose], lambda t, _: target)

    def apply_forces(self, system: RodType, time: float = 0.0) -> None:
        target_position, target_orientation = self.target(time, system)
        if self.save_counter % self.save_every == 0:
            self.target_history.append((target_position, target_orientation))
            self.save_counter = 0
        self.save_counter += 1

        self.compute_node_force(
            system.external_forces,
            system.external_torques,
            system.position_collection,
            system.director_collection,
            self.linear_gain,
            self.angular_gain,
            time,
            self.ramp_up_time,
            target_position,
            target_orientation,
            self.elem_index,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_node_force(
        external_forces: NDArray[np.float64],
        external_torques: NDArray[np.float64],
        positions: NDArray[np.float64],
        orientations: NDArray[np.float64],
        linear_gain: float,
        angular_gain: float,
        time: float,
        ramp_up_time: float,
        target_position: NDArray[np.float64],
        target_orientation: NDArray[np.float64],
        index: int,
    ) -> None:
        factor = min(1.0, time / ramp_up_time)

        # Linear
        position = 0.5 * (positions[..., index] + positions[..., index + 1])
        force = target_position - position
        external_forces[..., index] += 0.5 * linear_gain * factor * force
        external_forces[..., index + 1] += 0.5 * linear_gain * factor * force

        # Angular
        orientation = orientations[..., index]
        rotation = orientation.T @ target_orientation
        angle = np.arccos((np.trace(rotation) - 1) / 2 - 1e-10)
        vector = (1.0 / (2 * np.sin(angle) + 1e-14)) * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
        torque = factor * angular_gain * angle * vector

        external_torques[..., index] -= orientation @ torque


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


if __name__ == "__main__":
    # Options
    GENERATE_2D_VIDEO = False
    GENERATE_3D_VIDEO = True

    simulator = SoftRodSimulator()
    recorded_history: dict[str, list[Any]] = defaultdict(list)
    final_time = 5
    dt = 0.0002

    # setting up test params
    n_elem = 50
    start = np.zeros((3,))
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 1.2
    base_radius = 0.025
    density = 2000
    youngs_modulus = 1e6
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (2 * (poisson_ratio + 1.0))

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
    simulator.append(stretchable_rod)

    run_time = 4

    def base_target(t: float, rod: RodType) -> Pose:
        target_position = direction * base_length - 5 * base_radius * normal
        if t <= run_time / 2:
            ratio = min(2 * t / run_time, 1.0)
            angular_ratio = ratio * np.pi * 2
            position = target_position * ratio
            orientation_twist = np.array(
                [
                    [0, np.cos(angular_ratio), np.sin(angular_ratio)],
                    [0, -np.sin(angular_ratio), np.cos(angular_ratio)],
                    [1, 0, 0],
                ],
                dtype=float,
            )
        else:
            ratio = min(2 * (t - run_time / 2) / run_time, 1.0)
            R = 8
            position = np.array(
                [
                    target_position[0] * (1 - ratio),
                    -R * base_radius * np.cos(2 * ratio * 12) * (1 - ratio),
                    -R * base_radius * np.sin(2 * ratio * 12) * (1 - ratio),
                ]
            )
            angular_ratio = (1 - ratio) * np.pi * 2
            orientation_twist = np.array(
                [
                    [0, np.cos(angular_ratio), -np.sin(angular_ratio)],
                    [0, np.sin(angular_ratio), np.cos(angular_ratio)],
                    [1, 0, 0],
                ],
                dtype=float,
            )
        return position, orientation_twist

    # Control point
    p = 3e3
    pt = 5e0
    simulator.add_forcing_to(stretchable_rod).using(
        TargetPoseProportionalControl,
        elem_index=0,
        p_linear_value=p,
        p_angular_value=pt,
        target=base_target,
        ramp_up_time=1e-6,
        target_history=recorded_history["base_pose"],
    )

    # Boundary conditions
    simulator.constrain(stretchable_rod).using(
        ea.FixedConstraint, constrained_position_idx=(-1, -20)
    )

    # Self contact
    simulator.detect_contact_between(stretchable_rod, stretchable_rod).using(
        ea.RodSelfContact, k=1e4, nu=3
    )

    # Gravity
    simulator.add_forcing_to(stretchable_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.80665])
    )

    # Damping
    damping_constant = 5.0
    simulator.dampen(stretchable_rod).using(
        ea.AnalyticalLinearDamper,
        translational_damping_constant=damping_constant,
        rotational_damping_constant=damping_constant * 0.01,
        time_step=dt,
    )
    simulator.dampen(stretchable_rod).using(ea.LaplaceDissipationFilter, filter_order=5)

    simulator.collect_diagnostics(stretchable_rod).using(
        AxialStretchingCallBack, callback_params=recorded_history
    )

    # Finalize and run the simulation
    simulator.finalize()
    timestepper = ea.PositionVerlet()
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(timestepper, simulator, final_time, total_steps)


    # # Plot knot topological quantities
    # time = np.asarray(recorded_history["time"])
    # positions = np.asarray(recorded_history["position"])
    # orientations = np.asarray(recorded_history["orientation"])
    # radii = np.asarray(recorded_history["radius"])
    # total_twist, _ = ea.compute_twist(positions, orientations[:, 0, ...])
    # total_writhe = ea.compute_writhe(positions, np.float64(base_length), "next_tangent")
    # total_link = ea.compute_link(
    #     positions,
    #     orientations[:, 0, ...],
    #     radii,
    #     np.float64(base_length),
    #     "next_tangent",
    # )

    # plt.figure()
    # plt.plot(time, total_twist, label="twist")
    # plt.plot(time, total_writhe, label="writhe")
    # plt.plot(time, total_link, label="link")
    # plt.legend()
    # plt.xlabel("time")
    # plt.ylabel("link-writhe-twist quantity")
    # plt.savefig("LWT.png", dpi=300)
    # plt.close("all")
