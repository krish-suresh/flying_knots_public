# Architecture

This document describes how the code in this repository is organized and where each idea from the paper lives. It is meant as a reading guide; for the algorithmic motivation, defer to the paper.

## System overview

The system is a four-stage learning pipeline (Fig. 3 of the paper). A single human demonstration of the flying knot is captured with motion capture, cleaned, and converted into an initial feed-forward command for the xArm7. The Task-Level Iterative Learning Control (ILC) loop then refines the command by executing it (in simulation or on hardware), measuring the rope state at the *critical point* (the moment of rope self-collision), and solving a quadratic program (QP) to compute a command update.

```
                  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
demonstration ──▶ │ human_capture│──▶ │  clean_demo  │──▶ │   compute_ik     │──▶ │     learning     │
                  └──────────────┘    └──────────────┘    └──────────────────┘    └──────────────────┘
                   Vicon mocap         label, segment,      trajectory-opt IK         Task-Level ILC
                   acquisition         find critical pt     → initial command         loop (Alg. 1)
```

The ILC loop's per-iteration step:

1. Execute the current Bézier command **u**(t) — on hardware (`main/run_trajectory.py`) or in simulation (`simulation/forward_model.py`).
2. Measure (or simulate) the resulting rope state trajectory **x**(t).
3. Compute the *critical-point* task error **x̃**(t<sub>c</sub>) and linearize the system model **M** about the current command.
4. Solve the inverse-model QP (Eq. 3) for the command update **Δu**.
5. Apply the update: **u**<sub>k+1</sub> ← **u**<sub>k</sub> − **Δu**.

## Module-by-module

### `main/` — workflow entry points

- [`human_capture.py`](../main/human_capture.py) — captures a human demonstration via Vicon. Records hand markers and rope markers.
- [`clean_demo.py`](../main/clean_demo.py) — labels the rope-marker chain frame-by-frame using greedy assignment, removes outliers, manually annotates the *critical point* (collision frame), and runs a small search to pick the start/end of the demo. Produces a `HumanDemo` pickle.
- [`compute_ik.py`](../main/compute_ik.py) — solves the trajectory optimization in Eq. 4 / Appendix F to convert the demonstrated hand pose trajectory into an initial xArm7 Bézier command. Uses Drake's SNOPT bindings.
- [`learning.py`](../main/learning.py) — the Task-Level ILC main loop (Algorithm 1). Executes the current command, gathers the trial data, calls the inverse model to compute Δu, and writes a new trial pickle per iteration.
- [`run_trajectory.py`](../main/run_trajectory.py) — hardware execution helper. Streams a Bézier command to the xArm7 and captures synchronized joint + mocap data.
- [`visualize.py`](../main/visualize.py) — animates trials in [Viser](https://github.com/nerfstudio-project/viser).
- [`hardware_track_debug.py`](../main/hardware_track_debug.py) — diagnostic tool for inspecting hardware mocap captures and rope-marker tracking quality.

### `common/` — shared utilities

- [`data.py`](../common/data.py) — core data classes (`HumanDemo`, `RopeTrajectory`, `XarmTrajectory`, `XarmRopeMocapData`, `LearningState`) and serialization helpers.
- [`config.py`](../common/config.py) — YAML loading and the `FLYING_KNOT_DATA` environment-variable lookup.
- [`math.py`](../common/math.py) — Bézier helpers (`make_bezier`, `sample_bezier_and_derivative`), rotation-error / log-map, low-pass filter, finite differences, angular-velocity estimation, and a trapezoidal motion profile.
- [`solver.py`](../common/solver.py) — generic Newton solver with line search and a KKT-gradient helper for implicit-function differentiation.
- [`spline.py`](../common/spline.py) — quintic-polynomial bases used for the initial-state search.
- [`tracker.py`](../common/tracker.py) — frame-to-frame marker correspondence using greedy depth-first assignment plus linear-sum assignment fallback (key to labeling the rope chain).
- [`mocap.py`](../common/mocap.py) — `MultiObjectTracker` and the Vicon client wrapper.
- [`estimation.py`](../common/estimation.py) — end-effector pose estimation from mocap markers.
- [`visualize.py`](../common/visualize.py) — Viser-based 3D playback primitives for rope + xArm + mocap.
- [`params.py`](../common/params.py) — the `ParticleRopeParams` dataclass (mass, stiffness, damping, link count, etc.) shared by forward and inverse models.

### `simulation/` — rope dynamics and the inverse model

- [`particle_dynamics.py`](../simulation/particle_dynamics.py) — the point-mass rope model: stiffness and damping forces along the chain (and a cylindrical collision force used in some experiments).
- [`forward_model.py`](../simulation/forward_model.py) — forward integration of the rope dynamics. `forward_particle_model` rolls the rope state out under a given command; `dynamics_residual` / `dynamics_traj_kkt` expose the implicit dynamics for use by the inverse model. Also contains the rope/xArm animation helpers used by `main/visualize.py`.
- [`inverse_model.py`](../simulation/inverse_model.py) — the optimization-based inverse model (Eq. 3 of the paper). Builds the critical-point cost diagonal **Q** and the follow-through cost **Q**<sub>ft</sub>, linearizes the dynamics, assembles the QP, and calls Clarabel via the trust-region update in `common/solver.py`.
- [`drake_rope.py`](../simulation/drake_rope.py) — a higher-fidelity rope model implemented as a chain of rigid bodies in Drake's MultibodyPlant. Used for visualization and (optionally) as an alternative simulator.
- [`elastica_rope.py`](../simulation/elastica_rope.py) — an alternative Cosserat-rod simulator built on [PyElastica](https://github.com/GazzolaLab/PyElastica). Not used by the main ILC loop.
- [`spring_mass_damper.py`](../simulation/spring_mass_damper.py) — a minimal 1-D test case for the dynamics integrator.
- [`kin_test.py`](../simulation/kin_test.py) — kinematics smoke test.

### `xarm7/` — robot interface

- [`kinematics.py`](../xarm7/kinematics.py) — Drake `MultibodyPlant` setup for the xArm7, forward kinematics, and the Bézier-IK solver used by `compute_ik.py`.
- [`interface.py`](../xarm7/interface.py) — converts a Bézier trajectory into the xArm's native `.traj` format and uploads it via HTTP for playback.
- [`socket_data.py`](../xarm7/socket_data.py) — real-time joint-state capture thread reading from the xArm's socket API.
- [`visualize.py`](../xarm7/visualize.py) — xArm visuals for Viser.
- [`data.py`](../xarm7/data/) — packaged default joint-parameter gcode.

### `models/` — robot and rope assets

- [`xarm_description/`](../models/xarm_description/) — Drake-compatible xArm7 URDF/SDF with collision and visual meshes, adapted from UFactory's official [xarm_ros](https://github.com/xArm-Developer/xarm_ros) package (BSD-3-Clause). See [`models/xarm_description/NOTICE.md`](../models/xarm_description/NOTICE.md) for attribution.
- [`rope/`](../models/rope/) — SDF template and procedural generator for the rope rigid-body chain (`rope_model_generator.py`).
- [`handles/`](../models/handles/) — CAD meshes for the rope end-handles.

### `config/` — YAML configs

- [`config/hardware/`](../config/hardware/) — robot + Vicon IPs, FPS, mocap-frame names, command buffer times.
- [`config/learning/`](../config/learning/) — ILC parameters: critical-point position/velocity weights, control cost, follow-through weights, trust-region radius.
- [`config/simulation/`](../config/simulation/) — rope-dynamics parameters: stiffness, damping, end-mass, link count, link length.
- [`config/demo/`](../config/demo/) — task and Vicon-acquisition metadata.
- [`config/ik/`](../config/ik/) — solver knobs for the trajectory-optimization IK.
- [`config/mocap_objects/`](../config/mocap_objects/) — calibrated transforms between Vicon frames and physical frames.

## Paper → code map

| Paper concept | Reference | File(s) | Symbol |
|---|---|---|---|
| Task-Level ILC main loop | Algorithm 1, §IV-B | [`main/learning.py`](../main/learning.py) | top-level loop in `main(...)` |
| Critical-point objective | Eq. 3a, §IV-C | [`simulation/inverse_model.py`](../simulation/inverse_model.py) | `inverse_particle_model_constraints` (Q assembly) |
| Inverse-model QP (Eq. 3) | §IV-G | [`simulation/inverse_model.py`](../simulation/inverse_model.py), [`common/solver.py`](../common/solver.py) | `tracking_ilc_update_constraints` |
| Linearized dynamics constraint **Δx = M Δu** | Eq. 3b | [`simulation/forward_model.py`](../simulation/forward_model.py) | `dynamics_traj_kkt` + implicit-function gradient in `common/solver.py:implicit_function_kkt_gradient` |
| Follow-through tracking cost **Q**<sub>ft</sub> | Eq. 7–10, Appendix A2 | [`simulation/inverse_model.py`](../simulation/inverse_model.py) | follow-through term in `inverse_particle_model_constraints` |
| Robot model (kinematic chain) | §IV-E | [`xarm7/kinematics.py`](../xarm7/kinematics.py) | `xarm_plant_3d`, `xarm_forward_kinematics` |
| Rope dynamics (Eq. 11–13) | §IV-F, Appendix C | [`simulation/particle_dynamics.py`](../simulation/particle_dynamics.py), [`simulation/forward_model.py`](../simulation/forward_model.py) | `stiffness_force_3d`, `damping_force_3d`, `dynamics_step`, `serial_distance_constraint` |
| Bézier command parametrization | Appendix D | [`common/math.py`](../common/math.py) | `make_bezier`, `sample_bezier_and_derivative` |
| Demonstration timing search (5-step procedure) | Appendix E | [`main/clean_demo.py`](../main/clean_demo.py) | `__main__` pipeline |
| Initial-guess trajectory-opt IK (Eq. 4) | §IV-I, Appendix F | [`main/compute_ik.py`](../main/compute_ik.py), [`xarm7/kinematics.py`](../xarm7/kinematics.py) | `xarm_ik_3d_bezier` |
| Hand-tracking error **e**<sub>h</sub> | Appendix F | [`common/math.py`](../common/math.py) | `rot_log_error` + IK cost in `compute_ik.py` |
| Mocap acquisition pipeline | §V-A | [`main/human_capture.py`](../main/human_capture.py), [`common/mocap.py`](../common/mocap.py) | `MultiObjectTracker` |
| Rope-marker labeling | §V-B | [`common/tracker.py`](../common/tracker.py), [`main/clean_demo.py`](../main/clean_demo.py) | greedy DFS + linear-sum assignment |
| Rope model parameters (Table I) | §IV-F | [`config/simulation/particle_model_overhand.yaml`](../config/simulation/particle_model_overhand.yaml) | — |
| Inverse-model parameters (Table III) | Appendix A | [`config/learning/`](../config/learning/) | `particle_overhand_learning.yaml`, `real_overhand_learning.yaml` |

## Data layout

The pipeline reads and writes from a single root directory (`$FLYING_KNOT_DATA`, default `~/flying_knot_data`):

```
$FLYING_KNOT_DATA/
├── human/<trial>/      # raw + cleaned human demonstration (HumanDemo pickle + YAML)
├── commands/<cmd>/     # generated initial Bézier commands
├── hardware/<trial>/   # raw hardware trial captures (mocap + joint states)
├── learning/<run>/     # ILC iteration outputs (one trial pickle per iteration)
└── simulation/<run>/   # simulated rope trials
```

Folder names are timestamped (`YYYYMMDD-HHMMSS-<nanoid>`). The helpers in [`common/data.py`](../common/data.py) and [`common/config.py`](../common/config.py) generate trial names and resolve the data root.

## What is not included

This release is intentionally a snapshot of the research code, not a productized system. The following are required to actually run the pipeline and are *not* in the repository:

- Hardware: an xArm7 and a Vicon motion-capture system.
- Mocap setup. The configs in [`config/mocap_objects/`](../config/mocap_objects/) reference lab-specific Vicon subject calibrations.
- Captured demonstration data (Coming soon!)

