import numpy as np

from pydrake.all import *


def main():
    builder = DiagramBuilder()

    # Continuous-time plant
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    plant : MultibodyPlant

    ###########################################################################
    # 1. Box: free "kinematic" body we will drive along a sinusoid in x
    ###########################################################################
    box_size = [0.2, 0.2, 0.2]  # x, y, z
    box_mass = 1.0
    box_inertia = UnitInertia.SolidBox(*box_size)
    box_spatial_inertia = SpatialInertia(
        mass=box_mass,
        p_PScm_E=[0.0, 0.0, 0.0],
        G_SP_E=box_inertia,
    )

    box_body = plant.AddRigidBody("box", box_spatial_inertia)
    
    plant.AddWeldConstraint(plant.world_body(), RigidTransform(), box_body, RigidTransform())

    # NOTE: We do NOT weld the box to world. That makes it a free body.
    # We'll explicitly set its pose and spatial velocity in the loop below.

    # Visual + collision geom for the box
    box_geometry = Box(*box_size)
    plant.RegisterVisualGeometry(
        box_body,
        RigidTransform(),  # centered on body frame
        box_geometry,
        "box_visual",
        [0.5, 0.5, 0.9, 1.0],  # RGBA
    )
    plant.RegisterCollisionGeometry(
        box_body,
        RigidTransform(),
        box_geometry,
        "box_collision",
        CoulombFriction(0.9, 0.8),
    )


    ###########################################################################
    # 2. Pendulum attached to box
    ###########################################################################
    pend_length = 0.6
    pend_radius = 0.02
    pend_mass = 0.5
    pendulum_spatial_inertia = SpatialInertia.SolidCylinderWithMass(pend_mass, pend_radius, pend_length, [0,0,1])
    pendulum_spatial_inertia.

    pendulum_body = plant.AddRigidBody("pendulum", pendulum_spatial_inertia)

    pendulum_geometry = Cylinder(pend_radius, pend_length)
    X_BP = RigidTransform([0.0, 0.0, -pend_length / 2.0])
    plant.RegisterVisualGeometry(
        pendulum_body,
        X_BP,
        pendulum_geometry,
        "pendulum_visual",
        [0.9, 0.4, 0.4, 1.0],
    )
    plant.RegisterCollisionGeometry(
        pendulum_body,
        X_BP,
        pendulum_geometry,
        "pendulum_collision",
        CoulombFriction(0.9, 0.8),
    )

    box_pin_frame = plant.AddFrame(
        FixedOffsetFrame(
            "box_pin",
            box_body.body_frame(),
            RigidTransform([0.0, 0.0, box_size[2] / 2.0]),
        )
    )
    pendulum_joint = plant.AddJoint(
        RevoluteJoint(
            "pendulum_joint",
            box_pin_frame,
            pendulum_body.body_frame(),
            [0.0, 1.0, 0.0],
        )
    )

    ###########################################################################
    # 3. Gravity and finalize
    ###########################################################################
    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -9.81])
    plant.Finalize()

    ###########################################################################
    # 4. Meshcat visualization
    ###########################################################################
    meshcat = StartMeshcat()  # prints URL; open in browser

    # MeshcatVisualizer.AddToBuilder(
    #     builder,
    #     scene_graph,
    #     meshcat,
    # )
    
    AddDefaultVisualization(builder, meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    # Initial conditions: box at x = 0, z = 1; pendulum released with small angle
    # We'll immediately overwrite box pose in the loop, but this is a nice start.
    X_WB0 = RigidTransform([0.0, 0.0, 1.0])
    plant.SetFreeBodyPose(plant_context, box_body, X_WB0)
    pendulum_joint.set_angle(plant_context, 0.15)


    ###########################################################################
    # 5. Helper: set box pose/velocity to sinusoid
    ###########################################################################
    A = 0.5      # amplitude [m]
    omega = 1.0  # rad/s

    def set_box_kinematics(t):
        """Prescribe box motion: x(t) = A * sin(omega t)."""
        x = A * np.sin(omega * t)
        xd = A * omega * np.cos(omega * t)  # dx/dt

        # Pose: translate along x; keep y=0, z=1
        X_WB = RigidTransform([x, 0.0, 1.0])
        plant.SetFreeBodyPose(plant_context, box_body, X_WB)

        # Spatial velocity: pure translation along x, no rotation
        V_WB = SpatialVelocity(
            np.array([0.0, 0.0, 0.0]),       # angular velocity w
            np.array([xd, 0.0, 0.0]),        # translational velocity v
        )
        plant.SetFreeBodySpatialVelocity(box_body, V_WB, plant_context)

    ###########################################################################
    # 6. Simulate with small steps, prescribing box motion each step
    ###########################################################################
    simulator = Simulator(diagram, context)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    meshcat : Meshcat
    meshcat.StartRecording()

    t_final = 10.0
    dt = 0.01

    while simulator.get_context().get_time() < t_final:
        t = simulator.get_context().get_time()
        # Set kinematic box pose/velocity for current time
        # set_box_kinematics(t)
        simulator.AdvanceTo(t + dt)

    meshcat.PublishRecording()
    while True:
        pass


if __name__ == "__main__":
    main()
