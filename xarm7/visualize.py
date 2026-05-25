import viser
from pydrake.all import RigidTransform, MultibodyPlant, Context, PiecewisePolynomial, PiecewisePose
import numpy as np
from xarm7.kinematics import xarm_plant_3d
import trimesh
import logging
import os


def add_xarm_visual(server: viser.ViserServer, name: str = "xarm", color=(0, 255, 0), opacity=None, handle=None, visible=True):
    # plant = xarm_plant_3d()
    # mesh_names = [
    #     plant.get_body(idx).name()
    #     for idx in plant.GetBodyIndices(plant.GetModelInstanceByName("xarm7"))
    #     if "false" not in plant.get_body(idx).name()
    # ]
    mesh_names = ['link_base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7']
    xarm_meshes = [
        trimesh.load_mesh(f"models/xarm_description/meshes/xarm7/visual/{n}.obj")
        for n in mesh_names
    ]


    server.scene.add_frame("origin", axes_radius=0.01, visible=False)

    mesh_handles = {}
    for n, m in zip(mesh_names, xarm_meshes):
        mesh_handles[n] = (server.scene.add_mesh_simple(
            f"{name}/{n}", m.vertices, m.faces, color=color, opacity=opacity, visible=visible
        ))

    if handle is not None:
        handle_mesh_dir = f"models/handles/meshes/{handle.name}"
        if not os.path.isdir(handle_mesh_dir):
            logging.warning("Handle mesh folder does not exist: %s", handle_mesh_dir)
        else:
            m = trimesh.load_mesh(f"{handle_mesh_dir}/{handle.name}.obj")
            plant = xarm_plant_3d()
            X_L7H = plant.CalcRelativeTransform(plant.CreateDefaultContext(), plant.GetFrameByName("link7"), plant.GetFrameByName("handle_attachment"))
            X_L7C :RigidTransform = X_L7H @ handle.cad_frame_to_handle_attachment_frame
            server.scene.add_mesh_simple(
                f"{name}/link7/handle", m.vertices, m.faces, color=color, opacity=opacity, position=X_L7C.translation(), wxyz=X_L7C.rotation().ToQuaternion().wxyz(), visible=visible
            )

    return mesh_handles

def set_xarm_visual(xarm_mesh_handles: dict[str, viser.MeshHandle], plant: MultibodyPlant, plant_context: Context):
    for n, m in xarm_mesh_handles.items():
        X_WL = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName(n))
        m.position = X_WL.translation()
        m.wxyz = X_WL.rotation().ToQuaternion().wxyz()

def get_xarm_visual_traj(xarm_mesh_handles: dict[str, viser.MeshHandle], arm_traj : PiecewisePolynomial):
    plant = xarm_plant_3d()
    plant_context = plant.CreateDefaultContext()
    sample_times = np.linspace(0, arm_traj.end_time(), int(arm_traj.end_time()*250)+1)
    
    body_poses = [[] for _ in range(8)]
    for j in arm_traj.vector_values(sample_times).T:
        plant.SetPositions(plant_context, j)
        for i, (n, m) in enumerate(xarm_mesh_handles.items()):
            X_WL = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName(n))
            body_poses[i].append(X_WL)
    
    output = []
    for m, p in zip(xarm_mesh_handles.values(), body_poses):
        output.append((m, PiecewisePose.MakeLinear(sample_times, p)))
    
    return output
        
