import xml.etree.ElementTree as ET
import numpy as np
import os
from enum import Enum
from dataclasses import dataclass, field

from pydrake.all import UnitInertia, SpatialInertia
import yaml

def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
        

class LinkConnectionType(Enum):
    kBallRpy = 1
    kLinearBushingBallConstraint = 2
    kLinearBushingRpy = 3
    kRevolute = 4

@dataclass
class RopeParams:
    link_connection_type : LinkConnectionType = LinkConnectionType.kBallRpy
    mass_per_unit_len: float = 0.0390909091 # kg/m of rope
    rope_radius: float = 0.009/2 # [m]
    link_length: float = 0.05 # [m]
    num_links:int = 20
    link_mass: float = mass_per_unit_len*link_length
    link_extra_scale: float = 1.0

    unit_inertia : UnitInertia = SpatialInertia.SolidCylinderWithMass(link_mass, rope_radius, link_length, np.array([1,0,0])).get_unit_inertia()*link_mass

    torque_stiffness:list = field(default_factory=lambda: [0,0,0])
    torque_damping:list = field(default_factory=lambda: [0,0,0])
    force_stiffness:list = field(default_factory=lambda: [0,0,0])
    force_damping:list = field(default_factory=lambda: [0,0,0])
    ball_damping:float = 0.01
    ball_stiffness:float = 0.01
    ball_stiffness_roll:float = 0.01

    lead:bool = False
    lead_mass:float = 0.05
    lead_radius:float = 0.025
    lead_inertia : UnitInertia = UnitInertia()

    self_contact: bool = True
    collision: bool = True

    rope_color_rgba : list = field(default_factory=lambda: [0.5,0.5,0.5, 1])

    mass_at_link_center: bool = True

    mu_dynamic: float = 0.5

    @classmethod
    def from_yaml(cls, yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            params_yaml = yaml.safe_load(file)
        
        return RopeParams.from_dict(params_yaml)
        
    @classmethod
    def from_dict(cls, params_yaml):
        params = RopeParams()
        if params_yaml["link_connection_type"] == "linear_bushing_with_ball":
            params.link_connection_type = LinkConnectionType.kLinearBushingBallConstraint
        elif params_yaml["link_connection_type"] == "linear_bushing":
            params.link_connection_type = LinkConnectionType.kLinearBushingRpy
        elif params_yaml["link_connection_type"] == "ball":
            params.link_connection_type = LinkConnectionType.kBallRpy
        elif params_yaml["link_connection_type"] == "revolute":
            params.link_connection_type = LinkConnectionType.kRevolute
        else:
            raise RuntimeWarning("Unsupported link connection type")
        
        params.link_length = params_yaml["link_length"]
        if "mass_per_unit_len" in params_yaml:
            params.mass_per_unit_len = params_yaml["mass_per_unit_len"]
            params.link_mass = params.mass_per_unit_len*params.link_length
        elif "link_mass" in params_yaml:
            params.link_mass = params_yaml["link_mass"]
        else:
            raise RuntimeError("Need either mass per unit length or link mass")
        params.rope_radius = params_yaml["rope_radius"]
        params.num_links = params_yaml["num_links"]

        if params.link_connection_type == LinkConnectionType.kLinearBushingBallConstraint or params.link_connection_type == LinkConnectionType.kLinearBushingRpy:
            params.torque_stiffness = params_yaml["torque_stiffness"]
            params.torque_damping = params_yaml["torque_damping"]
            params.force_stiffness = params_yaml["force_stiffness"]
            params.force_damping = params_yaml["force_damping"]
        elif params.link_connection_type == LinkConnectionType.kBallRpy or params.link_connection_type == LinkConnectionType.kRevolute:
            params.ball_damping = params_yaml["ball_damping"]
            params.ball_stiffness = params_yaml["ball_stiffness"]
            params.ball_stiffness_roll = params_yaml["ball_stiffness_roll"]


        if params_yaml["lead"]:
            params.lead = True
            params.lead_mass = params_yaml["lead_mass"]
            params.lead_radius = params_yaml["lead_radius"]
            params.lead_inertia = SpatialInertia.SolidSphereWithMass(params.lead_mass, params.lead_radius).get_unit_inertia()*params.lead_mass
        
        if "link_inertia" in params_yaml:
            params.unit_inertia = UnitInertia(float(params_yaml["link_inertia"]["ixx"]), float(params_yaml["link_inertia"]["iyy"]), float(params_yaml["link_inertia"]["izz"]))
        else:
            params.unit_inertia = SpatialInertia.SolidCylinderWithMass(params.link_mass, params.rope_radius, params.link_length, np.array([1,0,0])).get_unit_inertia()*params.link_mass
        
        params.self_contact = params_yaml["self_contact"]

        if "rope_color_rgba" in params_yaml:
            params.rope_color_rgba = params_yaml["rope_color_rgba"]

        if "mass_at_link_center" in params_yaml:
            params.mass_at_link_center = params_yaml["mass_at_link_center"]

        if "collision" in params_yaml:
            params.collision = params_yaml["collision"]
        
        if "mu_dynamic" in params_yaml:
            params.mu_dynamic = params_yaml["mu_dynamic"]

        return params
    
    def to_yaml(self, yaml_file_path):
        if self.link_connection_type == LinkConnectionType.kLinearBushingBallConstraint:
            link_connection_str = "linear_bushing_with_ball"
        elif self.link_connection_type == LinkConnectionType.kLinearBushingRpy:
            link_connection_str = "linear_bushing"
        elif self.link_connection_type == LinkConnectionType.kBallRpy:
            link_connection_str = "ball"
        elif self.link_connection_type == LinkConnectionType.kRevolute:
            link_connection_str = "revolute"
        else:
            raise RuntimeWarning("Unsupported link connection type")

        data_to_dump = {
            "link_connection_type": link_connection_str,
            "mass_per_unit_len": self.mass_per_unit_len,
            "rope_radius": self.rope_radius,
            "link_length": self.link_length,
            "num_links": self.num_links,
            "torque_stiffness": self.torque_stiffness,
            "torque_damping": self.torque_damping,
            "force_stiffness": self.force_stiffness,
            "force_damping": self.force_damping,
            "ball_damping": self.ball_damping,
            "ball_stiffness": self.ball_stiffness,
            "lead_mass": self.lead_mass,
            "lead_radius": self.lead_radius,
            "self_contact": self.self_contact,
        }

        with open(yaml_file_path, 'w') as file:
            yaml.safe_dump(data_to_dump, file)



def generate_rope_sdf(params: RopeParams, template_path="models/rope/sdf/rope_template.sdf", prefix=""):
    tree = ET.parse(template_path)
    root = tree.getroot()
    root.set("xmlns:drake", "http://drake.mit.edu")
    model = root.find(".//model")
    model.append(ET.Comment("**BELOW AUTOGENERATED** by rope_model_generator.py"))
    model.append(ET.Comment("modifying by hand not recommended."))
    model.set("name", f"{prefix}rope")
    # print(list(model.iter()))
    i = 0
    last_link_name = None
    for i in range(params.num_links):
        link_name = f"{prefix}link_{i}"
        link = ET.SubElement(model, "link")
        link.set("name", link_name)
        pose = ET.SubElement(link, "pose")
        if i!=0:
            last_link_name = f"{prefix}link_{i-1}"
            pose.set("relative_to", last_link_name)
        
        if params.mass_at_link_center:
            pose.text = f"{params.link_length/2 if i==0 else params.link_length} 0 0 0 0 0"
        else:
            pose.text = f"{params.link_length} 0 0 0 0 0"

        link_inertial = ET.SubElement(link, "inertial")
        mass = ET.SubElement(link_inertial, "mass")
        mass.text = str(params.link_mass)
        link_inertia = ET.SubElement(link_inertial, "inertia")
        ixx = ET.SubElement(link_inertia, "ixx")
        ixx.text = str(params.unit_inertia[0, 0])
        ixy = ET.SubElement(link_inertia, "ixy")
        ixy.text = "0"
        ixz = ET.SubElement(link_inertia, "ixz")
        ixz.text = "0"
        iyy = ET.SubElement(link_inertia, "iyy")
        iyy.text = str(params.unit_inertia[1, 1])
        iyz = ET.SubElement(link_inertia, "iyz")
        iyz.text = "0"
        izz = ET.SubElement(link_inertia, "izz")
        izz.text = str(params.unit_inertia[2, 2])

        if i != 0: # Skip visual for first link
            link_visual = ET.SubElement(link, "visual")
            link_visual.set("name", f"{link_name}_visual")
            pose = ET.SubElement(link_visual, "pose")
            pose.set("degrees", "True")
            if params.mass_at_link_center:
                pose.text = f"0 0 0 0 90 0"
            else:
                pose.text = f"-{params.link_length/2} 0 0 0 90 0"
            visual_geometry = ET.SubElement(link_visual, "geometry")
            visual_capsule = ET.SubElement(visual_geometry, "capsule")
            ET.SubElement(visual_capsule, "radius").text = str(params.rope_radius)
            ET.SubElement(visual_capsule, "length").text = str(params.link_length*params.link_extra_scale)
            material = ET.SubElement(link_visual, "material")
            ET.SubElement(material, "diffuse").text = " ".join(str(x) for x in params.rope_color_rgba)

        if params.collision:
            link_collision = ET.SubElement(link, "collision")
            link_collision.set("name", f"{link_name}_collision")
            pose = ET.SubElement(link_collision, "pose")
            pose.set("degrees", "True")
            pose.text = f"0 0 0 0 90 0"
            collision_geometry = ET.SubElement(link_collision, "geometry")
            collision_capsule = ET.SubElement(collision_geometry, "capsule")
            ET.SubElement(collision_capsule, "radius").text = str(params.rope_radius)
            ET.SubElement(collision_capsule, "length").text = str(params.link_length*params.link_extra_scale)
            prox_props = ET.SubElement(link_collision, "drake:proximity_properties")
            ET.SubElement(prox_props, "drake:mu_dynamic").text = str(params.mu_dynamic)

        A_frame = ET.SubElement(model, "frame")
        A_frame.set("name", f"{link_name}_A")
        A_frame.set("attached_to", link_name)
        pose = ET.SubElement(A_frame, "pose")
        if params.mass_at_link_center:
            pose.text = f"-{params.link_length/2} 0 0 0 0 0"
        else:
            pose.text = f"-{params.link_length} 0 0 0 0 0"
        B_frame = ET.SubElement(model, "frame")
        B_frame.set("name", f"{link_name}_B")
        B_frame.set("attached_to", link_name)
        pose = ET.SubElement(B_frame, "pose")
        pose.text = f"{params.link_length/2} 0 0 0 0 0"

    for i in range(params.num_links-1):
        if params.link_connection_type == LinkConnectionType.kBallRpy:
            joint = ET.SubElement(model, "joint")
            joint.set("name", f"joint_{i}_{i+1}")
            joint.set("type", f"ball")
            ET.SubElement(joint, "parent").text = f"{prefix}link_{i}_B"
            ET.SubElement(joint, "child").text = f"{prefix}link_{i+1}_A"
            axis = ET.SubElement(joint, "axis")
            dynamics = ET.SubElement(axis, "dynamics")
            ET.SubElement(dynamics, "damping").text = str(params.ball_damping)
            ET.SubElement(dynamics, "spring_stiffness").text = str(params.ball_stiffness)
            # TODO Super hacky way to specify roll stiffness through sdf format
            ET.SubElement(dynamics, "spring_reference").text = str(params.ball_stiffness_roll) 
            limit = ET.SubElement(axis, "limit")
            ET.SubElement(limit, "effort").text = "0.0"
        elif params.link_connection_type == LinkConnectionType.kRevolute:
            joint = ET.SubElement(model, "joint")
            joint.set("name", f"{prefix}joint_{i}_{i+1}")
            joint.set("type", f"revolute")
            ET.SubElement(joint, "parent").text = f"{prefix}link_{i}_B"
            ET.SubElement(joint, "child").text = f"{prefix}link_{i+1}_A"
            axis = ET.SubElement(joint, "axis")
            xyz = ET.SubElement(axis, "xyz")
            xyz.text = "0 -1 0"
            xyz.set("expressed_in", "__model__")
            dynamics = ET.SubElement(axis, "dynamics")
            ET.SubElement(dynamics, "damping").text = str(params.ball_damping)
            ET.SubElement(dynamics, "spring_stiffness").text = str(params.ball_stiffness)
            limit = ET.SubElement(axis, "limit")
            ET.SubElement(limit, "effort").text = "0.0"
        elif params.link_connection_type == LinkConnectionType.kLinearBushingRpy or params.link_connection_type == LinkConnectionType.kLinearBushingBallConstraint:
            linear_bushing = ET.SubElement(model, "drake:linear_bushing_rpy")
            ET.SubElement(linear_bushing, "drake:bushing_frameA").text = f"{prefix}link_{i}_B"
            ET.SubElement(linear_bushing, "drake:bushing_frameC").text = f"{prefix}link_{i+1}_A"
            ET.SubElement(linear_bushing, "drake:bushing_torque_stiffness").text = " ".join(str(x) for x in params.torque_stiffness)
            ET.SubElement(linear_bushing, "drake:bushing_torque_damping").text = " ".join(str(x) for x in params.torque_damping)
            ET.SubElement(linear_bushing, "drake:bushing_force_stiffness").text = " ".join(str(x) for x in params.force_stiffness)
            ET.SubElement(linear_bushing, "drake:bushing_force_damping").text = " ".join(str(x) for x in params.force_damping)

            if params.link_connection_type == LinkConnectionType.kLinearBushingBallConstraint:
                ball_constraint = ET.SubElement(model, "drake:ball_constraint")
                ET.SubElement(ball_constraint, "drake:ball_constraint_body_A").text = f"{prefix}link_{i}"
                ET.SubElement(ball_constraint, "drake:ball_constraint_p_AP").text = f"{params.link_length/2} 0 0"
                ET.SubElement(ball_constraint, "drake:ball_constraint_body_B").text = f"{prefix}link_{i+1}"
                ET.SubElement(ball_constraint, "drake:ball_constraint_p_BQ").text = f"-{params.link_length/2} 0 0"

        if params.self_contact:
            collision_group = ET.SubElement(model, "drake:collision_filter_group")
            collision_group.set("name", f"{prefix}link_{i}_{i+1}_group")
            ET.SubElement(collision_group, "drake:member").text = f"{prefix}link_{i}"
            ET.SubElement(collision_group, "drake:member").text = f"{prefix}link_{i+1}"
            ET.SubElement(collision_group, "drake:ignored_collision_filter_group").text = f"{prefix}link_{i}_{i+1}_group"
    
    if not params.self_contact:
        collision_group = ET.SubElement(model, "drake:collision_filter_group")
        collision_group.set("name", f"rope_group")
        ET.SubElement(collision_group, "drake:ignored_collision_filter_group").text = f"rope_group"
        for i in range(params.num_links):
            ET.SubElement(collision_group, "drake:member").text = f"{prefix}link_{i}"

    if params.lead:
        link_name = f"rope_lead"
        link = ET.SubElement(model, "link")
        link.set("name", link_name)
        pose = ET.SubElement(link, "pose")
        pose.set("relative_to", f"{prefix}link_{params.num_links-1}")
        pose.text = f"{params.link_length/2} 0 0 0 0 0"

        link_inertial = ET.SubElement(link, "inertial")
        mass = ET.SubElement(link_inertial, "mass")
        mass.text = str(params.lead_mass)
        link_inertia = ET.SubElement(link_inertial, "inertia")
        ixx = ET.SubElement(link_inertia, "ixx")
        ixx.text = str(params.lead_inertia[0, 0])
        ixy = ET.SubElement(link_inertia, "ixy")
        ixy.text = "0"
        ixz = ET.SubElement(link_inertia, "ixz")
        ixz.text = "0"
        iyy = ET.SubElement(link_inertia, "iyy")
        iyy.text = str(params.lead_inertia[1, 1])
        iyz = ET.SubElement(link_inertia, "iyz")
        iyz.text = "0"
        izz = ET.SubElement(link_inertia, "izz")
        izz.text = str(params.lead_inertia[2, 2])
    
        # link_visual = ET.SubElement(link, "visual")
        # link_visual.set("name", f"{link_name}_visual")
        # pose = ET.SubElement(link_visual, "pose")
        # pose.set("relative_to", f"{prefix}link_{params.num_links-1}")
        # pose.text = f"{params.link_length/2} 0 0 0 0 0"
        # visual_geometry = ET.SubElement(link_visual, "geometry")
        # visual_capsule = ET.SubElement(visual_geometry, "sphere")
        # ET.SubElement(visual_capsule, "radius").text = str(params.lead_radius)

        # material = ET.SubElement(link_visual, "material")
        # ET.SubElement(material, "diffuse").text = " ".join(str(x) for x in params.rope_color_rgba)
        joint = ET.SubElement(model, "joint")
        joint.set("name", f"joint_lead")
        joint.set("type", f"fixed")
        ET.SubElement(joint, "parent").text = f"{prefix}link_{params.num_links-1}"
        ET.SubElement(joint, "child").text = f"rope_lead"

    indent(root)

    return ET.ElementTree(root)

def et_to_string(tree):
    return ET.tostring(tree.getroot(), encoding='utf8')

def save_et_to_file(tree, path):
    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the rope SDF model")
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(os.path.dirname(__file__), "rope_generated.sdf"),
        help="Output path for the generated SDF file",
    )
    args = parser.parse_args()

    params = RopeParams()
    params.link_connection_type = LinkConnectionType.kLinearBushingBallConstraint
    tree = generate_rope_sdf(params)
    save_et_to_file(tree, args.output)