import numpy as np
import pyvicon_datastream as pv
from pydrake.all import *
import pickle
from pyvicon_datastream import tools
import logging

class MultiObjectTracker(tools.ObjectTracker):
    def __init__(self, ip, object_names):
        super().__init__(ip)
        self.labeled_marker_names = None
        self.object_names = object_names

    def _get_object_positions(self, names : list):
        subject_count = self.vicon_client.get_subject_count()
        positions = [None] * len(names)
        for subj_idx in range(subject_count):
            subject_name = self.vicon_client.get_subject_name(subj_idx)

            if not subject_name in names: #Skip objects we are not interessted in
                continue
            
            name_idx = names.index(subject_name)

            segment_count = self.vicon_client.get_segment_count(subject_name)
            for seg_idx in range(segment_count):
                segment_name = self.vicon_client.get_segment_name(subject_name, seg_idx)
                segment_global_translation = self.vicon_client.get_segment_global_translation(subject_name, segment_name)
                segment_global_rotation     = self.vicon_client.get_segment_global_quaternion(subject_name, segment_name)

                if segment_global_translation is not None and segment_global_rotation is not None:
                    position_x = segment_global_translation[0]
                    position_y = segment_global_translation[1]
                    position_z = segment_global_translation[2]
                    q_w = segment_global_rotation[0]
                    q_x = segment_global_rotation[1]
                    q_y = segment_global_rotation[2]
                    q_z = segment_global_rotation[3]

                    position_entry = [
                        subject_name, 
                        segment_name, 
                        position_x,
                        position_y,
                        position_z,
                        q_w,
                        q_x,
                        q_y,
                        q_z
                    ]
                    positions[name_idx] = position_entry
        return positions

    def get_positions(self):
        if self.is_connected == True:
            frame = self.vicon_client.get_frame()
            if frame == pv.Result.Success:
                latency     = self.vicon_client.get_latency_total()
                framenumber = self.vicon_client.get_frame_number()
                position    = self._get_object_positions(self.object_names)
                return latency, framenumber, position
        return False

    def get_all_unlabeled_marker_positions(self):
        unlabeled_markers = []
        for i in range(self.vicon_client.get_unlabeled_marker_count()):
            unlabeled_markers.append(
                self.vicon_client.get_unlabeled_marker_global_translation(i)
            )
        return unlabeled_markers

    def get_all_labeled_marker_positions(self):
        if self.labeled_marker_names is None:
            self.labeled_marker_names = {}
            for obj_name in self.object_names:
                self.labeled_marker_names[obj_name] = []
                for i in range(self.vicon_client.get_marker_count(obj_name)):
                    self.labeled_marker_names[obj_name].append(
                        self.vicon_client.get_marker_name(obj_name, i)
                    )

        labeled_markers = {}
        for obj_name in self.object_names:
            labeled_markers[obj_name] = []
            for obj_marker_name in self.labeled_marker_names[obj_name]:
                labeled_markers[obj_name].append(
                    self.vicon_client.get_marker_global_translation(
                        obj_name, obj_marker_name
                    )
                )
        return labeled_markers

def setup_vicon(config) -> MultiObjectTracker:
    vicon_tracker_ip = config["vicon_ip"]
    tracker = MultiObjectTracker(vicon_tracker_ip, config["tracking_objects"])

    if not tracker.is_connected:
        logging.error(f"Failed to connect to Vicon DataStream at {vicon_tracker_ip}")
        raise Exception(f"Connection to {vicon_tracker_ip} failed")
    
    logging.info(f"Connected to Vicon DataStream at {vicon_tracker_ip}")

    if config["enable_unlabeled_marker_data"]:
        tracker.vicon_client.enable_unlabeled_marker_data()
    if config["enable_marker_data"]:
        tracker.vicon_client.enable_marker_data()
    
    return tracker