import socket
import time
import pickle
import struct
from dataclasses import dataclass
from typing import Optional
from xarm7.kinematics import *
from xarm7.visualize import set_xarm_visual, add_xarm_visual
import threading
from pydrake.all import StartMeshcat, Meshcat


def bytes_to_fp32(bytes_data, is_big_endian=False):
    return struct.unpack('>f' if is_big_endian else '<f', bytes_data)[0]

def bytes_to_fp32_list(bytes_data, n=0, is_big_endian=False):
    ret = []
    count = n if n > 0 else len(bytes_data) // 4
    for i in range(count):
        ret.append(bytes_to_fp32(bytes_data[i * 4: i * 4 + 4], is_big_endian))
    return ret

def bytes_to_u8(data, is_big_endian=False):
    return int.from_bytes(data[:1], byteorder='big' if is_big_endian else 'little')

def bytes_to_u16(data, is_big_endian=False):
    return int.from_bytes(data[:2], byteorder='big' if is_big_endian else 'little')

def bytes_to_u32(data, is_big_endian=False):
    return int.from_bytes(data[:4], byteorder='big' if is_big_endian else 'little')

def bytes_to_u64(data, is_big_endian=False):
    return int.from_bytes(data[:8], byteorder='big' if is_big_endian else 'little')

@dataclass
class XarmRawDataFrame:
    num_bytes: int                      # [1-4]   U32
    timestamp_us: int                   # [5-12]  U64 (μs)
    motion_status_mode: int             # [13]    U8
    num_cached_cmds: int                # [14-15] U16

    target_joint_pos: list[float]       # [33-60]  7*FP32  rad
    target_joint_vel: list[float]       # [61-88]  7*FP32  rad/s
    target_joint_acc: list[float]       # [89-116] 7*FP32  rad/s^2
    actual_joint_pos: list[float]       # [117-144] 7*FP32
    actual_joint_vel: list[float]       # [145-172] 7*FP32
    actual_joint_acc: list[float]       # [173-200] 7*FP32
    actual_joint_curr: list[float]      # [201-228] 7*FP32 A
    est_joint_torque: list[float]       # [229-256] 7*FP32 N·m

    target_tcp_pose: list[float]        # [425-448] 6*FP32 (mm & rad)
    target_tcp_speed: list[float]       # [449-472] 6*FP32 (mm/s & rad/s)
    actual_tcp_pose: list[float]        # [473-496] 6*FP32 (mm & rad)
    actual_tcp_speed: list[float]       # [497-520] 6*FP32 (mm/s & rad/s)
    est_tcp_torques: list[float]        # [521-544] 6*FP32 (N & N·m)

    ft_raw: Optional[list[float]]       # [689-712] 6*FP32 [Fx,Fy,Fz,Tx,Ty,Tz]
    ft_filt: Optional[list[float]]      # [713-736] 6*FP32 [Fx,Fy,Fz,Tx,Ty,Tz]

    @classmethod
    def parse_frame_bytes(cls, frame_bytes: bytes):
        if len(frame_bytes) < 4:
            raise ValueError("Frame too short for length field")

        num_bytes = bytes_to_u32(frame_bytes[:4], True)
        if len(frame_bytes) != num_bytes:
            raise ValueError(f"Length mismatch: header said {num_bytes}, got {len(frame_bytes)}")
        
        timestamp_us       = bytes_to_u64(frame_bytes[4:12], True)
        motion_status_mode = frame_bytes[12]

        num_cached_cmds    = bytes_to_u16(frame_bytes[13:15], True)

        target_joint_pos  = bytes_to_fp32_list(frame_bytes[32:60], 7)
        target_joint_vel  = bytes_to_fp32_list(frame_bytes[60:88], 7)
        target_joint_acc  = bytes_to_fp32_list(frame_bytes[88:116], 7)

        actual_joint_pos  = bytes_to_fp32_list(frame_bytes[116:144], 7)
        actual_joint_vel  = bytes_to_fp32_list(frame_bytes[144:172], 7)
        actual_joint_acc  = bytes_to_fp32_list(frame_bytes[172:200], 7)
        actual_joint_curr = bytes_to_fp32_list(frame_bytes[200:228], 7)

        est_joint_torque = bytes_to_fp32_list(frame_bytes[228:256], 7)

        target_tcp_pose = bytes_to_fp32_list(frame_bytes[424:448], 6)
        target_tcp_speed = bytes_to_fp32_list(frame_bytes[448:472], 6)
        actual_tcp_pose = bytes_to_fp32_list(frame_bytes[472:496], 6)
        actual_tcp_speed = bytes_to_fp32_list(frame_bytes[496:520], 6)
        est_tcp_torques = bytes_to_fp32_list(frame_bytes[520:544], 6)

        ft_raw = bytes_to_fp32_list(frame_bytes[688:712], 6)
        ft_filt = bytes_to_fp32_list(frame_bytes[712:736], 6)

        return XarmRawDataFrame(
            num_bytes=num_bytes,
            timestamp_us=timestamp_us,
            motion_status_mode=motion_status_mode,
            num_cached_cmds=num_cached_cmds,
            target_joint_pos=target_joint_pos,
            target_joint_vel=target_joint_vel,
            target_joint_acc=target_joint_acc,
            actual_joint_pos=actual_joint_pos,
            actual_joint_vel=actual_joint_vel,
            actual_joint_acc=actual_joint_acc,
            actual_joint_curr=actual_joint_curr,
            est_joint_torque=est_joint_torque,
            target_tcp_pose=target_tcp_pose,
            target_tcp_speed=target_tcp_speed,
            actual_tcp_pose=actual_tcp_pose,
            actual_tcp_speed=actual_tcp_speed,
            est_tcp_torques=est_tcp_torques,
            ft_raw=ft_raw,
            ft_filt=ft_filt,
        )
        

def capture_xarm_raw_data(stop_event: threading.Event, byte_data: list[bytes], verbose=False, visualize=False, robot_ip = '192.168.1.202'):
    if visualize:
        meshcat: Meshcat = StartMeshcat()
        plant = xarm_plant_3d("xarm7_no_hand")
        plant_context = plant.CreateDefaultContext()
        add_xarm_visual(meshcat)
    # Read Data
    
    robot_port = 30000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(True)
    sock.settimeout(1)
    sock.connect((robot_ip, robot_port))

    buffer = sock.recv(4)
    while len(buffer) < 4:
        buffer += sock.recv(4 - len(buffer))
    size = bytes_to_u32(buffer[:4], True)

    if verbose:
        print(f"Read header, frame size {size}")

    while not stop_event.is_set():
        buffer += sock.recv(size - len(buffer))
        if len(buffer) < size:
            continue

        data = buffer[:size]
        buffer = buffer[size:]
        byte_data.append(data)
        if verbose:
            parsed_data : XarmRawDataFrame = XarmRawDataFrame.parse_frame_bytes(data)
            print(f"Frame timestamp: {parsed_data.timestamp_us}")
        if visualize:
            parsed_data : XarmRawDataFrame = XarmRawDataFrame.parse_frame_bytes(data)
            plant.SetPositions(plant_context, parsed_data.actual_joint_pos)
            set_xarm_visual(meshcat, plant, plant_context)

if __name__ == "__main__":
    stop_event = threading.Event()
    data: list[bytes] = []
    t = threading.Thread(target=capture_xarm_raw_data, args=(stop_event, data, True, True))
    t.start()

    time.sleep(3)
    stop_event.set()
    t.join()  

    print(f"Captured {len(data)} frames")

    # #Save Data
    # with open("raw_data.pickle", 'wb') as f:
    #     pickle.dump(data, f,protocol=pickle.HIGHEST_PROTOCOL)

    # # Load and visualize
    # with open("raw_data.pickle", 'rb') as f:
    #     parsed_data_list: list[XarmRawDataFrame] = [XarmRawDataFrame.parse_frame_bytes(data) for data in pickle.load(f)]

    # fps = 250
    # animation = MeshcatAnimation(fps)
    # animate_xarm_states(meshcat, animation, plant, np.array([d.actual_joint_pos for d in parsed_data_list]), dt=1/fps)
    # meshcat.SetAnimation(animation)
    # input()
        

