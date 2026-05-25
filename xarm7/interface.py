from pydrake.all import *
from xarm.wrapper import XArmAPI
import pickle
import time
from datetime import datetime
import requests
import tarfile
import shutil



def xarm_send_trajectory(xarm : XArmAPI, traj: Trajectory, name, robot_ip="192.168.1.202", start_buffer=0, end_buffer=0):
    """ Send a resample_hz Hz sampled trajectory to xarm
    Just need to call xarm.playback_trajectory() after this
    """
    dt = 1/250
    assert len(traj.value(0)) == 7
    if os.path.exists("/tmp/traj"):
        shutil.rmtree("/tmp/traj")

    # only_idx = 6
    os.mkdir("/tmp/traj")
    with open(f"/tmp/traj/{name}.traj", "w") as f:
        f.write(f"# frequency=250.000000\n")
        start_cfg = traj.value(traj.start_time()).T[0]
        print(start_cfg)

        for _ in np.arange(0, start_buffer, dt):
            cmd = ",".join([f"{b:.6f}" for b in start_cfg])
            f.write(f"{cmd}\n")

        for a_ in traj.vector_values(np.arange(0, traj.end_time()+dt, dt)).T:
            a = a_
            cmd = ",".join([f"{b:.6f}" for b in a])
            f.write(f"{cmd}\n")

        end_cfg = traj.value(traj.end_time()).T[0]
        for _ in np.arange(0, end_buffer, dt):
            a = end_cfg
            cmd = ",".join([f"{b:.6f}" for b in a])
            f.write(f"{cmd}\n")
    
    with tarfile.open(f"/tmp/traj-{name}.tar.gz", "w:gz") as tar:
        tar.add("/tmp/traj", arcname=os.path.basename("/tmp/traj"))

    xarm.delete_trajectory(name)

    url = f"http://{robot_ip}:18333/project/upload"
    data = {
        "path": "test/xarm7/traj",
    }

    files = {
        "file": (f"traj-{name}.tar.gz", open(f"/tmp/traj-{name}.tar.gz", "rb"), "application/x-gzip"),
    }

    r = requests.post(
        url,
        data=data,
        files=files,
        headers={
            "Origin": f"http://{robot_ip}:18333",
            "Referer": f"http://{robot_ip}:18333/control?lang=en&channel=prod&path=%2Fcontrol",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "python-requests",
        },
        timeout=30,
    )
    print(r.status_code, r.text)


    xarm.load_trajectory(f"{name}.traj")


def xarm_move_to_trajectory_start(xarm: XArmAPI, traj: Trajectory, speed=50, mvacc=500):
    xarm.set_mode(0)
    xarm.set_state(0)
    xarm.set_servo_angle(angle=traj.value(0).T[0, -7:].tolist(), speed=speed, mvacc=mvacc, wait=True, is_radian=True)


def xarm_reset_rope(xarm: XArmAPI, traj: Trajectory):
    # TODO
    xarm.set_mode(0)
    xarm.set_state(0)
    print(arm.get_position(), arm.get_position(is_radian=True))
    xarm.set_servo_angle(angle=traj.value(0).T[0, -7:].tolist(), speed=10, mvacc=10, wait=True, is_radian=True)


if __name__ == "__main__":
    arm = XArmAPI("192.168.1.202")
    arm.motion_enable()
    arm.set_mode(0)
    arm.set_state(0)

    cmd = np.zeros((100, 7))
    cmd[:, 0] = -np.pi/2
    cmd[:, 0] = np.pi/2

    traj = PiecewisePolynomial.FirstOrderHold(np.linspace(0, 5, 100), cmd.T)

    xarm_command_file = "command.pickle"
    with open("command.pickle", 'rb') as f:
        traj : BsplineTrajectory = pickle.load(f)

    xarm_move_to_trajectory_start(arm, traj)
    input("Ready to move")
    xarm_send_trajectory(arm, traj, "abc")

    arm.playback_trajectory()