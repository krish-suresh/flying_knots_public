import viser
import time
import threading
import numpy as np
import trimesh
from pydrake.all import RollPitchYaw, RigidTransform,PiecewisePolynomial, PiecewisePose
from common.math import cart_to_polar_3d
from common.data import RopeTrajectory
import logging



class ViserAnimation:
    def __init__(self, end_time, server = None, visualization_fps = 200):
        if server is None:
            server = viser.ViserServer(port=8081)
        self.visualization_fps = visualization_fps
        self.server = server
        self.end_time = end_time
        self.update_callback = []
        with self.server.gui.add_folder("Playback"):
            self.gui_frame_slider = self.server.gui.add_slider(
                "Time", 0, end_time, 1 / self.visualization_fps, 0
            )
            self.gui_frame_step_buttons = server.gui.add_button_group(
                "Step", ["<<", "<", ">", ">>"]
            )
            self.gui_play_button = self.server.gui.add_button(
                "Play", icon=viser.Icon.PLAYER_PLAY_FILLED
            )
            self.gui_pause_button = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
            self.gui_play_speed = server.gui.add_number(
                "Speed", 1.5  # TODO hack to allow floats
            )

        @self.gui_play_button.on_click
        def _(_) -> None:
            self.gui_play_button.icon = viser.Icon.PLAYER_PLAY_FILLED
            self.gui_pause_button.icon = viser.Icon.PLAYER_PAUSE
            if self.gui_frame_slider.value == self.gui_frame_slider.max:
                self.gui_frame_slider.value = 0

        @self.gui_pause_button.on_click
        def _(_) -> None:
            self.gui_play_button.icon = viser.Icon.PLAYER_PLAY
            self.gui_pause_button.icon = viser.Icon.PLAYER_PAUSE_FILLED

        @self.gui_frame_step_buttons.on_click
        def _(_) -> None:
            match self.gui_frame_step_buttons.value:
                case "<<":
                    self.gui_frame_slider.value = max(
                        self.gui_frame_slider.min,
                        self.gui_frame_slider.value - 5.0 / visualization_fps,
                    )
                case "<":
                    self.gui_frame_slider.value = max(
                        self.gui_frame_slider.min,
                        self.gui_frame_slider.value - 1.0 / visualization_fps,
                    )
                case ">>":
                    self.gui_frame_slider.value = min(
                        self.gui_frame_slider.max,
                        self.gui_frame_slider.value + 5.0 / visualization_fps,
                    )
                case ">":
                    self.gui_frame_slider.value = min(
                        self.gui_frame_slider.max,
                        self.gui_frame_slider.value + 1.0 / visualization_fps,
                    )

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            # TODO load these from somewhere better
            # client.camera.wxyz = np.array(
            #     [0.61311877, -0.73316445, -0.22568724,  0.18873403]
            # )
            # client.camera.position = np.array([1.43572065, -1.35890189, 0.40179528])
            self.gui_play_speed.value = 1
    
    def play(self):
        last_frame_idx = None
        while True:
            frame_idx = self.gui_frame_slider.value
            playing = self.gui_play_button.icon == viser.Icon.PLAYER_PLAY_FILLED
            if not playing and frame_idx == last_frame_idx and not refresh:
                continue

            refresh = False
            last_frame_idx = frame_idx

            if playing:
                self.gui_frame_slider.value = min(
                    self.gui_frame_slider.max, self.gui_frame_slider.value + 1.0 / self.visualization_fps
                )
                # Pause if at end
                if self.gui_frame_slider.value == self.gui_frame_slider.max:
                    self.gui_play_button.icon = viser.Icon.PLAYER_PLAY
                    self.gui_pause_button.icon = viser.Icon.PLAYER_PAUSE_FILLED

            for f in self.update_callback:
                f(self.gui_frame_slider.value)

            play_speed = 1 if self.gui_play_speed.value == 0 else self.gui_play_speed.value
            time.sleep(1 / (play_speed * self.visualization_fps))
            
class ViserAnimationRealtime:
    def __init__(self, server : viser.ViserServer, visualization_fps = 60, default_play_speed=1.0):
        self.visualization_fps = visualization_fps
        self.default_play_speed = default_play_speed
        self.server = server
        self.animated_objects = {}
        self._animated_objects_lock = threading.Lock()
        self._refresh_needed = False
        
        with self.server.gui.add_folder("Playback"):
            self.gui_time_slider = self.server.gui.add_slider(
                "Time", 0, 0.001, 1 / self.visualization_fps, 0
            )
            self.gui_frame_step_buttons = server.gui.add_button_group(
                "Step", ["<<", "<", ">", ">>"]
            )
            self.gui_play_button = self.server.gui.add_button(
                "Play", icon=viser.Icon.PLAYER_PLAY
            )
            self.gui_pause_button = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE_FILLED)
            self.gui_play_speed = server.gui.add_number(
                "Speed", 1.5, step=0.001  # hack to allow floats
            )
            self.gui_speed_buttons = server.gui.add_button_group(
                "Speed Presets", ["0.1x", "0.25x", "0.5x", "1x"]
            )
        @self.gui_play_button.on_click
        def _(_) -> None:
            self.play()
            if self.gui_time_slider.value == self.gui_time_slider.max:
                self.gui_time_slider.value = 0

        @self.gui_pause_button.on_click
        def _(_) -> None:
            self.pause()

        @self.gui_frame_step_buttons.on_click
        def _(_) -> None:
            match self.gui_frame_step_buttons.value:
                case "<<":
                    self.gui_time_slider.value = max(
                        self.gui_time_slider.min,
                        self.gui_time_slider.value - 5.0 / visualization_fps,
                    )
                case "<":
                    self.gui_time_slider.value = max(
                        self.gui_time_slider.min,
                        self.gui_time_slider.value - 1.0 / visualization_fps,
                    )
                case ">>":
                    self.gui_time_slider.value = min(
                        self.gui_time_slider.max,
                        self.gui_time_slider.value + 5.0 / visualization_fps,
                    )
                case ">":
                    self.gui_time_slider.value = min(
                        self.gui_time_slider.max,
                        self.gui_time_slider.value + 1.0 / visualization_fps,
                    )

        @self.gui_speed_buttons.on_click
        def _(_) -> None:
            self.gui_play_speed.value = float(self.gui_speed_buttons.value[:-1])

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            # TODO load these from somewhere better
            # client.camera.wxyz = np.array(
            #     [0.61311877, -0.73316445, -0.22568724,  0.18873403]
            # )
            # client.camera.position = np.array([1.43572065, -1.35890189, 0.40179528])
            self.gui_play_speed.value = self.default_play_speed

        self._run_thread = threading.Thread(
            target=self.run,
            name="ViserAnimationRealtime.play",
            daemon=True,
        )
        self._run_thread.start()
    
    def add_animated_object(self, handle, traj: PiecewisePose):
        with self._animated_objects_lock:
            self.animated_objects[handle.name] = (handle, traj)
            self._refresh_needed = True
        if traj.end_time() > self.gui_time_slider.max:
            self.gui_time_slider.max = traj.end_time()

    def clear(self):
        with self._animated_objects_lock:
            self.animated_objects.clear()
            self._refresh_needed = True

    def reset(self):
        self.gui_time_slider.value = 0


    def play(self):
        self.gui_play_button.icon = viser.Icon.PLAYER_PLAY_FILLED
        self.gui_pause_button.icon = viser.Icon.PLAYER_PAUSE

    def pause(self):
        self.gui_play_button.icon = viser.Icon.PLAYER_PLAY
        self.gui_pause_button.icon = viser.Icon.PLAYER_PAUSE_FILLED
        
    
    def run(self):
        if (
            hasattr(self, "_run_thread")
            and self._run_thread.is_alive()
            and threading.current_thread() is not self._run_thread
        ):
            return
        t_last = None
        frame_period = 1.0 / self.visualization_fps
        last_tick = time.perf_counter()
        play_speed = 1
        while True:
            now = time.perf_counter()
            elapsed = now - last_tick
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
                continue
            last_tick = now
            play_speed = 1 if self.gui_play_speed.value == 0 else self.gui_play_speed.value
            t_current = self.gui_time_slider.value
            playing = self.gui_play_button.icon == viser.Icon.PLAYER_PLAY_FILLED
            with self._animated_objects_lock:
                refresh = self._refresh_needed
                self._refresh_needed = False
                animated_items = list(self.animated_objects.items())
            if not playing and t_current == t_last and not refresh:
                continue
            
            if t_current == 0.0:
                self.a = time.perf_counter()
            
            if playing:
                next_t = t_current + (play_speed * elapsed)
                if next_t >= self.gui_time_slider.max:
                    self.gui_time_slider.value = self.gui_time_slider.max
                    self.pause()
                    logging.debug(f"Visualization time: {time.perf_counter() - self.a}")
                else:
                    self.gui_time_slider.value = next_t
                # Pause if at end
            t_render = self.gui_time_slider.value
            t_last = t_render

            for name, (handle, pose_traj) in animated_items:
                X: RigidTransform = pose_traj.GetPose(t_render)
                handle.position = X.translation()
                handle.wxyz = X.rotation().ToQuaternion().wxyz()




def add_rope_visual(
    server: viser.ViserServer,
    rope_length,
    rope_radius,
    num_links=200,
    name: str = "rope",
    color=(200, 200, 200),
    visible=True,
    opacity=None,
):
    link_length = rope_length / num_links
    rope_link_mesh = trimesh.creation.capsule(link_length, rope_radius)

    rope_mesh_handles = []
    for i in range(int(num_links)):
        rope_mesh_handles.append(
                server.scene.add_mesh_simple(
                    f"{name}/link_{i}/capsule",
                    rope_link_mesh.vertices,
                    rope_link_mesh.faces,
                    visible=visible,
                    color=color,
                    opacity=opacity,
                )
        )

    return rope_mesh_handles


def set_rope_visual(
    rope_mesh_handles: list[viser.MeshHandle],
    rope_curve: PiecewisePolynomial
):
    rope_points = rope_curve.vector_values(np.linspace(0, 1, len(rope_mesh_handles)+1)).T
    for i, m in enumerate(rope_mesh_handles):
        m.position = (rope_points[i] + rope_points[i + 1]) / 2
        u, v = cart_to_polar_3d(rope_points[i + 1] - rope_points[i])
        m.wxyz = RollPitchYaw(0, np.pi / 2 - v, u).ToQuaternion().wxyz()

def get_rope_visual_traj(rope_mesh_handles: list[viser.MeshHandle], rope_traj : RopeTrajectory):
    sample_times = np.linspace(0, rope_traj.end_time, int(rope_traj.end_time*200)+1)
    body_poses = [[] for _ in range(len(rope_mesh_handles))]
    for t in sample_times:
        rope_curve = rope_traj.fit_curve_to_rope(t)
        rope_points = rope_curve.vector_values(np.linspace(0, 1, len(rope_mesh_handles)+1)).T
        for i, m in enumerate(rope_mesh_handles):
            pos = (rope_points[i] + rope_points[i + 1]) / 2
            u, v = cart_to_polar_3d(rope_points[i + 1] - rope_points[i])
            rpy = RollPitchYaw(0, np.pi / 2 - v, u)
            
            body_poses[i].append(RigidTransform(rpy, pos))

    output = []
    for m, p in zip(rope_mesh_handles, body_poses):
        output.append((m, PiecewisePose.MakeLinear(sample_times, p)))
    
    return output

if __name__ == "__main__":
    server = viser.ViserServer(port=8081)
    
    box = server.scene.add_box("a")
    traj = PiecewisePose.MakeLinear([0, 1], [RigidTransform([0,0,0]), RigidTransform(RollPitchYaw(np.pi, 0,0), [1,1,1])])
    
    animation = ViserAnimationRealtime(server)
    animation.add_animated_object(box, traj)

    server.sleep_forever()
