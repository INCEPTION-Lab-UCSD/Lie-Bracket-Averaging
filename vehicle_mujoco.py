import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

import vehicle_trajectories

MODE_COLORS = {
    1: np.array([0.86, 0.22, 0.18, 1.0]),
    2: np.array([0.95, 0.62, 0.12, 1.0]),
    3: np.array([0.16, 0.55, 0.85, 1.0]),
}

MODE_LABELS = {
    1: "Reversed Measurements",
    2: "Blind to Measurements",
    3: "Normal Measurements",
}

MODE_COLOR_NAMES = {
    1: "red",
    2: "orange",
    3: "blue",
}

TARGET_CENTER_SIZE = 0.10


@dataclass
class VehiclePose:
    position: np.ndarray
    yaw: float
    mode: int


@dataclass
class VehiclePlayback:
    label: str
    solution: object


class MuJoCoVehicleVisualizer:
    """
    MuJoCo playback for existing hybrid vehicle trajectory solutions.

    The paper's kinematics are still integrated by VehicleTrajectorySimulation.
    MuJoCo is used here as a visualization scene: each playback frame overwrites
    the free-joint poses of simple vehicle bodies with sampled hybrid states.
    """

    def __init__(
        self,
        playbacks,
        x_p_goal,
        target_radius=1.0,
        trail_samples=140,
        vehicle_length=0.7,
        vehicle_width=0.35,
    ):
        self.playbacks = list(playbacks)
        if not self.playbacks:
            raise ValueError("playbacks must contain at least one VehiclePlayback")

        self.x_p_goal = np.asarray(x_p_goal, dtype=float)
        self.target_radius = float(target_radius)
        self.trail_samples = int(trail_samples)
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)

        self._samples = [
            self._sample_path(playback.solution, self.trail_samples)
            for playback in self.playbacks
        ]
        self.model = mujoco.MjModel.from_xml_string(self._build_xml())
        self.data = mujoco.MjData(self.model)

        self._vehicle_qpos_addrs = [
            self.model.joint(f"vehicle_{idx}_freejoint").qposadr[0]
            for idx in range(len(self.playbacks))
        ]
        self._chassis_geom_ids = [
            self.model.geom(f"chassis_{idx}").id for idx in range(len(self.playbacks))
        ]
        self._nose_geom_ids = [
            self.model.geom(f"nose_{idx}").id for idx in range(len(self.playbacks))
        ]

        self._set_poses(self.t_start)

    @property
    def t_start(self):
        return min(playback.solution.t[0] for playback in self.playbacks)

    @property
    def t_end(self):
        return max(playback.solution.t[-1] for playback in self.playbacks)

    def _build_xml(self):
        all_xy = np.concatenate([samples[:2] for samples in self._samples], axis=1)
        goal = self.x_p_goal

        min_xy = np.minimum(np.min(all_xy, axis=1), goal) - 2.0
        max_xy = np.maximum(np.max(all_xy, axis=1), goal) + 2.0
        center = 0.5 * (min_xy + max_xy)
        span = np.maximum(max_xy - min_xy, 4.0)
        floor_size = max(span) / 2.0 + 1.0
        camera_distance = max(span) * 1.15
        camera_height = max(span) * 0.85 + 3.0

        trail_geoms = "\n".join(
            self._trail_geoms(playback_idx, samples)
            for playback_idx, samples in enumerate(self._samples)
        )
        start_geoms = "\n".join(
            self._start_geom(playback_idx, samples, self.target_radius / 4)
            for playback_idx, samples in enumerate(self._samples)
        )
        vehicle_bodies = "\n".join(
            self._vehicle_body(playback_idx)
            for playback_idx in range(len(self.playbacks))
        )

        return f"""
<mujoco model="vehicle_trajectory_visualization">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.70 0.70 0.70" diffuse="0.55 0.55 0.55" specular="0.12 0.12 0.12"/>
    <rgba haze="1 1 1 1"/>
    <global offwidth="1280" offheight="720"/>
    <map znear="0.01" zfar="100"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="flat" width="32" height="32"
             rgb1="1 1 1" rgb2="1 1 1"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.96 0.96 0.96" rgb2="0.86 0.88 0.89"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.12"/>
  </asset>
  <worldbody>
    <light name="key" pos="{center[0]:.6f} {center[1]:.6f} 8" dir="0 0 -1"/>
    <camera name="overview"
            pos="{center[0]:.6f} {center[1] - camera_distance:.6f} {camera_height:.6f}"
            xyaxes="1 0 0 0 0.58 0.82"/>
    <geom name="floor" type="plane" pos="{center[0]:.6f} {center[1]:.6f} 0"
          size="{floor_size:.6f} {floor_size:.6f} 0.1" material="grid"/>
    <geom name="target" type="cylinder"
          pos="{goal[0]:.6f} {goal[1]:.6f} 0.012"
          size="{self.target_radius:.6f} 0.012"
          rgba="0.10 0.70 0.34 0.28"/>
    <geom name="target_center" type="sphere"
          pos="{goal[0]:.6f} {goal[1]:.6f} 0.10"
          size="{TARGET_CENTER_SIZE:.6f}" rgba="0.06 0.40 0.20 0.95"/>
{start_geoms}
{trail_geoms}
{vehicle_bodies}
  </worldbody>
</mujoco>
"""

    def _vehicle_body(self, playback_idx):
        wheel_radius = self.vehicle_width / 3
        wheel_half_width = self.vehicle_width / 12
        fork_height = wheel_radius * 1.35
        direction_length = self.vehicle_length * 0.55
        axle_radius = wheel_radius * 0.22
        hub_radius = wheel_radius * 0.32
        return f"""
    <body name="vehicle_{playback_idx}" pos="0 0 0.12">
      <freejoint name="vehicle_{playback_idx}_freejoint"/>
      <geom name="chassis_{playback_idx}" type="cylinder"
            euler="1.570796326795 0 0"
            size="{wheel_radius:.6f} {wheel_half_width:.6f}"
            rgba="0.16 0.55 0.85 1"/>
      <geom name="nose_{playback_idx}" type="capsule"
            fromto="0 0 {fork_height:.6f} {direction_length:.6f} 0 {fork_height:.6f}"
            size="{axle_radius:.6f}"
            rgba="0.05 0.15 0.20 1"/>
      <geom name="axle_{playback_idx}" type="cylinder"
            euler="1.570796326795 0 0"
            size="{axle_radius:.6f} {wheel_half_width * 1.35:.6f}"
            rgba="0.02 0.02 0.02 1"/>
      <geom name="hub_{playback_idx}" type="sphere"
            size="{hub_radius:.6f}" rgba="0.96 0.96 0.96 1"/>
    </body>
"""

    @staticmethod
    def _sample_path(solution, samples):
        times = np.linspace(solution.t[0], solution.t[-1], samples)
        return solution(times)

    def _trail_geoms(self, playback_idx, samples):
        geoms = []
        for sample_idx in range(samples.shape[1] - 1):
            x0, y0 = samples[0, sample_idx], samples[1, sample_idx]
            x1, y1 = samples[0, sample_idx + 1], samples[1, sample_idx + 1]
            if np.hypot(x1 - x0, y1 - y0) < 1e-6:
                continue
            mode = int(round(samples[5, sample_idx]))
            color = MODE_COLORS.get(mode, MODE_COLORS[3])
            rgba = " ".join(f"{value:.3f}" for value in color[:3]) + " 0.82"
            geoms.append(
                f'    <geom name="trail_{playback_idx}_{sample_idx}" type="capsule" '
                f'fromto="{x0:.6f} {y0:.6f} 0.055 {x1:.6f} {y1:.6f} 0.055" '
                f'size="0.025" rgba="{rgba}"/>'
            )
        return "\n".join(geoms)

    @staticmethod
    def _start_geom(playback_idx, samples, start_marker_radius):
        x0, y0 = samples[0, 0], samples[1, 0]
        return (
            f'    <geom name="start_{playback_idx}" type="cylinder" '
            f'pos="{x0:.6f} {y0:.6f} 0.035" '
            f'size="{start_marker_radius:.6f} 0.035" '
            f'rgba="0 0 0 1"/>'
        )

    @staticmethod
    def pose_at(solution, t):
        clamped_t = float(np.clip(t, solution.t[0], solution.t[-1]))
        state = solution(clamped_t)
        direction = state[2:4]
        yaw = math.atan2(direction[1], direction[0])
        return VehiclePose(
            position=np.array([state[0], state[1], 0.16], dtype=float),
            yaw=yaw,
            mode=int(round(state[5])),
        )

    def _set_poses(self, t):
        for idx, playback in enumerate(self.playbacks):
            pose = self.pose_at(playback.solution, t)
            qpos_addr = self._vehicle_qpos_addrs[idx]
            qpos = self.data.qpos[qpos_addr : qpos_addr + 7]
            qpos[:3] = pose.position
            qpos[3:] = self._yaw_quaternion(pose.yaw)

            color = MODE_COLORS.get(pose.mode, MODE_COLORS[3])
            self.model.geom_rgba[self._chassis_geom_ids[idx]] = color
            self.model.geom_rgba[self._nose_geom_ids[idx]] = np.array(
                [
                    max(color[0] - 0.08, 0.0),
                    max(color[1] - 0.08, 0.0),
                    max(color[2] - 0.08, 0.0),
                    1.0,
                ]
            )
        mujoco.mj_forward(self.model, self.data)

    def run(self, fps=60.0, realtime_factor=1.0):
        import mujoco.viewer

        playback_t = self.t_start
        dt = 1.0 / fps

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = self.model.camera("overview").id

            while viewer.is_running() and playback_t <= self.t_end:
                loop_start = time.time()
                with viewer.lock():
                    self._set_poses(playback_t)
                viewer.set_texts(self._viewer_texts(playback_t))
                viewer.sync()

                playback_t += dt * realtime_factor
                sleep_time = dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _viewer_texts(self, t):
        mode_left = "Mode 1\nMode 2\nMode 3"
        mode_right = "\n".join(self._mode_label_text(mode) for mode in (1, 2, 3))
        playback_left = "\n".join(playback.label for playback in self.playbacks)
        playback_right = []
        for playback in self.playbacks:
            playback_right.append(self._playback_status_text(playback, t))

        return [
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                mode_left,
                mode_right,
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                playback_left,
                "\n".join(playback_right),
            ),
        ]

    def render_snapshot(self, output_path, t=None, width=1280, height=720):
        output_path = Path(output_path)
        snapshot_t = self.t_end if t is None else t
        self._set_poses(snapshot_t)

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera="overview")
        image = renderer.render()
        renderer.close()

        self._save_snapshot_with_text(output_path, image, snapshot_t, width, height)
        return output_path

    def render_animation(
        self,
        output_path,
        fps=24.0,
        duration=10.0,
        width=1280,
        height=720,
    ):
        import matplotlib
        import matplotlib.animation as animation
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        if not animation.writers.is_available("ffmpeg"):
            import imageio_ffmpeg

            matplotlib.rcParams["animation.ffmpeg_path"] = (
                imageio_ffmpeg.get_ffmpeg_exe()
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_count = max(2, int(round(float(fps) * float(duration))))
        times = np.linspace(self.t_start, self.t_end, frame_count)

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.set_facecolor("white")
        image_artist = ax.imshow(np.full((height, width, 3), 255, dtype=np.uint8))
        ax.axis("off")
        ax.text(
            18,
            24,
            self._mode_legend_text(),
            va="top",
            ha="left",
            fontsize=12,
            color="black",
            bbox=self._text_box_style(),
        )
        active_text = ax.text(
            18,
            height - 24,
            "",
            va="bottom",
            ha="left",
            fontsize=12,
            color="black",
            bbox=self._text_box_style(),
        )
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=3200,
            extra_args=["-pix_fmt", "yuv420p"],
        )
        try:
            with writer.saving(fig, str(output_path), dpi=100):
                for frame_t in times:
                    self._set_poses(frame_t)
                    renderer.update_scene(self.data, camera="overview")
                    image_artist.set_data(renderer.render())
                    active_text.set_text(self._active_modes_text(frame_t))
                    writer.grab_frame()
        finally:
            renderer.close()
            fig.clear()

        return output_path

    def _save_snapshot_with_text(self, output_path, image, t, width, height):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.set_facecolor("white")
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            18,
            24,
            self._mode_legend_text(),
            va="top",
            ha="left",
            fontsize=12,
            color="black",
            bbox=self._text_box_style(),
        )
        ax.text(
            18,
            height - 24,
            self._active_modes_text(t),
            va="bottom",
            ha="left",
            fontsize=12,
            color="black",
            bbox=self._text_box_style(),
        )
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(output_path, dpi=100)

    @staticmethod
    def _text_box_style():
        return {
            "facecolor": "white",
            "alpha": 0.82,
            "pad": 8,
            "edgecolor": "0.35",
            "linewidth": 0.9,
        }

    @staticmethod
    def _mode_legend_text():
        return "\n".join(
            f"Mode {mode}: {MuJoCoVehicleVisualizer._mode_label_text(mode)}"
            for mode in (1, 2, 3)
        )

    @staticmethod
    def _mode_label_text(mode):
        return f"{MODE_LABELS[mode]} ({MODE_COLOR_NAMES[mode]})"

    def _active_modes_text(self, t):
        return "\n".join(
            f"{playback.label}: {self._playback_status_text(playback, t)}"
            for playback in self.playbacks
        )

    def _playback_status_text(self, playback, t):
        mode = self.pose_at(playback.solution, t).mode
        elapsed = self._elapsed_time(playback, t)
        return f"t = {elapsed:5.2f} s | Mode {mode}: {MODE_LABELS[mode]}"

    @staticmethod
    def _elapsed_time(playback, t):
        return float(
            np.clip(t, playback.solution.t[0], playback.solution.t[-1])
            - playback.solution.t[0]
        )

    @staticmethod
    def _yaw_quaternion(yaw):
        half_yaw = 0.5 * yaw
        return np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)])


def build_default_solutions():
    epsilon = 1 / np.sqrt(10 * np.pi)
    mode_schedule = [
        (0.0, 2),
        (1.0, 3),
        (4.5, 1),
        (5.0, 3),
        (8.0, 1),
        (8.5, 2),
        (10.0, 1),
        (10.5, 3),
        (14.0, 1),
        (14.5, 2),
    ]
    mode_schedule_diverge = [
        (0.0, 2),
        (1.0, 1),
        (6.0, 3),
        (8.0, 2),
        (12.0, 3),
        (13.5, 2),
        (14.5, 1),
    ]
    x_0_p_converge = np.array([-4, 4])
    x_0_p_diverge = np.array([-4, -4])
    x_p_goal = np.array([0.0, 0.0])

    converge_simulation = vehicle_trajectories.VehicleTrajectorySimulation(
        x_0_p_converge,
        x_p_goal,
        epsilon,
        0.0,
        15.0,
        mode_schedule=mode_schedule,
    )
    diverge_simulation = vehicle_trajectories.VehicleTrajectorySimulation(
        x_0_p_diverge,
        x_p_goal,
        epsilon,
        0.0,
        15.0,
        mode_schedule=mode_schedule_diverge,
    )

    return x_p_goal, [
        VehiclePlayback("Converging trajectory", converge_simulation.solve()),
        VehiclePlayback("Diverging trajectory", diverge_simulation.solve()),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the two vehicle trajectories in MuJoCo."
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Render one PNG instead of opening the interactive MuJoCo viewer.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Render an MP4 instead of opening the interactive MuJoCo viewer.",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--video-fps", type=float, default=24.0)
    parser.add_argument("--video-duration", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.snapshot is not None and args.video is not None:
        parser.error("use either --snapshot or --video, not both")

    x_p_goal, playbacks = build_default_solutions()
    visualizer = MuJoCoVehicleVisualizer(playbacks, x_p_goal)

    if args.video is not None:
        output_path = visualizer.render_animation(
            args.video,
            fps=args.video_fps,
            duration=args.video_duration,
            width=args.width,
            height=args.height,
        )
        print(f"wrote {output_path}")
    elif args.snapshot is not None:
        output_path = visualizer.render_snapshot(
            args.snapshot,
            width=args.width,
            height=args.height,
        )
        print(f"wrote {output_path}")
    else:
        try:
            visualizer.run(fps=args.fps, realtime_factor=args.realtime_factor)
        except RuntimeError as exc:
            if "requires that the Python script be run under `mjpython`" in str(exc):
                raise SystemExit(
                    "The interactive MuJoCo viewer must be launched with mjpython on macOS.\n"
                    "From the project root, run:\n\n"
                    "  ./run_vehicle_mujoco.sh\n\n"
                    "Pass viewer options through the wrapper, for example:\n\n"
                    "  ./run_vehicle_mujoco.sh --fps 60 --realtime-factor 2"
                ) from None
            raise


if __name__ == "__main__":
    main()
