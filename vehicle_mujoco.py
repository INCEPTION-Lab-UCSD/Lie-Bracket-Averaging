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
VEHICLE_TRAJECTORY_ALPHA = 1.00
VEHICLE_TRAJECTORY_Z = 0.095
VEHICLE_TRAJECTORY_RADIUS = 0.030
LEVEL_CURVE_COUNT = 10
LEVEL_CURVE_SEGMENTS = 144
LEVEL_PATCH_Z = 0.014
LEVEL_PATCH_ALPHA = 0.54
LEVEL_CURVE_Z = 0.032
LEVEL_CURVE_RADIUS = 0.008


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
        trail_samples=200,
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
        self._level_cost_min = 0.0
        self._level_cost_max = 0.0

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
        self._seat_geom_ids = [
            self.model.geom(f"seat_{idx}").id for idx in range(len(self.playbacks))
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

        buffer = 2.0
        min_xy = np.minimum(np.min(all_xy, axis=1), goal) - buffer
        max_xy = np.maximum(np.max(all_xy, axis=1), goal) + buffer
        center = 0.5 * (min_xy + max_xy)
        span = np.maximum(max_xy - min_xy, 4.0)
        floor_size = max(span) / 2.0 + 1.0
        camera_distance = max(span) * 1.15
        camera_height = max(span) * 0.85 + 3.0
        floor_min_xy = center - floor_size
        floor_max_xy = center + floor_size
        level_assets, level_geoms = self._level_sets(floor_min_xy, floor_max_xy)

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
  <option timestep="0.01"/>
  <visual>
    <headlight ambient="0.70 0.70 0.70" diffuse="0.55 0.55 0.55" specular="0.12 0.12 0.12"/>
    <rgba haze="1 1 1 1"/>
    <global offwidth="1920" offheight="1080"/>
    <map znear="0.01" zfar="100"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="flat" width="32" height="32"
             rgb1="1 1 1" rgb2="1 1 1"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.96 0.96 0.96" rgb2="0.86 0.88 0.89"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.0"/>
{level_assets}
  </asset>
  <worldbody>

    <camera name="overview"
            pos="{center[0]:.6f} {center[1] - camera_distance:.6f} {camera_height:.6f}"
            xyaxes="1 0 0 0 0.58 0.82"/>
    <geom name="floor" type="plane" pos="{center[0]:.6f} {center[1]:.6f} 0"
          size="{floor_size:.6f} {floor_size:.6f} 0.1" material="grid"/>
{level_geoms}
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
        wheel_radius = self.vehicle_width / 1.25
        wheel_half_width = self.vehicle_width / 8
        axle_height = wheel_radius * 2.5
        axle_radius = wheel_radius * 0.10
        hub_radius = wheel_radius * 0.32
        seat_half_length = self.vehicle_length * 0.28
        seat_half_width = self.vehicle_width * 0.35
        seat_half_height = self.vehicle_width * 0.08
        return f"""
    <body name="vehicle_{playback_idx}" pos="0 0 {wheel_radius:.6f}">
      <freejoint name="vehicle_{playback_idx}_freejoint"/>
      <geom name="chassis_{playback_idx}" type="cylinder"
            euler="1.570796326795 0 0"
            size="{wheel_radius:.6f} {wheel_half_width:.6f}"
            rgba="0.16 0.55 0.85 1"/>
      <geom name="axle_{playback_idx}" type="cylinder"
            pos="0 0 {axle_height / 2:.6f}"
            size="{axle_radius:.6f} {axle_height / 2:.6f}"
            rgba="0.02 0.02 0.02 1"/>
      <geom name="seat_{playback_idx}" type="box"
            pos="0 0 {axle_height + seat_half_height:.6f}"
            size="{seat_half_length:.6f} {seat_half_width:.6f} {seat_half_height:.6f}"
            rgba="0.05 0.15 0.20 1"/>
      <geom name="hub_{playback_idx}" type="sphere"
            size="{hub_radius:.6f}"
            rgba="0.96 0.96 0.96 1"/>
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
            rgba = (
                " ".join(f"{value:.3f}" for value in color[:3])
                + f" {VEHICLE_TRAJECTORY_ALPHA:.3f}"
            )
            geoms.append(
                f'    <geom name="trail_{playback_idx}_{sample_idx}" type="capsule" '
                f'fromto="{x0:.6f} {y0:.6f} {VEHICLE_TRAJECTORY_Z:.6f} '
                f'{x1:.6f} {y1:.6f} {VEHICLE_TRAJECTORY_Z:.6f}" '
                f'size="{VEHICLE_TRAJECTORY_RADIUS:.6f}" rgba="{rgba}"/>'
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

    def pose_at(self, solution, t):
        clamped_t = float(np.clip(t, solution.t[0], solution.t[-1]))
        state = solution(clamped_t)
        direction = state[2:4]
        yaw = math.atan2(direction[1], direction[0])
        return VehiclePose(
            position=np.array([state[0], state[1], self.vehicle_width], dtype=float),
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
            self.model.geom_rgba[self._seat_geom_ids[idx]] = np.array(
                [
                    max(color[0] - 0.08, 0.0),
                    max(color[1] - 0.08, 0.0),
                    max(color[2] - 0.08, 0.0),
                    1.0,
                ]
            )
        mujoco.mj_forward(self.model, self.data)

    def _level_sets(self, min_xy, max_xy):
        min_xy = np.asarray(min_xy, dtype=float)
        max_xy = np.asarray(max_xy, dtype=float)
        corners = np.array(
            [
                [min_xy[0], min_xy[1]],
                [min_xy[0], max_xy[1]],
                [max_xy[0], min_xy[1]],
                [max_xy[0], max_xy[1]],
            ]
        )
        max_radius = float(np.max(np.linalg.norm(corners - self.x_p_goal, axis=1)))
        if max_radius <= 0.0:
            return "", ""
        self._level_cost_max = 0.5 * max_radius**2

        level_radii = np.sqrt(np.linspace(0.0, max_radius**2, LEVEL_CURVE_COUNT + 1))
        angles = np.linspace(0.0, 2.0 * math.pi, LEVEL_CURVE_SEGMENTS + 1)
        circle_offsets = np.column_stack((np.cos(angles), np.sin(angles)))

        assets = []
        geoms = []
        for level_idx in range(LEVEL_CURVE_COUNT):
            inner_radius = level_radii[level_idx]
            outer_radius = level_radii[level_idx + 1]
            vertices, faces = self._level_patch_mesh(
                min_xy,
                max_xy,
                inner_radius,
                outer_radius,
                circle_offsets,
            )
            if vertices:
                vertex_attr = " ".join(f"{x:.6f} {y:.6f} 0" for x, y in vertices)
                face_attr = " ".join(f"{i0} {i1} {i2}" for i0, i1, i2 in faces)
                assets.append(
                    f'    <mesh name="level_patch_{level_idx}" '
                    f'inertia="shell" vertex="{vertex_attr}" face="{face_attr}"/>'
                )
                rgba = self._level_patch_rgba(level_idx)
                geoms.append(
                    f'    <geom name="level_patch_{level_idx}" type="mesh" '
                    f'mesh="level_patch_{level_idx}" pos="0 0 {LEVEL_PATCH_Z:.6f}" '
                    f'rgba="{rgba}" contype="0" conaffinity="0"/>'
                )

        for level_idx, radius in enumerate(level_radii[1:-1], start=1):
            points = self.x_p_goal + radius * circle_offsets
            rgba = self._level_curve_rgba(level_idx)
            for segment_idx in range(LEVEL_CURVE_SEGMENTS):
                p0 = points[segment_idx]
                p1 = points[segment_idx + 1]
                midpoint = 0.5 * (p0 + p1)
                if np.any(midpoint < min_xy) or np.any(midpoint > max_xy):
                    continue
                geoms.append(
                    f'    <geom name="level_curve_{level_idx}_{segment_idx}" '
                    f'type="capsule" '
                    f'fromto="{p0[0]:.6f} {p0[1]:.6f} {LEVEL_CURVE_Z:.6f} '
                    f'{p1[0]:.6f} {p1[1]:.6f} {LEVEL_CURVE_Z:.6f}" '
                    f'size="{LEVEL_CURVE_RADIUS:.6f}" rgba="{rgba}" '
                    f'contype="0" conaffinity="0"/>'
                )
        return "\n".join(assets), "\n".join(geoms)

    def _level_patch_mesh(
        self,
        min_xy,
        max_xy,
        inner_radius,
        outer_radius,
        circle_offsets,
    ):
        vertices = []
        faces = []
        for segment_idx in range(LEVEL_CURVE_SEGMENTS):
            inner_0 = self.x_p_goal + inner_radius * circle_offsets[segment_idx]
            inner_1 = self.x_p_goal + inner_radius * circle_offsets[segment_idx + 1]
            outer_0 = self.x_p_goal + outer_radius * circle_offsets[segment_idx]
            outer_1 = self.x_p_goal + outer_radius * circle_offsets[segment_idx + 1]
            if inner_radius <= 1e-9:
                patch = [self.x_p_goal, outer_0, outer_1]
            else:
                patch = [inner_0, outer_0, outer_1, inner_1]
            clipped_patch = self._clip_polygon_to_bounds(patch, min_xy, max_xy)
            if len(clipped_patch) < 3:
                continue

            start_idx = len(vertices)
            vertices.extend(
                (float(point[0]), float(point[1])) for point in clipped_patch
            )
            for point_idx in range(1, len(clipped_patch) - 1):
                faces.append(
                    (start_idx, start_idx + point_idx, start_idx + point_idx + 1)
                )
        return vertices, faces

    @staticmethod
    def _clip_polygon_to_bounds(polygon, min_xy, max_xy):
        clipped = [np.asarray(point, dtype=float) for point in polygon]
        for axis, lower, keep_greater in (
            (0, min_xy[0], True),
            (0, max_xy[0], False),
            (1, min_xy[1], True),
            (1, max_xy[1], False),
        ):
            clipped = MuJoCoVehicleVisualizer._clip_polygon_against_edge(
                clipped,
                axis,
                lower,
                keep_greater,
            )
            if not clipped:
                break
        return clipped

    @staticmethod
    def _clip_polygon_against_edge(polygon, axis, boundary, keep_greater):
        clipped = []
        previous = polygon[-1]
        previous_inside = (
            previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        )
        for current in polygon:
            current_inside = (
                current[axis] >= boundary if keep_greater else current[axis] <= boundary
            )
            if current_inside != previous_inside:
                clipped.append(
                    MuJoCoVehicleVisualizer._edge_intersection(
                        previous,
                        current,
                        axis,
                        boundary,
                    )
                )
            if current_inside:
                clipped.append(current)
            previous = current
            previous_inside = current_inside
        return clipped

    @staticmethod
    def _edge_intersection(p0, p1, axis, boundary):
        delta = p1[axis] - p0[axis]
        if abs(delta) < 1e-12:
            return p0.copy()
        alpha = (boundary - p0[axis]) / delta
        return p0 + alpha * (p1 - p0)

    @staticmethod
    def _level_patch_rgba(level_idx):
        gray = MuJoCoVehicleVisualizer._level_patch_gray(level_idx)
        return f"{gray:.3f} {gray:.3f} {gray:.3f} {LEVEL_PATCH_ALPHA:.3f}"

    @staticmethod
    def _level_patch_gray(level_idx):
        fade = level_idx / max(LEVEL_CURVE_COUNT - 1, 1)
        return 0.90 - 0.52 * fade

    @staticmethod
    def _level_curve_rgba(level_idx):
        fade = level_idx / max(LEVEL_CURVE_COUNT - 1, 1)
        gray = 0.58 - 0.22 * fade
        return f"{gray:.3f} {gray:.3f} {gray:.3f} 0.62"

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
        image_artist = ax.imshow(
            np.full((height, width, 3), 255, dtype=np.uint8),
            extent=(-0.5, width - 0.5, height - 0.5, -0.5),
        )
        ax.axis("off")
        ax.text(
            18,
            48,
            self._mode_legend_text(),
            va="top",
            ha="left",
            fontsize=12,
            color="black",
            bbox=self._text_box_style(),
        )
        active_text = ax.text(
            18,
            height - 48,
            "",
            va="bottom",
            ha="left",
            fontsize=12,
            color="black",
            bbox=self._text_box_style(),
        )
        self._add_cost_colorbar_overlay(ax, width, height)
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
                    image_artist.set_data(self._reframe_render(renderer.render()))
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
        ax.imshow(
            self._reframe_render(image),
            extent=(-0.5, width - 0.5, height - 0.5, -0.5),
        )
        ax.axis("off")
        ax.text(
            18,
            48,
            self._mode_legend_text(),
            va="top",
            ha="left",
            fontsize=12,
            color="black",
            bbox=self._text_box_style(),
        )
        ax.text(
            18,
            height - 48,
            self._active_modes_text(t),
            va="bottom",
            ha="left",
            fontsize=12,
            color="black",
            bbox=self._text_box_style(),
        )
        self._add_cost_colorbar_overlay(ax, width, height)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(output_path, dpi=100)

    def _add_cost_colorbar_overlay(self, ax, width, height):
        from matplotlib.patches import Rectangle

        if self._level_cost_max <= self._level_cost_min:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        bar_height = max(160.0, height * 0.34)
        bar_width = max(18.0, width * 0.018)
        right_margin = max(52.0, width * 0.045)
        bar_x1 = width - right_margin
        bar_x0 = bar_x1 - bar_width
        bar_y0 = max(94.0, height * 0.17)
        bar_y1 = bar_y0 + bar_height

        tick_label_width = max(44.0, width * 0.045)
        box_pad_x = 14.0
        box_x0 = bar_x0 - tick_label_width - box_pad_x
        box_y0 = bar_y0 - 34.0
        box_width = tick_label_width + bar_width + 2.0 * box_pad_x
        box_height = bar_height + 54.0
        ax.add_patch(
            Rectangle(
                (box_x0, box_y0),
                box_width,
                box_height,
                facecolor="white",
                alpha=0.82,
                edgecolor="0.35",
                linewidth=0.9,
                zorder=6,
            )
        )

        grays = np.array(
            [
                self._level_patch_gray(level_idx)
                for level_idx in reversed(range(LEVEL_CURVE_COUNT))
            ]
        ).reshape(LEVEL_CURVE_COUNT, 1)
        ax.imshow(
            grays,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            extent=(bar_x0, bar_x1, bar_y1, bar_y0),
            zorder=7,
        )
        ax.add_patch(
            Rectangle(
                (bar_x0, bar_y0),
                bar_width,
                bar_height,
                fill=False,
                edgecolor="0.25",
                linewidth=0.9,
                zorder=8,
            )
        )

        ax.text(
            0.5 * (bar_x0 + bar_x1),
            bar_y0 - 12.0,
            r"$J(x_p)$",
            ha="center",
            va="bottom",
            fontsize=11,
            color="black",
            zorder=9,
        )

        tick_x0 = bar_x0 - 5.0
        tick_x1 = bar_x0
        for value, y in (
            (self._level_cost_max, bar_y0),
            (
                0.5 * (self._level_cost_min + self._level_cost_max),
                0.5 * (bar_y0 + bar_y1),
            ),
            (self._level_cost_min, bar_y1),
        ):
            ax.plot(
                [tick_x0, tick_x1],
                [y, y],
                color="0.2",
                linewidth=0.8,
                zorder=9,
            )
            ax.text(
                tick_x0 - 4.0,
                y,
                self._format_cost_tick(value),
                ha="right",
                va="center",
                fontsize=9,
                color="black",
                zorder=9,
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    @staticmethod
    def _format_cost_tick(value):
        value = float(value)
        if abs(value) >= 10.0:
            return f"{value:.0f}"
        if abs(value) >= 1.0:
            return f"{value:.1f}"
        return f"{value:.2f}"

    @staticmethod
    def _reframe_render(image, zoom=1.10, upward_shift_fraction=0.05):
        if zoom <= 1.0:
            return image

        height, width = image.shape[:2]
        crop_height = int(round(height / zoom))
        crop_width = int(round(width / zoom))
        y_shift = int(round(height * upward_shift_fraction))
        y0 = height // 2 - crop_height // 2 + y_shift
        x0 = width // 2 - crop_width // 2
        y0 = int(np.clip(y0, 0, height - crop_height))
        x0 = int(np.clip(x0, 0, width - crop_width))
        return image[y0 : y0 + crop_height, x0 : x0 + crop_width]

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
    parser.add_argument("--video-duration", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
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
