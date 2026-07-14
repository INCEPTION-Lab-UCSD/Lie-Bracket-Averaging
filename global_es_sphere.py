import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, to_rgba
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import hybrid_solution

# Global Variables
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

CHARCOAL_THEME = {
    "figure": "#111318",
    "axes": "#181B22",
    "text": "#E8EAED",
    "grid": "#3A3F4B",
    "trajectory": "#FF4D4D",
    "initial": "#E8EAED",
    "target": "#6EE7B7",
    "edge": "#111318",
    "cmap": "viridis",
}

MODE_VIEW_LABELS = {
    1: "Mode 1",
    2: "Mode 2",
}

SPHERE_FIGSIZE = (9.5, 7.2)
SPHERE_AXIS_PADDING = 1.12
TRAJECTORY_LINEWIDTH = 0.9
TRAJECTORY_OUTLINE_LINEWIDTH = TRAJECTORY_LINEWIDTH + 0.45
TRAJECTORY_OUTLINE_ALPHA = 0.42
TRAJECTORY_RENDERING = {
    "antialiased": True,
    "solid_capstyle": "round",
    "solid_joinstyle": "round",
    "snap": False,
}
ANIMATION_N_TIME = 1200
ANIMATION_SAVE_DPI = 220
ANIMATION_BITRATE = 6000


class Global_ES_Sphere:
    def __init__(self, x0, delta, omega, alpha, kappa, epsilon, t_1, t_2):
        self.x0 = self._normalize_sphere_state(x0)
        self.delta = delta
        self.omega = np.asarray(omega, dtype=float)
        self.alpha = alpha
        self.kappa = kappa
        self.epsilon = epsilon
        self.t_1 = t_1
        self.t_2 = t_2

    @staticmethod
    def _normalize_sphere_state(x):
        x = np.asarray(x, dtype=float).copy()
        norm = np.linalg.norm(x[:3])
        if norm == 0.0:
            raise ValueError("sphere state must have nonzero position component")
        x[:3] /= norm
        return x

    def skew_symmetric(self, x):
        return np.array(
            [[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype=float
        )

    def minimize(self, x):
        x_p = x[0:3]
        J1 = self.cost(
            self.diffeomorphism(
                x_p,
                1,
            )
        )
        J2 = self.cost(
            self.diffeomorphism(
                x_p,
                2,
            )
        )

        J = np.array([J1, J2])
        return np.min(J)

    def b(self, x):
        I = np.eye(3)
        return np.cross(I, x)

    def diffeomorphism(self, x_p, q):
        J = self.cost(x_p)

        k_q = {1: 0.25 / np.sqrt(2), 2: -0.25 / np.sqrt(2)}

        if J <= self.alpha:
            return x_p
        return np.dot(
            expm(
                k_q[int(q)]
                * (J - self.alpha) ** 2
                * (self.skew_symmetric(e1) + self.skew_symmetric(e2))
            ),
            x_p,
        )

    def jump_map(self, x):
        J1 = self.cost(self.diffeomorphism(x[0:3], 1))
        J2 = self.cost(self.diffeomorphism(x[0:3], 2))

        J = np.array([J1, J2])
        x_plus = x.copy()
        x_plus[3] = int(np.argmin(J)) + 1
        x_plus[4] += 1
        return x_plus

    def solve(self, x=None, t=None, rtol=1e-6, atol=1e-8):
        # set starting time as t_1 if not otherwise specified
        y0 = self.x0.copy() if x is None else self._normalize_sphere_state(x)
        if t is None:
            t = self.t_1
        solution_segments = []
        jump_event = self._jump_event()
        while t < self.t_2:
            sol = solve_ivp(
                fun=self.flow_map,
                t_span=(t, self.t_2),
                y0=y0,
                method="RK45",
                rtol=rtol,
                atol=atol,
                dense_output=True,
                events=jump_event,
                max_step=self.epsilon / 10,
            )
            if not sol.success:
                raise RuntimeError(f"Solution failed: {sol.message}")

            solution_segments.append(sol)
            t = sol.t[-1]
            y0 = sol.y[:, -1].copy()

            if sol.t_events[0].size == 0 or t >= self.t_2:
                break

            y0 = self.jump_map(y0)
            t = np.nextafter(t, self.t_2)

        return hybrid_solution.HybridSolution(solution_segments)

    def _jump_event(self):
        def event(t, x):
            return self.jump_condition(t, x)

        event.terminal = True
        event.direction = 1
        return event

    def jump_condition(self, t, x):
        Jq = self.cost(self.diffeomorphism(x[0:3], x[3]))
        J_min = self.minimize(x)
        return Jq - J_min - self.delta

    def flow_map(self, t, x):
        x_p = x[0:3]
        q = int(x[3])
        u = self.feedback_controller(t, x_p, q)
        dx_p = np.sum(self.b(x_p) * u[:, None], axis=0)
        dxdt = np.append(dx_p, [0.0, 0.0])

        return dxdt

    def cost(self, x):
        x_p = x[:3]
        return 1 - np.dot(x_p, e3)

    def feedback_controller(self, t, x, q):
        J = self.cost(self.diffeomorphism(x[:3], q))
        return (
            1
            / self.epsilon
            * np.sqrt(2 * np.pi * self.omega)
            * np.cos(2 * np.pi * self.omega * t / self.epsilon**2 + self.kappa * J)
            * np.pow(self.kappa, -0.5)
        )

    @staticmethod
    def _apply_charcoal_3d_style(ax, theme=CHARCOAL_THEME):
        ax.set_facecolor(theme["axes"])
        ax.tick_params(colors=theme["text"])
        ax.xaxis.label.set_color(theme["text"])
        ax.yaxis.label.set_color(theme["text"])
        ax.zaxis.label.set_color(theme["text"])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(to_rgba(theme["axes"], 1.0))
            axis.pane.set_edgecolor(to_rgba(theme["axes"], 0.0))
            axis.line.set_color(to_rgba(theme["axes"], 0.0))
            axis._axinfo["grid"]["color"] = to_rgba(theme["axes"], 0.0)
            axis._axinfo["grid"]["linewidth"] = 0.0
            axis._axinfo["axisline"]["color"] = to_rgba(theme["axes"], 0.0)
            axis._axinfo["axisline"]["linewidth"] = 0.0
        ax.grid(False)

    @staticmethod
    def _apply_charcoal_legend_style(legend, theme=CHARCOAL_THEME):
        if legend is None:
            return
        legend.get_frame().set_facecolor(theme["axes"])
        legend.get_frame().set_edgecolor(theme["grid"])
        legend.get_frame().set_alpha(0.95)
        for text in legend.get_texts():
            text.set_color(theme["text"])

    @staticmethod
    def _apply_charcoal_colorbar_style(colorbar, theme=CHARCOAL_THEME):
        colorbar.ax.set_facecolor(theme["figure"])
        colorbar.ax.tick_params(colors=theme["text"])
        colorbar.ax.yaxis.label.set_color(theme["text"])
        colorbar.outline.set_edgecolor(theme["grid"])

    @staticmethod
    def _apply_charcoal_2d_style(ax, theme=CHARCOAL_THEME):
        ax.set_facecolor(theme["axes"])
        ax.tick_params(colors=theme["text"])
        ax.xaxis.label.set_color(theme["text"])
        ax.yaxis.label.set_color(theme["text"])
        for spine in ax.spines.values():
            spine.set_color(theme["grid"])
        ax.grid(True, color=theme["grid"], alpha=0.35, linewidth=0.8)

    def _plot_mode_panel(self, ax_mode, times, q_values, theme=CHARCOAL_THEME):
        self._apply_charcoal_2d_style(ax_mode, theme)
        ax_mode.step(
            times,
            q_values,
            where="post",
            color=theme["text"],
            linewidth=1.8,
        )
        ax_mode.set_xlabel(r"$t$")

        ax_mode.set_yticks([1, 2])
        ax_mode.set_yticklabels([MODE_VIEW_LABELS[1], MODE_VIEW_LABELS[2]])
        ax_mode.set_ylim(0.7, 2.3)
        ax_mode.set_xlim(times[0], times[-1])
        ax_mode.margins(x=0)

    @staticmethod
    def _camera_facing_mask(ax, points):
        """Return the points on the camera-facing hemisphere."""
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[:, None]

        elev = np.deg2rad(ax.elev)
        azim = np.deg2rad(ax.azim)
        view_direction = np.array(
            [
                np.cos(elev) * np.cos(azim),
                np.cos(elev) * np.sin(azim),
                np.sin(elev),
            ]
        )
        return view_direction @ points >= -1e-12

    def _update_sphere_visibility(self, fig, ax):
        state = getattr(fig, "_global_es_visibility_state", None)
        if state is None:
            return

        trajectory = state["trajectory"][:, : state["trajectory_count"]]
        visible_trajectory = trajectory.copy()
        visible_trajectory[:, ~self._camera_facing_mask(ax, trajectory)] = np.nan
        for line in state["trajectory_lines"]:
            line.set_data(visible_trajectory[0], visible_trajectory[1])
            line.set_3d_properties(visible_trajectory[2])

        for marker in state["markers"]:
            is_visible = (
                marker["active"] and self._camera_facing_mask(ax, marker["point"])[0]
            )
            marker["artist"].set_visible(bool(is_visible))

        state["view"] = (ax.elev, ax.azim, ax.roll)

    def _connect_sphere_visibility_updates(self, fig, ax):
        def refresh_after_view_change(event):
            if event.canvas is not fig.canvas:
                return
            state = fig._global_es_visibility_state
            view = (ax.elev, ax.azim, ax.roll)
            if view == state["view"]:
                return
            self._update_sphere_visibility(fig, ax)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("draw_event", refresh_after_view_change)

    def plot_sphere_simulation(
        self,
        solution,
        x_target=e3,
        r=1.0,
        n_grid=100,
        n_time=500,
    ):
        x_target = np.asarray(x_target, dtype=float)
        if np.linalg.norm(x_target) != 0.0:
            x_target = r * x_target / np.linalg.norm(x_target)

        times = np.linspace(solution.t[0], solution.t[-1], n_time)
        states = solution(times)
        x_points = states[:3]
        q_values = np.rint(states[3]).astype(int)
        x_points = r * x_points / np.linalg.norm(x_points, axis=0)
        x0 = x_points[:, 0]

        u = np.linspace(0, 2 * np.pi, n_grid, dtype=float)
        v = np.linspace(0, np.pi, n_grid, dtype=float)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones_like(u), np.cos(v))

        j_values = 1.0 - z / r
        norm = Normalize(vmin=0.0, vmax=2.0)
        theme = CHARCOAL_THEME
        sphere_cmap = plt.get_cmap(theme["cmap"])
        facecolors = sphere_cmap(norm(j_values))

        fig = plt.figure(figsize=SPHERE_FIGSIZE)
        fig.patch.set_facecolor(theme["figure"])
        grid_spec = fig.add_gridspec(
            2,
            4,
            width_ratios=[1.0, 6.2, 0.22, 1.0],
            height_ratios=[7.4, 0.8],
            hspace=0.04,
            wspace=0.08,
            left=0.03,
            right=0.97,
            top=0.990,
            bottom=0.055,
        )
        ax = fig.add_subplot(grid_spec[0, 1], projection="3d", computed_zorder=False)
        ax_colorbar = fig.add_subplot(grid_spec[0, 2])
        ax_mode = fig.add_subplot(grid_spec[1, 1])
        ax.set_anchor("C")
        self._apply_charcoal_3d_style(ax, theme)
        ax.plot_surface(
            x,
            y,
            z,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            edgecolor=to_rgba(theme["grid"], 0.35),
            linewidth=0.2,
            antialiased=False,
            shade=False,
            zorder=1,
        )

        trajectory = x_points
        (trajectory_outline,) = ax.plot3D(
            trajectory[0],
            trajectory[1],
            trajectory[2],
            color=theme["edge"],
            linewidth=TRAJECTORY_OUTLINE_LINEWIDTH,
            alpha=TRAJECTORY_OUTLINE_ALPHA,
            zorder=9,
            label="_nolegend_",
            **TRAJECTORY_RENDERING,
        )
        (trajectory_line,) = ax.plot3D(
            trajectory[0],
            trajectory[1],
            trajectory[2],
            color=theme["trajectory"],
            linewidth=TRAJECTORY_LINEWIDTH,
            zorder=10,
            label=r"$x(t)$",
            **TRAJECTORY_RENDERING,
        )

        initial_marker = ax.scatter(
            x0[0],
            x0[1],
            x0[2],
            marker="o",
            s=120,
            color=theme["initial"],
            edgecolor=theme["edge"],
            linewidth=1.0,
            depthshade=False,
            zorder=11,
            label=r"$x(0)$",
        )
        target_marker = ax.scatter(
            x_target[0],
            x_target[1],
            x_target[2],
            marker="o",
            s=190,
            color=theme["target"],
            edgecolor=theme["edge"],
            linewidth=1.0,
            depthshade=False,
            zorder=11,
            label=r"$x^*$",
        )

        mappable = cm.ScalarMappable(norm=norm, cmap=sphere_cmap)
        mappable.set_array(j_values)
        colorbar = fig.colorbar(mappable, cax=ax_colorbar)
        colorbar.set_label(r"Cost: $J(x)$", rotation=270, labelpad=18)
        self._apply_charcoal_colorbar_style(colorbar, theme)

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$x_3$")
        ax.set_xlim([-SPHERE_AXIS_PADDING * r, SPHERE_AXIS_PADDING * r])
        ax.set_ylim([-SPHERE_AXIS_PADDING * r, SPHERE_AXIS_PADDING * r])
        ax.set_zlim([-SPHERE_AXIS_PADDING * r, SPHERE_AXIS_PADDING * r])
        ax.set_box_aspect((1, 1, 1), zoom=1.0)
        ax.set_proj_type("ortho")
        ax.view_init(elev=10, azim=-45)
        self._apply_charcoal_legend_style(ax.legend(loc="upper left"), theme)

        self._plot_mode_panel(ax_mode, times, q_values, theme)
        fig._global_es_mode_axis = ax_mode
        fig._global_es_trajectory_outline = trajectory_outline
        fig._global_es_trajectory_line = trajectory_line
        fig._global_es_visibility_state = {
            "trajectory": trajectory,
            "trajectory_count": trajectory.shape[1],
            "trajectory_lines": (trajectory_outline, trajectory_line),
            "markers": [
                {"artist": initial_marker, "point": x0, "active": True},
                {
                    "artist": target_marker,
                    "point": x_target,
                    "active": True,
                },
            ],
            "view": None,
        }
        self._update_sphere_visibility(fig, ax)
        self._connect_sphere_visibility_updates(fig, ax)

        return fig, ax

    @staticmethod
    def _hide_3d_axis_labels_and_ticks(ax):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    def animate_solution(
        self,
        solution,
        x_target=e3,
        r=1.0,
        n_grid=100,
        n_time=ANIMATION_N_TIME,
        frame_step=200,
        interval=40,
        repeat_delay=1200,
        save_path=None,
        fps=30,
        save_dpi=ANIMATION_SAVE_DPI,
        bitrate=ANIMATION_BITRATE,
    ):
        if frame_step <= 0:
            raise ValueError("frame_step must be positive")

        fig, ax = self.plot_sphere_simulation(
            solution,
            x_target=x_target,
            r=r,
            n_grid=n_grid,
            n_time=n_time,
        )

        self._hide_3d_axis_labels_and_ticks(ax)

        ax_mode = getattr(fig, "_global_es_mode_axis", None)
        trajectory_outline = getattr(fig, "_global_es_trajectory_outline", None)
        trajectory_line = getattr(fig, "_global_es_trajectory_line", None)
        if trajectory_outline is None:
            (trajectory_outline,) = ax.plot3D(
                [],
                [],
                [],
                color=CHARCOAL_THEME["edge"],
                linewidth=TRAJECTORY_OUTLINE_LINEWIDTH,
                alpha=TRAJECTORY_OUTLINE_ALPHA,
                zorder=9,
                **TRAJECTORY_RENDERING,
            )
        else:
            trajectory_outline.set_data([], [])
            trajectory_outline.set_3d_properties([])

        if trajectory_line is None:
            (trajectory_line,) = ax.plot3D(
                [],
                [],
                [],
                color=CHARCOAL_THEME["trajectory"],
                linewidth=TRAJECTORY_LINEWIDTH,
                zorder=10,
                **TRAJECTORY_RENDERING,
            )
        else:
            trajectory_line.set_data([], [])
            trajectory_line.set_3d_properties([])

        times = np.linspace(solution.t[0], solution.t[-1], n_time)
        states = solution(times)
        x_points = states[:3]
        q_values = np.rint(states[3]).astype(int)
        x_points = r * x_points / np.linalg.norm(x_points, axis=0)
        trajectory = x_points

        mode_line = None
        if ax_mode is not None and ax_mode.lines:
            mode_line = ax_mode.lines[0]
            mode_line.set_data([], [])

        current_point = ax.scatter(
            [],
            [],
            [],
            marker="o",
            s=95,
            color=CHARCOAL_THEME["trajectory"],
            edgecolor=CHARCOAL_THEME["edge"],
            linewidth=0.8,
            depthshade=False,
            zorder=12,
        )
        visibility_state = fig._global_es_visibility_state
        visibility_state["trajectory"] = trajectory
        visibility_state["trajectory_count"] = 0
        current_marker = {
            "artist": current_point,
            "point": trajectory[:, 0],
            "active": True,
        }
        visibility_state["markers"].append(current_marker)

        mode_text = ax.text2D(
            0.97,
            0.94,
            "",
            transform=ax.transAxes,
            color=CHARCOAL_THEME["text"],
            fontsize=11,
            ha="right",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": CHARCOAL_THEME["axes"],
                "edgecolor": CHARCOAL_THEME["grid"],
                "alpha": 0.95,
            },
        )

        frame_indices = np.linspace(0, n_time - 1, frame_step, dtype=int)
        if frame_indices[-1] != n_time - 1:
            frame_indices = np.append(frame_indices, n_time - 1)

        def update(frame_idx):
            idx = frame_indices[frame_idx] + 1
            active_mode = q_values[idx - 1]
            visibility_state["trajectory_count"] = idx
            current_marker["point"] = trajectory[:, idx - 1]
            current_point._offsets3d = (
                [trajectory[0, idx - 1]],
                [trajectory[1, idx - 1]],
                [trajectory[2, idx - 1]],
            )
            self._update_sphere_visibility(fig, ax)
            mode_text.set_text(f"{MODE_VIEW_LABELS[int(active_mode)]}")
            if mode_line is not None:
                mode_line.set_data(times[:idx], q_values[:idx])
            artists = [trajectory_outline, trajectory_line, current_point, mode_text]
            if mode_line is not None:
                artists.append(mode_line)
            return tuple(artists)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frame_indices),
            interval=interval,
            repeat=True,
            repeat_delay=repeat_delay,
            blit=False,
        )
        update(0)

        if save_path is not None:
            if not animation.writers.is_available("ffmpeg"):
                import imageio_ffmpeg

                plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate)
            with plt.rc_context({"path.simplify": False}):
                ani.save(save_path, writer=writer, dpi=save_dpi)

        return ani
