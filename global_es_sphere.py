import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import hybrid_solution

# Global Variables
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])


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
        x_points = solution(times)[:3]
        x_points = r * x_points / np.linalg.norm(x_points, axis=0)
        x0 = x_points[:, 0]

        u = np.linspace(0, 2 * np.pi, n_grid, dtype=float)
        v = np.linspace(0, np.pi, n_grid, dtype=float)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones_like(u), np.cos(v))

        j_values = 1.0 - z / r
        norm = Normalize(vmin=0.0, vmax=2.0)
        facecolors = cm.gray(norm(j_values))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            x,
            y,
            z,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0.3,
            antialiased=False,
            shade=False,
            zorder=1,
        )

        trajectory_radius = 1.01 * r
        trajectory = trajectory_radius * x_points / r
        marker_radius = 1.04 * r
        x0_marker = marker_radius * x0 / r
        x_target_marker = marker_radius * x_target / r
        ax.plot3D(
            trajectory[0],
            trajectory[1],
            trajectory[2],
            color="red",
            linewidth=1.8,
            zorder=10,
            label=r"$x(t)$",
        )
        ax.scatter(
            x0_marker[0],
            x0_marker[1],
            x0_marker[2],
            marker="o",
            s=80,
            color="black",
            depthshade=False,
            zorder=11,
            label=r"$x(0)$",
        )
        ax.scatter(
            x_target_marker[0],
            x_target_marker[1],
            x_target_marker[2],
            marker="o",
            s=160,
            color="green",
            depthshade=False,
            zorder=11,
            label=r"$x^*$",
        )

        mappable = cm.ScalarMappable(norm=norm, cmap=cm.gray)
        mappable.set_array(j_values)
        colorbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(r"$J(x)$", rotation=270, labelpad=18)

        ax.set_xlim([-1.5 * r, 1.5 * r])
        ax.set_ylim([-1.5 * r, 1.5 * r])
        ax.set_zlim([-1.5 * r, 1.5 * r])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=10, azim=-45)
        ax.legend(loc="upper left")

        return fig, ax

    def animate_solution(
        self,
        solution,
        x_target=e3,
        r=1.0,
        n_grid=100,
        n_time=500,
        frame_step=200,
        interval=40,
        repeat_delay=1200,
        save_path=None,
        fps=30,
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

        times = np.linspace(solution.t[0], solution.t[-1], n_time)
        x_points = solution(times)[:3]
        x_points = r * x_points / np.linalg.norm(x_points, axis=0)
        trajectory = 1.01 * x_points

        trajectory_line = ax.lines[-1]
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])

        current_point = ax.scatter(
            [],
            [],
            [],
            marker="o",
            s=55,
            color="red",
            depthshade=False,
            zorder=12,
        )

        frame_indices = np.linspace(0, n_time - 1, frame_step, dtype=int)
        if frame_indices[-1] != n_time - 1:
            frame_indices = np.append(frame_indices, n_time - 1)

        def update(frame_idx):
            idx = frame_indices[frame_idx] + 1
            trajectory_line.set_data(trajectory[0, :idx], trajectory[1, :idx])
            trajectory_line.set_3d_properties(trajectory[2, :idx])
            current_point._offsets3d = (
                [trajectory[0, idx - 1]],
                [trajectory[1, idx - 1]],
                [trajectory[2, idx - 1]],
            )
            return trajectory_line, current_point

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
            writer = animation.FFMpegWriter(fps=fps)
            ani.save(save_path, writer=writer, dpi=160)

        return ani
