import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy


class VehicleTrajectorySimulation:
    def __init__(self, x_0_p, x_p_goal, epsilon, t_1, t_2, mode_schedule=None):
        """
        State vector: x = [x_1, x_2, x_3, x_4, tau_2, z_1]

        x_1, x_2  - continuous states
        x_3, x_4  - angle parameters
        tau_2     - fast time scale, tau_2 = t / epsilon**2
        z_1       - discrete logic mode, z_1 in Q = {1, 2, 3}, per (4c)

        z_1 is constant along flows (its derivative is always zero, see
        kinematics()) and only changes at discrete switching instants. This is
        implemented as a hybrid simulation: the horizon [t_1, t_2] is split into
        segments according to self.mode_schedule, solve_ivp is run independently
        on each segment with z_1 frozen at that segment's mode, and the segments
        are stitched together into a single HybridSolution.
        """
        self.x_0 = np.append(x_0_p, [1.0, 0.0, 0.0, 0.0])
        self.x_p_goal = x_p_goal
        self.epsilon = epsilon
        self.t_1 = t_1
        self.t_2 = t_2
        self.mode_schedule = self._validate_schedule(mode_schedule)
        self.time_horizon = t_2 - t_1

    def _validate_schedule(self, mode_schedule):
        schedule = sorted(mode_schedule, key=lambda item: item[0])

        if schedule[0][0] != self.t_1:
            raise ValueError(
                f"mode_schedule must start at t_1={self.t_1}, "
                f"got first switch time {schedule[0][0]}"
            )

        times = [t for t, _ in schedule]
        if any(t2 <= t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("mode_schedule switch times must be strictly increasing")

        if times[-1] >= self.t_2:
            raise ValueError(f"last switch time {times[-1]} must be < t_2={self.t_2}")

        for _, mode in schedule:
            if mode not in (1, 2, 3):
                raise ValueError(f"mode {mode} not in Q = {{1, 2, 3}}")

        return schedule

    def J_x(self, x):
        x = np.asarray(x)  # (2,) during simulation, (2, 400, 400) during plotting
        N = x.shape[0]  # always 2, regardless of meshgrid or not

        goal = np.asarray(self.x_p_goal).reshape((N,) + (1,) * (x.ndim - 1))
        # simulation: goal stays (2,)
        # meshgrid:   goal becomes (2, 1, 1) → broadcasts over (2, 400, 400)

        return 1 / N * np.sum((x - goal) ** N, axis=0)
        # simulation: sums over shape (2,)         → scalar       ✓
        # meshgrid:   sums over shape (2,400,400)  → (400, 400)   ✓

    # x = [x_1, x_2, x_3, x_4, tau_2, z_1]
    def kinematics(self, t, x):
        epsilon = self.epsilon

        control = self.feedback_controller(t, x)

        # angle parameters
        x_3 = x[2]
        x_4 = x[3]
        return np.array(
            [
                1 / epsilon * x_3 * control,
                1 / epsilon * x_4 * control,
                1 / epsilon * x_4,
                -1 / epsilon * x_3,
                1 / epsilon**2,
                0,
            ]
        )

    def solve(self, t=None, rtol=1e-6, atol=1e-8):
        t_end = self.t_2 if t is None else t

        boundaries = [
            time for time, _ in self.mode_schedule if self.mode_schedule is not None
        ]

        boundaries.append(t_end)
        modes = [mode for _, mode in self.mode_schedule][: len(boundaries) - 1]

        segments = []
        y0 = self.x_0.copy()

        for i in range(len(boundaries) - 1):

            seg_t1, seg_t2 = boundaries[i], boundaries[i + 1]
            mode = modes[i]
            y0[5] = mode  # discrete jump in z_1 at the start of this segment

            sol = scipy.integrate.solve_ivp(
                fun=self.kinematics,
                t_span=(seg_t1, seg_t2),
                y0=y0,
                rtol=rtol,
                atol=atol,
                dense_output=True,
            )

            if not sol.success:
                raise RuntimeError(
                    f"Solution failed on segment [{seg_t1}, {seg_t2}] "
                    f"(mode {mode}): {sol.message}"
                )

            segments.append(sol)
            y0 = sol.y[:, -1].copy()  # carry continuous states into next segment

        return HybridSolution(segments)

    def feedback_controller(self, t, x):
        """
        Input: (time, current_state)
        return mode-dependent control
        """
        epsilon = self.epsilon
        tau_2 = t / epsilon**2
        x_p = x[:2]
        J = self.J_x(x_p)
        z_1 = int(round(x[5]))

        return np.cos(tau_2 + (z_1 - 2) * J)

    def plot_trajectory(self, HybridSolutions, target_circle_size=1.0, padding=1.0):
        fig, ax = plt.subplots(figsize=(7, 7))

        x_1_max = -math.inf
        x_1_min = math.inf
        x_2_max = -math.inf
        x_2_min = math.inf

        target_circle = patches.Circle(
            self.x_p_goal,
            radius=target_circle_size,
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            zorder=2,
        )

        ax.add_patch(target_circle)

        for solution in HybridSolutions:
            x_1 = solution.y[0]
            x_2 = solution.y[1]
            x_1_max = max(x_1_max, np.max(x_1))
            x_1_min = min(x_1_min, np.min(x_1))
            x_2_max = max(x_2_max, np.max(x_2))
            x_2_min = min(x_2_min, np.min(x_2))

            start_patch = patches.Circle((x_1[0], x_2[0]), radius=0.2, color="black")
            ax.add_patch(start_patch)

            t_end = solution.t[-1]

            if (
                np.linalg.norm(solution(t_end)[:2] - self.x_p_goal)
                <= target_circle_size
            ):
                ax.plot(x_1, x_2, color="blue", zorder=4)
            else:
                ax.plot(x_1, x_2, color="red", zorder=4)

        x_1 = np.linspace(x_1_min, x_1_max, 400)
        x_2 = np.linspace(x_2_min, x_2_max, 400)

        X1, X2 = np.meshgrid(x_1, x_2)

        J_grid = self.J_x([X1, X2])
        contour_plot = plt.contour(
            X1,
            X2,
            J_grid,
            cmap="gray_r",
            levels=15,
        )
        cbar = fig.colorbar(contour_plot, ax=ax)
        cbar.set_label(r"$J(x_p)$", rotation=0, labelpad=15)

        ax.set_xlabel(r"$x_1$", fontsize=13)
        ax.set_ylabel(r"$x_2$", fontsize=13)

        plt.xlim(x_1_min - padding, x_1_max + padding)
        plt.ylim(x_2_min - padding, x_2_max + padding)

        plt.tight_layout()
        plt.show()

    def verify_solution(self):
        pass


class HybridSolution:
    """
    Wraps the sequence of per-segment scipy.integrate solve_ivp results
    produced by VehicleTrajectorySimultion.solve() into a single piecewise
    trajectory, so the hybrid solution can be queried like a normal
    solve_ivp result (callable dense output, plus concatenated .t / .y).
    """

    def __init__(self, segments):
        self.segments = segments
        self.t = np.concatenate([seg.t for seg in segments])
        self.y = np.concatenate([seg.y for seg in segments], axis=1)
        self.switch_times = [seg.t[0] for seg in segments[1:]]

    def __call__(self, t):
        """Evaluate the stitched trajectory at scalar or array time t."""
        t_arr = np.atleast_1d(t).astype(float)
        out = np.empty((self.segments[0].y.shape[0], len(t_arr)))
        for i, ti in enumerate(t_arr):
            seg = self._segment_for_time(ti)
            out[:, i] = seg.sol(ti)
        return out[:, 0] if np.isscalar(t) else out

    def _segment_for_time(self, t):
        for seg in self.segments:
            if seg.t[0] - 1e-12 <= t <= seg.t[-1] + 1e-12:
                return seg
        # clamp out-of-range queries to the nearest boundary segment
        return self.segments[0] if t < self.segments[0].t[0] else self.segments[-1]

    def mode_at(self, t):
        """Return the active z_1 mode (rounded to int) at time t."""
        y = self(t)
        if y.ndim == 1:
            return int(round(y[5]))
        return np.round(y[5]).astype(int)
