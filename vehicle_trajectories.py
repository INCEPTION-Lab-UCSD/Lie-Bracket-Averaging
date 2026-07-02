import math

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy

import hybrid_solution


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
        self.x_p_goal = x_p_goal
        self.epsilon = epsilon
        self.t_1 = t_1
        self.t_2 = t_2

        self.time_horizon = t_2 - t_1

        self.mode_schedule = self._validate_schedule(mode_schedule)
        self.x_0 = np.append(x_0_p, [1.0, 0.0, 0.0, self.mode_schedule[0][1]])

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

        return hybrid_solution.HybridSolution(segments)

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
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7, 7),
            gridspec_kw={"width_ratios": [1, 3]},  # ← mode narrow on left
        )
        ax_mode, ax_trajectory = axes  # ← mode first now
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
        ax_trajectory.add_patch(target_circle)
        for solution in HybridSolutions:
            x_1 = solution.y[0]
            x_2 = solution.y[1]
            x_1_max = max(x_1_max, np.max(x_1))
            x_1_min = min(x_1_min, np.min(x_1))
            x_2_max = max(x_2_max, np.max(x_2))
            x_2_min = min(x_2_min, np.min(x_2))
            start_patch = patches.Circle(
                (x_1[0], x_2[0]), radius=0.35, color="black", zorder=5
            )
            ax_trajectory.add_patch(start_patch)
            t_end = solution.t[-1]
            color, label = (
                ("blue", "converging")
                if np.linalg.norm(solution(t_end)[:2] - self.x_p_goal)
                <= target_circle_size
                else ("red", "diverging")
            )
            ax_trajectory.plot(x_1, x_2, color=color, zorder=4, label=label)
            t_data = []
            z1_data = []
            for seg in solution.segments:
                z1_val = int(round(seg.y[5, 0]))
                t_data.extend([seg.t[0], seg.t[-1]])
                z1_data.extend([z1_val, z1_val])
            ax_mode.plot(z1_data, t_data, color=color, linewidth=2)
        ax_trajectory.legend(loc="upper right")
        x_1 = np.linspace(x_1_min - padding, x_1_max + padding, 1000)
        x_2 = np.linspace(x_2_min - padding, x_2_max + padding, 1000)
        X1, X2 = np.meshgrid(x_1, x_2)
        J_grid = self.J_x([X1, X2])
        contour_plot = ax_trajectory.contourf(
            X1,
            X2,
            J_grid,
            cmap="gray_r",
            levels=15,
            zorder=1,
        )
        cbar = fig.colorbar(contour_plot, ax=ax_trajectory)
        cbar.set_label(r"$J(x_p)$", rotation=0, labelpad=15)
        ax_trajectory.set_xlabel(r"$x_1$", fontsize=13)
        ax_trajectory.set_ylabel(r"$x_2$", fontsize=13)
        ax_trajectory.set_xlim(x_1_min - padding, x_1_max + padding)
        ax_trajectory.set_ylim(x_2_min - padding, x_2_max + padding)
        t_max = HybridSolutions[0].t[-1]
        tick_mark_length = t_max // 3
        tick_marks = [tick_mark_length * i for i in range(4)]
        ax_mode.set_xlabel(r"$z_1(t)$")
        ax_mode.set_ylabel(r"$t$")
        ax_mode.set_xticks([1, 2, 3])
        ax_mode.set_yticks(tick_marks)
        plt.tight_layout()
        return fig, axes

    def generate_random_mode_schedule(
        self,
        eta_1=1.0,  # avg switches per unit time → controls switch frequency
        eta_2=None,  # if given, enforces AAT constraint (8b) on time in Qu={1,2}
        N_0=2,  # jump budget slack (unused directly but kept for clarity)
        T_0=1.0,  # activation time slack for AAT budget
        seed=None,
    ):
        """
        Randomly generate a mode_schedule admissible for the hybrid automaton (6).

        Switch times are sampled from an exponential distribution with mean 1/eta_1,
        naturally producing an average dwell-time of 1/eta_1 between switches.
        At each switch, the new mode is drawn uniformly from Q\\{current_mode},
        matching the jump map z+_1 ∈ Q\\{z_1} in (6b).

        If eta_2 is provided, the total time spent in Qu={1,2} is tracked and
        capped at eta_2*(t_2-t_1)+T_0 to enforce the AAT constraint (8b). When
        the budget is nearly exhausted, the next mode is forced into Qs={3}.
        """
        rng = np.random.default_rng(seed)
        Q = [1, 2, 3]
        Qu = {1, 2}  # unstable modes: spoofed, no measurement
        Qs = {3}  # stable mode:   nominal operation

        # Qu budget from constraint (8b): T♯(t_1,t_2) ≤ η₂(t_2−t_1) + T°
        Qu_budget = eta_2 * (self.t_2 - self.t_1) + T_0 if eta_2 is not None else np.inf

        # Random initial mode at t_1
        mode = int(rng.choice(Q))
        schedule = [(self.t_1, mode)]
        t = self.t_1
        time_in_Qu = 0.0

        while True:
            # Dwell time drawn from Exp(eta_1), mean = 1/eta_1
            dwell = rng.exponential(scale=1.0 / eta_1)

            # If currently in Qu, cap dwell at remaining budget
            if mode in Qu and eta_2 is not None:
                remaining = Qu_budget - time_in_Qu
                dwell = min(dwell, remaining)

            t_next = t + dwell
            if t_next >= self.t_2:
                break

            # Accumulate time spent in Qu
            if mode in Qu:
                time_in_Qu += dwell

            # Available modes: always exclude current (jump map z+_1 ∈ Q\{z_1})
            available = [m for m in Q if m != mode]

            # If Qu budget exhausted, force next mode into Qs
            if eta_2 is not None and time_in_Qu >= Qu_budget - 1e-9:
                available = [m for m in available if m in Qs]
                if not available:
                    break  # already in Qs and budget gone, stop switching

            mode = int(rng.choice(available))
            schedule.append((t_next, mode))
            t = t_next

        return schedule

    def _trajectory_headings(self, x1, x2):
        if len(x1) < 2:
            return np.zeros_like(x1)

        dx = np.gradient(x1)
        dy = np.gradient(x2)
        speed = np.hypot(dx, dy)
        headings = np.arctan2(dy, dx)

        for i, speed_i in enumerate(speed):
            if speed_i > 1e-9 and np.isfinite(headings[i]):
                continue
            headings[i] = headings[i - 1] if i > 0 else 0.0

        return headings

    def _add_cartoon_vehicle(self, ax, color, zorder=7):
        length = 0.82
        width = 0.44
        wheel_radius = 0.07

        body = patches.FancyBboxPatch(
            (-length / 2, -width / 2),
            length,
            width,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=zorder,
        )
        cabin = patches.Polygon(
            [
                (-0.14, -0.16),
                (0.18, -0.15),
                (0.31, 0.0),
                (0.18, 0.15),
                (-0.14, 0.16),
            ],
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            alpha=0.9,
            zorder=zorder + 1,
        )
        hood = patches.Polygon(
            [(length / 2, 0.0), (0.27, -0.13), (0.27, 0.13)],
            closed=True,
            facecolor="#f7d65a",
            edgecolor="black",
            linewidth=0.8,
            zorder=zorder + 2,
        )
        wheels = [
            patches.Circle(
                (x, y),
                wheel_radius,
                facecolor="black",
                edgecolor="white",
                linewidth=0.5,
                zorder=zorder + 2,
            )
            for x in (-0.23, 0.23)
            for y in (-width / 2, width / 2)
        ]

        artists = [body, *wheels, cabin, hood]
        for artist in artists:
            ax.add_patch(artist)

        self._set_cartoon_vehicle_pose(ax, artists, 0.0, 0.0, 0.0)
        return artists

    def _set_cartoon_vehicle_pose(self, ax, artists, x, y, heading):
        vehicle_transform = transforms.Affine2D().rotate(heading).translate(x, y)
        for artist in artists:
            artist.set_transform(vehicle_transform + ax.transData)

    def animate_solution(
        self,
        HybridSolutions,
        frame_step=200,
        interval=40,
        repeat_delay=1200,
        target_circle_size=1.0,
        padding=1.0,
    ):
        if frame_step <= 0:
            raise ValueError("frame_step must be positive")

        fig, axes = self.plot_trajectory(HybridSolutions, target_circle_size, padding)
        ax_mode, ax_trajectory = axes

        # plot_trajectory drew full static lines; grab and clear them for animation
        mode_lines = list(ax_mode.lines)  # one per solution
        trajectory_lines = list(ax_trajectory.lines)  # one per solution
        for line in [*mode_lines, *trajectory_lines]:
            line.set_data([], [])

        # --- Precompute trajectories at a uniform time grid ---
        t_min = min(sol.t[0] for sol in HybridSolutions)
        t_max = max(sol.t[-1] for sol in HybridSolutions)
        n_points = max(len(sol.t) for sol in HybridSolutions)
        all_times = np.linspace(t_min, t_max, n_points)

        precomputed = []
        for sol in HybridSolutions:
            t_clipped = np.clip(all_times, sol.t[0], sol.t[-1])
            xy = sol(t_clipped)  # shape (6, n_points) via dense output
            precomputed.append(
                {
                    "x1": xy[0],
                    "x2": xy[1],
                    "heading": self._trajectory_headings(xy[0], xy[1]),
                    "z1": np.round(xy[5]).astype(
                        int
                    ),  # z1 is piecewise constant; round avoids float drift
                }
            )

        vehicle_artists = [
            self._add_cartoon_vehicle(
                ax_trajectory,
                trajectory_line.get_color(),
                zorder=7 + i,
            )
            for i, trajectory_line in enumerate(trajectory_lines)
        ]
        vehicle_artist_list = [
            artist for vehicle in vehicle_artists for artist in vehicle
        ]

        # Frame indices: n evenly spaced steps through all_times, always ending at last point
        frame_indices = np.linspace(0, n_points - 1, frame_step, dtype=int)
        if frame_indices[-1] != n_points - 1:
            frame_indices = np.append(frame_indices, n_points - 1)

        def update(frame_idx):
            idx = frame_indices[frame_idx] + 1
            for i, data in enumerate(precomputed):
                # Trajectory: x1 on x-axis, x2 on y-axis
                trajectory_lines[i].set_data(data["x1"][:idx], data["x2"][:idx])
                # Mode plot: z1 on x-axis, t on y-axis
                mode_lines[i].set_data(data["z1"][:idx], all_times[:idx])
                self._set_cartoon_vehicle_pose(
                    ax_trajectory,
                    vehicle_artists[i],
                    data["x1"][idx - 1],
                    data["x2"][idx - 1],
                    data["heading"][idx - 1],
                )
            return (*mode_lines, *trajectory_lines, *vehicle_artist_list)

        ani = animation.FuncAnimation(
            fig,
            update,
            len(frame_indices),
            interval=interval,
            repeat=True,
            repeat_delay=repeat_delay,
            blit=False,
        )
        update(0)
        return ani

    def verify_solution(self):
        pass
