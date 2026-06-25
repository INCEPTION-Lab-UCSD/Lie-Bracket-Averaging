from itertools import product

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import hybrid_solution


def generate_random_mode_schedule(t_1, t_2, num_modes, eta_1=1.0, N_0=1, seed=None):
    """
    Generate a random admissible mode schedule for Q = {1, ..., num_modes}.

    The first dwell includes the jump-count slack N_0; subsequent dwells use
    the average dwell time 1 / eta_1.
    """
    if eta_1 <= 0:
        raise ValueError("eta_1 must be positive")
    if num_modes < 1:
        raise ValueError("num_modes must be at least 1")
    if N_0 < 0:
        raise ValueError("N_0 must be nonnegative")
    if t_2 <= t_1:
        raise ValueError("t_2 must be greater than t_1")

    rng = np.random.default_rng(seed)
    Q = list(range(1, num_modes + 1))

    z_1 = int(rng.choice(Q))
    schedule = [(t_1, z_1)]
    if num_modes == 1:
        return schedule

    t = t_1
    # First dwell includes N_0 slack: (1 + N_0) / eta_1.
    dwell = (1.0 + N_0) / eta_1

    while True:
        t_next = t + dwell
        if t_next >= t_2:
            break

        available = [m for m in Q if m != z_1]
        z_1 = int(rng.choice(available))
        schedule.append((t_next, z_1))
        t = t_next
        dwell = 1.0 / eta_1

    return schedule


class Oscillator_Synchronization:
    def __init__(
        self,
        r,
        epsilon,
        kappa,
        omega,
        t_1,
        t_2,
        tau=None,
        mode_schedule=None,
        mode_schedule_config=None,
        graphs=None,
        tau_1=0.0,
        tau_2=0.0,
    ):
        self.r = r
        self.epsilon = epsilon
        self.kappa = kappa
        self.omega = omega
        self.t_1 = t_1
        self.t_2 = t_2

        # Generate starting angles ξ_i ∈ [0, 2π) for each oscillator
        self.xi0 = np.random.uniform(0.0, 2 * np.pi, size=r)

        if tau is None:
            self.tau = self.generate_control_directions()
        else:
            self.tau = tau

        # N_1 is the number of elements of Tau
        self.N_1 = len(self.tau)

        if graphs is None:
            self.graphs = [self.generate_default_graph()]
        else:
            self.graphs = graphs

        # N_2 is the number of graphs
        self.N_2 = len(self.graphs)

        if mode_schedule is not None and mode_schedule_config is not None:
            raise ValueError(
                "Provide either mode_schedule or mode_schedule_config, not both"
            )

        if mode_schedule is None:
            config = {} if mode_schedule_config is None else dict(mode_schedule_config)
            self.mode_schedule = self.generate_random_mode_schedule(**config)
        else:
            self.mode_schedule = self._validate_schedule(mode_schedule)

        # Full continuous state: [ξ_1, ..., ξ_r, τ_1, τ_2]
        self.state_0 = np.append(self.xi0, [tau_1, tau_2])

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def generate_oscillator(self):
        """
        Sample a random angle ξ ∈ [0, 2π) and return the corresponding
        Cartesian point on S^1 via the coordinate map:
            x_1 = -cos(ξ),  x_2 = sin(ξ)
        """
        xi = np.random.uniform(0.0, 2 * np.pi)
        return self.xi_to_cartesian(xi)

    def xi_to_cartesian(self, xi):
        """
        Map polar angle(s) ξ to Cartesian coordinates on S^1.

            x_1 = -cos(ξ),  x_2 = sin(ξ)

        Parameters
        ----------
        xi : float or np.ndarray
            Angle(s) in [0, 2π).

        Returns
        -------
        np.ndarray, shape (..., 2)
            Cartesian coordinates on S^1.
        """
        xi = np.asarray(xi)
        return np.stack([-np.cos(xi), np.sin(xi)], axis=-1)

    def embed_torus(self, state):
        """
        Embed the current state's oscillator angles into R^{2r} via:
            φ(ξ) = (x_1(ξ_1), ..., x_r(ξ_r)),  x_i = (-cos(ξ_i), sin(ξ_i))

        Returns
        -------
        np.ndarray, shape (2r,)
            Flattened Cartesian embedding of all r oscillators.
        """
        xi = state[: self.r]
        return self.xi_to_cartesian(xi).flatten()

    # ------------------------------------------------------------------
    # Control directions
    # ------------------------------------------------------------------

    def sign_combinations(self, r=None):
        if r is None:
            r = self.r
        return list(product([1, -1], repeat=r))

    def bijection(self, z_1):
        """
        Bijection σ: {0, ..., N1-1} → J ⊆ {+1, -1}^r.
        Maps an integer index to the corresponding α ∈ J.
        """
        idx = z_1 - 1

        tau_index = idx % self.N_1
        graph_index = idx // self.N_1

        return tau_index, graph_index

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def generate_default_graph(self):
        """
        Complete graph on r nodes: every oscillator connected to every other.
        Always connected and undirected, satisfying Assumption 5.
        For N_2 = 1 (static topology), this is the only graph used.
        """
        return [[j for j in range(self.r) if j != i] for i in range(self.r)]

    def generate_control_directions(self):
        return list(product([1, -1], repeat=self.r))

    def validate_graph(self, graph):
        """Check that a graph is undirected and connected (Assumption 5)."""
        r = self.r
        for i in range(r):
            for j in graph[i]:
                assert (
                    i in graph[j]
                ), f"Graph not undirected: edge ({i},{j}) missing reverse"
        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(graph[node])
        assert visited == set(range(r)), "Graph is not connected"

    # ------------------------------------------------------------------
    # Feedback controller — polar form (eq. 26)
    # ------------------------------------------------------------------
    def feedback_controller(self, state, graph_index):
        """
        Compute control inputs u_{i,k} in polar coordinates (eq. 26):

            u_{i,k}(ξ, τ_2) = ε^{-1} √2 w_i cos(w_i τ_2 + κ J_{i,k}(ξ)) κ^{-1/2}

        with the polar distance function:

            J_{i,k}(ξ) = Σ_{j ∈ N^i_k} (1 - cos(ξ_i - ξ_j))

        Parameters
        ----------
        state : np.ndarray, shape (r + 2,)
            Full state [ξ_1, ..., ξ_r, τ_1, τ_2].
        graph_index : int
            Index k ∈ {0, ..., N_2 - 1} selecting the active graph G_k.
        alpha : np.ndarray, shape (r,)
            Control direction vector α ∈ {+1, -1}^r.
        omega : np.ndarray, shape (r,) or None
            Probing frequencies. If None, linearly spaced in [ω_1, ω_2].

        Returns
        -------
        np.ndarray, shape (r,)
            Control inputs u_i for each oscillator.
        """
        xi = state[: self.r]  # angles ξ_i
        tau_2 = state[-1]  # probing timer

        graph = self.graphs[graph_index]

        u = np.zeros(self.r)
        for i in range(self.r):
            neighbors = graph[i]

            # Polar sync cost: J_{i,k}(ξ) = Σ_{j ∈ N^i_k} (1 - cos(ξ_i - ξ_j))
            J_ik = sum(1.0 - np.cos(xi[i] - xi[j]) for j in neighbors)

            w_i = self.omega[i]

            # u_{i,k} = ε^{-1} √2 w_i cos(w_i τ_2 + κ J_{i,k}) κ^{-1/2}
            u[i] = (
                (1.0 / self.epsilon)
                * np.sqrt(2 * w_i)
                * np.cos(w_i * tau_2 + self.kappa * J_ik)
                * self.kappa ** (-0.5)
            )

        return u

    # ------------------------------------------------------------------
    # Dynamics — polar form (eq. 25)
    # ------------------------------------------------------------------

    def dynamics(self, t, state, tau_index, graph_index):
        """
        Evaluate the closed-loop vector field in polar coordinates (eq. 25):

            ξ̇_i = 1 + α_i u_{i,k}(ξ, τ_2)
            τ̇_1 = 1     (dwell-time timer; reset to 0 at each mode switch)
            τ̇_2 = 1     (probing timer; never reset)

        Parameters
        ----------
        state : np.ndarray, shape (r + 2,)
            Full state [ξ_1, ..., ξ_r, τ_1, τ_2].
        graph_index : int
            Index k selecting the active graph G_k.
        alpha : np.ndarray, shape (r,)
            Control direction vector.
        omega : np.ndarray or None
            Probing frequencies.

        Returns
        -------
        np.ndarray, shape (r + 2,)
            Time derivative of the full state [ξ̇_1,...,ξ̇_r, τ̇_1, τ̇_2].
        """
        u = self.feedback_controller(state, graph_index)
        alpha = self.tau[tau_index]

        xi_dot = 1.0 + alpha * u  # shape (r,), eq. 25
        tau_1_dot = 1 / self.epsilon
        tau_2_dot = 1 / self.epsilon**2

        return np.append(xi_dot, [tau_1_dot, tau_2_dot])

    def solve(self, t=None, rtol=1e-6, atol=1e-8):

        t_end = self.t_2 if t is None else t
        if self.mode_schedule is not None:
            boundaries = [time for time, _ in self.mode_schedule]
        else:
            raise ValueError("mode schedule was not generated or inputted")

        boundaries.append(t_end)
        if self.mode_schedule is not None:
            modes = [mode for _, mode in self.mode_schedule][: len(boundaries) - 1]
        else:
            raise ValueError("mode schedule was not generated or inputted")

        segment_solutions = []
        state_0 = self.state_0

        for i in range(len(boundaries) - 1):
            seg_t1, seg_t2 = boundaries[i], boundaries[i + 1]
            mode = modes[i]
            tau_index, graph_index = self.bijection(mode)
            sol = solve_ivp(
                fun=self.dynamics,
                t_span=(seg_t1, seg_t2),
                y0=state_0,
                rtol=rtol,
                atol=atol,
                dense_output=True,
                args=(tau_index, graph_index),
            )
            if not sol.success:
                raise RuntimeError(
                    f"Solution failed on segment [{seg_t1}, {seg_t2}] "
                    f"(mode {mode}): {sol.message}"
                )

            segment_solutions.append(sol)
            state_0 = sol.y[:, -1].copy()
        return hybrid_solution.HybridSolution(segment_solutions)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def synchronization_cost_grid(self, xi_1, xi_2, graph_index=0):
        """
        Evaluate the single-graph synchronization cost on a phase grid.

        For r = 2 and the complete graph this is J(xi) = 1 - cos(xi_1 - xi_2),
        with range [0, 2].
        """
        if self.r != 2:
            raise ValueError("phase-plane cost plots require r == 2")

        graph = self.graphs[graph_index]
        xi = [xi_1, xi_2]
        cost = np.zeros_like(xi_1, dtype=float)

        for i, neighbors in enumerate(graph):
            for j in neighbors:
                if i < j:
                    cost += 1.0 - np.cos(xi[i] - xi[j])

        return cost

    def _plot_wrapped_phase_curve(self, ax, xi, **plot_kwargs):
        x_data, y_data = self._wrapped_phase_line_data(xi)
        ax.plot(x_data, y_data, **plot_kwargs)

    def _wrapped_phase_line_data(self, xi):
        jumps = np.any(np.abs(np.diff(xi, axis=1)) > np.pi, axis=0)
        x_data = []
        y_data = []

        for i in range(xi.shape[1]):
            if i > 0 and jumps[i - 1]:
                x_data.append(np.nan)
                y_data.append(np.nan)
            x_data.append(xi[0, i])
            y_data.append(xi[1, i])

        return np.array(x_data), np.array(y_data)

    def _mode_trace_data_until(self, t_current):
        t_end = min(t_current, self.t_2)
        boundaries = [time for time, _ in self.mode_schedule] + [self.t_2]
        modes = [mode for _, mode in self.mode_schedule]

        z_values = [modes[0]]
        t_values = [boundaries[0]]
        for i, mode in enumerate(modes):
            segment_end = min(boundaries[i + 1], t_end)
            if segment_end < boundaries[i]:
                break

            z_values.append(mode)
            t_values.append(segment_end)
            if segment_end >= t_end:
                break

            if i + 1 < len(modes):
                z_values.append(modes[i + 1])
                t_values.append(segment_end)

        return z_values, t_values

    def plot_solution(
        self,
        solution,
        graph_index=0,
        n_grid=300,
        n_time=2000,
        trajectory_color="red",
    ):
        """
        Plot z_1(t) and the wrapped two-oscillator phase trajectory.

        The right panel is intended for a fixed single graph. If multiple graphs
        are present, pass the graph_index whose cost surface should be shown.
        """
        if self.r != 2:
            raise ValueError("this plot is defined for the two-oscillator case r == 2")
        if not 0 <= graph_index < self.N_2:
            raise ValueError(f"graph_index must be in [0, {self.N_2 - 1}]")

        fig, (ax_mode, ax_phase) = plt.subplots(
            1,
            2,
            figsize=(8, 5),
            gridspec_kw={"width_ratios": [1, 3]},
        )

        t_end = solution.t[-1]
        boundaries = [time for time, _ in self.mode_schedule] + [t_end]
        modes = [mode for _, mode in self.mode_schedule]

        z_values = [modes[0]]
        t_values = [boundaries[0]]
        for i, mode in enumerate(modes):
            segment_end = boundaries[i + 1]
            z_values.append(mode)
            t_values.append(segment_end)
            if i + 1 < len(modes):
                z_values.append(modes[i + 1])
                t_values.append(segment_end)

        ax_mode.plot(z_values, t_values, color="black", linewidth=4)
        ax_mode.set_xlabel(r"$z_1(t)$")
        ax_mode.set_ylabel(r"$t$")
        ax_mode.set_xlim(0.5, self.N_1 * self.N_2 + 0.5)
        ax_mode.set_ylim(boundaries[0], t_end)
        ax_mode.set_xticks(range(1, self.N_1 * self.N_2 + 1))
        ax_mode.grid(True, alpha=0.3)

        phase_max = 2 * np.pi
        xi_values = np.linspace(0.0, phase_max, n_grid)
        xi_1, xi_2 = np.meshgrid(xi_values, xi_values)
        cost = self.synchronization_cost_grid(xi_1, xi_2, graph_index)
        image = ax_phase.imshow(
            cost,
            extent=(0.0, phase_max, 0.0, phase_max),
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=max(2.0, float(np.max(cost))),
            aspect="equal",
        )

        t_eval = np.linspace(solution.t[0], t_end, n_time)
        xi = np.mod(solution(t_eval)[:2], phase_max)
        self._plot_wrapped_phase_curve(
            ax_phase,
            xi,
            color=trajectory_color,
            linewidth=3,
            label=r"$\xi(t)$",
        )
        ax_phase.scatter(
            xi[0, 0],
            xi[1, 0],
            color="black",
            s=45,
            zorder=4,
            label=r"$\xi(0)$",
        )

        ax_phase.set_xlabel(r"$\xi_1$")
        ax_phase.set_ylabel(r"$\xi_2$")
        ax_phase.set_xlim(0.0, phase_max)
        ax_phase.set_ylim(0.0, phase_max)
        ax_phase.set_xticks(range(0, 7))
        ax_phase.set_yticks(range(0, 7))
        ax_phase.legend(loc="upper left", bbox_to_anchor=(0.0, 1.1), ncol=2)

        colorbar = fig.colorbar(image, ax=ax_phase, fraction=0.046, pad=0.04)
        colorbar.set_label(r"$J(\xi)$", rotation=270, labelpad=18)

        fig.tight_layout()
        return fig, (ax_mode, ax_phase)

    def animate_solution(
        self,
        solution,
        graph_index=0,
        n_grid=300,
        n_time=2000,
        frame_step=200,
        interval=40,
        repeat_delay=1200,
        trajectory_color="red",
    ):
        """
        Animate z_1(t) and the wrapped two-oscillator phase trajectory.
        """
        if frame_step <= 0:
            raise ValueError("frame_step must be positive")
        if self.r != 2:
            raise ValueError("this animation is defined for r == 2")
        if not 0 <= graph_index < self.N_2:
            raise ValueError(f"graph_index must be in [0, {self.N_2 - 1}]")

        fig, (ax_mode, ax_phase) = plt.subplots(
            1,
            2,
            figsize=(8, 5),
            gridspec_kw={"width_ratios": [1, 3]},
        )

        t_start = solution.t[0]
        t_end = solution.t[-1]

        ax_mode.set_xlabel(r"$z_1(t)$")
        ax_mode.set_ylabel(r"$t$")
        ax_mode.set_xlim(0.5, self.N_1 * self.N_2 + 0.5)
        ax_mode.set_ylim(t_start, t_end)
        ax_mode.set_xticks(range(1, self.N_1 * self.N_2 + 1))
        ax_mode.grid(True, alpha=0.3)

        phase_max = 2 * np.pi
        xi_values = np.linspace(0.0, phase_max, n_grid)
        xi_1, xi_2 = np.meshgrid(xi_values, xi_values)
        cost = self.synchronization_cost_grid(xi_1, xi_2, graph_index)
        image = ax_phase.imshow(
            cost,
            extent=(0.0, phase_max, 0.0, phase_max),
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=max(2.0, float(np.max(cost))),
            aspect="equal",
        )

        ax_phase.set_xlabel(r"$\xi_1$")
        ax_phase.set_ylabel(r"$\xi_2$")
        ax_phase.set_xlim(0.0, phase_max)
        ax_phase.set_ylim(0.0, phase_max)
        ax_phase.set_xticks(range(0, 7))
        ax_phase.set_yticks(range(0, 7))

        t_eval = np.linspace(t_start, t_end, n_time)
        xi = np.mod(solution(t_eval)[:2], phase_max)

        (mode_line,) = ax_mode.plot([], [], color="black", linewidth=4)
        (phase_line,) = ax_phase.plot(
            [],
            [],
            color=trajectory_color,
            linewidth=3,
            label=r"$\xi(t)$",
        )
        current_point = ax_phase.scatter(
            [],
            [],
            color=trajectory_color,
            s=35,
            zorder=5,
        )
        initial_point = ax_phase.scatter(
            xi[0, 0],
            xi[1, 0],
            color="black",
            s=45,
            zorder=4,
            label=r"$\xi(0)$",
        )
        ax_phase.legend(loc="upper left", bbox_to_anchor=(0.0, 1.1), ncol=2)

        colorbar = fig.colorbar(image, ax=ax_phase, fraction=0.046, pad=0.04)
        colorbar.set_label(r"$J(\xi)$", rotation=270, labelpad=18)
        fig.tight_layout()

        frame_indices = np.linspace(0, n_time - 1, frame_step, dtype=int)
        if frame_indices[-1] != n_time - 1:
            frame_indices = np.append(frame_indices, n_time - 1)

        def update(frame_idx):
            idx = frame_indices[frame_idx] + 1

            mode_x, mode_y = self._mode_trace_data_until(t_eval[idx - 1])
            mode_line.set_data(mode_x, mode_y)

            phase_x, phase_y = self._wrapped_phase_line_data(xi[:, :idx])
            phase_line.set_data(phase_x, phase_y)
            current_point.set_offsets([[xi[0, idx - 1], xi[1, idx - 1]]])

            return mode_line, phase_line, current_point, initial_point

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
        return ani

    # ------------------------------------------------------------------
    # Mode schedule
    # ------------------------------------------------------------------
    def _validate_schedule(self, mode_schedule):
        if mode_schedule is None:
            raise ValueError("mode_schedule must not be None")

        schedule = sorted(mode_schedule, key=lambda item: item[0])
        if not schedule:
            raise ValueError("mode_schedule must contain at least one entry")

        if schedule[0][0] != self.t_1:
            raise ValueError(
                f"mode_schedule must start at t_1={self.t_1}, "
                f"got first switch time {schedule[0][0]}"
            )

        times = [time for time, _ in schedule]
        if any(t_next <= t for t, t_next in zip(times, times[1:])):
            raise ValueError("mode_schedule switch times must be strictly increasing")

        if times[-1] >= self.t_2:
            raise ValueError(f"last switch time {times[-1]} must be < t_2={self.t_2}")

        max_mode = self.N_1 * self.N_2
        normalized_schedule = []
        for time, mode in schedule:
            mode_int = int(mode)
            if mode != mode_int or mode_int not in range(1, max_mode + 1):
                raise ValueError(f"mode {mode} not in Q = {{1, ..., {max_mode}}}")
            normalized_schedule.append((time, mode_int))

        return normalized_schedule

    def generate_random_mode_schedule(self, eta_1=1.0, N_0=1, seed=None):
        """
        Generate a mode schedule for the hybrid automaton (6).

        Q = {1, ..., N_1*N_2}. The auxiliary timer z_2 flows at rate eta_1
        and triggers a jump when z_2 >= 1, giving average dwell time 1/eta_1.
        This satisfies the ADT constraint (8a): N#(t1,t2) <= eta_1*(t2-t1) + N_0.

        Parameters
        ----------
        eta_1 : float
            Average switch rate (switches per unit time).
        N_0 : float
            Initial jump counter value (slack N° in eq. 8a).
        seed : int or None
            Random seed.

        Returns
        -------
        list of (float, int)
            (t_switch, z_1) pairs covering [t_1, t_2].
        """
        schedule = generate_random_mode_schedule(
            self.t_1,
            self.t_2,
            self.N_1 * self.N_2,
            eta_1=eta_1,
            N_0=N_0,
            seed=seed,
        )
        return self._validate_schedule(schedule)
