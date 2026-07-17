from itertools import product

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

import hybrid_solution

DEEP_NAVY_THEME = {
    "figure": "#08111F",
    "axes": "#0E1A2B",
    "text": "#F2F5F8",
    "grid": "#26384F",
    "trajectory": "#FFB000",
    "arrow": "#5CC8FF",
    "circle": "#F2F5F8",
    "initial": "#F2F5F8",
    "edge": "#08111F",
    "cmap": "magma",
}

CHARCOAL_THEME = {
    "figure": "#111318",
    "axes": "#181B22",
    "text": "#E8EAED",
    "grid": "#3A3F4B",
    "trajectory": "#FF4D4D",
    "arrow": "#4DA3FF",
    "circle": "#F2F5F8",
    "initial": "#F2F5F8",
    "edge": "#08111F",
    "cmap": "viridis",
}


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


class OscillatorSynchronization:
    def __init__(
        self,
        r,
        epsilon,
        kappa,
        omega,
        t_1,
        t_2,
        tau=None,
        xi0=None,
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

        # Generate starting angles eta_i ∈ [0, 2π) for each oscillator
        if xi0 is None:
            self.xi0 = np.random.uniform(0.0, 2 * np.pi, size=r)
            print(self.xi0)
        else:
            self.xi0 = xi0

        if tau is None:
            self.tau = self.generate_control_directions()
        else:
            self.tau = tau

        # N_1 is the number of elements of Tau
        self.N_1 = len(self.tau)

        if graphs is None:
            self.graphs = [self.generate_default_graph()]
        else:
            self.graphs = self.normalize_graphs(graphs)

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

    def embed_torus_3d(self, xi_1, xi_2, major_radius=2.0, minor_radius=0.7):
        """
        Embed T^2 into R^3 for visualization as a standard torus.
        """
        x = (major_radius + minor_radius * np.cos(xi_2)) * np.cos(xi_1)
        y = (major_radius + minor_radius * np.cos(xi_2)) * np.sin(xi_1)
        z = minor_radius * np.sin(xi_2)
        return x, y, z

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

    def normalize_graphs(self, graphs):
        return [self.normalize_graph(graph) for graph in graphs]

    def normalize_graph(self, graph):
        """
        Convert supported graph inputs into a zero-based adjacency list.

        Supported inputs:
        - zero-based adjacency list: [[1, 2], [0], ...]
        - one-based edge list: [(3, 1), (1, 3), ...]
        - zero-based edge list: [(2, 0), (0, 2), ...]
        """
        if self._looks_like_edge_list(graph):
            return self._edge_list_to_adjacency(graph)

        adjacency = []
        if len(graph) != self.r:
            raise ValueError(
                f"adjacency graph must have {self.r} nodes, got {len(graph)}"
            )

        for i, neighbors in enumerate(graph):
            normalized_neighbors = []
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if not 0 <= neighbor < self.r:
                    raise ValueError(
                        f"neighbor {neighbor} for node {i} is not in "
                        f"{{0, ..., {self.r - 1}}}"
                    )
                if neighbor == i:
                    raise ValueError(f"self-loop ({i}, {neighbor}) is not allowed")
                normalized_neighbors.append(neighbor)
            adjacency.append(sorted(set(normalized_neighbors)))

        self.validate_graph(adjacency)
        return adjacency

    def _looks_like_edge_list(self, graph):
        return all(
            isinstance(edge, tuple)
            and len(edge) == 2
            and all(isinstance(node, (int, np.integer)) for node in edge)
            for edge in graph
        )

    def _edge_list_to_adjacency(self, graph):
        if not graph:
            raise ValueError("edge-list graph must contain at least one edge")

        nodes = [int(node) for edge in graph for node in edge]
        min_node = min(nodes)
        max_node = max(nodes)

        if 1 <= min_node and max_node <= self.r:
            index_offset = 1
        elif 0 <= min_node and max_node < self.r:
            index_offset = 0
        else:
            raise ValueError(
                "edge-list nodes must be either zero-based in "
                f"{{0, ..., {self.r - 1}}} or one-based in {{1, ..., {self.r}}}"
            )

        adjacency_sets = [set() for _ in range(self.r)]
        for source, target in graph:
            source = int(source) - index_offset
            target = int(target) - index_offset
            if source == target:
                raise ValueError(f"self-loop ({source}, {target}) is not allowed")
            adjacency_sets[source].add(target)

        adjacency = [sorted(neighbors) for neighbors in adjacency_sets]
        self.validate_graph(adjacency)
        return adjacency

    def validate_graph(self, graph):
        """Check that a graph is undirected and connected (Assumption 5)."""
        if self._looks_like_edge_list(graph):
            graph = self._edge_list_to_adjacency(graph)

        r = self.r
        if len(graph) != r:
            raise ValueError(f"graph must have {r} nodes, got {len(graph)}")

        for i in range(r):
            for j in graph[i]:
                if not 0 <= j < r:
                    raise ValueError(
                        f"neighbor {j} for node {i} is not in {{0, ..., {r - 1}}}"
                    )
                if i not in graph[j]:
                    raise ValueError(
                        f"Graph not undirected: edge ({i}, {j}) missing reverse"
                    )

        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(graph[node])
        if visited != set(range(r)):
            raise ValueError("Graph is not connected")

        return True

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

            deta_i = 1 + α_i u_{i,k}(eta, tau_2)
            dtau_1 = 1     (dwell-time timer; reset to 0 at each mode switch)
            dtau_2 = 1     (probing timer; never reset)

        Parameters
        ----------
        state : np.ndarray, shape (r + 2,)
            Full state [eta_1, ..., eta_r, tau_1, tau_2].
        Returns
        -------
        np.ndarray, shape (r + 2,)
            Time derivative of the full state [deta_1,...,deta_r, dtau_1, dtau_2].
        """
        u = self.feedback_controller(state, graph_index)
        alpha = self.tau[tau_index]

        xi_dot = 1.0 + alpha * u  # shape (r,), eq. 25
        tau_1_dot = 1 / self.epsilon
        tau_2_dot = 1 / self.epsilon**2

        return np.append(xi_dot, [tau_1_dot, tau_2_dot])

    def solve(self, t=None, rtol=1e-6, atol=1e-8):

        t_end = self.t_2 if t is None else t
        if self.mode_schedule is None:
            raise ValueError("mode schedule was not generated or inputted")

        boundaries = [time for time, _ in self.mode_schedule]
        boundaries.append(t_end)
        modes = [mode for _, mode in self.mode_schedule][: len(boundaries) - 1]

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

    def _cost_color_norm(self, cost, cost_gamma=1.8):
        if cost_gamma is None:
            return colors.Normalize(
                vmin=0.0,
                vmax=max(2.0, float(np.max(cost))),
            )
        if cost_gamma <= 0:
            raise ValueError("cost_gamma must be positive")

        return colors.PowerNorm(
            gamma=cost_gamma,
            vmin=0.0,
            vmax=max(2.0, float(np.max(cost))),
        )

    @staticmethod
    def _apply_dark_2d_axis_style(ax, theme=CHARCOAL_THEME):
        ax.set_facecolor(theme["axes"])
        ax.tick_params(colors=theme["text"])
        ax.xaxis.label.set_color(theme["text"])
        ax.yaxis.label.set_color(theme["text"])
        ax.title.set_color(theme["text"])
        for spine in ax.spines.values():
            spine.set_color(theme["grid"])

    @staticmethod
    def _apply_dark_legend_style(legend, theme=CHARCOAL_THEME):
        if legend is None:
            return
        legend.get_frame().set_facecolor(theme["axes"])
        legend.get_frame().set_edgecolor(theme["grid"])
        legend.get_frame().set_alpha(0.95)
        for text in legend.get_texts():
            text.set_color(theme["text"])

    @staticmethod
    def _apply_dark_colorbar_style(colorbar, theme=CHARCOAL_THEME):
        colorbar.ax.set_facecolor(theme["figure"])
        colorbar.ax.tick_params(colors=theme["text"])
        colorbar.ax.yaxis.label.set_color(theme["text"])
        colorbar.outline.set_edgecolor(theme["grid"])

    @staticmethod
    def _apply_dark_3d_axis_style(ax, theme=CHARCOAL_THEME):
        ax.set_facecolor(theme["axes"])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(colors.to_rgba(theme["axes"], 1.0))
            axis.pane.set_edgecolor(colors.to_rgba(theme["axes"], 1.0))
            axis.line.set_color(colors.to_rgba(theme["axes"], 0.0))
            axis._axinfo["grid"]["color"] = colors.to_rgba(theme["axes"], 0.0)
            axis._axinfo["grid"]["linewidth"] = 0.0
            axis._axinfo["axisline"]["color"] = colors.to_rgba(theme["axes"], 0.0)
            axis._axinfo["axisline"]["linewidth"] = 0.0
        ax.grid(False)

    def _setup_oscillator_state_axis(
        self,
        ax,
        oscillator_index,
        t_start,
        t_end,
        theme=None,
    ):
        ax.set_title(rf"Oscillator {oscillator_index + 1} State", fontsize=11, pad=6)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(rf"$\xi_{oscillator_index + 1}$", labelpad=-5)
        ax.set_xlim(t_start, t_end)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, 0.0, np.pi])
        ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        if theme is not None:
            self._apply_dark_2d_axis_style(ax, theme)
            ax.grid(True, color=theme["grid"], alpha=0.6)
        else:
            ax.grid(True, alpha=0.3)

    @staticmethod
    def _wrapped_phase_trace(phase):
        """Return a wrapped phase trace with discontinuities omitted."""
        trace = np.asarray(phase, dtype=float).copy()
        jumps = np.abs(np.diff(trace)) > np.pi
        trace[1:][jumps] = np.nan
        return trace

    def _mode_at_time(self, t):
        mode = self.mode_schedule[0][1]
        for switch_time, switch_mode in self.mode_schedule:
            if t + 1e-12 < switch_time:
                break
            mode = switch_mode
        return mode

    def _alpha_for_mode(self, mode):
        tau_index, _ = self.bijection(mode)
        return np.asarray(self.tau[tau_index], dtype=float)

    def _setup_unit_circle_axis(self, ax, oscillator_index, theme=None):
        theta = np.linspace(0.0, 2 * np.pi, 400)
        circle = self.xi_to_cartesian(theta)
        circle_color = theme["circle"] if theme is not None else "0.25"
        grid_color = theme["grid"] if theme is not None else "0.82"
        ax.plot(circle[:, 0], circle[:, 1], color=circle_color, linewidth=1.6)
        ax.axhline(0.0, color=grid_color, linewidth=0.8, alpha=0.8, zorder=0)
        ax.axvline(0.0, color=grid_color, linewidth=0.8, alpha=0.8, zorder=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.28, 1.28)
        ax.set_ylim(-1.28, 1.28)
        ax.set_xticks([])
        ax.set_yticks([])
        if theme is not None:
            self._apply_dark_2d_axis_style(ax, theme)
        title = ax.set_title(rf"Oscillator {oscillator_index + 1}", fontsize=11)
        if theme is not None:
            title.set_color(theme["text"])
        return title

    def _setup_unit_circle_axes(self, axes, point_color="red", theme=None):
        points = []
        arrows = []
        titles = []
        for i, ax in enumerate(axes):
            titles.append(self._setup_unit_circle_axis(ax, i, theme=theme))
            point = ax.scatter(
                [],
                [],
                color=point_color,
                edgecolor=theme["edge"] if theme is not None else "black",
                linewidth=0.6,
                s=70,
                zorder=4,
            )
            arrow = ax.quiver(
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.014,
                color=theme["arrow"] if theme is not None else "#1f77b4",
                clip_on=False,
                zorder=5,
            )
            points.append(point)
            arrows.append(arrow)
        self._add_current_direction_legend(axes[0], theme=theme)
        return points, arrows, titles

    @staticmethod
    def _add_current_direction_legend(ax, theme=None):
        control_direction_handle = Line2D(
            [0],
            [0],
            color=theme["arrow"] if theme is not None else "#1f77b4",
            marker=">",
            markersize=8,
            linewidth=2,
            linestyle="None",
            label="Oscillator Direction",
        )
        legend = ax.legend(
            handles=[control_direction_handle],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            frameon=True,
            fontsize=9,
        )
        if theme is not None:
            OscillatorSynchronization._apply_dark_legend_style(legend, theme)

    def _current_direction_arrow(self, xi_value, direction_value):
        tangent = np.array([np.sin(xi_value), np.cos(xi_value)])
        return 0.42 * np.sign(direction_value) * tangent

    def _update_unit_circle_artists(
        self,
        points,
        arrows,
        titles,
        xi_values,
        direction,
        alpha,
    ):
        cartesian = self.xi_to_cartesian(xi_values)

        for i, point in enumerate(points):
            point.set_offsets([cartesian[i]])
            arrow = self._current_direction_arrow(xi_values[i], direction[i])
            arrows[i].set_offsets([cartesian[i]])
            arrows[i].set_UVC([arrow[0]], [arrow[1]])
            titles[i].set_text(rf"Oscillator {i + 1}: Control Direction={alpha[i]:.0f}")

    def current_direction(self, state, mode):
        tau_index, graph_index = self.bijection(mode)
        return self.dynamics(0.0, state, tau_index, graph_index)[: self.r]

    @staticmethod
    def _hide_3d_axis_labels_and_ticks(ax):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    def plot_solution_3d(
        self,
        solution,
        graph_index=0,
        n_grid=120,
        n_time=2000,
        major_radius=2.0,
        minor_radius=0.7,
        trajectory_color=CHARCOAL_THEME["trajectory"],
        surface_alpha=0.55,
        trajectory_lift=0.0,
        elev=25,
        azim=-60,
        cost_gamma=1.8,
    ):
        """
        Plot z_1(t) and the trajectory on a 3D torus embedding of T^2.
        """
        if self.r != 2:
            raise ValueError("this plot is defined for the two-oscillator case r == 2")
        if not 0 <= graph_index < self.N_2:
            raise ValueError(f"graph_index must be in [0, {self.N_2 - 1}]")

        theme = CHARCOAL_THEME
        fig = plt.figure(figsize=(9.6, 7))
        fig.patch.set_facecolor(theme["figure"])
        grid_spec = fig.add_gridspec(
            3,
            5,
            width_ratios=[1.35, 1.45, 1.45, 1.45, 1.45],
            height_ratios=[1.25, 1.25, 0.55],
            hspace=0.55,
            wspace=0.18,
        )
        ax_circle_1 = fig.add_subplot(grid_spec[0, 0])
        ax_circle_2 = fig.add_subplot(grid_spec[1, 0])
        ax_torus = fig.add_subplot(grid_spec[:2, 1:], projection="3d")
        ax_state_1 = fig.add_subplot(grid_spec[2, 1:3])
        ax_state_2 = fig.add_subplot(grid_spec[2, 3:], sharex=ax_state_1)

        t_end = solution.t[-1]
        circle_points, circle_arrows, circle_titles = self._setup_unit_circle_axes(
            (ax_circle_1, ax_circle_2),
            point_color=trajectory_color,
            theme=theme,
        )

        phase_max = 2 * np.pi
        xi_values = np.linspace(0.0, phase_max, n_grid)
        xi_1, xi_2 = np.meshgrid(xi_values, xi_values)
        cost = self.synchronization_cost_grid(xi_1, xi_2, graph_index)
        surface_x, surface_y, surface_z = self.embed_torus_3d(
            xi_1,
            xi_2,
            major_radius=major_radius,
            minor_radius=minor_radius,
        )

        norm = self._cost_color_norm(cost, cost_gamma)
        torus_cmap = plt.get_cmap(theme["cmap"])
        facecolors = torus_cmap(norm(cost))
        facecolors[..., -1] = surface_alpha

        ax_torus.plot_surface(
            surface_x,
            surface_y,
            surface_z,
            facecolors=facecolors,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            shade=False,
        )

        t_eval = np.linspace(solution.t[0], t_end, n_time)
        xi = np.mod(solution(t_eval)[:2], phase_max)
        traj_x, traj_y, traj_z = self.embed_torus_3d(
            xi[0],
            xi[1],
            major_radius=major_radius,
            minor_radius=minor_radius + trajectory_lift,
        )
        ax_torus.plot(
            traj_x,
            traj_y,
            traj_z,
            color=trajectory_color,
            linewidth=3,
            label=r"$\xi(t)$",
        )
        ax_torus.scatter(
            traj_x[0],
            traj_y[0],
            traj_z[0],
            color=theme["initial"],
            edgecolor=theme["edge"],
            linewidth=0.6,
            s=45,
            depthshade=False,
            label=r"$\xi(0)$",
        )
        ax_torus.scatter(
            traj_x[-1],
            traj_y[-1],
            traj_z[-1],
            color=trajectory_color,
            edgecolor="black",
            linewidth=0.6,
            s=45,
            depthshade=False,
            label=r"$\xi(t_f)$",
        )

        axis_limit = major_radius + minor_radius + trajectory_lift
        ax_torus.set_xlim(-axis_limit, axis_limit)
        ax_torus.set_ylim(-axis_limit, axis_limit)
        ax_torus.set_zlim(
            -(minor_radius + trajectory_lift),
            minor_radius + trajectory_lift,
        )
        ax_torus.set_box_aspect(
            (1, 1, (minor_radius + trajectory_lift) / axis_limit),
            zoom=1.2,
        )
        self._apply_dark_3d_axis_style(ax_torus, theme)
        self._hide_3d_axis_labels_and_ticks(ax_torus)
        ax_torus.view_init(elev=elev, azim=azim)
        self._apply_dark_legend_style(ax_torus.legend(loc="upper left"), theme)

        mappable = cm.ScalarMappable(norm=norm, cmap=torus_cmap)
        mappable.set_array(cost)
        colorbar = fig.colorbar(
            mappable,
            ax=ax_torus,
            fraction=0.032,
            pad=0.08,
            shrink=0.92,
        )
        colorbar.set_label(r"$J(\xi)$", rotation=270, labelpad=18)
        self._apply_dark_colorbar_style(colorbar, theme)

        state_phases = np.angle(np.exp(1j * xi))
        for oscillator_index, ax_state in enumerate((ax_state_1, ax_state_2)):
            self._setup_oscillator_state_axis(
                ax_state,
                oscillator_index,
                solution.t[0],
                t_end,
                theme=theme,
            )
            state_trace = self._wrapped_phase_trace(state_phases[oscillator_index])
            ax_state.plot(t_eval, state_trace, color=trajectory_color, linewidth=2.5)
            ax_state.plot(
                [t_eval[-1]],
                [state_phases[oscillator_index, -1]],
                marker="o",
                color=trajectory_color,
                markersize=5,
            )

        state_end = solution(t_end)
        mode_end = self._mode_at_time(t_end)
        alpha_end = self._alpha_for_mode(mode_end)
        direction_end = self.current_direction(state_end, mode_end)
        self._update_unit_circle_artists(
            circle_points,
            circle_arrows,
            circle_titles,
            xi[:, -1],
            direction_end,
            alpha_end,
        )

        fig.subplots_adjust(
            left=0.06,
            right=0.95,
            top=0.93,
            bottom=0.10,
            wspace=0.18,
            hspace=0.55,
        )
        return fig, ((ax_circle_1, ax_circle_2), ax_torus, (ax_state_1, ax_state_2))

    def animate_solution_3d(
        self,
        solution,
        graph_index=0,
        n_grid=120,
        n_time=2000,
        frame_step=500,
        interval=40,
        repeat_delay=1200,
        major_radius=2.0,
        minor_radius=0.7,
        trajectory_color=CHARCOAL_THEME["trajectory"],
        surface_alpha=0.55,
        trajectory_lift=0.0,
        elev=25,
        azim=-60,
        cost_gamma=None,
    ):
        """
        Animate two unit-circle oscillator states and the 3D torus trajectory.
        """
        if frame_step <= 0:
            raise ValueError("frame_step must be positive")
        if self.r != 2:
            raise ValueError("this animation is defined for r == 2")
        if not 0 <= graph_index < self.N_2:
            raise ValueError(f"graph_index must be in [0, {self.N_2 - 1}]")

        theme = CHARCOAL_THEME
        fig = plt.figure(figsize=(9.6, 7))
        fig.patch.set_facecolor(theme["figure"])
        grid_spec = fig.add_gridspec(
            3,
            4,
            width_ratios=[1.45, 1.45, 1.45, 0.2],
            height_ratios=[1.25, 1.25, 0.65],
            hspace=0.55,
            wspace=0.25,
        )
        ax_circle_1 = fig.add_subplot(grid_spec[0, 0])
        ax_circle_2 = fig.add_subplot(grid_spec[1, 0])
        ax_torus = fig.add_subplot(grid_spec[:2, 1:-1], projection="3d")
        ax_state_1 = fig.add_subplot(grid_spec[2, 1])
        ax_state_2 = fig.add_subplot(grid_spec[2, 2], sharex=ax_state_1)

        t_start = solution.t[0]
        t_end = solution.t[-1]

        circle_points, circle_arrows, circle_titles = self._setup_unit_circle_axes(
            (ax_circle_1, ax_circle_2),
            point_color=trajectory_color,
            theme=theme,
        )

        phase_max = 2 * np.pi
        xi_values = np.linspace(0.0, phase_max, n_grid)
        xi_1, xi_2 = np.meshgrid(xi_values, xi_values)
        cost = self.synchronization_cost_grid(xi_1, xi_2, graph_index)
        surface_x, surface_y, surface_z = self.embed_torus_3d(
            xi_1,
            xi_2,
            major_radius=major_radius,
            minor_radius=minor_radius,
        )

        norm = self._cost_color_norm(cost, cost_gamma)
        torus_cmap = plt.get_cmap(theme["cmap"])
        facecolors = torus_cmap(norm(cost))
        facecolors[..., -1] = surface_alpha

        ax_torus.plot_surface(
            surface_x,
            surface_y,
            surface_z,
            facecolors=facecolors,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            shade=False,
        )

        axis_limit = major_radius + minor_radius + trajectory_lift
        ax_torus.set_xlim(-axis_limit, axis_limit)
        ax_torus.set_ylim(-axis_limit, axis_limit)
        ax_torus.set_zlim(
            -(minor_radius + trajectory_lift),
            minor_radius + trajectory_lift,
        )
        ax_torus.set_box_aspect(
            (1, 1, (minor_radius + trajectory_lift) / axis_limit),
            zoom=0.95,
        )
        self._apply_dark_3d_axis_style(ax_torus, theme)
        self._hide_3d_axis_labels_and_ticks(ax_torus)
        ax_torus.view_init(elev=elev, azim=azim)

        t_eval = np.linspace(t_start, t_end, n_time)
        xi = np.mod(solution(t_eval)[:2], phase_max)
        state_phases = np.angle(np.exp(1j * xi))
        state_traces = tuple(self._wrapped_phase_trace(phase) for phase in state_phases)
        for oscillator_index, ax_state in enumerate((ax_state_1, ax_state_2)):
            self._setup_oscillator_state_axis(
                ax_state,
                oscillator_index,
                t_start,
                t_end,
                theme=theme,
            )
        traj_x, traj_y, traj_z = self.embed_torus_3d(
            xi[0],
            xi[1],
            major_radius=major_radius,
            minor_radius=minor_radius + trajectory_lift,
        )

        (trajectory_line,) = ax_torus.plot(
            [],
            [],
            [],
            color=trajectory_color,
            linewidth=3,
            label=r"$\xi(t)$",
        )
        current_point = ax_torus.scatter(
            [],
            [],
            [],
            color=trajectory_color,
            edgecolor=theme["edge"],
            linewidth=0.6,
            s=35,
            depthshade=False,
        )
        initial_point = ax_torus.scatter(
            traj_x[0],
            traj_y[0],
            traj_z[0],
            color=theme["initial"],
            edgecolor=theme["edge"],
            linewidth=0.6,
            s=45,
            depthshade=False,
            label=r"$\xi(0)$",
        )
        state_lines = []
        state_points = []
        for ax_state in (ax_state_1, ax_state_2):
            (state_line,) = ax_state.plot([], [], color=trajectory_color, linewidth=2.5)
            (state_point,) = ax_state.plot(
                [], [], marker="o", color=trajectory_color, markersize=5
            )
            state_lines.append(state_line)
            state_points.append(state_point)
        self._apply_dark_legend_style(ax_torus.legend(loc="upper left"), theme)

        mappable = cm.ScalarMappable(norm=norm, cmap=torus_cmap)
        mappable.set_array(cost)
        colorbar = fig.colorbar(
            mappable,
            ax=ax_torus,
            fraction=0.032,
            pad=0.08,
            shrink=0.92,
        )

        colorbar.set_label(r"Cost: $J(\xi)$", rotation=270, labelpad=18)
        self._apply_dark_colorbar_style(colorbar, theme)
        fig.subplots_adjust(
            left=0.06,
            right=0.95,
            top=0.93,
            bottom=0.10,
            wspace=0.18,
            hspace=0.55,
        )

        frame_indices = np.linspace(0, n_time - 1, frame_step, dtype=int)
        if frame_indices[-1] != n_time - 1:
            frame_indices = np.append(frame_indices, n_time - 1)

        def update(frame_idx):
            idx = frame_indices[frame_idx] + 1

            trajectory_line.set_data(traj_x[:idx], traj_y[:idx])
            trajectory_line.set_3d_properties(traj_z[:idx])
            current_point._offsets3d = (
                [traj_x[idx - 1]],
                [traj_y[idx - 1]],
                [traj_z[idx - 1]],
            )
            for oscillator_index, (state_line, state_point) in enumerate(
                zip(state_lines, state_points)
            ):
                state_line.set_data(t_eval[:idx], state_traces[oscillator_index][:idx])
                state_point.set_data(
                    [t_eval[idx - 1]],
                    [state_phases[oscillator_index, idx - 1]],
                )
            t_current = t_eval[idx - 1]
            mode = self._mode_at_time(t_current)
            state = solution(t_current)
            alpha = self._alpha_for_mode(mode)
            direction = self.current_direction(state, mode)

            self._update_unit_circle_artists(
                circle_points,
                circle_arrows,
                circle_titles,
                xi[:, idx - 1],
                direction,
                alpha,
            )

            return (
                *circle_points,
                *circle_arrows,
                trajectory_line,
                current_point,
                initial_point,
                *state_lines,
                *state_points,
            )

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


Oscillator_Synchronization = OscillatorSynchronization
