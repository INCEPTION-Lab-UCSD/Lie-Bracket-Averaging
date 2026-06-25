import math
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import hybrid_solution


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

        if mode_schedule is None:
            self.mode_schedule = self.generate_random_mode_schedule()
        else:
            self.mode_schedule = mode_schedule

        # Full continuous state: [ξ_1, ..., ξ_r, τ_1, τ_2, z_1]
        # ξ_i ∈ [0, 2π) are the polar angles; τ_1 is the dwell-time timer
        # (reset to 0 at each mode switch); τ_2 is the probing timer (never reset)
        self.state_0 = np.append(self.xi0, [tau_1, tau_2, self.mode_schedule[0][1]])

        if graphs is None:
            self.graphs = [self.generate_default_graph()]
        else:
            self.graphs = graphs

        # N_2 is the number of graphs
        self.N_2 = len(self.graphs)

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

    def bijection(self, alpha_index, tau):
        """
        Bijection σ: {0, ..., N1-1} → J ⊆ {+1, -1}^r.
        Maps an integer index to the corresponding α ∈ J.
        """
        return np.array(tau[alpha_index], dtype=float)

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
        tau_2 = state[-2]  # probing timer

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

    def dynamics(self, t, state, graph_index, alpha, omega=None):
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

        xi_dot = 1.0 + alpha * u  # shape (r,), eq. 25
        tau_1_dot = 1 / self.epsilon
        tau_2_dot = 1 / self.epsilon**2

        return np.append(xi_dot, [tau_1_dot, tau_2_dot])

    def solve(self, t=None, rtol=1e-6, atol=1e-8):
        t_end = self.t_2 if t is None else t
        boundaries = [time for time, _ in self.mode_schedule]

        boundaries.append(t_end)
        modes = [mode for _, mode in self.mode_schedule][: len(boundaries) - 1]

        segment_solutions = []
        state_0 = self.state_0

        for i in range(len(boundaries) - 1):
            seg_t1, seg_t2 = boundaries[i], boundaries[i + 1]
            mode = modes[i]
            state_0[-1] = mode

            sol = solve_ivp(
                fun=self.dynamics,
                t_span=(seg_t1, seg_t2),
                y0=state_0,
                rtol=rtol,
                atol=atol,
                dense_output=True,
                args=(),
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
    # Mode schedule
    # ------------------------------------------------------------------
    def generate_random_mode_schedule(
        self,
        eta_1=1.0,
        eta_2=None,
        N_0=2,
        T_0=1.0,
        seed=None,
    ):
        z_1 = np.random.randint(1, int(self.N_1 * self.N_2))
        pass
