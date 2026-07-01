import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import hybrid_solution

# Global Variables
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])


class Global_ES_Sphere:
    def __init__(self, x0, delta, omega, alpha, kappa, epsilon, t_1, t_2):
        self.x0 = np.asarray(x0, dtype=float)
        self.delta = delta
        self.omega = np.asarray(omega, dtype=float)
        self.alpha = alpha
        self.kappa = kappa
        self.epsilon = epsilon
        self.t_1 = t_1
        self.t_2 = t_2

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

        k_q = {1: 0.5, 2: -0.5}

        if J <= self.alpha:
            return x_p
        return np.dot(
            expm(
                k_q[int(q)]
                * (J - 1) ** 2
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
        y0 = self.x0.copy() if x is None else np.asarray(x, dtype=float).copy()
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
        tau_2 = t / self.epsilon
        return (
            1
            / self.epsilon
            * np.sqrt(2 * self.omega)
            * np.cos(self.omega * tau_2 + self.kappa * (q - 2) * J)
            * np.pow(self.kappa, -0.5)
        )

    def plot_sphere_simulation(self, simulation):

        pass
