import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# Global Variables
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])


class Global_ES_Sphere:
    def __init__(self, x0, delta, omega, alpha, kappa, epsilon, t_1, t_2):
        self.x0 = x0
        self.delta = delta
        self.omega = omega
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
        J1 = self.cost(
            self.diffeomorphism(
                x,
                1,
            )
        )
        J2 = self.cost(
            self.diffeomorphism(
                x,
                2,
            )
        )

        J = np.array([J1, J2])
        return np.min(J)

    def b(self, x):
        I = np.eye(3)
        return np.cross(I, x)

    def diffeomorphism(self, x, q):
        J = self.cost(x)

        k_q = {1: 0.5, 2: -0.5}

        if J <= self.alpha:
            return x
        return (
            expm(
                k_q[q]
                * (J - 1) ** 2
                * (self.skew_symmetric(e1) + self.skew_symmetric(e2))
            )
            * x
        )

    def jump_map(self, x):
        J1 = self.cost(self.diffeomorphism(x[0:3], 1))
        J2 = self.cost(self.diffeomorphism(x[0:3], 2))

        J = np.array([J1, J2])

        # x[3] = np.where(J == np.min(J)) + 1
        x[4] = x[4] + 1
        return x

    def solve(self, x, t=None, rtol=1e-6, atol=1e-8):
        # set starting time as t_1 if not otherwise specified
        if t == None:
            t = self.t_1
        while t < self.t_2:
            sol = solve_ivp(
                fun=self.flow_map,
                t_span=(t, self.t_2),
                y0=self.x0,
                method="RK45",
                rtol=rtol,
                atol=atol,
                dense_output=True,
                events=self.jump_condition,
                terminal=True,
            )
            if sol.success:
                x[3] = self.jump_map(x)
                t = sol.t[-1]

    def jump_condition(self, t, x):
        Jq = self.cost(self.diffeomorphism(x[0:3], x[3]))
        J_min = self.minimize(x)
        return Jq - J_min >= self.delta

    def flow_map(self, t, x):
        x_p = x[0:3]
        q = x[3]
        u = self.feedback_controller(t, x_p, q)
        dxdt = np.sum(self.b(x) * u)

        return dxdt

    def cost(self, x):
        return 1 - np.dot(x, e3)

    def feedback_controller(self, t, x, q):
        J = self.cost(self.diffeomorphism(x, q))
        tau_2 = t / self.epsilon
        return (
            1
            / self.epsilon
            * np.sqrt(2 * self.omega)
            * np.cos(self.omega * tau_2 + self.kappa * (q - 2) * J)
            * np.pow(self.kappa, -0.5)
        )
