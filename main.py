import math
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import global_es_sphere
import oscillator_synchronization
import vehicle_trajectories


def run_vehicle_trajectory_simulation():
    output_path = Path("Animations") / "vehicle_trajectory.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epsilon = 1 / np.sqrt(10 * np.pi)
    t_1 = 0.0
    t_2 = 15.0
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

    vehicle_trajectory_simulation_converge = (
        vehicle_trajectories.VehicleTrajectorySimulation(
            x_0_p_converge, x_p_goal, epsilon, t_1, t_2, mode_schedule=mode_schedule
        )
    )

    hybrid_converge_solution = vehicle_trajectory_simulation_converge.solve()

    vehicle_trajectory_simulation_diverge = (
        vehicle_trajectories.VehicleTrajectorySimulation(
            x_0_p_diverge,
            x_p_goal,
            epsilon,
            t_1,
            t_2,
            mode_schedule=mode_schedule_diverge,
        )
    )

    hybrid_diverge_solution = vehicle_trajectory_simulation_diverge.solve()

    # fig, axes = vehicle_trajectory_simulation_converge.plot_trajectory(
    #     [hybrid_converge_solution, hybrid_diverge_solution], padding=2.0
    # )

    ani = vehicle_trajectory_simulation_converge.animate_solution(
        [hybrid_converge_solution, hybrid_diverge_solution],
        frame_step=500,
        save_path=str(output_path),
    )
    print(f"Saved animation to {output_path}")
    plt.show()


def run_oscillator_synchronization_simulation():
    r = 2
    N_2 = 1
    kappa = 10
    epsilon = 1 / np.sqrt(10 * np.pi)
    omega = np.array([1, 2])
    eta_1 = 2.5
    N_o = 1
    t_1 = 0.0
    t_2 = 20.0

    oscillator = oscillator_synchronization.Oscillator_Synchronization(
        r,
        epsilon,
        kappa,
        omega,
        t_1,
        t_2,
        mode_schedule_config={"eta_1": eta_1, "N_0": N_o},
    )

    solution = oscillator.solve()
    # ani = oscillator.animate_solution(solution)

    ani = oscillator.animate_solution_3d(solution, frame_step=500)
    plt.show()


def run_oscillator_synchronization_simulation_multi_graph():
    r = 4
    eta_1 = 1.5
    N_o = 1
    graphs = [
        [(3, 1), (1, 3), (2, 1), (1, 2), (4, 2), (2, 4)],
        [
            (1, 2),
            (2, 1),
            (1, 4),
            (4, 1),
            (4, 3),
            (3, 4),
            (3, 2),
            (2, 3),
            (4, 2),
            (2, 4),
        ],
        [(3, 4), (4, 3), (4, 2), (2, 4), (2, 1), (1, 2)],
    ]
    kappa = 10
    epsilon = 1 / np.sqrt(10 * np.pi)
    omega = np.array([1, 4 / 3, 5 / 3, 2])
    t_1 = 0.0
    t_2 = 10.0
    tau = [(1, 1, -1, 1), (-1, 1, 1, 1), (-1, 1, -1, -1), (-1, -1, 1, 1)]

    oscillator = oscillator_synchronization.Oscillator_Synchronization(
        r,
        epsilon,
        kappa,
        omega,
        t_1,
        t_2,
        graphs=graphs,
        tau=tau,
        mode_schedule_config={"eta_1": eta_1, "N_0": N_o},
    )
    solution = oscillator.solve()
    ani = oscillator.animate_cartesian_components(solution)
    plt.show()


def run_global_es_sphere():
    alpha = 1
    delta = 1 / 5
    epsilon = 1 / np.sqrt(8 * np.pi)
    omega = np.array([2, 3, 1])
    kappa = 4
    x0 = np.array([-0.11, 0.11, -0.98])
    x0 = np.append(x0, [2, 0.0])
    x_target = np.array([0, 0, 1], dtype=float)
    t_1 = 0.0
    t_2 = 15.0
    sphere_simulation = global_es_sphere.Global_ES_Sphere(
        x0, delta, omega, alpha, kappa, epsilon, t_1, t_2
    )
    solution = sphere_simulation.solve(x0, t_1)
    # sphere_simulation.plot_sphere_simulation(solution, x_target)
    ani = sphere_simulation.animate_solution(solution, x_target)
    plt.show()
    return ani


if __name__ == "__main__":
    # run_vehicle_trajectory_simulation()
    run_oscillator_synchronization_simulation()
    # run_oscillator_synchronization_simulation_multi_graph()
    # run_global_es_sphere()
