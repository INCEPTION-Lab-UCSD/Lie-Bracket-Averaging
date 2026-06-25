import matplotlib.pyplot as plt
import numpy as np

import oscillator_synchronization
import vehicle_trajectories


def run_vehicle_trajectory_simulation():
    epsilon = 1 / np.sqrt(10 * np.pi)
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
            x_0_p_converge, x_p_goal, epsilon, 0.0, 15.0, mode_schedule=mode_schedule
        )
    )

    hybrid_converge_solution = vehicle_trajectory_simulation_converge.solve()

    vehicle_trajectory_simulation_diverge = (
        vehicle_trajectories.VehicleTrajectorySimulation(
            x_0_p_diverge,
            x_p_goal,
            epsilon,
            0.0,
            15.0,
            mode_schedule=mode_schedule_diverge,
        )
    )

    hybrid_diverge_solution = vehicle_trajectory_simulation_diverge.solve()

    # fig, axes = vehicle_trajectory_simulation_converge.plot_trajectory(
    #     [hybrid_converge_solution, hybrid_diverge_solution], padding=2.0
    # )

    ani = vehicle_trajectory_simulation_converge.animate_solution(
        [hybrid_converge_solution, hybrid_diverge_solution]
    )
    plt.show()


def run_oscillator_synchronization_simulation():
    r = 2
    N_2 = 1
    kappa = 10
    epsilon = 1 / np.sqrt(10 * np.pi)
    omega = np.array([1, 2])
    eta_1 = 2.5
    N_o = 1

    oscillator = oscillator_synchronization.Oscillator_Synchronization(
        r,
        epsilon,
        kappa,
        omega,
        0.0,
        10.0,
        mode_schedule_config={"eta_1": eta_1, "N_0": N_o},
    )
    solution = oscillator.solve()
    ani = oscillator.animate_solution(solution)
    plt.show()


if __name__ == "__main__":
    # run_vehicle_trajectory_simulation()
    run_oscillator_synchronization_simulation()
