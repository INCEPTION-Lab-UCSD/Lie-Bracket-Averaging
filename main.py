import numpy as np

import vehicle_trajectories


def main():
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
        (14.5, 1.0),
    ]
    x_0_p_converge = np.array([-4, 4])
    x_0_p_diverge = np.array([-4, -4])
    x_p_goal = np.array([0.0, 0.0])

    vehicle_trajectory_simulation_converge = (
        vehicle_trajectories.VehicleTrajectorySimulation(
            x_0_p_converge, x_p_goal, epsilon, 0, 15, mode_schedule=mode_schedule
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

    vehicle_trajectory_simulation_converge.plot_trajectory(
        [hybrid_converge_solution, hybrid_diverge_solution], padding=2.0
    )


if __name__ == "__main__":
    main()
