# Lie-Bracket Averaging

Python simulations and visualizations for examples from "On Lie-Bracket
Averaging for Hybrid Dynamical Systems With Applications to Model-Free Control
and Optimization."

## Setup

Install the project dependencies:

```bash
uv sync
```

Run all commands from the repository root. The examples use Matplotlib for
plots and animations, so a Python environment with GUI support is required for
interactive windows.

## Running simulations and viewing plots

The simulation entry points are defined in `main.py`. Run the default example:

```bash
uv run python main.py
```

By default, this runs `run_oscillator_synchronization_simulation()` and opens
the animated plot with `plt.show()`. Close the plot window to return to the
terminal.

To run a different example, edit the `if __name__ == "__main__":` block in
`main.py` and uncomment the function you want to run:

```python
if __name__ == "__main__":
    run_vehicle_trajectory_simulation()
    # run_oscillator_synchronization_simulation()
    # run_oscillator_synchronization_simulation_multi_graph()
    # run_global_es_sphere()
```

Available examples:

- `run_vehicle_trajectory_simulation()` solves two vehicle trajectory examples,
  opens the Matplotlib animation, and saves an MP4 to
  `Animations/vehicle_trajectory.mp4`.
- `run_oscillator_synchronization_simulation()` runs the two-oscillator
  synchronization example and opens the 3D animation.
- `run_oscillator_synchronization_simulation_multi_graph()` runs the
  multi-graph oscillator example and opens the Cartesian component animation.
- `run_global_es_sphere()` runs the global extremum-seeking example on the
  sphere and opens the animation.

Some modules also expose plotting helpers such as `plot_trajectory()`,
`plot_solution()`, `plot_solution_3d()`, and `plot_sphere_simulation()`. These
can be called from a Python session or by swapping the corresponding
`animate_*()` call in `main.py` for the `plot_*()` call and then running
`uv run python main.py`.

## MuJoCo vehicle visualization

The vehicle trajectory kinematics are still solved by `VehicleTrajectorySimulation`.
MuJoCo is used only as a visualization/playback layer for the resulting hybrid
solutions. The MuJoCo view shows both vehicle trajectories from
`run_vehicle_trajectory_simulation()` in `main.py`.

Open the interactive MuJoCo viewer:

```bash
./run_vehicle_mujoco.sh
```

Pass playback options through the wrapper:

```bash
./run_vehicle_mujoco.sh --fps 60 --realtime-factor 2
```

Render a non-interactive PNG snapshot:

```bash
uv run python vehicle_mujoco.py --snapshot vehicle_mujoco_snapshot.png
```

Render a 20-second MP4 animation:

```bash
uv run python vehicle_mujoco.py --video Animations/vehicle_mujoco_20s.mp4 --video-duration 20
```

Use `.gif` instead of `.mp4` in the `--video` path to render a GIF. Snapshots
can be taken at a specific simulation time with `--snapshot-time`.

The rendered snapshot and animation overlays show:

- A top legend for `Reversed Measurements`, `Blind to Measurements`, and
  `Normal Measurements`.
- A bottom timeline for the `Converging` and `Diverging` trajectories, with the
  current-time marker showing when each measurement condition is active.
- A right-side level-curve legend for the cost `J(x_p)`.

## Citation

- M. Abdelgalil, J. Poveda [**On Lie-Bracket Averaging for Hybrid Dynamical Systems With Applications to Model-Free Control and Optimization**](https://ieeexplore.ieee.org/document/10839324). IEEE Transactions on Automatic Control, 2025.

```bibtex
@article{10839324,
  title={On Lie-Bracket Averaging for Hybrid Dynamical Systems With Applications to Model-Free Control and Optimization},
  author={Abdelgalil, Mahmoud and Poveda, Jorge I.},
  journal={IEEE Transactions on Automatic Control},
  year={2025},
  volume={70},
  number={7},
  pages={4655-4670},
  publisher={IEEE}
}
```

## License

This project is distributed under the [MIT License](LICENSE).
