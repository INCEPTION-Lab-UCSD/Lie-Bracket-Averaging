# Lie Bracket Averaging

## MuJoCo vehicle visualization

The vehicle trajectory kinematics are still solved by `VehicleTrajectorySimulation`.
MuJoCo is used only as a visualization/playback layer for the resulting hybrid
solutions. The MuJoCo view shows both vehicle trajectories from
`run_vehicle_trajectory_simulation()` in `main.py`.

Install the project dependencies:

```bash
uv sync
```

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

Mode labels shown in the viewer:

```text
Mode 1: Reversed Measurements (red)
Mode 2: Blind to Measurements (orange)
Mode 3: Normal Measurements (blue)
```
