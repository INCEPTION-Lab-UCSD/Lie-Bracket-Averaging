from __future__ import annotations

import argparse
import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "poveda_meeting_animations"
LOCAL_MODULES = {
    "global_es_sphere",
    "hybrid_solution",
    "oscillator_synchronization",
    "vehicle_trajectories",
}


@contextmanager
def code_root_modules(code_root: Path):
    code_root = code_root.resolve()
    old_path = list(sys.path)
    old_modules = {
        name: sys.modules[name] for name in LOCAL_MODULES if name in sys.modules
    }

    for name in LOCAL_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(code_root))

    try:
        yield {
            "sphere": importlib.import_module("global_es_sphere"),
            "oscillator": importlib.import_module("oscillator_synchronization"),
            "vehicle": importlib.import_module("vehicle_trajectories"),
        }
    finally:
        for name in LOCAL_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(old_modules)
        sys.path = old_path


def ensure_ffmpeg():
    if animation.writers.is_available("ffmpeg"):
        return

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib cannot find ffmpeg and imageio-ffmpeg is not installed."
        ) from exc

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def save_animation(ani, output_path: Path, *, fps=24, dpi=120):
    ensure_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2400,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    ani.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(ani._fig)
    print(output_path)


def build_vehicle_animation(
    vehicle_module,
    output_dir: Path,
    output_name="main_vehicle_trajectory.mp4",
):
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

    x_p_goal = np.array([0.0, 0.0])
    sim_converge = vehicle_module.VehicleTrajectorySimulation(
        np.array([-4, 4]),
        x_p_goal,
        epsilon,
        t_1,
        t_2,
        mode_schedule=mode_schedule,
    )
    sim_diverge = vehicle_module.VehicleTrajectorySimulation(
        np.array([-4, -4]),
        x_p_goal,
        epsilon,
        t_1,
        t_2,
        mode_schedule=mode_schedule_diverge,
    )
    solution_converge = sim_converge.solve()
    solution_diverge = sim_diverge.solve()
    ani = sim_converge.animate_solution(
        [solution_converge, solution_diverge],
        frame_step=180,
        interval=40,
    )
    save_animation(ani, output_dir / output_name)


def build_two_oscillator(oscillator_module):
    np.random.seed(7)
    return oscillator_module.Oscillator_Synchronization(
        2,
        1 / np.sqrt(10 * np.pi),
        10,
        np.array([1, 2]),
        0.0,
        20.0,
        mode_schedule_config={"eta_1": 2.5, "N_0": 1},
    )


def build_multi_graph_oscillator(oscillator_module):
    np.random.seed(11)
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
    tau = [(1, 1, -1, 1), (-1, 1, 1, 1), (-1, 1, -1, -1), (-1, -1, 1, 1)]
    return oscillator_module.Oscillator_Synchronization(
        4,
        1 / np.sqrt(10 * np.pi),
        10,
        np.array([1, 4 / 3, 5 / 3, 2]),
        0.0,
        10.0,
        graphs=graphs,
        tau=tau,
        mode_schedule_config={"eta_1": 1.5, "N_0": 1},
    )


def build_main_oscillator_animations(oscillator_module, output_dir: Path):
    oscillator = build_two_oscillator(oscillator_module)
    solution = oscillator.solve()

    ani = oscillator.animate_solution_3d(
        solution,
        n_grid=90,
        n_time=900,
        frame_step=180,
        interval=40,
    )
    save_animation(ani, output_dir / "main_oscillator_torus_3d.mp4")

    oscillator = build_two_oscillator(oscillator_module)
    hide_blue_current_direction_arrows(oscillator)
    solution = oscillator.solve()
    ani = oscillator.animate_solution_3d(
        solution,
        n_grid=90,
        n_time=900,
        frame_step=180,
        interval=40,
    )
    save_animation(ani, output_dir / "main_oscillator_torus_3d_without_blue_arrow.mp4")

    multi_graph = build_multi_graph_oscillator(oscillator_module)
    multi_graph_solution = multi_graph.solve()
    ani = multi_graph.animate_cartesian_components(
        multi_graph_solution,
        n_time=900,
        frame_step=180,
        interval=40,
    )
    save_animation(ani, output_dir / "main_oscillator_cartesian_components.mp4")


def build_sphere_animation(sphere_module, output_dir: Path):
    x0 = np.array([-0.11, 0.11, -0.98, 2.0, 0.0])
    sphere = sphere_module.Global_ES_Sphere(
        x0,
        1 / 5,
        np.array([2, 3, 1]),
        1,
        4,
        1 / np.sqrt(8 * np.pi),
        0.0,
        15.0,
    )
    solution = sphere.solve(x0, 0.0)
    ani = sphere.animate_solution(
        solution,
        np.array([0.0, 0.0, 1.0]),
        n_grid=80,
        n_time=650,
        frame_step=180,
        interval=40,
    )
    save_animation(ani, output_dir / "main_global_es_sphere.mp4")


def hide_blue_current_direction_arrows(oscillator):
    setup_unit_circle_axes = oscillator._setup_unit_circle_axes

    def setup_without_arrows(axes, point_color="red"):
        points, arrows, titles = setup_unit_circle_axes(axes, point_color=point_color)
        for arrow in arrows:
            arrow.set_visible(False)
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None:
                legend.set_visible(False)
        return points, arrows, titles

    def update_without_arrows(points, arrows, titles, xi_values, direction, alpha):
        cartesian = oscillator.xi_to_cartesian(xi_values)
        for i, point in enumerate(points):
            point.set_offsets([cartesian[i]])
            arrows[i].set_offsets([cartesian[i]])
            arrows[i].set_UVC([0.0], [0.0])
            arrows[i].set_visible(False)
            titles[i].set_text(rf"Oscillator {i + 1}")

    oscillator._setup_unit_circle_axes = setup_without_arrows
    oscillator._update_unit_circle_artists = update_without_arrows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render meeting-ready MP4 animations from the main codebase."
    )
    parser.add_argument("--main-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--only",
        choices=("all", "vehicle-extras"),
        default="all",
        help="Use vehicle-extras to render only the additional vehicle MP4.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.only == "all":
        with code_root_modules(args.main_root) as modules:
            build_vehicle_animation(modules["vehicle"], output_dir)
            build_main_oscillator_animations(modules["oscillator"], output_dir)
            build_sphere_animation(modules["sphere"], output_dir)

    with code_root_modules(args.main_root) as modules:
        build_vehicle_animation(
            modules["vehicle"],
            output_dir,
            "vehicle_branch_mode_colored_rover.mp4",
        )


if __name__ == "__main__":
    main()
