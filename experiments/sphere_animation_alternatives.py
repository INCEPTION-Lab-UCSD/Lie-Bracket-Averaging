from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from global_es_sphere import Global_ES_Sphere, e3


@dataclass(frozen=True)
class SphereScenario:
    name: str
    alpha: float
    delta: float
    epsilon: float
    omega: np.ndarray
    kappa: float
    x0: np.ndarray
    t_1: float
    t_2: float


def default_scenario() -> SphereScenario:
    return SphereScenario(
        name="default",
        alpha=1.0,
        delta=1 / 5,
        epsilon=1 / np.sqrt(8 * np.pi),
        omega=np.array([2.0, 3.0, 1.0]),
        kappa=4.0,
        x0=np.array([-0.11, 0.11, -0.98, 2.0, 0.0]),
        t_1=0.0,
        t_2=15.0,
    )


def jump_scenario() -> SphereScenario:
    return SphereScenario(
        name="jump-demo",
        alpha=1.0,
        delta=1 / 8,
        epsilon=1 / np.sqrt(8 * np.pi),
        omega=np.array([2.0, 3.0, 1.0]),
        kappa=4.0,
        x0=np.array([-0.286, 0.286, -0.914, 2.0, 0.0]),
        t_1=0.0,
        t_2=12.0,
    )


def solve_scenario(scenario: SphereScenario):
    simulation = Global_ES_Sphere(
        scenario.x0,
        scenario.delta,
        scenario.omega,
        scenario.alpha,
        scenario.kappa,
        scenario.epsilon,
        scenario.t_1,
        scenario.t_2,
    )
    solution = simulation.solve(scenario.x0, scenario.t_1)
    return simulation, solution


def sample_solution(simulation, solution, n_time=600, radius=1.0):
    times = np.linspace(solution.t[0], solution.t[-1], n_time)
    states = solution(times)
    positions = states[:3]
    positions = radius * positions / np.linalg.norm(positions, axis=0)
    modes = np.round(states[3]).astype(int)
    jumps = states[4]
    cost = 1.0 - positions[2] / radius

    j_modes = np.empty((2, n_time))
    for i, x in enumerate(positions.T):
        j_modes[0, i] = simulation.cost(simulation.diffeomorphism(x, 1))
        j_modes[1, i] = simulation.cost(simulation.diffeomorphism(x, 2))

    active_cost = j_modes[modes - 1, np.arange(n_time)]
    min_cost = np.min(j_modes, axis=0)
    jump_margin = active_cost - min_cost - simulation.delta

    return {
        "times": times,
        "states": states,
        "positions": positions,
        "modes": modes,
        "jumps": jumps,
        "cost": cost,
        "j_modes": j_modes,
        "active_cost": active_cost,
        "min_cost": min_cost,
        "jump_margin": jump_margin,
        "switch_times": list(solution.switch_times),
    }


def sphere_mesh(radius=1.0, n_grid=72):
    u = np.linspace(0.0, 2 * np.pi, n_grid)
    v = np.linspace(0.0, np.pi, n_grid)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    cost = 1.0 - z / radius
    return x, y, z, cost


def configure_3d_axis(ax, radius=1.0, elev=16, azim=-48):
    limit = 1.35 * radius
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")
    ax.grid(True, alpha=0.25)


def add_cost_sphere(ax, style, radius=1.0, n_grid=72):
    x, y, z, cost = sphere_mesh(radius, n_grid)
    norm = Normalize(vmin=0.0, vmax=2.0)

    if style == "paper":
        cmap = cm.gray
        alpha = 1.0
        linewidth = 0.18
    elif style == "high_contrast":
        cmap = cm.cividis
        alpha = 0.88
        linewidth = 0.12
    else:
        cmap = cm.viridis
        alpha = 0.72
        linewidth = 0.08

    facecolors = cmap(norm(cost))
    facecolors[..., -1] = alpha
    surface = ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        linewidth=linewidth,
        antialiased=False,
        shade=False,
        zorder=1,
    )

    if style == "transparent":
        ax.plot_wireframe(
            x,
            y,
            z,
            rstride=6,
            cstride=6,
            color="0.15",
            linewidth=0.35,
            alpha=0.22,
            zorder=2,
        )

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(cost)
    return surface, mappable


def add_visible_markers(ax, data, target=e3, radius=1.0, target_color="#f2c230"):
    marker_radius = 1.08 * radius
    x0 = marker_radius * data["positions"][:, 0] / radius
    target = np.asarray(target, dtype=float)
    target = marker_radius * target / np.linalg.norm(target)

    ax.scatter(
        [x0[0]],
        [x0[1]],
        [x0[2]],
        marker="o",
        s=250,
        color="white",
        edgecolors="black",
        linewidths=1.6,
        depthshade=False,
        zorder=12,
        label=r"$x(0)$",
    )
    ax.scatter(
        [target[0]],
        [target[1]],
        [target[2]],
        marker="*",
        s=420,
        color=target_color,
        edgecolors="black",
        linewidths=1.1,
        depthshade=False,
        zorder=13,
        label=r"$x^*$",
    )
    ax.text(x0[0], x0[1], x0[2] - 0.12, "start", color="black", ha="center")
    ax.text(target[0], target[1], target[2] + 0.10, "target", color="black", ha="center")


def build_base_sphere_figure(data, style="paper", with_colorbar=True, figsize=(8, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    _, mappable = add_cost_sphere(ax, style)
    add_visible_markers(ax, data)
    configure_3d_axis(ax)
    if with_colorbar:
        colorbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(r"$J(x)$", rotation=270, labelpad=18)
    return fig, ax


def add_animation_artists(ax, data, line_color="red"):
    trajectory = 1.025 * data["positions"]
    (line,) = ax.plot3D(
        [],
        [],
        [],
        color=line_color,
        linewidth=2.5,
        zorder=10,
        label=r"$x(t)$",
    )
    point = ax.scatter(
        [],
        [],
        [],
        marker="o",
        s=80,
        color=line_color,
        edgecolors="white",
        linewidths=0.8,
        depthshade=False,
        zorder=14,
    )
    ax.legend(loc="upper left", framealpha=0.92)
    return trajectory, line, point


def frame_indices(n_time, frame_step):
    indices = np.linspace(0, n_time - 1, frame_step, dtype=int)
    if indices[-1] != n_time - 1:
        indices = np.append(indices, n_time - 1)
    return indices


def animate_sphere_only(data, style, save_path=None, frame_step=160, fps=24):
    fig, ax = build_base_sphere_figure(data, style=style)
    line_color = "#d7191c" if style == "paper" else "#ff3b30"
    trajectory, line, point = add_animation_artists(ax, data, line_color=line_color)
    indices = frame_indices(trajectory.shape[1], frame_step)

    def update(frame):
        idx = indices[frame] + 1
        line.set_data(trajectory[0, :idx], trajectory[1, :idx])
        line.set_3d_properties(trajectory[2, :idx])
        point._offsets3d = (
            [trajectory[0, idx - 1]],
            [trajectory[1, idx - 1]],
            [trajectory[2, idx - 1]],
        )
        return line, point

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(indices),
        interval=1000 / fps,
        repeat=True,
        blit=False,
    )
    update(0)
    save_animation(ani, save_path, fps)
    return ani


def animate_cost_dashboard(data, save_path=None, frame_step=160, fps=24):
    fig = plt.figure(figsize=(11.5, 6.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.25)
    ax_sphere = fig.add_subplot(grid[0, 0], projection="3d")
    ax_cost = fig.add_subplot(grid[0, 1])

    _, mappable = add_cost_sphere(ax_sphere, "high_contrast")
    add_visible_markers(ax_sphere, data)
    configure_3d_axis(ax_sphere, elev=18, azim=-48)
    fig.colorbar(mappable, ax=ax_sphere, fraction=0.046, pad=0.03).set_label(
        r"$J(x)$", rotation=270, labelpad=15
    )
    trajectory, line, point = add_animation_artists(
        ax_sphere, data, line_color="#f03b20"
    )

    times = data["times"]
    cost = data["cost"]
    ax_cost.plot(times, np.full_like(times, cost[0]), color="#2c7bb6", linewidth=1.8, label=r"$J(x(0))$")
    ax_cost.plot(times, cost, color="0.78", linewidth=1.2, label=r"$J(x(t))$ full")
    (cost_line,) = ax_cost.plot([], [], color="#d7191c", linewidth=2.2, label=r"$J(x(t))$")
    cost_dot = ax_cost.scatter([], [], s=50, color="#d7191c", zorder=5)
    cursor = ax_cost.axvline(times[0], color="0.2", linewidth=0.9, alpha=0.65)
    ax_cost.set_xlim(times[0], times[-1])
    ax_cost.set_ylim(-0.05, max(2.05, 1.05 * np.max(cost)))
    ax_cost.set_xlabel(r"$t$")
    ax_cost.set_ylabel(r"$J(x)$")
    ax_cost.grid(True, alpha=0.28)
    ax_cost.legend(loc="upper right", framealpha=0.9)

    indices = frame_indices(trajectory.shape[1], frame_step)

    def update(frame):
        idx = indices[frame] + 1
        line.set_data(trajectory[0, :idx], trajectory[1, :idx])
        line.set_3d_properties(trajectory[2, :idx])
        point._offsets3d = (
            [trajectory[0, idx - 1]],
            [trajectory[1, idx - 1]],
            [trajectory[2, idx - 1]],
        )
        cost_line.set_data(times[:idx], cost[:idx])
        cost_dot.set_offsets([[times[idx - 1], cost[idx - 1]]])
        cursor.set_xdata([times[idx - 1], times[idx - 1]])
        return line, point, cost_line, cost_dot, cursor

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(indices),
        interval=1000 / fps,
        repeat=True,
        blit=False,
    )
    update(0)
    save_animation(ani, save_path, fps)
    return ani


def animate_jump_dashboard(data, save_path=None, frame_step=180, fps=24):
    fig = plt.figure(figsize=(12.5, 7.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.35, wspace=0.25)
    ax_sphere = fig.add_subplot(grid[:, 0], projection="3d")
    ax_cost = fig.add_subplot(grid[0, 1])
    ax_margin = fig.add_subplot(grid[1, 1])
    ax_mode = ax_margin.twinx()

    _, mappable = add_cost_sphere(ax_sphere, "transparent")
    add_visible_markers(ax_sphere, data)
    configure_3d_axis(ax_sphere, elev=18, azim=-50)
    fig.colorbar(mappable, ax=ax_sphere, fraction=0.046, pad=0.03).set_label(
        r"$J(x)$", rotation=270, labelpad=15
    )
    trajectory, line, point = add_animation_artists(
        ax_sphere, data, line_color="#e31a1c"
    )

    times = data["times"]
    cost = data["cost"]
    j1, j2 = data["j_modes"]
    margin = data["jump_margin"]
    modes = data["modes"]

    ax_cost.plot(times, j1, color="#5e81ff", linewidth=1.1, alpha=0.65, label=r"$\tilde J_1$")
    ax_cost.plot(times, j2, color="#ff9d3a", linewidth=1.1, alpha=0.65, label=r"$\tilde J_2$")
    ax_cost.plot(times, cost, color="0.82", linewidth=1.0, label=r"$J(x)$ full")
    (cost_line,) = ax_cost.plot([], [], color="#d7191c", linewidth=2.0, label=r"$J(x)$")
    cost_dot = ax_cost.scatter([], [], s=45, color="#d7191c", zorder=6)
    cost_cursor = ax_cost.axvline(times[0], color="0.2", linewidth=0.9, alpha=0.65)
    for switch_time in data["switch_times"]:
        ax_cost.axvline(switch_time, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_cost.set_xlim(times[0], times[-1])
    ax_cost.set_ylim(-0.05, max(2.05, 1.05 * np.max([np.max(j1), np.max(j2), np.max(cost)])))
    ax_cost.set_ylabel("cost")
    ax_cost.grid(True, alpha=0.28)
    ax_cost.legend(loc="upper right", framealpha=0.9, fontsize=9)

    ax_margin.axhline(0.0, color="black", linewidth=1.0, alpha=0.7, label="jump threshold")
    ax_margin.plot(times, margin, color="0.75", linewidth=1.0)
    (margin_line,) = ax_margin.plot([], [], color="#e31a1c", linewidth=2.0, label="jump margin")
    margin_dot = ax_margin.scatter([], [], s=45, color="#e31a1c", zorder=6)
    margin_cursor = ax_margin.axvline(times[0], color="0.2", linewidth=0.9, alpha=0.65)
    (mode_line,) = ax_mode.plot([], [], color="#238b45", linewidth=1.7, drawstyle="steps-post", label=r"$q(t)$")
    for switch_time in data["switch_times"]:
        ax_margin.axvline(switch_time, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_margin.set_xlim(times[0], times[-1])
    pad = 0.08
    ax_margin.set_ylim(np.min(margin) - pad, max(0.05, np.max(margin) + pad))
    ax_margin.set_xlabel(r"$t$")
    ax_margin.set_ylabel(r"$\tilde J_q-\min_i \tilde J_i-\delta$")
    ax_margin.grid(True, alpha=0.28)
    ax_mode.set_ylim(0.75, 2.25)
    ax_mode.set_yticks([1, 2])
    ax_mode.set_ylabel(r"$q(t)$")
    lines = [ax_margin.lines[0], margin_line, mode_line]
    labels = ["jump threshold", "jump margin", r"$q(t)$"]
    ax_margin.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=9)

    indices = frame_indices(trajectory.shape[1], frame_step)

    def update(frame):
        idx = indices[frame] + 1
        t_now = times[idx - 1]
        line.set_data(trajectory[0, :idx], trajectory[1, :idx])
        line.set_3d_properties(trajectory[2, :idx])
        point._offsets3d = (
            [trajectory[0, idx - 1]],
            [trajectory[1, idx - 1]],
            [trajectory[2, idx - 1]],
        )
        cost_line.set_data(times[:idx], cost[:idx])
        cost_dot.set_offsets([[t_now, cost[idx - 1]]])
        cost_cursor.set_xdata([t_now, t_now])
        margin_line.set_data(times[:idx], margin[:idx])
        margin_dot.set_offsets([[t_now, margin[idx - 1]]])
        margin_cursor.set_xdata([t_now, t_now])
        mode_line.set_data(times[:idx], modes[:idx])
        return (
            line,
            point,
            cost_line,
            cost_dot,
            cost_cursor,
            margin_line,
            margin_dot,
            margin_cursor,
            mode_line,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(indices),
        interval=1000 / fps,
        repeat=True,
        blit=False,
    )
    update(0)
    save_animation(ani, save_path, fps)
    return ani


def add_full_trajectory(ax, data, line_color="#d7191c"):
    trajectory = 1.025 * data["positions"]
    ax.plot3D(
        trajectory[0],
        trajectory[1],
        trajectory[2],
        color=line_color,
        linewidth=2.5,
        zorder=10,
        label=r"$x(t)$",
    )
    ax.scatter(
        [trajectory[0, -1]],
        [trajectory[1, -1]],
        [trajectory[2, -1]],
        marker="o",
        s=95,
        color=line_color,
        edgecolors="white",
        linewidths=0.9,
        depthshade=False,
        zorder=14,
        label=r"$x(t_f)$",
    )
    ax.legend(loc="upper left", framealpha=0.92)


def save_sphere_png(data, style, save_path):
    fig, ax = build_base_sphere_figure(data, style=style)
    line_color = "#d7191c" if style == "paper" else "#ff3b30"
    add_full_trajectory(ax, data, line_color=line_color)
    save_png(fig, save_path)
    plt.close(fig)


def save_cost_dashboard_png(data, save_path):
    fig = plt.figure(figsize=(11.5, 6.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.25)
    ax_sphere = fig.add_subplot(grid[0, 0], projection="3d")
    ax_cost = fig.add_subplot(grid[0, 1])

    _, mappable = add_cost_sphere(ax_sphere, "high_contrast")
    add_visible_markers(ax_sphere, data)
    configure_3d_axis(ax_sphere, elev=18, azim=-48)
    fig.colorbar(mappable, ax=ax_sphere, fraction=0.046, pad=0.03).set_label(
        r"$J(x)$", rotation=270, labelpad=15
    )
    add_full_trajectory(ax_sphere, data, line_color="#f03b20")

    times = data["times"]
    cost = data["cost"]
    ax_cost.plot(times, np.full_like(times, cost[0]), color="#2c7bb6", linewidth=1.8, label=r"$J(x(0))$")
    ax_cost.plot(times, cost, color="#d7191c", linewidth=2.2, label=r"$J(x(t))$")
    ax_cost.scatter([times[-1]], [cost[-1]], s=55, color="#d7191c", zorder=5)
    ax_cost.axvline(times[-1], color="0.2", linewidth=0.9, alpha=0.65)
    ax_cost.set_xlim(times[0], times[-1])
    ax_cost.set_ylim(-0.05, max(2.05, 1.05 * np.max(cost)))
    ax_cost.set_xlabel(r"$t$")
    ax_cost.set_ylabel(r"$J(x)$")
    ax_cost.grid(True, alpha=0.28)
    ax_cost.legend(loc="upper right", framealpha=0.9)

    save_png(fig, save_path)
    plt.close(fig)


def save_jump_dashboard_png(data, save_path):
    fig = plt.figure(figsize=(12.5, 7.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.35, wspace=0.25)
    ax_sphere = fig.add_subplot(grid[:, 0], projection="3d")
    ax_cost = fig.add_subplot(grid[0, 1])
    ax_margin = fig.add_subplot(grid[1, 1])
    ax_mode = ax_margin.twinx()

    _, mappable = add_cost_sphere(ax_sphere, "transparent")
    add_visible_markers(ax_sphere, data)
    configure_3d_axis(ax_sphere, elev=18, azim=-50)
    fig.colorbar(mappable, ax=ax_sphere, fraction=0.046, pad=0.03).set_label(
        r"$J(x)$", rotation=270, labelpad=15
    )
    add_full_trajectory(ax_sphere, data, line_color="#e31a1c")

    times = data["times"]
    cost = data["cost"]
    j1, j2 = data["j_modes"]
    margin = data["jump_margin"]
    modes = data["modes"]

    ax_cost.plot(times, j1, color="#5e81ff", linewidth=1.1, alpha=0.7, label=r"$\tilde J_1$")
    ax_cost.plot(times, j2, color="#ff9d3a", linewidth=1.1, alpha=0.7, label=r"$\tilde J_2$")
    ax_cost.plot(times, cost, color="#d7191c", linewidth=2.0, label=r"$J(x)$")
    for switch_time in data["switch_times"]:
        ax_cost.axvline(switch_time, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_cost.set_xlim(times[0], times[-1])
    ax_cost.set_ylim(-0.05, max(2.05, 1.05 * np.max([np.max(j1), np.max(j2), np.max(cost)])))
    ax_cost.set_ylabel("cost")
    ax_cost.grid(True, alpha=0.28)
    ax_cost.legend(loc="upper right", framealpha=0.9, fontsize=9)

    ax_margin.axhline(0.0, color="black", linewidth=1.0, alpha=0.7, label="jump threshold")
    ax_margin.plot(times, margin, color="#e31a1c", linewidth=2.0, label="jump margin")
    for switch_time in data["switch_times"]:
        ax_margin.axvline(switch_time, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_margin.set_xlim(times[0], times[-1])
    pad = 0.08
    ax_margin.set_ylim(np.min(margin) - pad, max(0.05, np.max(margin) + pad))
    ax_margin.set_xlabel(r"$t$")
    ax_margin.set_ylabel(r"$\tilde J_q-\min_i \tilde J_i-\delta$")
    ax_margin.grid(True, alpha=0.28)

    mode_line = ax_mode.plot(
        times,
        modes,
        color="#238b45",
        linewidth=1.7,
        drawstyle="steps-post",
        label=r"$q(t)$",
    )[0]
    ax_mode.set_ylim(0.75, 2.25)
    ax_mode.set_yticks([1, 2])
    ax_mode.set_ylabel(r"$q(t)$")
    lines = [ax_margin.lines[0], ax_margin.lines[1], mode_line]
    labels = ["jump threshold", "jump margin", r"$q(t)$"]
    ax_margin.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=9)

    save_png(fig, save_path)
    plt.close(fig)


def save_variant_png(variant, data, save_path):
    if variant == "paper_markers":
        save_sphere_png(data, "paper", save_path)
    elif variant == "high_contrast":
        save_sphere_png(data, "high_contrast", save_path)
    elif variant == "cost_dashboard":
        save_cost_dashboard_png(data, save_path)
    elif variant == "jump_dashboard":
        save_jump_dashboard_png(data, save_path)
    else:
        raise ValueError(f"unknown variant: {variant}")


def save_png(fig, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")


def save_animation(ani, save_path, fps):
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not animation.writers.is_available("ffmpeg"):
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(save_path, writer=writer, dpi=120)


def make_variant(variant, output_dir, preview=False):
    output_dir = Path(output_dir)
    n_grid = 34 if preview else 84
    n_time = 90 if preview else 700
    frame_step = 18 if preview else 220
    fps = 12 if preview else 30

    if variant == "jump_dashboard":
        scenario = jump_scenario()
    else:
        scenario = default_scenario()

    simulation, solution = solve_scenario(scenario)
    data = sample_solution(simulation, solution, n_time=n_time)
    save_path = output_dir / f"{variant}.mp4"
    png_path = output_dir / f"{variant}.png"

    if variant == "paper_markers":
        ani = animate_sphere_only(data, "paper", save_path, frame_step, fps)
    elif variant == "high_contrast":
        ani = animate_sphere_only(data, "high_contrast", save_path, frame_step, fps)
    elif variant == "cost_dashboard":
        ani = animate_cost_dashboard(data, save_path, frame_step, fps)
    elif variant == "jump_dashboard":
        ani = animate_jump_dashboard(data, save_path, frame_step, fps)
    else:
        raise ValueError(f"unknown variant: {variant}")

    plt.close(ani._fig)
    save_variant_png(variant, data, png_path)
    final_state = solution(scenario.t_2)
    return {
        "variant": variant,
        "path": save_path,
        "png_path": png_path,
        "scenario": scenario.name,
        "final_state": final_state,
        "switch_times": list(solution.switch_times),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        default="all",
        choices=["all", "paper_markers", "high_contrast", "cost_dashboard", "jump_dashboard"],
    )
    parser.add_argument(
        "--output-dir",
        default="figures/sphere_animation_variants",
        help="Directory for rendered MP4 previews.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render short low-resolution previews for fast iteration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    variants = (
        ["paper_markers", "high_contrast", "cost_dashboard", "jump_dashboard"]
        if args.variant == "all"
        else [args.variant]
    )
    for variant in variants:
        result = make_variant(variant, args.output_dir, preview=args.preview)
        final_state = np.array2string(result["final_state"], precision=4)
        print(
            f"{variant}: wrote {result['path']} | "
            f"png={result['png_path']} | "
            f"scenario={result['scenario']} | switches={result['switch_times']} | "
            f"final={final_state}"
        )


if __name__ == "__main__":
    main()
