"""Build publication figures and tables from the first paper experiment suite."""

from __future__ import annotations

import csv
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from experiments.research.common import (
    OUTPUT_ROOT,
    aligned_control_samples,
    load_pickle,
    write_json,
)
from experiments.research.paper_1_external_wrench.run import (
    CONTROL_COMPARISON_WINDOW_S,
)


DEFAULT_INPUT = OUTPUT_ROOT / "paper_1_external_wrench"
COLORS = {"ignore": "#C84A45", "estimate": "#2368A2", "oracle": "#2D7D46"}
LABELS = {
    "ignore": "без учета воздействия",
    "estimate": "EKF + MPC",
    "oracle": "известное воздействие",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def control_scheme(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.0, 3.0))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3)
    axis.axis("off")
    boxes = [
        (0.25, 1.75, 1.45, 0.62, "Визуальные и\nинерциальные данные", "#E8EEF4"),
        (2.15, 1.75, 1.25, 0.62, "Расширенный\nEKF", "#DDEAF5"),
        (3.85, 1.75, 1.35, 0.62, "Оценка состояния\nи воздействия", "#DDEAF5"),
        (5.70, 1.75, 1.15, 0.62, "MPC", "#E4F0E7"),
        (7.35, 1.75, 1.15, 0.62, "Распределение\nтяги", "#F2E9D7"),
        (8.95, 1.75, 0.80, 0.62, "Робот", "#ECECEC"),
    ]
    for x, y, w, h, label, color in boxes:
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=color, ec="#333333", lw=0.9))
        axis.text(x + w / 2, y + h / 2, label, ha="center", va="center")
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        axis.add_patch(
            FancyArrowPatch(
                (left[0] + left[2], 2.06),
                (right[0], 2.06),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=1.0,
            )
        )
    axis.add_patch(
        FancyArrowPatch(
            (9.35, 1.72),
            (2.80, 1.03),
            connectionstyle="arc3,rad=-0.17",
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.0,
        )
    )
    axis.text(6.8, 0.38, "измеряемое движение", ha="center")
    axis.add_patch(
        FancyArrowPatch(
            (4.55, 1.72),
            (6.15, 1.30),
            connectionstyle="arc3,rad=0.0",
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.0,
            color="#2368A2",
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (6.15, 1.30),
            (6.15, 1.72),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.0,
            color="#2368A2",
        )
    )
    axis.text(5.20, 1.03, "оцененное воздействие", ha="center", color="#2368A2")
    _save(fig, output, "paper1_control_scheme")


def wrench_estimation(input_root: Path, output: Path) -> None:
    snapshot_path = next(
        input_root.glob(
            "exp1_estimation/profile-step/mode-estimate/force_ratio-0.25/noise_multiplier-1.0/seed-0/snapshot.pkl"
        )
    )
    snapshot = load_pickle(snapshot_path)
    t, _real, _observed, _reference, true_wrench, estimated_wrench = (
        aligned_control_samples(snapshot)
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.3), sharex=True)
    component_colors = ("#2368A2", "#C84A45", "#2D7D46")
    for i, (color, label) in enumerate(zip(component_colors, ("x", "y", "z"), strict=True)):
        axes[0].plot(t, true_wrench[:, i], color=color, lw=1.5, label=f"$F_{label}$, заданная")
        axes[0].plot(
            t,
            estimated_wrench[:, i],
            color=color,
            lw=1.0,
            ls="--",
            label=f"$F_{label}$, EKF",
        )
        axes[1].plot(
            t,
            true_wrench[:, i + 3],
            color=color,
            lw=1.5,
            label=f"$M_{label}$, заданный",
        )
        axes[1].plot(
            t,
            estimated_wrench[:, i + 3],
            color=color,
            lw=1.0,
            ls="--",
            label=f"$M_{label}$, EKF",
        )
    for axis in axes:
        axis.axvline(12.0, color="#555555", lw=0.8, ls=":")
        axis.axvspan(45.0, 60.0, color="#A8A8A8", alpha=0.14)
        axis.legend(ncol=3, loc="upper center")
    axes[0].set_ylabel("Сила, Н")
    axes[1].set_ylabel("Момент, Н·м")
    axes[1].set_xlabel("Время, с")
    axes[1].set_xlim(0.0, 60.0)
    axes[0].set_title("Ступенчатое внешнее силомоментное воздействие")
    _save(fig, output, "paper1_wrench_estimation")
    settled = (t >= 45.0) & (t <= 60.0)
    force_error = np.linalg.norm(estimated_wrench[settled, :3] - true_wrench[settled, :3], axis=1)
    moment_error = np.linalg.norm(estimated_wrench[settled, 3:] - true_wrench[settled, 3:], axis=1)
    write_json(
        output / "paper1_wrench_estimation_values.json",
        {
            "settled_window_s": [45.0, 60.0],
            "force_rmse_n": float(np.sqrt(np.mean(force_error**2))),
            "moment_rmse_nm": float(np.sqrt(np.mean(moment_error**2))),
            "fy_slope_n_per_s": float(np.polyfit(t[settled], estimated_wrench[settled, 1], 1)[0]),
        },
    )


def control_comparison(input_root: Path, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    table = {}
    for mode in ("ignore", "estimate", "oracle"):
        path = (
            input_root
            / "exp2_control_comparison"
            / "profile-constant"
            / f"mode-{mode}"
            / "force_ratio-0.375"
            / "noise_multiplier-0.5"
            / "seed-0"
            / "snapshot.pkl"
        )
        snapshot = load_pickle(path)
        history = snapshot["control_history"]
        errors = [(t, error) for t, error in zip(history.ts, history.error, strict=False) if error is not None]
        t = np.asarray([item[0] for item in errors], dtype=float)
        position = np.asarray([np.linalg.norm(item[1].p) for item in errors], dtype=float)
        orientation = np.degrees(np.asarray([abs(float(item[1].q[0])) for item in errors], dtype=float))
        selected = (t >= CONTROL_COMPARISON_WINDOW_S[0]) & (t <= CONTROL_COMPARISON_WINDOW_S[1])
        t = t[selected]
        position = position[selected]
        orientation = orientation[selected]
        axes[0].plot(t, position, color=COLORS[mode], lw=1.2, label=LABELS[mode])
        axes[1].plot(t, orientation, color=COLORS[mode], lw=1.2, label=LABELS[mode])
        table[mode] = {
            "position_rmse_m": float(np.sqrt(np.mean(position**2))),
            "orientation_rmse_deg": float(np.sqrt(np.mean(orientation**2))),
        }
    axes[0].set_ylabel("Ошибка положения, м")
    axes[1].set_ylabel("Ошибка ориентации, град")
    axes[1].set_xlabel("Время, с")
    axes[1].set_xlim(*CONTROL_COMPARISON_WINDOW_S)
    axes[0].legend(ncol=3, loc="upper center")
    axes[0].set_title("Первый перелет при постоянном внешнем воздействии")
    _save(fig, output, "paper1_control_comparison")
    write_json(output / "paper1_control_comparison_values.json", table)


def robustness(input_root: Path, output: Path) -> None:
    with (input_root / "metrics.csv").open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["experiment"] == "exp3_robustness"]
    force_ratios = (0.125, 0.25, 0.375)
    noise_levels = (0.5, 1.0, 2.0)
    position = np.full((3, 3), np.nan)
    force = np.full((3, 3), np.nan)
    for i, noise in enumerate(noise_levels):
        for j, ratio in enumerate(force_ratios):
            subset = [
                row for row in rows if float(row["noise_multiplier"]) == noise and float(row["force_ratio"]) == ratio
            ]
            position[i, j] = np.mean([float(row["tracking_position_rmse_m"]) for row in subset])
            force[i, j] = np.mean([float(row["force_estimation_rmse_n"]) for row in subset])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)
    for axis, values, title, fmt in (
        (axes[0], position, "RMSE положения, м", ".3f"),
        (axes[1], force, "RMSE оценки силы, Н", ".3f"),
    ):
        image = axis.imshow(values, cmap="Blues", aspect="auto")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                axis.text(
                    j,
                    i,
                    format(values[i, j], fmt),
                    ha="center",
                    va="center",
                    color="#111111",
                )
        axis.set_xticks(range(3), [str(value).replace(".", ",") for value in force_ratios])
        axis.set_yticks(range(3), [str(value).replace(".", ",") for value in noise_levels])
        axis.set_xlabel("Относительная величина силы")
        axis.set_ylabel("Множитель шума")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.75)
    _save(fig, output, "paper1_robustness")


@click.command(context_settings={"show_default": True})
@click.option(
    "--input",
    "input_dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_INPUT,
)
@click.option("--output", "output_dir", type=click.Path(path_type=Path, file_okay=False))
def main(input_dir: Path, output_dir: Path | None) -> None:
    """Render article figures from external-wrench experiment results."""
    output = output_dir or input_dir / "figures"
    _style()
    control_scheme(output)
    wrench_estimation(input_dir, output)
    control_comparison(input_dir, output)
    robustness(input_dir, output)


if __name__ == "__main__":
    main()
