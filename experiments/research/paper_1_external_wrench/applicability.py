"""Pre-registered EnvMPC applicability search with truth-based metrics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import numpy as np
from scipy.optimize import linprog

from belka.common import RobotParams3D, RobotState3D
from experiments.research.common import OUTPUT_ROOT, write_json, write_metrics_csv
from experiments.research.paper_1_external_wrench.run import (
    BASE_SCENARIO,
    _base_mapping,
    refresh_saved_metrics,
    run_prepared_case,
)


DEFAULT_OUTPUT = OUTPUT_ROOT / "paper_1_external_wrench_corrected"
MODES = ("ignore", "estimate")
CONFIRMATION_MODES = (*MODES, "oracle")
COMPOSITIONS = ("force_only", "moment_only", "mixed")
MIXED_MOMENT_LEVER_M = 0.10


@dataclass(frozen=True)
class EvaluationCase:
    """One fully specified external-wrench evaluation case."""

    stage: str
    candidate_id: str
    composition: str
    mode: str
    force_direction: tuple[float, float, float]
    moment_direction: tuple[float, float, float]
    relative_scale: float
    modulation_frequency_hz: float
    modulation_amplitude: float
    seed: int

    def __post_init__(self) -> None:
        """Validate the case before solver or simulator creation."""
        if self.stage not in {"screening", "bandwidth", "confirmation"}:
            raise ValueError(f"unknown stage: {self.stage}")
        if self.composition not in COMPOSITIONS:
            raise ValueError(f"unknown wrench composition: {self.composition}")
        if self.mode not in CONFIRMATION_MODES:
            raise ValueError(f"unknown controller mode: {self.mode}")
        if self.stage != "confirmation" and self.mode == "oracle":
            raise ValueError("oracle mode is reserved for independent confirmation")
        if not np.isfinite(self.relative_scale) or not 0.0 < self.relative_scale <= 1.0:
            raise ValueError("relative_scale must be finite and in (0, 1]")
        if (
            not np.isfinite(self.modulation_frequency_hz)
            or self.modulation_frequency_hz < 0.0
            or not np.isfinite(self.modulation_amplitude)
            or self.modulation_amplitude < 0.0
        ):
            raise ValueError("modulation frequency/amplitude must be finite and non-negative")
        force = np.asarray(self.force_direction, dtype=float)  # world-frame force direction
        moment = np.asarray(self.moment_direction, dtype=float)  # world-frame moment direction
        if force.shape != (3,) or moment.shape != (3,) or np.any(~np.isfinite(np.r_[force, moment])):
            raise ValueError("force_direction and moment_direction must contain three finite values")
        active_force = self.composition in {"force_only", "mixed"}
        active_moment = self.composition in {"moment_only", "mixed"}
        if active_force != bool(np.linalg.norm(force) > 1e-12):
            raise ValueError("force direction activity does not match composition")
        if active_moment != bool(np.linalg.norm(moment) > 1e-12):
            raise ValueError("moment direction activity does not match composition")
        if active_force and not np.isclose(np.linalg.norm(force), 1.0, atol=1e-9):
            raise ValueError("active force_direction must be unit length")
        if active_moment and not np.isclose(np.linalg.norm(moment), 1.0, atol=1e-9):
            raise ValueError("active moment_direction must be unit length")


def wrench_basis(case: EvaluationCase) -> np.ndarray:
    """Return the world-frame wrench produced by one unit of lambda."""
    force = np.asarray(case.force_direction, dtype=float)  # force per lambda
    moment = np.asarray(case.moment_direction, dtype=float)  # moment direction
    if case.composition == "force_only":
        return np.r_[force, np.zeros(3)]
    if case.composition == "moment_only":
        return np.r_[np.zeros(3), moment]
    return np.r_[force, MIXED_MOMENT_LEVER_M * moment]


def compensable_wrench_limit(
    params: RobotParams3D, case: EvaluationCase
) -> dict[str, object]:
    """Maximize compensable lambda under individual and impeller-group force limits."""
    basis = wrench_basis(case)  # external world/body wrench per lambda at identity
    allocation = np.asarray(params.allocation_matrix, dtype=float)  # body wrench per force
    c = np.r_[np.zeros(params.nu), -1.0]  # maximize lambda
    A_eq = np.column_stack((allocation, basis))  # actuator wrench + external wrench = 0
    side_rows = []
    for group in params.side_force_groups:
        row = np.zeros(params.nu + 1, dtype=float)  # one impeller summed-force row
        row[list(group)] = 1.0
        side_rows.append(row)
    result = linprog(
        c=c,
        A_ub=np.asarray(side_rows, dtype=float),
        b_ub=np.asarray(params.side_force_limits_n, dtype=float),
        A_eq=A_eq,
        b_eq=np.zeros(6, dtype=float),
        bounds=[(0.0, float(cap)) for cap in params.per_thruster_force_caps_n]
        + [(0.0, None)],
        method="highs",
    )
    if not result.success:
        raise ValueError(f"direction is not statically compensable: {result.message}")
    forces = np.asarray(result.x[: params.nu], dtype=float)  # optimal thruster forces [N]
    lambda_max = float(result.x[-1])  # maximum external-wrench scale
    if not np.isfinite(lambda_max) or lambda_max <= 1e-10:
        raise ValueError("direction has no positive finite compensable lambda")
    residual = allocation @ forces + lambda_max * basis  # static wrench balance
    group_totals = np.asarray(
        [np.sum(forces[list(group)]) for group in params.side_force_groups], dtype=float
    )
    active_constraints = [
        f"thruster_{index + 1}"
        for index, (force, cap) in enumerate(
            zip(forces, params.per_thruster_force_caps_n, strict=True)
        )
        if np.isclose(force, cap, rtol=1e-7, atol=1e-9)
    ]
    active_constraints.extend(
        f"group_{index + 1}"
        for index, (total, limit) in enumerate(
            zip(group_totals, params.side_force_limits_n, strict=True)
        )
        if np.isclose(total, limit, rtol=1e-7, atol=1e-9)
    )
    return {
        "lambda_max": lambda_max,
        "lambda_unit": "N*m" if case.composition == "moment_only" else "N",
        "basis_wrench": basis,
        "thruster_forces_n": forces,
        "group_totals_n": group_totals,
        "active_constraints": active_constraints,
        "balance_residual_norm": float(np.linalg.norm(residual)),
    }


def screening_cases() -> list[EvaluationCase]:
    """Return the pre-registered 36-case static screening matrix."""
    zero = (0.0, 0.0, 0.0)
    axes = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }
    directions = [
        *(('force_only', f"force_{name}", direction, zero) for name, direction in axes.items()),
        *(('moment_only', f"moment_{name}", zero, direction) for name, direction in axes.items()),
        ("mixed", "mixed_xy", axes["x"], axes["y"]),
        ("mixed", "mixed_yz", axes["y"], axes["z"]),
        ("mixed", "mixed_zx", axes["z"], axes["x"]),
    ]
    return [
        EvaluationCase(
            stage="screening",
            candidate_id=f"{name}_s{int(scale * 100):02d}",
            composition=composition,
            mode=mode,
            force_direction=force,
            moment_direction=moment,
            relative_scale=scale,
            modulation_frequency_hz=0.0,
            modulation_amplitude=0.0,
            seed=0,
        )
        for composition, name, force, moment in directions
        for scale in (0.35, 0.55)
        for mode in MODES
    ]


def _case_output(root: Path, case: EvaluationCase) -> Path:
    """Return a stable output directory for one campaign case."""
    return root / case.stage / case.candidate_id / case.mode


def _manifest_payload(stage: str, cases: list[EvaluationCase]) -> dict[str, object]:
    """Build a hash-stable stage manifest before any case is executed."""
    body = json.loads(
        json.dumps(
            {"schema_version": 1, "stage": stage, "cases": [asdict(case) for case in cases]}
        )
    )
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "sha256": hashlib.sha256(encoded).hexdigest()}


def write_locked_manifest(root: Path, stage: str, cases: list[EvaluationCase]) -> Path:
    """Write a manifest once and reject a changed case matrix on resume."""
    path = root / stage / "manifest.json"
    payload = _manifest_payload(stage, cases)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError(f"existing {stage} manifest differs from requested campaign")
    write_json(path, payload)
    return path


def _wind_profile(case: EvaluationCase, limit: dict[str, object]):
    """Build the deterministic external world-frame wrench for one case."""
    base = (
        case.relative_scale
        * float(limit["lambda_max"])
        * np.asarray(limit["basis_wrench"], dtype=float)
    )  # static external world-frame wrench

    def value(t: float, state: RobotState3D | None) -> np.ndarray:
        del state
        scale = 1.0 + case.modulation_amplitude * np.sin(
            2.0 * np.pi * case.modulation_frequency_hz * t
        )
        return scale * base

    value.summary = {
        "profile": "static" if case.modulation_frequency_hz == 0.0 else "sinusoidal",
        "base_wrench": base.tolist(),
        "modulation_frequency_hz": case.modulation_frequency_hz,
        "modulation_amplitude": case.modulation_amplitude,
        **limit,
    }
    return value


def run_evaluation_case(
    case: EvaluationCase, output_root: Path, *, force: bool = False
) -> dict:
    """Execute one case from its locked manifest and persist truth-based metrics."""
    output_dir = _case_output(output_root, case)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text())
    params = RobotParams3D.from_json(BASE_SCENARIO)
    limit = compensable_wrench_limit(params, case)
    case_mapping = {**asdict(case), "static_allocation": limit}
    mapping = _base_mapping(case.seed, noise_multiplier=0.5)
    mapping["name"] = f"paper1_applicability_{case.stage}_{case.candidate_id}_{case.mode}"
    mapping["description"] = "Pre-registered truth-based EnvMPC applicability case."
    mapping["run"]["t_max"] = 45.0 if case.stage == "screening" else 60.0
    mapping["wind"]["force_ratio"] = 0.0
    return run_prepared_case(
        experiment=f"applicability_{case.stage}",
        case=case_mapping,
        output_dir=output_dir,
        mapping=mapping,
        wind=_wind_profile(case, limit),
        metric_window=(15.0, float(mapping["run"]["t_max"])),
    )


def refresh_stage_metrics(
    cases: list[EvaluationCase], output_root: Path
) -> list[dict]:
    """Apply the pre-registered post-convergence window to saved stage snapshots."""
    rows = []
    for case in cases:
        output_dir = _case_output(output_root, case)
        scenario = json.loads((output_dir / "scenario.json").read_text())
        rows.append(
            refresh_saved_metrics(
                output_dir, (15.0, float(scenario["run"]["t_max"]))
            )
        )
    write_metrics_csv(output_root / cases[0].stage / "metrics.csv", rows)
    return rows


def run_stage(
    cases: list[EvaluationCase], output_root: Path, *, force: bool = False
) -> list[dict]:
    """Run every case in one locked stage and write its rectangular metric table."""
    if not cases:
        raise ValueError("stage must contain at least one case")
    stage = cases[0].stage
    if any(case.stage != stage for case in cases):
        raise ValueError("all cases in one stage must share the same stage")
    write_locked_manifest(output_root, stage, cases)
    rows = []
    for index, case in enumerate(cases, start=1):
        click.echo(f"[{stage} {index}/{len(cases)}] {case.candidate_id} {case.mode}")
        rows.append(run_evaluation_case(case, output_root, force=force))
        write_metrics_csv(output_root / stage / "metrics.csv", rows)
    return rows


def _primary_metric(composition: str) -> str:
    """Return the truth-based primary metric for one wrench composition."""
    return (
        "tracking_position_rmse_m"
        if composition == "force_only"
        else "tracking_orientation_rmse_rad"
    )


def paired_effects(rows: list[dict]) -> list[dict]:
    """Pair MPC and EnvMPC rows and compute pre-registered improvement criteria."""
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["composition"], row["candidate_id"]), {})[
            row["mode"]
        ] = row
    effects = []
    for (composition, candidate_id), modes in sorted(grouped.items()):
        if not all(mode in modes for mode in MODES):
            continue
        mpc = modes["ignore"]
        env = modes["estimate"]
        position = 100.0 * (
            float(mpc["tracking_position_rmse_m"])
            - float(env["tracking_position_rmse_m"])
        ) / float(mpc["tracking_position_rmse_m"])
        orientation = 100.0 * (
            float(mpc["tracking_orientation_rmse_rad"])
            - float(env["tracking_orientation_rmse_rad"])
        ) / float(mpc["tracking_orientation_rmse_rad"])
        improvement = (
            position
            if composition == "force_only"
            else orientation
            if composition == "moment_only"
            else 0.5 * (position + orientation)
        )
        admissible = (
            improvement > 0.0
            and float(env["thruster_saturation_fraction"]) == 0.0
            and float(env["slew_violation_max_fraction"]) <= 1e-9
            and float(env["group_violation_max_n"]) <= 1e-9
            and float(env["solver_nonstandard_status_fraction"]) == 0.0
            and float(env["solver_time_p95_ms"]) < 50.0
        )
        effects.append(
            {
                "composition": composition,
                "candidate_id": candidate_id,
                "relative_scale": env["relative_scale"],
                "modulation_frequency_hz": env["modulation_frequency_hz"],
                "position_improvement_percent": position,
                "orientation_improvement_percent": orientation,
                "primary_improvement_percent": improvement,
                "admissible": admissible,
                "force_direction": env["force_direction"],
                "moment_direction": env["moment_direction"],
                "seed": env["seed"],
            }
        )
    return effects


def select_best(rows: list[dict], output_path: Path) -> dict[str, dict]:
    """Select at most one admissible candidate per composition without substitution."""
    effects = paired_effects(rows)
    selected = {}
    for composition in COMPOSITIONS:
        candidates = [
            row for row in effects if row["composition"] == composition and row["admissible"]
        ]
        if candidates:
            selected[composition] = max(
                candidates, key=lambda row: float(row["primary_improvement_percent"])
            )
    write_json(output_path, {"effects": effects, "selected": selected})
    return selected


def bandwidth_cases(selected: dict[str, dict]) -> list[EvaluationCase]:
    """Expand three selected static candidates into the fixed 18-case frequency stage."""
    if set(selected) != set(COMPOSITIONS):
        raise RuntimeError(f"bandwidth requires one admissible candidate per class, got {sorted(selected)}")
    return [
        EvaluationCase(
            stage="bandwidth",
            candidate_id=f"{composition}_f{str(frequency).replace('.', 'p')}",
            composition=composition,
            mode=mode,
            force_direction=tuple(selected[composition]["force_direction"]),
            moment_direction=tuple(selected[composition]["moment_direction"]),
            relative_scale=float(selected[composition]["relative_scale"]),
            modulation_frequency_hz=frequency,
            modulation_amplitude=0.25,
            seed=0,
        )
        for composition in COMPOSITIONS
        for frequency in (0.005, 0.015, 0.03)
        for mode in MODES
    ]


def confirmation_cases(selected: dict[str, dict]) -> list[EvaluationCase]:
    """Build the locked 45-case holdout matrix from five unseen seeded directions."""
    if set(selected) != set(COMPOSITIONS):
        raise RuntimeError(f"confirmation requires one admissible candidate per class, got {sorted(selected)}")
    cases = []
    for composition in COMPOSITIONS:
        for seed in range(101, 106):
            rng = np.random.default_rng(seed + 1000 * COMPOSITIONS.index(composition))
            force = rng.normal(size=3) if composition != "moment_only" else np.zeros(3)
            moment = rng.normal(size=3) if composition != "force_only" else np.zeros(3)
            if np.linalg.norm(force):
                force /= np.linalg.norm(force)
            if np.linalg.norm(moment):
                moment /= np.linalg.norm(moment)
            for mode in CONFIRMATION_MODES:
                cases.append(
                    EvaluationCase(
                        stage="confirmation",
                        candidate_id=f"{composition}_holdout_{seed}",
                        composition=composition,
                        mode=mode,
                        force_direction=tuple(float(value) for value in force),
                        moment_direction=tuple(float(value) for value in moment),
                        relative_scale=float(selected[composition]["relative_scale"]),
                        modulation_frequency_hz=float(
                            selected[composition]["modulation_frequency_hz"]
                        ),
                        modulation_amplitude=0.25,
                        seed=seed,
                    )
                )
    return cases


def confirmation_selection(
    screening_selected: dict[str, dict], bandwidth_selected: dict[str, dict]
) -> dict[str, dict]:
    """Use a passed bandwidth case or the already admissible static fallback."""
    if set(screening_selected) != set(COMPOSITIONS):
        raise RuntimeError("confirmation fallback requires all static classes")
    selected = {}
    for composition in COMPOSITIONS:
        if composition in bandwidth_selected:
            selected[composition] = {
                **bandwidth_selected[composition],
                "selection_source": "bandwidth",
            }
        else:
            selected[composition] = {
                **screening_selected[composition],
                "modulation_frequency_hz": 0.0,
                "selection_source": "admissible_static_fallback",
            }
    return selected


def confirmation_effects(rows: list[dict]) -> list[dict]:
    """Compute holdout effects and whether EnvMPC lies closer to the oracle result."""
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["composition"], row["candidate_id"]), {})[
            row["mode"]
        ] = row
    output = []
    for (composition, candidate_id), modes in sorted(grouped.items()):
        if not all(mode in modes for mode in CONFIRMATION_MODES):
            continue
        position = {
            mode: float(values["tracking_position_rmse_m"])
            for mode, values in modes.items()
        }
        orientation = {
            mode: float(values["tracking_orientation_rmse_rad"])
            for mode, values in modes.items()
        }
        position_improvement = 100.0 * (
            position["ignore"] - position["estimate"]
        ) / position["ignore"]
        orientation_improvement = 100.0 * (
            orientation["ignore"] - orientation["estimate"]
        ) / orientation["ignore"]
        if composition == "force_only":
            metric = "tracking_position_rmse_m"
            improvement = position_improvement
            env_closer = abs(position["estimate"] - position["oracle"]) < abs(
                position["ignore"] - position["oracle"]
            )
        elif composition == "moment_only":
            metric = "tracking_orientation_rmse_rad"
            improvement = orientation_improvement
            env_closer = abs(orientation["estimate"] - orientation["oracle"]) < abs(
                orientation["ignore"] - orientation["oracle"]
            )
        else:
            metric = "mean_relative_position_orientation_improvement"
            improvement = 0.5 * (position_improvement + orientation_improvement)
            mpc_gap = 0.5 * (
                abs(position["ignore"] - position["oracle"]) / position["ignore"]
                + abs(orientation["ignore"] - orientation["oracle"])
                / orientation["ignore"]
            )
            env_gap = 0.5 * (
                abs(position["estimate"] - position["oracle"])
                / position["ignore"]
                + abs(orientation["estimate"] - orientation["oracle"])
                / orientation["ignore"]
            )
            env_closer = env_gap < mpc_gap
        output.append(
            {
                "composition": composition,
                "candidate_id": candidate_id,
                "primary_metric": metric,
                "position_improvement_percent": position_improvement,
                "orientation_improvement_percent": orientation_improvement,
                "improvement_percent": improvement,
                "env_closer_to_oracle": env_closer,
            }
        )
    return output


def plot_applicability(root: Path, screening: list[dict], bandwidth: list[dict], confirmation: list[dict]) -> Path:
    """Plot discovery maps and independent confirmation without mixing their claims."""
    import matplotlib.pyplot as plt

    screen_effects = paired_effects(screening)
    band_effects = paired_effects(bandwidth)
    confirm_effects = confirmation_effects(confirmation)
    colors = {"force_only": "#2563eb", "moment_only": "#dc2626", "mixed": "#059669"}
    labels = {"force_only": "только сила", "moment_only": "только момент", "mixed": "сила и момент"}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    for composition in COMPOSITIONS:
        rows = [row for row in screen_effects if row["composition"] == composition]
        axes[0].scatter(
            [row["relative_scale"] for row in rows],
            [row["primary_improvement_percent"] for row in rows],
            label=labels[composition],
            color=colors[composition],
        )
        rows = [row for row in band_effects if row["composition"] == composition]
        axes[1].plot(
            [row["modulation_frequency_hz"] for row in rows],
            [row["primary_improvement_percent"] for row in rows],
            "o-",
            color=colors[composition],
        )
        rows = [row for row in confirm_effects if row["composition"] == composition]
        axes[2].scatter(
            [COMPOSITIONS.index(composition)] * len(rows),
            [row["improvement_percent"] for row in rows],
            color=colors[composition],
        )
    axes[0].set(
        xlabel=r"$\lambda/\lambda_{max}$",
        ylabel="Улучшение, %",
        title="Статические воздействия",
    )
    axes[1].set(xlabel="Частота, Гц", title="Изменение воздействия")
    axes[2].set(title="Независимое подтверждение", ylabel="Улучшение, %")
    axes[2].set_xticks(range(3), [labels[name] for name in COMPOSITIONS], rotation=15)
    for axis in axes[1:]:
        axis.set_yscale("symlog", linthresh=20.0)
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    path = root / "envmpc_applicability.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def run_all(output_root: Path, *, force: bool = False) -> None:
    """Run the fixed 36 + 18 + 45 campaign and produce its final artifacts."""
    screening = run_stage(screening_cases(), output_root, force=force)
    selected_screening = select_best(
        screening, output_root / "screening" / "selection.json"
    )
    bandwidth = run_stage(
        bandwidth_cases(selected_screening), output_root, force=force
    )
    selected_bandwidth = select_best(
        bandwidth, output_root / "bandwidth" / "selection.json"
    )
    selected_confirmation = confirmation_selection(
        selected_screening, selected_bandwidth
    )
    write_json(
        output_root / "confirmation" / "selection.json", selected_confirmation
    )
    confirmation = run_stage(
        confirmation_cases(selected_confirmation), output_root, force=force
    )
    effects = confirmation_effects(confirmation)
    write_metrics_csv(output_root / "confirmation" / "effects.csv", effects)
    write_json(output_root / "confirmation" / "effects.json", effects)
    plot_applicability(output_root, screening, bandwidth, confirmation)


@click.command(context_settings={"show_default": True})
@click.option("--output", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_OUTPUT)
@click.option("--force", is_flag=True, help="Recompute cases with existing summaries.")
def main(output: Path, force: bool) -> None:
    """Execute the complete pre-registered EnvMPC applicability campaign."""
    run_all(output, force=force)


if __name__ == "__main__":
    main()
