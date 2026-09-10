"""Shared metrics and artifact helpers for paper-specific experiments."""

from __future__ import annotations

import csv
import json
import pickle
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from belka.common import quat_distance
from belka.actuation import valve_openings_to_forces


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "out" / "for_papers"


def git_revision() -> str:
    """Return the simulation revision recorded with generated artifacts."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def json_value(value: Any) -> Any:
    """Convert numpy and dataclass values into JSON-compatible objects."""
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    """Write one deterministic UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a rectangular metric table with a stable column order."""
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_value(row.get(key, "")) for key in fieldnames})


def write_run_log(
    path: Path,
    *,
    experiment: str,
    case: dict[str, Any],
    revision: str,
    artifacts: list[str],
) -> None:
    """Record a compact, machine-readable index of one completed case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"indexed_utc={datetime.now(timezone.utc).isoformat()}",
        "status=completed",
        f"simulation_revision={revision}",
        f"experiment={experiment}",
        f"case={json.dumps(json_value(case), ensure_ascii=False, sort_keys=True)}",
        f"artifacts={','.join(artifacts)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_pickle(path: Path) -> Any:
    """Load a paper experiment snapshot."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def aligned_control_samples(
    snapshot: dict[str, Any],
) -> tuple[np.ndarray, list, list, list, np.ndarray, np.ndarray]:
    """Align simulator truth, estimate, exact reference and wrench samples."""
    sim_history = snapshot["sim_history"]
    control_history = snapshot["control_history"]
    control_t = np.asarray(control_history.ts, dtype=float)
    sim_t = np.asarray(sim_history.ts, dtype=float)
    references = getattr(control_history, "reference_state", [])
    n = min(len(control_t), len(control_history.observed_state), len(references))
    if n == 0 or sim_t.size == 0:
        return np.empty(0), [], [], [], np.empty((0, 6)), np.empty((0, 6))
    control_t = control_t[:n]
    indices = np.searchsorted(sim_t, control_t, side="right") - 1
    indices = np.clip(indices, 0, len(sim_history.real_state) - 1)
    real_states = [sim_history.real_state[int(index)] for index in indices]
    observed_states = control_history.observed_state[:n]
    reference_states = references[:n]
    true_wrench = wrench_array(
        [sim_history.external_wrench[int(index)] for index in indices]
    )
    estimated_wrench = wrench_array(control_history.estimations[:n])
    return (
        control_t,
        real_states,
        observed_states,
        reference_states,
        true_wrench,
        estimated_wrench,
    )


def wrench_array(values: list[Any]) -> np.ndarray:
    """Convert wrench-like values into an N by 6 array."""
    rows = []
    for value in values:
        if value is None:
            rows.append(np.full(6, np.nan, dtype=float))
        elif hasattr(value, "to_array"):
            rows.append(np.asarray(value.to_array(), dtype=float))
        else:
            rows.append(np.asarray(value, dtype=float))
    return np.asarray(rows, dtype=float)


def flight_metrics(
    snapshot: dict[str, Any],
    solver_times: list[float] | None = None,
    solver_statuses: list[int] | None = None,
    time_window: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Calculate truth-based tracking, estimated tracking and observer metrics."""
    control_history = snapshot["control_history"]
    full_control_t = np.asarray(control_history.ts, dtype=float)
    references = getattr(control_history, "reference_state", None)
    if references is None or len(references) != len(full_control_t):
        raise ValueError(
            "exact reference_state is required for truth-based tracking metrics"
        )
    if time_window is None:
        control_mask = np.ones(full_control_t.shape, dtype=bool)
    else:
        start_t, end_t = time_window
        if end_t < start_t:
            raise ValueError("time_window end must not precede its start")
        control_mask = (full_control_t >= start_t) & (full_control_t <= end_t)

    control_t = full_control_t[control_mask]
    commands = np.asarray(
        [
            forces.to_array()
            for selected, forces in zip(
                control_mask, control_history.control_forces, strict=True
            )
            if selected
        ],
        dtype=float,
    )
    params = snapshot["effective_robot_params"]
    if commands.size:
        effort = (
            float(np.trapezoid(np.sum(commands * commands, axis=1), x=control_t))
            if len(control_t) > 1
            else 0.0
        )
        saturation = float(np.mean(commands >= 1.0 - 1e-8))
        if len(commands) > 1:
            delta = np.abs(np.diff(commands, axis=0))
            dt = np.maximum(np.diff(control_t), 1e-9)[:, None]
            slew_excess = np.maximum(
                delta - float(params.valve_response_per_s) * dt,
                0.0,
            )
            slew_violation = float(np.max(slew_excess))
        else:
            slew_violation = 0.0
        actual_forces = np.asarray(
            [
                valve_openings_to_forces(opening, params).to_array()
                for opening in control_history.real_forces
            ],
            dtype=float,
        )[control_mask]
        group_excess = []
        for group, side_limit in zip(
            params.side_force_groups,
            params.side_force_limits_n,
            strict=True,
        ):
            group_excess.append(
                np.maximum(
                    np.sum(actual_forces[:, list(group)], axis=1) - side_limit,
                    0.0,
                )
            )
        group_violation = float(
            max((np.max(values) for values in group_excess), default=0.0)
        )
    else:
        effort = saturation = slew_violation = group_violation = float("nan")

    (
        aligned_t,
        real_states,
        observed_states,
        reference_states,
        true_wrench,
        estimated_wrench,
    ) = aligned_control_samples(snapshot)
    if time_window is not None and aligned_t.size:
        aligned_mask = (aligned_t >= time_window[0]) & (aligned_t <= time_window[1])
        real_states = [
            state
            for selected, state in zip(aligned_mask, real_states, strict=True)
            if selected
        ]
        observed_states = [
            state
            for selected, state in zip(aligned_mask, observed_states, strict=True)
            if selected
        ]
        reference_states = [
            state
            for selected, state in zip(aligned_mask, reference_states, strict=True)
            if selected
        ]
        true_wrench = true_wrench[aligned_mask]
        estimated_wrench = estimated_wrench[aligned_mask]
    if any(state is None for state in reference_states):
        raise ValueError(
            "exact reference_state is missing in selected controller samples"
        )
    if real_states:
        position_error = np.asarray(
            [
                np.linalg.norm(real.p - reference.p)
                for real, reference in zip(real_states, reference_states, strict=True)
            ],
            dtype=float,
        )
        orientation_error = np.asarray(
            [
                quat_distance(real.q, reference.q)
                for real, reference in zip(real_states, reference_states, strict=True)
            ],
            dtype=float,
        )
        estimated_position_error = np.asarray(
            [
                np.linalg.norm(observed.p - reference.p)
                for observed, reference in zip(
                    observed_states, reference_states, strict=True
                )
            ],
            dtype=float,
        )
        estimated_orientation_error = np.asarray(
            [
                quat_distance(observed.q, reference.q)
                for observed, reference in zip(
                    observed_states, reference_states, strict=True
                )
            ],
            dtype=float,
        )
        observer_position = np.asarray(
            [
                np.linalg.norm(observed.p - real.p)
                for real, observed in zip(real_states, observed_states, strict=True)
            ],
            dtype=float,
        )
        observer_orientation = np.asarray(
            [
                quat_distance(observed.q, real.q)
                for real, observed in zip(real_states, observed_states, strict=True)
            ],
            dtype=float,
        )
    else:
        position_error = orientation_error = np.empty(0)
        estimated_position_error = estimated_orientation_error = np.empty(0)
        observer_position = observer_orientation = np.empty(0)

    valid_wrench = (
        true_wrench.size
        and estimated_wrench.size
        and not np.isnan(estimated_wrench).all()
    )
    wrench_error = (
        estimated_wrench - true_wrench if valid_wrench else np.empty((0, 6))
    )
    force_error = (
        np.linalg.norm(wrench_error[:, :3], axis=1)
        if wrench_error.size
        else np.empty(0)
    )
    moment_error = (
        np.linalg.norm(wrench_error[:, 3:], axis=1)
        if wrench_error.size
        else np.empty(0)
    )
    solver = np.asarray(solver_times or [], dtype=float)
    statuses = np.asarray(solver_statuses or [], dtype=int)
    if time_window is not None and solver.size == full_control_t.size:
        solver = solver[control_mask]
    if time_window is not None and statuses.size == full_control_t.size:
        statuses = statuses[control_mask]

    def rmse(values: np.ndarray) -> float:
        return (
            float(np.sqrt(np.nanmean(values * values))) if values.size else float("nan")
        )

    def percentile(values: np.ndarray, q: float) -> float:
        return float(np.nanpercentile(values, q)) if values.size else float("nan")

    metrics = {
        "tracking_position_rmse_m": rmse(position_error),
        "tracking_position_p95_m": percentile(position_error, 95.0),
        "tracking_position_final_m": float(position_error[-1])
        if position_error.size
        else float("nan"),
        "tracking_orientation_rmse_rad": rmse(orientation_error),
        "tracking_orientation_p95_rad": percentile(orientation_error, 95.0),
        "tracking_orientation_final_rad": float(orientation_error[-1])
        if orientation_error.size
        else float("nan"),
        "estimated_tracking_position_rmse_m": rmse(estimated_position_error),
        "estimated_tracking_position_p95_m": percentile(
            estimated_position_error, 95.0
        ),
        "estimated_tracking_position_final_m": float(estimated_position_error[-1])
        if estimated_position_error.size
        else float("nan"),
        "estimated_tracking_orientation_rmse_rad": rmse(
            estimated_orientation_error
        ),
        "estimated_tracking_orientation_p95_rad": percentile(
            estimated_orientation_error, 95.0
        ),
        "estimated_tracking_orientation_final_rad": float(
            estimated_orientation_error[-1]
        )
        if estimated_orientation_error.size
        else float("nan"),
        "observer_position_rmse_m": rmse(observer_position),
        "observer_orientation_rmse_rad": rmse(observer_orientation),
        "force_estimation_rmse_n": rmse(force_error),
        "force_estimation_mae_n": float(np.nanmean(force_error))
        if force_error.size
        else float("nan"),
        "moment_estimation_rmse_nm": rmse(moment_error),
        "moment_estimation_mae_nm": float(np.nanmean(moment_error))
        if moment_error.size
        else float("nan"),
        "control_effort_normalized2_s": effort,
        "thruster_saturation_fraction": saturation,
        "slew_violation_max_fraction": slew_violation,
        "group_violation_max_n": group_violation,
        "solver_time_median_ms": float(1000.0 * np.median(solver))
        if solver.size
        else float("nan"),
        "solver_time_p95_ms": float(1000.0 * np.percentile(solver, 95.0))
        if solver.size
        else float("nan"),
        "solver_nonstandard_status_fraction": float(
            np.mean(~np.isin(statuses, (0, 2)))
        )
        if statuses.size
        else float("nan"),
        "solver_max_iter_status_fraction": float(np.mean(statuses == 2))
        if statuses.size
        else float("nan"),
    }
    component_names = ("fx", "fy", "fz", "mx", "my", "mz")
    component_units = ("n", "n", "n", "nm", "nm", "nm")
    for index, (name, unit) in enumerate(
        zip(component_names, component_units, strict=True)
    ):
        component = wrench_error[:, index] if wrench_error.size else np.empty(0)
        metrics[f"wrench_{name}_rmse_{unit}"] = rmse(component)
        metrics[f"wrench_{name}_bias_{unit}"] = (
            float(np.nanmean(component)) if component.size else float("nan")
        )
    return metrics


def aggregate(
    rows: list[dict[str, Any]], group_keys: list[str]
) -> list[dict[str, Any]]:
    """Aggregate numeric metrics into mean and sample standard deviation."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(group_keys, key, strict=True))
        result["n"] = len(group)
        common_keys = set.intersection(*(set(item) for item in group))
        numeric_keys = sorted(
            name
            for name in common_keys
            if name not in group_keys
            and all(isinstance(item[name], (int, float, np.number)) for item in group)
        )
        for name in numeric_keys:
            values = np.asarray([item[name] for item in group], dtype=float)
            finite = values[np.isfinite(values)]
            result[f"{name}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            result[f"{name}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
        output.append(result)
    return output
