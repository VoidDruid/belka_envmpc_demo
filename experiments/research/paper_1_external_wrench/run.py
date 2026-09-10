"""Run reproducible 6 DoF experiments for external-wrench estimation and control."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Callable

import numpy as np
import click

from belka.common import RobotParams3D, RobotState3D
from belka.orientation import state_from_presented_position_angles
from belka.shared.observers import WrenchEstimate3D
from experiments.config import Scenario3DConfig
from experiments.scenario_runner import (
    FlightLogic3D,
    _build_inertial_noise,
    _build_visual_noise,
    _build_wind,
    save_3d_run_artifacts,
)
from experiments.research.common import (
    OUTPUT_ROOT,
    aggregate,
    aligned_control_samples,
    flight_metrics,
    git_revision,
    load_pickle,
    write_json,
    write_metrics_csv,
    write_run_log,
)
from sim import SimParams
from sim.core import SimIO
from sim.noise import make_engine_noise
from sim.shared import DummyViewer


HERE = Path(__file__).resolve().parent
BASE_SCENARIO = HERE.parents[1] / "3D" / "cube.json"
DEFAULT_OUTPUT = OUTPUT_ROOT / "paper_1_external_wrench"
MODES = ("ignore", "estimate", "oracle")
PROFILES = ("constant", "step", "slow")
ESTIMATION_HOLD_DURATION_S = 60.0
CONTROL_COMPARISON_WINDOW_S = (15.0, 40.0)
SOLVER_METRIC_KEYS = (
    "solver_time_median_ms",
    "solver_time_p95_ms",
    "solver_nonstandard_status_fraction",
    "solver_max_iter_status_fraction",
)


def controller_wrench_input(
    mode: str,
    estimate: WrenchEstimate3D | None,
    true_wrench: np.ndarray,
) -> WrenchEstimate3D | None:
    """Select the wrench visible to MPC without leaking plant truth outside oracle mode."""
    if mode == "ignore":
        return WrenchEstimate3D.zeros()
    if mode == "estimate":
        return estimate
    if mode == "oracle":
        return WrenchEstimate3D.from_array(true_wrench)
    raise ValueError(f"Unknown wrench mode: {mode}")


class ComparisonFlightLogic3D(FlightLogic3D):
    """Use one EKF state estimate while varying wrench information supplied to MPC."""

    def __init__(self, *args, wrench_mode: str, true_wrench: Callable, **kwargs) -> None:
        if wrench_mode not in MODES:
            raise ValueError(f"Unknown wrench mode: {wrench_mode}")
        super().__init__(*args, **kwargs)
        self.wrench_mode = wrench_mode
        self.true_wrench = true_wrench
        self.control_step_times: list[float] = []
        self._raw_estimation = None

    def observe(self, t, input_data, actual_actuation):
        state = super().observe(t, input_data, actual_actuation)
        self._raw_estimation = self._estimation
        return state

    def think(self, t, actual_actuation):
        raw_estimation = self._raw_estimation
        controller_estimation = controller_wrench_input(
            self.wrench_mode,
            raw_estimation,
            self.true_wrench(t, None),
        )
        self._estimation = controller_estimation
        history_size = 0 if self.controller is None else len(self.controller.history)
        started = time.perf_counter()
        control = super().think(t, actual_actuation)
        self.control_step_times.append(time.perf_counter() - started)
        if self.controller is not None and len(self.controller.history) > history_size:
            self.controller.history.estimations[-1] = raw_estimation
        self._estimation = raw_estimation
        return control


def _base_mapping(seed: int, noise_multiplier: float) -> dict:
    mapping = json.loads(BASE_SCENARIO.read_text())
    mapping["name"] = "paper_1_external_wrench"
    mapping["description"] = "Paper experiment: 6 DoF EKF and disturbance-aware MPC."
    mapping["cargo"]["cargo_m"] = 0.0
    mapping["run"].update(
        {
            "initial_p": [0.0, 0.0, 1.0],
            "initial_angles": [0.0, 0.0, 0.0],
            "target_p": [0.65, 0.50, 1.25],
            "target_angles": [0.15, -0.12, 0.35],
            "trajectory_t": 15.0,
            "hold_time": 5.0,
            "t_max": 45.0,
            "seed": int(seed),
        }
    )
    mapping["cargo_estimation"]["enabled"] = False
    for name in (
        "position_bias_m",
        "position_slow_amplitude_m",
        "position_fast_amplitude_m",
        "attitude_bias_rad",
        "attitude_slow_amplitude_rad",
        "attitude_fast_amplitude_rad",
        "omega_std",
        "accel_std",
    ):
        value = mapping["sensor_noise"][name]
        if isinstance(value, str):
            numeric = {
                "2.2*deg": np.deg2rad(2.2),
                "2.0*deg": np.deg2rad(2.0),
                "0.3*deg": np.deg2rad(0.3),
            }[value]
        else:
            numeric = float(value)
        mapping["sensor_noise"][name] = numeric * noise_multiplier
    return mapping


def _configure_experiment_mapping(mapping: dict, experiment: str) -> dict:
    """Apply experiment-specific timing without changing the common robot model."""
    if experiment == "exp1_estimation":
        mapping["run"].update(
            {
                "target_p": list(mapping["run"]["initial_p"]),
                "target_angles": list(mapping["run"]["initial_angles"]),
                "hold_time": ESTIMATION_HOLD_DURATION_S + 1.0,
                "t_max": ESTIMATION_HOLD_DURATION_S,
            }
        )
        mapping["description"] += " Estimator characterization uses endpoint hold."
    return mapping


def _profile(params: RobotParams3D, profile: str, force_ratio: float, seed: int):
    rng = np.random.default_rng(seed + 9173)
    force_direction = rng.normal(size=3)
    force_direction /= np.linalg.norm(force_direction)
    moment_direction = rng.normal(size=3)
    moment_direction /= np.linalg.norm(moment_direction)
    arms = np.linalg.norm(
        np.cross(params.thruster_positions_body, params.thruster_directions_body),
        axis=1,
    )
    force = force_ratio * params.max_thruster_force_n * force_direction
    moment = 0.7 * np.linalg.norm(force) * 0.5 * np.max(arms) * moment_direction
    base = np.concatenate([force, moment])

    def value(t: float, state: RobotState3D | None) -> np.ndarray:
        del state
        if profile == "constant":
            return base.copy()
        if profile == "step":
            return np.zeros(6, dtype=float) if t < 12.0 else base.copy()
        if profile == "slow":
            scale = 1.0 + 0.25 * np.sin(2.0 * np.pi * 0.025 * t)
            transverse = 0.08 * np.sin(2.0 * np.pi * 0.11 * t + 0.7)
            return scale * base + transverse * np.roll(base, 1)
        raise ValueError(f"Unknown profile: {profile}")

    value.summary = {
        "profile": profile,
        "base_wrench": base.tolist(),
        "force_ratio": force_ratio,
    }
    return value


def _case_output(root: Path, experiment: str, case: dict) -> Path:
    tokens = [experiment]
    for name in ("profile", "mode", "force_ratio", "noise_multiplier", "seed"):
        if name in case:
            tokens.append(f"{name}-{case[name]}")
    return root.joinpath(*tokens)


def _cases(experiment: str) -> list[dict]:
    if experiment == "exp1_estimation":
        return [
            {
                "profile": profile,
                "mode": "estimate",
                "force_ratio": 0.25,
                "noise_multiplier": 1.0,
                "seed": seed,
            }
            for profile in PROFILES
            for seed in range(5)
        ]
    if experiment == "exp2_control_comparison":
        return [
            {
                "profile": "constant",
                "mode": mode,
                "force_ratio": 0.375,
                "noise_multiplier": 0.5,
                "seed": seed,
            }
            for mode in MODES
            for seed in range(5)
        ]
    if experiment == "exp3_robustness":
        return [
            {
                "profile": "random",
                "mode": "estimate",
                "force_ratio": force_ratio,
                "noise_multiplier": noise,
                "seed": seed,
            }
            for force_ratio in (0.125, 0.25, 0.375)
            for noise in (0.5, 1.0, 2.0)
            for seed in range(5)
        ]
    raise ValueError(f"Unknown experiment: {experiment}")


def _settling_time(snapshot: dict, profile: str) -> float:
    if profile == "slow":
        return float("nan")
    t, _real, _observed, _reference, true_wrench, estimated_wrench = (
        aligned_control_samples(snapshot)
    )
    if not len(t):
        return float("nan")
    start_t = 12.0 if profile == "step" else 0.0
    mask = t >= start_t
    error = np.linalg.norm(estimated_wrench[:, :3] - true_wrench[:, :3], axis=1)
    reference = np.linalg.norm(true_wrench[:, :3], axis=1)
    accepted = error <= 0.1 * reference + 0.01
    indices = np.flatnonzero(mask)
    window = 20
    for index in indices:
        if index + window <= len(accepted) and np.all(accepted[index : index + window]):
            return float(t[index] - start_t)
    return float("nan")


def run_case(experiment: str, case: dict, output_root: Path, force: bool = False) -> dict:
    output_dir = _case_output(output_root, experiment, case)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = _configure_experiment_mapping(_base_mapping(case["seed"], case["noise_multiplier"]), experiment)
    mapping["name"] = f"paper1_{experiment}_{case['mode']}"
    mapping["wind"]["force_ratio"] = case["force_ratio"]
    scenario_path = output_dir / "scenario.json"
    write_json(scenario_path, mapping)
    config = Scenario3DConfig.load(scenario_path)
    np.random.seed(config.run.seed)
    params = config.robot_params
    wind = (
        _build_wind(config, params)
        if case["profile"] == "random"
        else _profile(params, case["profile"], case["force_ratio"], case["seed"])
    )
    return run_prepared_case(
        experiment=experiment,
        case=case,
        output_dir=output_dir,
        mapping=mapping,
        wind=wind,
        metric_window=CONTROL_COMPARISON_WINDOW_S
        if experiment == "exp2_control_comparison"
        else None,
        settling_profile=case["profile"],
    )


def run_prepared_case(
    *,
    experiment: str,
    case: dict,
    output_dir: Path,
    mapping: dict,
    wind: Callable,
    metric_window: tuple[float, float] | None = None,
    settling_profile: str | None = None,
) -> dict:
    """Execute and persist one fully specified external-wrench case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    scenario_path = output_dir / "scenario.json"
    write_json(scenario_path, mapping)
    config = Scenario3DConfig.load(scenario_path)
    np.random.seed(config.run.seed)
    params = config.robot_params
    initial_state = state_from_presented_position_angles(
        config.run.initial_p, config.run.initial_angles, params.get_presented_frame()
    )
    target = state_from_presented_position_angles(
        config.run.target_p, config.run.target_angles, params.get_presented_frame()
    )
    logic = ComparisonFlightLogic3D(
        target,
        T=config.run.trajectory_t,
        hold_time=config.run.hold_time,
        home_state=initial_state,
        show_front_arrows=False,
        controller_params=params,
        wrench_mode=case["mode"],
        true_wrench=wind,
    )
    sim = SimIO(
        sim_params=SimParams(
            t_max=config.run.t_max,
            full_odom_hz=config.run.full_odom_hz,
            highfreq_odom_hz=config.run.highfreq_odom_hz,
        ),
        initial_state=initial_state,
        json_path=scenario_path,
        output_dir=output_dir,
        highlevel_control=logic,
        visual_noise=_build_visual_noise(config),
        inertial_noise=_build_inertial_noise(config),
        engine_noise=make_engine_noise(config.engine_noise.std(params), params.nu),
        external_wrench=wind,
        show_perceived_position=False,
        show_front_arrows=False,
    )
    sim.viewer = DummyViewer()
    while sim.is_running():
        sim.step()
    logic.finish()
    save_3d_run_artifacts(
        sim=sim,
        control_history=logic.history,
        scenario=mapping,
        scenario_json_path=scenario_path,
        target=target,
        save_plots=False,
        effective_robot_params=params,
    )
    snapshot = load_pickle(output_dir / "snapshot.pkl")
    if logic.controller is None:
        raise RuntimeError("paper run finished without configured controller")
    metrics = flight_metrics(
        snapshot,
        logic.controller.solve_times,
        logic.controller.solve_statuses,
        time_window=metric_window,
    )
    if metric_window is not None:
        metrics["metric_window_start_s"] = metric_window[0]
        metrics["metric_window_end_s"] = metric_window[1]
    if settling_profile is not None:
        settling_time = _settling_time(snapshot, settling_profile)
        if np.isfinite(settling_time):
            metrics["force_settling_time_s"] = settling_time
    summary = {
        "experiment": experiment,
        **case,
        **metrics,
        "output_dir": str(output_dir),
    }
    if logic.observer is None:
        raise RuntimeError("paper run finished without configured observer/controller")
    observer_params = logic.observer.params
    observer_config = {
        field.name: getattr(observer_params, field.name)
        for field in fields(observer_params)
        if field.name != "robot_params"
    }
    observer_config.update(
        {
            "state": "[p_world,q_body_to_world,v_world,omega_body,F_world,M_world]",
            "error_state": "[dp,dtheta,dv,domega,dF,dM]",
            "visual_measurement": "[p_world,q_body_to_world]",
            "inertial_measurement": "[q_body_to_world,omega_body,a_body]",
            "visual_rate_hz": config.run.full_odom_hz,
            "inertial_rate_hz": config.run.highfreq_odom_hz,
            "initial_covariance_diagonal": np.diag(
                logic.observer._initial_covariance()
            ),
            "process_covariance_diagonal_at_control_dt": np.diag(
                logic.observer._process_noise(1.0 / config.run.highfreq_odom_hz)
            ),
            "measurement_covariance_visual_diagonal": np.diag(
                logic.observer._measurement_noise(has_visual=True)
            ),
            "measurement_covariance_inertial_diagonal": np.diag(
                logic.observer._measurement_noise(has_visual=False)
            ),
            "imu_bias_states": False,
            "covariance_update": "Joseph then right-injection reset",
        }
    )
    controller_config = {
        "horizon": logic.controller.horizon,
        "dt_s": logic.controller.dt,
        "state_weight": logic.controller.state_weight,
        "thruster_weight": logic.controller.thruster_weight,
        "thruster_delta_weight": logic.controller.thruster_delta_weight,
        "terminal_weight": logic.controller.terminal_weight,
        "solver_options": logic.controller.effective_solver_options,
    }
    write_json(summary_path, summary)
    write_metrics_csv(output_dir / "metrics.csv", [summary])
    revision = git_revision()
    artifacts = [
        "scenario.json",
        "model.xml",
        "snapshot.pkl",
        "summary.json",
        "metrics.csv",
        "manifest.json",
    ]
    write_json(
        output_dir / "manifest.json",
        {
            "simulation_revision": revision,
            "experiment": experiment,
            "case": case,
            "scenario": mapping,
            "observer": observer_config,
            "controller": controller_config,
            "runtime_samples": len(sim.history.ts),
            "control_samples": len(logic.history.ts),
            "runtime_snapshots": ["snapshot.pkl"],
            "artifacts": artifacts + ["run.log"],
        },
    )
    write_run_log(
        output_dir / "run.log",
        experiment=experiment,
        case=case,
        revision=revision,
        artifacts=artifacts,
    )
    return summary


def refresh_saved_metrics(
    output_dir: Path, time_window: tuple[float, float]
) -> dict:
    """Recompute windowed physical metrics without rerunning a saved simulation."""
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    solver_metrics = {name: summary[name] for name in SOLVER_METRIC_KEYS}
    metrics = flight_metrics(
        load_pickle(output_dir / "snapshot.pkl"), time_window=time_window
    )
    metrics.update(solver_metrics)
    metrics["metric_window_start_s"] = time_window[0]
    metrics["metric_window_end_s"] = time_window[1]
    summary.update(metrics)
    write_json(summary_path, summary)
    write_metrics_csv(output_dir / "metrics.csv", [summary])
    return summary


def _ensure_case_artifacts(output_dir: Path, experiment: str, case: dict, row: dict) -> None:
    """Backfill the common per-case artifact index without rerunning a simulation."""
    write_metrics_csv(output_dir / "metrics.csv", [row])
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = str(manifest.get("simulation_revision") or git_revision())
    artifacts = [
        "scenario.json",
        "model.xml",
        "snapshot.pkl",
        "summary.json",
        "metrics.csv",
        "manifest.json",
    ]
    manifest.update(
        {
            "simulation_revision": revision,
            "experiment": experiment,
            "case": case,
            "runtime_snapshots": ["snapshot.pkl"],
            "artifacts": artifacts + ["run.log"],
        }
    )
    write_json(manifest_path, manifest)
    if not (output_dir / "run.log").exists():
        write_run_log(
            output_dir / "run.log",
            experiment=experiment,
            case=case,
            revision=revision,
            artifacts=artifacts,
        )


def collect_results(output_root: Path) -> list[dict]:
    rows = []
    for experiment in ("exp1_estimation", "exp2_control_comparison", "exp3_robustness"):
        for case in _cases(experiment):
            summary_path = _case_output(output_root, experiment, case) / "summary.json"
            if summary_path.exists():
                row = json.loads(summary_path.read_text())
                settling_time = row.get("force_settling_time_s")
                if settling_time is not None and not np.isfinite(settling_time):
                    row.pop("force_settling_time_s")
                    write_json(summary_path, row)
                _ensure_case_artifacts(summary_path.parent, experiment, case, row)
                rows.append(row)
    write_metrics_csv(output_root / "metrics.csv", rows)
    if rows:
        aggregate_rows = aggregate(rows, ["experiment", "profile", "mode", "force_ratio", "noise_multiplier"])
        write_metrics_csv(output_root / "aggregate_metrics.csv", aggregate_rows)
        write_json(output_root / "summary.json", aggregate_rows)
    return rows


@click.command(context_settings={"show_default": True})
@click.option(
    "--experiment",
    type=click.Choice(("exp1_estimation", "exp2_control_comparison", "exp3_robustness", "all")),
    default="all",
)
@click.option("--seed", type=int, help="Run only one seed from the selected experiment.")
@click.option("--output", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_OUTPUT)
@click.option("--force", is_flag=True, help="Recompute cases with existing summaries.")
def main(experiment: str, seed: int | None, output: Path, force: bool) -> None:
    """Run external-wrench paper experiments."""
    experiments = (
        ("exp1_estimation", "exp2_control_comparison", "exp3_robustness") if experiment == "all" else (experiment,)
    )
    for experiment in experiments:
        cases = _cases(experiment)
        if seed is not None:
            cases = [case for case in cases if case["seed"] == seed]
        for index, case in enumerate(cases, start=1):
            click.echo(f"[{experiment} {index}/{len(cases)}] {case}")
            run_case(experiment, deepcopy(case), output, force=force)
    collect_results(output)


if __name__ == "__main__":
    main()
