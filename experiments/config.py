"""Typed JSON configuration for 3D payload mobility scenarios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from belka.common import CargoParams, RobotParams3D
from belka.numeric_config import (
    parse_bool,
    parse_int,
    parse_numeric,
    parse_numeric_vector,
    require_exact_keys,
)


def _validate_rates(full_odom_hz: float, highfreq_odom_hz: float) -> None:
    """Validate simulator-owned odometry rates."""
    for field_name, value in [
        ("full_odom_hz", full_odom_hz),
        ("highfreq_odom_hz", highfreq_odom_hz),
    ]:
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"run.{field_name} must be finite and positive")
    ratio = (
        highfreq_odom_hz / full_odom_hz
    )  # high-rate samples per full odometry sample
    if highfreq_odom_hz + 1e-12 < full_odom_hz or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            "run.highfreq_odom_hz must be an integer multiple of full_odom_hz"
        )


@dataclass(frozen=True)
class RunConfig3D:
    """Simulation timing and endpoint target fields from a 3D scenario JSON."""

    initial_p: np.ndarray
    initial_angles: np.ndarray
    target_p: np.ndarray
    target_angles: np.ndarray
    trajectory_t: float
    hold_time: float
    t_max: float
    full_odom_hz: float
    highfreq_odom_hz: float
    seed: int

    @classmethod
    def from_mapping(cls, cfg: dict) -> "RunConfig3D":
        """Parse a top-level run block into numeric fields."""
        require_exact_keys(
            cfg,
            {
                "initial_p",
                "initial_angles",
                "target_p",
                "target_angles",
                "trajectory_t",
                "hold_time",
                "t_max",
                "full_odom_hz",
                "highfreq_odom_hz",
                "seed",
            },
            field_name="run",
        )
        full_odom_hz = parse_numeric(
            cfg["full_odom_hz"], field_name="run.full_odom_hz"
        )  # visual odometry rate
        highfreq_odom_hz = parse_numeric(
            cfg["highfreq_odom_hz"], field_name="run.highfreq_odom_hz"
        )  # IMU rate
        _validate_rates(full_odom_hz, highfreq_odom_hz)
        vectors = {
            name: np.asarray(
                parse_numeric_vector(cfg[name], field_name=f"run.{name}"),
                dtype=float,
            )
            for name in ("initial_p", "initial_angles", "target_p", "target_angles")
        }
        if any(
            value.shape != (3,) or np.any(~np.isfinite(value))
            for value in vectors.values()
        ):
            raise ValueError("run state vectors must each contain three finite values")
        trajectory_t = parse_numeric(cfg["trajectory_t"], field_name="run.trajectory_t")
        hold_time = parse_numeric(cfg["hold_time"], field_name="run.hold_time")
        t_max = parse_numeric(cfg["t_max"], field_name="run.t_max")
        if any(
            not np.isfinite(value) or value <= 0.0 for value in (trajectory_t, t_max)
        ):
            raise ValueError(
                "run.trajectory_t and run.t_max must be finite and positive"
            )
        if not np.isfinite(hold_time) or hold_time < 0.0:
            raise ValueError("run.hold_time must be finite and non-negative")
        if t_max < trajectory_t:
            raise ValueError("run.t_max must not be shorter than run.trajectory_t")
        return cls(
            initial_p=vectors["initial_p"],
            initial_angles=vectors["initial_angles"],
            target_p=vectors["target_p"],
            target_angles=vectors["target_angles"],
            trajectory_t=trajectory_t,
            hold_time=hold_time,
            t_max=t_max,
            full_odom_hz=full_odom_hz,
            highfreq_odom_hz=highfreq_odom_hz,
            seed=parse_int(cfg["seed"], field_name="run.seed"),
        )


@dataclass(frozen=True)
class VisualizationConfig3D:
    """Viewer overlay flags owned by the simulator and trajectory marker layer."""

    show_perceived_position: bool
    show_front_arrows: bool

    @classmethod
    def from_mapping(cls, cfg: dict) -> "VisualizationConfig3D":
        """Parse top-level visualization flags."""
        require_exact_keys(
            cfg,
            {"show_perceived_position", "show_front_arrows"},
            field_name="visualization",
        )
        return cls(
            show_perceived_position=parse_bool(
                cfg["show_perceived_position"],
                field_name="visualization.show_perceived_position",
            ),
            show_front_arrows=parse_bool(
                cfg["show_front_arrows"],
                field_name="visualization.show_front_arrows",
            ),
        )


@dataclass(frozen=True)
class CargoEstimationRunConfig3D:
    """Top-level switch and simulated heavy-fit duration."""

    enabled: bool
    simulated_fit_duration_s: float

    @classmethod
    def from_mapping(cls, cfg: dict) -> "CargoEstimationRunConfig3D":
        """Parse cargo-estimation config from JSON."""
        require_exact_keys(
            cfg,
            {"enabled", "simulated_fit_duration_s"},
            field_name="cargo_estimation",
        )
        simulated_fit_duration_s = parse_numeric(
            cfg["simulated_fit_duration_s"],
            field_name="cargo_estimation.simulated_fit_duration_s",
        )  # simulated long-fit duration
        if simulated_fit_duration_s < 0.0 or not np.isfinite(simulated_fit_duration_s):
            raise ValueError(
                "cargo_estimation.simulated_fit_duration_s must be finite and non-negative"
            )
        return cls(
            enabled=parse_bool(cfg["enabled"], field_name="cargo_estimation.enabled"),
            simulated_fit_duration_s=simulated_fit_duration_s,
        )


@dataclass(frozen=True)
class WindConfig3D:
    """Random external-wrench generator settings known only to the simulator."""

    mean_freq: float
    jitter_freq: float
    force_ratio: float
    secondary_ratio: float
    tertiary_ratio: float
    moment_lever_scale: float
    wave_amplitude_ratio: float
    jitter_tube_ratio: float

    @classmethod
    def from_mapping(cls, cfg: dict) -> "WindConfig3D":
        """Parse random wind field parameters."""
        require_exact_keys(
            cfg,
            {
                "mean_freq",
                "jitter_freq",
                "force_ratio",
                "secondary_ratio",
                "tertiary_ratio",
                "moment_lever_scale",
                "wave_amplitude_ratio",
                "jitter_tube_ratio",
            },
            field_name="wind",
        )
        values = cls(
            mean_freq=parse_numeric(cfg["mean_freq"], field_name="wind.mean_freq"),
            jitter_freq=parse_numeric(
                cfg["jitter_freq"], field_name="wind.jitter_freq"
            ),
            force_ratio=parse_numeric(
                cfg["force_ratio"], field_name="wind.force_ratio"
            ),
            secondary_ratio=parse_numeric(
                cfg["secondary_ratio"], field_name="wind.secondary_ratio"
            ),
            tertiary_ratio=parse_numeric(
                cfg["tertiary_ratio"], field_name="wind.tertiary_ratio"
            ),
            moment_lever_scale=parse_numeric(
                cfg["moment_lever_scale"], field_name="wind.moment_lever_scale"
            ),
            wave_amplitude_ratio=parse_numeric(
                cfg["wave_amplitude_ratio"], field_name="wind.wave_amplitude_ratio"
            ),
            jitter_tube_ratio=parse_numeric(
                cfg["jitter_tube_ratio"], field_name="wind.jitter_tube_ratio"
            ),
        )
        if any(value < 0.0 for value in values.__dict__.values()):
            raise ValueError("wind values must be non-negative")
        return values


@dataclass(frozen=True)
class SensorNoiseConfig3D:
    """Biased 3D sensor-noise generator settings known to the simulator."""

    position_bias_m: float
    position_slow_amplitude_m: float
    position_fast_amplitude_m: float
    attitude_bias_rad: float
    attitude_slow_amplitude_rad: float
    attitude_fast_amplitude_rad: float
    position_slow_freq: float
    position_fast_freq: float
    attitude_slow_freq: float
    attitude_fast_freq: float
    omega_std: float
    accel_std: float

    @classmethod
    def from_mapping(cls, cfg: dict) -> "SensorNoiseConfig3D":
        """Parse biased sensor-noise parameters."""
        require_exact_keys(cfg, cls.__dataclass_fields__, field_name="sensor_noise")
        values = cls(
            position_bias_m=parse_numeric(
                cfg["position_bias_m"], field_name="sensor_noise.position_bias_m"
            ),
            position_slow_amplitude_m=parse_numeric(
                cfg["position_slow_amplitude_m"],
                field_name="sensor_noise.position_slow_amplitude_m",
            ),
            position_fast_amplitude_m=parse_numeric(
                cfg["position_fast_amplitude_m"],
                field_name="sensor_noise.position_fast_amplitude_m",
            ),
            attitude_bias_rad=parse_numeric(
                cfg["attitude_bias_rad"], field_name="sensor_noise.attitude_bias_rad"
            ),
            attitude_slow_amplitude_rad=parse_numeric(
                cfg["attitude_slow_amplitude_rad"],
                field_name="sensor_noise.attitude_slow_amplitude_rad",
            ),
            attitude_fast_amplitude_rad=parse_numeric(
                cfg["attitude_fast_amplitude_rad"],
                field_name="sensor_noise.attitude_fast_amplitude_rad",
            ),
            position_slow_freq=parse_numeric(
                cfg["position_slow_freq"], field_name="sensor_noise.position_slow_freq"
            ),
            position_fast_freq=parse_numeric(
                cfg["position_fast_freq"], field_name="sensor_noise.position_fast_freq"
            ),
            attitude_slow_freq=parse_numeric(
                cfg["attitude_slow_freq"], field_name="sensor_noise.attitude_slow_freq"
            ),
            attitude_fast_freq=parse_numeric(
                cfg["attitude_fast_freq"], field_name="sensor_noise.attitude_fast_freq"
            ),
            omega_std=parse_numeric(
                cfg["omega_std"], field_name="sensor_noise.omega_std"
            ),
            accel_std=parse_numeric(
                cfg["accel_std"], field_name="sensor_noise.accel_std"
            ),
        )
        if any(value < 0.0 for value in values.__dict__.values()):
            raise ValueError("sensor_noise values must be non-negative")
        return values


@dataclass(frozen=True)
class EngineNoiseConfig3D:
    """Engine-noise settings expressed relative to current robot max thrust."""

    std_ratio: float

    @classmethod
    def from_mapping(cls, cfg: dict) -> "EngineNoiseConfig3D":
        """Parse per-thruster additive noise scale."""
        require_exact_keys(cfg, {"std_ratio"}, field_name="engine_noise")
        std_ratio = parse_numeric(
            cfg["std_ratio"], field_name="engine_noise.std_ratio"
        )  # std / max_force
        if std_ratio < 0.0 or not np.isfinite(std_ratio):
            raise ValueError("engine_noise.std_ratio must be finite and non-negative")
        return cls(std_ratio=std_ratio)

    def std(self, params: RobotParams3D) -> float:
        """Return absolute per-thruster noise std for active robot params."""
        return float(self.std_ratio * params.max_thruster_force_n)


@dataclass(frozen=True)
class Scenario3DConfig:
    """Complete typed 3D scenario config plus raw JSON for artifacts."""

    path: Path
    raw: dict
    name: str
    description: str
    robot_params: RobotParams3D
    cargo_params: CargoParams
    run: RunConfig3D
    visualization: VisualizationConfig3D
    cargo_estimation: CargoEstimationRunConfig3D
    wind: WindConfig3D
    sensor_noise: SensorNoiseConfig3D
    engine_noise: EngineNoiseConfig3D

    @classmethod
    def load(cls, path: str | Path) -> "Scenario3DConfig":
        """Load a scenario JSON and parse all typed config blocks."""
        path = Path(path)
        with open(path) as f:
            raw = json.load(f)
        require_exact_keys(
            raw,
            {
                "name",
                "description",
                "robot",
                "cargo",
                "run",
                "visualization",
                "cargo_estimation",
                "wind",
                "sensor_noise",
                "engine_noise",
            },
            field_name="scenario",
        )
        if not isinstance(raw["name"], str) or not raw["name"].strip():
            raise ValueError("scenario.name must be a non-empty string")
        if not isinstance(raw["description"], str):
            raise TypeError("scenario.description must be a string")
        return cls(
            path=path,
            raw=raw,
            name=str(raw["name"]),
            description=str(raw["description"]),
            robot_params=RobotParams3D.from_json(path),
            cargo_params=CargoParams.from_json(path),
            run=RunConfig3D.from_mapping(raw["run"]),
            visualization=VisualizationConfig3D.from_mapping(raw["visualization"]),
            cargo_estimation=CargoEstimationRunConfig3D.from_mapping(
                raw["cargo_estimation"]
            ),
            wind=WindConfig3D.from_mapping(raw["wind"]),
            sensor_noise=SensorNoiseConfig3D.from_mapping(raw["sensor_noise"]),
            engine_noise=EngineNoiseConfig3D.from_mapping(raw["engine_noise"]),
        )
