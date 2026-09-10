"""3D external force generators: force and moment in world-frame."""

import numpy as np

from belka.common import RobotParams3D, RobotState3D


AXIS_NAMES = ("x", "y", "z")


def make_random_wind_field_3d(
    params: RobotParams3D,
    *,
    mean_freq: float = 0.004,
    jitter_freq: float = 0.35,
    force_ratio: float = 0.25,
    secondary_ratio: float = 0.7,
    tertiary_ratio: float = 0.3,
    moment_lever_scale: float = 0.5,
    wave_amplitude_ratio: float = 0.25,
    jitter_tube_ratio: float = 0.12,
):
    """Create wind as baseline wrench with a very slow wave and bounded jitter tube."""
    if not 0.0 <= wave_amplitude_ratio < 0.5:
        raise ValueError("wave_amplitude_ratio must be in [0, 0.5)")
    primary_force = float(
        force_ratio * params.max_thruster_force_n
    )  # основная внешняя сила
    ranked_force_magnitudes = np.array(
        [
            primary_force,
            secondary_ratio * primary_force,
            tertiary_ratio * primary_force,
        ],
        dtype=float,
    )

    torque_arms = np.linalg.norm(
        np.cross(params.thruster_positions_body, params.thruster_directions_body),
        axis=1,
    )  # плечи движителей относительно body origin
    effective_lever = moment_lever_scale * float(
        np.max(torque_arms)
    )  # уменьшенное плечо внешнего момента
    ranked_moment_magnitudes = ranked_force_magnitudes * effective_lever

    axes = np.random.permutation(3)  # порядок world axes: primary, secondary, tertiary
    signs_f = np.random.choice((-1.0, 1.0), size=3)  # знаки компонент силы
    signs_m = np.random.choice((-1.0, 1.0), size=3)  # знаки компонент момента
    mean_force = np.zeros(3, dtype=float)  # nominal world-frame сила
    mean_moment = np.zeros(3, dtype=float)  # nominal world-frame момент
    force_magnitudes = np.zeros(3, dtype=float)  # magnitudes indexed by world axis
    moment_magnitudes = np.zeros(
        3, dtype=float
    )  # moment magnitudes indexed by world axis
    for rank, axis in enumerate(axes):
        mean_force[axis] = signs_f[rank] * ranked_force_magnitudes[rank]
        mean_moment[axis] = signs_m[rank] * ranked_moment_magnitudes[rank]
        force_magnitudes[axis] = ranked_force_magnitudes[rank]
        moment_magnitudes[axis] = ranked_moment_magnitudes[rank]
    mean_phase = np.random.uniform(0.0, 2.0 * np.pi)  # фаза общего very-slow multiplier
    force_jitter_phase = np.random.uniform(
        0.0, 2.0 * np.pi, size=3
    )  # фазы force jitter
    moment_jitter_phase = np.random.uniform(
        0.0, 2.0 * np.pi, size=3
    )  # фазы moment jitter
    jitter_freq_scale = np.array(
        [1.0, 1.37, 0.73], dtype=float
    )  # разные быстрые частоты по осям

    def wind_field(t: float, state: RobotState3D) -> np.ndarray:
        """Return world-frame external wrench [Fx, Fy, Fz, Mx, My, Mz]."""
        del state
        wave_scale = 1.0 + wave_amplitude_ratio * np.sin(
            2.0 * np.pi * mean_freq * t + mean_phase
        )  # slow baseline multiplier
        jitter_angle = (
            2.0 * np.pi * jitter_freq * jitter_freq_scale * t
        )  # per-axis jitter phase
        F = wave_scale * mean_force + jitter_tube_ratio * force_magnitudes * np.sin(
            jitter_angle + force_jitter_phase
        )  # world-frame force
        M = wave_scale * mean_moment + jitter_tube_ratio * moment_magnitudes * np.sin(
            jitter_angle + moment_jitter_phase
        )  # world-frame moment
        return np.concatenate([F, M])

    wind_field.summary = {
        "primary_axis": AXIS_NAMES[int(axes[0])],
        "secondary_axis": AXIS_NAMES[int(axes[1])],
        "tertiary_axis": AXIS_NAMES[int(axes[2])],
        "ranked_force_magnitudes": ranked_force_magnitudes,
        "ranked_moment_magnitudes": ranked_moment_magnitudes,
        "baseline_force": mean_force,
        "baseline_moment": mean_moment,
        "effective_lever": effective_lever,
        "mean_freq": float(mean_freq),
        "jitter_freq": float(jitter_freq),
        "wave_amplitude_ratio": float(wave_amplitude_ratio),
        "jitter_tube_ratio": float(jitter_tube_ratio),
    }
    return wind_field
