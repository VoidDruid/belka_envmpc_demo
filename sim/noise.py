"""3D noise generators: biased odometry noise and additive engine noise."""

import numpy as np

from belka.common import (
    InertialOdom3D,
    RobotState3D,
    ThrusterForces,
    VisualOdom3D,
    quat_multiply,
    quat_normalize,
)


def make_visual_noise(
    *,
    position_bias_m: float = 0.020,
    position_slow_amplitude_m: float = 0.032,
    position_fast_amplitude_m: float = 0.005,
    attitude_bias_rad: float = np.deg2rad(2.2),
    attitude_slow_amplitude_rad: float = np.deg2rad(2.0),
    attitude_fast_amplitude_rad: float = np.deg2rad(0.3),
    position_slow_freq: float = 0.013,
    position_fast_freq: float = 0.09,
    attitude_slow_freq: float = 0.011,
    attitude_fast_freq: float = 0.07,
):
    """Create visual-odometry noise for position and attitude measurements."""
    position_direction = _random_unit_vector()  # направление position offset
    attitude_axis = _random_unit_vector()  # ось attitude offset
    position_phase = np.random.uniform(
        0.0, 2.0 * np.pi, size=2
    )  # фазы position sinusoids
    attitude_phase = np.random.uniform(
        0.0, 2.0 * np.pi, size=2
    )  # фазы attitude sinusoids

    def visual_noise(t: float, odom: VisualOdom3D) -> VisualOdom3D:
        """Return one noisy visual odometry sample."""
        pos_mag = abs(
            position_bias_m
            + position_slow_amplitude_m
            * np.sin(2.0 * np.pi * position_slow_freq * t + position_phase[0])
            + position_fast_amplitude_m
            * np.sin(2.0 * np.pi * position_fast_freq * t + position_phase[1])
        )  # модуль position error
        angle = abs(
            attitude_bias_rad
            + attitude_slow_amplitude_rad
            * np.sin(2.0 * np.pi * attitude_slow_freq * t + attitude_phase[0])
            + attitude_fast_amplitude_rad
            * np.sin(2.0 * np.pi * attitude_fast_freq * t + attitude_phase[1])
        )  # модуль attitude error
        noisy_p = odom.p + position_direction * pos_mag
        noise_q = _rotation_noise_quat(attitude_axis, angle)
        noisy_q = quat_normalize(quat_multiply(odom.q, noise_q))
        return VisualOdom3D(p=noisy_p, q=noisy_q)

    visual_noise.summary = {
        "position_median_target_m": 0.02,
        "position_p90_target_m": 0.05,
        "attitude_median_target_rad": np.deg2rad(2.0),
        "attitude_p90_target_rad": np.deg2rad(4.0),
    }
    return visual_noise


def make_inertial_noise(
    *,
    attitude_bias_rad: float = np.deg2rad(0.4),
    attitude_slow_amplitude_rad: float = np.deg2rad(0.3),
    attitude_fast_amplitude_rad: float = np.deg2rad(0.1),
    attitude_slow_freq: float = 0.017,
    attitude_fast_freq: float = 0.11,
    omega_std: float = 0.03,
    accel_std: float = 0.04,
):
    """Create IMU-like noise for attitude, body gyro and body-frame linear acceleration."""
    attitude_axis = _random_unit_vector()  # ось inertial attitude offset
    omega_direction = _random_unit_vector()  # направление gyro bias
    accel_direction = _random_unit_vector()  # направление accelerometer bias
    attitude_phase = np.random.uniform(
        0.0, 2.0 * np.pi, size=2
    )  # фазы attitude sinusoids
    omega_phase = np.random.uniform(0.0, 2.0 * np.pi, size=2)  # фазы gyro sinusoids
    accel_phase = np.random.uniform(
        0.0, 2.0 * np.pi, size=2
    )  # фазы accelerometer sinusoids

    def inertial_noise(t: float, odom: InertialOdom3D) -> InertialOdom3D:
        """Return one noisy inertial odometry sample."""
        angle = abs(
            attitude_bias_rad
            + attitude_slow_amplitude_rad
            * np.sin(2.0 * np.pi * attitude_slow_freq * t + attitude_phase[0])
            + attitude_fast_amplitude_rad
            * np.sin(2.0 * np.pi * attitude_fast_freq * t + attitude_phase[1])
        )  # модуль inertial attitude error
        omega_bias = omega_direction * (
            omega_std
            + 0.6
            * omega_std
            * np.sin(2.0 * np.pi * attitude_slow_freq * t + omega_phase[0])
            + 0.2
            * omega_std
            * np.sin(2.0 * np.pi * attitude_fast_freq * t + omega_phase[1])
        )  # body-frame gyro bias+jitter tube
        accel_bias = accel_direction * (
            accel_std
            + 0.6
            * accel_std
            * np.sin(2.0 * np.pi * attitude_slow_freq * t + accel_phase[0])
            + 0.2
            * accel_std
            * np.sin(2.0 * np.pi * attitude_fast_freq * t + accel_phase[1])
        )  # body-frame accelerometer bias+jitter tube
        noise_q = _rotation_noise_quat(attitude_axis, angle)
        noisy_q = quat_normalize(quat_multiply(odom.q, noise_q))
        noisy_omega = (
            odom.omega + omega_bias + np.random.normal(0.0, 0.25 * omega_std, 3)
        )
        noisy_a_body = (
            odom.a_body + accel_bias + np.random.normal(0.0, 0.25 * accel_std, 3)
        )
        return InertialOdom3D(q=noisy_q, omega=noisy_omega, a_body=noisy_a_body)

    inertial_noise.summary = {
        "omega_std": float(omega_std),
        "accel_std": float(accel_std),
    }
    return inertial_noise


def make_engine_noise(std_fs: float, nu: int = 12):
    """Создать генератор additive Gaussian noise для 3D движителей."""

    def random_disturbances(t: float, state: RobotState3D) -> ThrusterForces:
        """Return one random 3D thruster disturbance sample."""
        noise = np.random.normal(0, std_fs, nu)
        return ThrusterForces(values=noise)

    random_disturbances.summary = {"std_per_thruster": float(std_fs), "nu": int(nu)}
    return random_disturbances


def _random_unit_vector() -> np.ndarray:
    """Draw a random 3D unit vector."""
    v = np.random.randn(3)  # ненормированный случайный вектор
    return v / (float(np.linalg.norm(v)) + 1e-12)


def _rotation_noise_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert a small axis-angle sensor error into quaternion [w,x,y,z]."""
    half = 0.5 * float(angle)  # половина угла для quaternion
    return np.array(
        [
            np.cos(half),
            np.sin(half) * axis[0],
            np.sin(half) * axis[1],
            np.sin(half) * axis[2],
        ],
        dtype=float,
    )
