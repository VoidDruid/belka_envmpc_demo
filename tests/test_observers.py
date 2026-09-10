import json
from pathlib import Path

import numpy as np
import pytest

from belka.common import (
    InertialOdom3D,
    OdomInput3D,
    RobotParams3D,
    RobotState3D,
    ThrusterForces,
    ValveOpenings,
    VisualOdom3D,
    quat_distance,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_rot_matrix,
)
from belka.observers import (
    KalmanEstimator3DParams,
    KalmanObserver3D,
)
from belka.actuation import valve_openings_to_forces


pytestmark = pytest.mark.unit
JSON_3D = Path(__file__).resolve().parents[1] / "sim" / "models" / "belka3d.json"


def _params_json(tmp_path: Path, **robot_updates) -> Path:
    """Create a test-local 3D robot JSON with explicit robot-block edits."""
    cfg = json.loads(JSON_3D.read_text())
    cfg["robot"].update(robot_updates)
    path = tmp_path / "belka3d.json"
    path.write_text(json.dumps(cfg))
    return path


def _rotation_vector_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Map a small rotation vector to a unit quaternion for reset finite differences."""
    angle = float(np.linalg.norm(rotation))  # rotation angle [rad]
    if angle < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotation / angle  # unit rotation axis
    return np.concatenate(([np.cos(0.5 * angle)], axis * np.sin(0.5 * angle)))


def _quaternion_rotation_vector(q: np.ndarray) -> np.ndarray:
    """Map a near-identity quaternion to its shortest rotation vector."""
    q = quat_normalize(q)  # relative attitude quaternion
    if q[0] < 0.0:
        q = -q
    vector_norm = float(np.linalg.norm(q[1:4]))  # norm of quaternion vector part
    if vector_norm < 1e-15:
        return np.zeros(3, dtype=float)
    angle = 2.0 * np.arctan2(vector_norm, float(q[0]))  # shortest angle [rad]
    return q[1:4] * (angle / vector_norm)


def test_kalman_observer_3d_recovers_constant_world_wrench():
    params = RobotParams3D.from_json(JSON_3D)
    observer = KalmanObserver3D(
        KalmanEstimator3DParams(
            robot_params=params,
            initial_force_std=1.0,
            initial_moment_std=1.0,
            process_force_std=1e-3,
            process_moment_std=1e-3,
            measurement_visual_position_std=0.02,
            measurement_visual_orientation_std=0.02,
            measurement_inertial_orientation_std=0.02,
            measurement_accel_std=0.01,
            measurement_omega_std=0.01,
        )
    )
    zero_openings = ValveOpenings.zeros(params.nu)
    F_world = np.array([0.12, -0.08, 0.05], dtype=float)
    M_world = np.array([0.0, 0.0, 0.03], dtype=float)
    a_world = F_world / params.m  # world-frame линейное ускорение
    alpha_body = np.linalg.inv(params.I_body) @ M_world  # body-frame угловое ускорение
    dt = 0.05

    for i in range(100):
        t = i * dt  # absolute simulation time
        yaw = 0.5 * alpha_body[2] * t**2  # yaw при постоянном Mz
        q = np.array(
            [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=float
        )  # attitude quaternion
        state = RobotState3D(
            p=0.5 * a_world * t**2,
            q=q,
            v=a_world * t,
            omega=alpha_body * t,
            a=a_world,
            alpha=alpha_body,
        )
        _observed, estimate = observer.process_state(
            t,
            0.0 if i == 0 else dt,
            state,
            zero_openings,
        )

    estimate_array = estimate.to_array()
    np.testing.assert_allclose(estimate_array[0:3], F_world, atol=0.02)
    np.testing.assert_allclose(estimate_array[3:6], M_world, atol=0.01)
    np.testing.assert_allclose(observer.P, observer.P.T, atol=1e-12)
    assert np.linalg.eigvalsh(observer.P).min() >= -1e-12


def test_kalman_observer_3d_covariance_reset_jacobian_matches_right_injection():
    dtheta = np.array([1.2e-4, -0.8e-4, 0.5e-4])
    analytic = KalmanObserver3D._covariance_reset_jacobian(dtheta)[3:6, 3:6]
    eps = 1e-7
    numeric = np.empty((3, 3), dtype=float)
    q_injected_inverse = quat_conjugate(_rotation_vector_quaternion(dtheta))
    for axis in range(3):
        step = np.zeros(3, dtype=float)
        step[axis] = eps
        q_plus = quat_multiply(
            q_injected_inverse,
            _rotation_vector_quaternion(dtheta + step),
        )
        q_minus = quat_multiply(
            q_injected_inverse,
            _rotation_vector_quaternion(dtheta - step),
        )
        numeric[:, axis] = (
            _quaternion_rotation_vector(q_plus)
            - _quaternion_rotation_vector(q_minus)
        ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=2e-8)


def test_kalman_observer_3d_covariance_reset_preserves_psd_and_zero_is_identity():
    np.testing.assert_allclose(
        KalmanObserver3D._covariance_reset_jacobian(np.zeros(3)), np.eye(18)
    )
    rng = np.random.default_rng(42)
    factor = rng.normal(size=(18, 18))
    corrected = factor @ factor.T
    G = KalmanObserver3D._covariance_reset_jacobian(
        np.array([0.03, -0.02, 0.01])
    )
    reset = G @ corrected @ G.T
    np.testing.assert_allclose(reset, reset.T, atol=1e-12)
    assert np.linalg.eigvalsh(reset).min() >= -1e-11


def test_kalman_observer_3d_process_jacobian_matches_error_state_finite_difference_with_custom_geometry(
    tmp_path,
):
    """Regression: analytic F must use RobotParams3D allocation matrix, including custom geometry."""
    params = RobotParams3D.from_json(
        _params_json(tmp_path, side_L=0.42, side_w=0.21, l=0.07)
    )
    observer = KalmanObserver3D(KalmanEstimator3DParams(robot_params=params))
    q = np.array(
        [0.94, 0.12, -0.25, 0.19], dtype=float
    )  # nominal quaternion before normalization
    z = np.concatenate(
        [
            np.array([0.3, -0.2, 0.15], dtype=float),
            q / np.linalg.norm(q),
            np.array([0.08, -0.03, 0.05], dtype=float),
            np.array([0.07, -0.04, 0.05], dtype=float),
            np.array([0.11, -0.09, 0.04], dtype=float),
            np.array([0.03, 0.02, -0.025], dtype=float),
        ]
    )
    u = ThrusterForces.from_array(
        np.linspace(0.05, 0.55, params.nu)
    )  # nonuniform thruster vector
    dt = 0.037  # prediction step
    eps = 1e-6  # finite-difference perturbation

    analytic = observer._process_jacobian(z, dt, u)
    nominal_next = observer._process_model(z, dt, u)  # nominal propagated state
    numeric = np.zeros((18, 18), dtype=float)  # finite-difference F matrix
    for idx in range(18):
        dx = np.zeros(18, dtype=float)  # one-axis error-state perturbation
        dx[idx] = eps
        plus_next = observer._process_model(observer._apply_error_state(z, dx), dt, u)
        minus_next = observer._process_model(observer._apply_error_state(z, -dx), dt, u)
        err_plus = _error_state_between(
            observer, nominal_next, plus_next
        )  # propagated +eps error
        err_minus = _error_state_between(
            observer, nominal_next, minus_next
        )  # propagated -eps error
        numeric[:, idx] = (err_plus - err_minus) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=5e-5)


def test_kalman_observer_3d_measurement_jacobian_matches_error_state_finite_difference_with_custom_geometry(
    tmp_path,
):
    """Regression: analytic H must match pose/accel/gyro measurement perturbations."""
    params = RobotParams3D.from_json(
        _params_json(tmp_path, side_L=0.41, side_w=0.205, l=0.06)
    )
    observer = KalmanObserver3D(KalmanEstimator3DParams(robot_params=params))
    q = np.array(
        [0.97, -0.08, 0.15, 0.17], dtype=float
    )  # nominal quaternion before normalization
    z = np.concatenate(
        [
            np.array([-0.1, 0.25, 0.05], dtype=float),
            q / np.linalg.norm(q),
            np.array([0.04, 0.02, -0.01], dtype=float),
            np.array([-0.02, 0.06, -0.03], dtype=float),
            np.array([0.09, 0.04, -0.07], dtype=float),
            np.array([-0.02, 0.015, 0.035], dtype=float),
        ]
    )
    u = ThrusterForces.from_array(
        np.linspace(0.02, 0.5, params.nu)
    )  # nonuniform thruster vector
    eps = 1e-6  # finite-difference perturbation

    observer.z = z.copy()
    analytic = observer._measurement_jacobian(z, u, has_visual=True)
    numeric = np.zeros((15, 18), dtype=float)  # finite-difference H matrix
    for idx in range(18):
        dx = np.zeros(18, dtype=float)  # one-axis error-state perturbation
        dx[idx] = eps
        plus_residual = _measurement_residual_from_z(
            observer, z, observer._apply_error_state(z, dx), u
        )
        minus_residual = _measurement_residual_from_z(
            observer, z, observer._apply_error_state(z, -dx), u
        )
        numeric[:, idx] = (plus_residual - minus_residual) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=5e-5)


def test_kalman_observer_3d_handles_antipodal_quaternion_measurement():
    params = RobotParams3D.from_json(JSON_3D)
    observer = KalmanObserver3D(KalmanEstimator3DParams(robot_params=params))
    q = np.array([-1.0, 0.0, 0.0, 0.0], dtype=float)  # antipodal identity quaternion
    state = RobotState3D(q=q)

    observed, estimate = observer.process_state(
        0.0, 0.0, state, ValveOpenings.zeros(params.nu)
    )

    assert quat_distance(observed.q, q) < 1e-12
    np.testing.assert_allclose(
        estimate.to_array(), np.zeros(6, dtype=float), atol=1e-12
    )


def test_kalman_observer_3d_initializes_external_wrench_from_preflight_estimate():
    params = RobotParams3D.from_json(JSON_3D)
    initial_wrench = np.array([0.1, -0.05, 0.02, 0.003, -0.004, 0.002], dtype=float)
    observer = KalmanObserver3D(
        KalmanEstimator3DParams(
            robot_params=params,
            initial_external_wrench=initial_wrench,
            initial_force_std=1e-9,
            initial_moment_std=1e-9,
        )
    )

    _observed, estimate = observer.process_state(
        0.0, 0.0, RobotState3D(), ValveOpenings.zeros(params.nu)
    )

    np.testing.assert_allclose(estimate.to_array(), initial_wrench, atol=1e-10)


def test_kalman_observer_3d_handoff_state_survives_inertial_only_tick():
    params = RobotParams3D.from_json(JSON_3D)
    observer = KalmanObserver3D(KalmanEstimator3DParams(robot_params=params))
    initial = RobotState3D(
        p=np.array([0.2, -0.1, 1.0], dtype=float),
        v=np.array([0.01, -0.02, 0.03], dtype=float),
    )
    observer.initialize(initial)
    inertial_only = OdomInput3D(
        visual=None,
        inertial=InertialOdom3D(
            q=initial.q.copy(),
            omega=initial.omega.copy(),
            a_body=np.zeros(3, dtype=float),
        ),
    )

    observed, _estimate = observer.process_state(
        1.0,
        0.0,
        inertial_only,
        ValveOpenings.zeros(params.nu),
    )

    np.testing.assert_allclose(observed.p, initial.p, atol=1e-12)
    np.testing.assert_allclose(observed.v, initial.v, atol=1e-12)


def test_kalman_observer_converts_actual_openings_with_controller_visible_model():
    params = RobotParams3D.from_json(JSON_3D)
    observer = KalmanObserver3D(KalmanEstimator3DParams(robot_params=params))
    observer.initialize(RobotState3D())
    openings = ValveOpenings.from_array(np.linspace(0.1, 0.9, params.nu))
    captured: list[ThrusterForces] = []

    observer._predict = lambda dt, forces: captured.append(forces)
    observer._update = lambda odom, forces: captured.append(forces)

    observer.process_state(0.1, 0.1, RobotState3D(), openings)

    expected = valve_openings_to_forces(openings, params)
    assert len(captured) == 2
    for forces in captured:
        np.testing.assert_allclose(forces.to_array(), expected.to_array(), atol=1e-12)


def _error_state_between(
    observer: KalmanObserver3D, reference_z: np.ndarray, actual_z: np.ndarray
) -> np.ndarray:
    """Return right-error-state vector that maps reference_z to actual_z."""
    error = np.zeros(18, dtype=float)  # [dp,dtheta,dv,domega,dF,dM]
    error[0:3] = actual_z[0:3] - reference_z[0:3]
    error[3:6] = observer._orientation_error(actual_z[3:7], reference_z[3:7])
    error[6:9] = actual_z[7:10] - reference_z[7:10]
    error[9:12] = actual_z[10:13] - reference_z[10:13]
    error[12:15] = actual_z[13:16] - reference_z[13:16]
    error[15:18] = actual_z[16:19] - reference_z[16:19]
    return error


def _measurement_residual_from_z(
    observer: KalmanObserver3D,
    nominal_z: np.ndarray,
    measurement_z: np.ndarray,
    u: ThrusterForces,
) -> np.ndarray:
    """Build a synthetic measurement from measurement_z and evaluate residual against nominal_z."""
    observer.z = nominal_z.copy()
    q = measurement_z[3:7]  # synthetic measurement attitude quaternion
    R_body_to_world = quat_to_rot_matrix(q)  # body->world rotation matrix
    a_world = observer._world_acceleration(
        measurement_z, u
    )  # synthetic world-frame acceleration
    measurement = OdomInput3D(
        visual=VisualOdom3D(p=measurement_z[0:3], q=q),
        inertial=InertialOdom3D(
            q=q,
            omega=measurement_z[10:13],
            a_body=R_body_to_world.T @ a_world,
        ),
    )
    return observer._measurement_residual(measurement, u)
