from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from belka.actuation import valve_openings_to_forces
from belka.common import (
    InertialOdom3D,
    OdomInput3D,
    ThrusterForces,
    ValveOpenings,
    RobotParams3D,
    RobotState3D,
    VisualOdom3D,
    quat_conjugate,
    quat_derivative,
    quat_multiply,
    quat_normalize,
    quat_to_rot_matrix,
)
from belka.model import body_origin_acceleration_3d
from belka.shared.observers import WrenchEstimate3D


@dataclass(frozen=True)
class KalmanEstimator3DParams:
    """Noise and initialization parameters for 6DoF EKF disturbance observer."""

    robot_params: RobotParams3D
    initial_external_wrench: np.ndarray | None = None
    initial_position_std: float = 0.05
    initial_orientation_std: float = 0.10
    initial_velocity_std: float = 0.20
    initial_omega_std: float = 0.10
    initial_force_std: float = 0.10
    initial_moment_std: float = 0.10
    process_position_std: float = 1e-4
    process_orientation_std: float = 1e-4
    process_velocity_std: float = 2e-3
    process_omega_std: float = 2e-3
    process_force_std: float = 2e-3
    process_moment_std: float = 2e-3
    measurement_visual_position_std: float = 0.03
    measurement_visual_orientation_std: float = 0.04
    measurement_inertial_orientation_std: float = 0.02
    measurement_accel_std: float = 0.04
    measurement_omega_std: float = 0.03


class KalmanObserver3D:
    """EKF observer estimating 6DoF state and constant world-frame external wrench."""

    def __init__(self, params: KalmanEstimator3DParams):
        """Store EKF parameters and initialize lazy state/covariance buffers."""
        self.params = params
        self.I_body = params.robot_params.I_body  # тензор инерции body-frame
        self.I_inv = np.linalg.inv(self.I_body)  # I_body⁻¹
        self.z: np.ndarray | None = (
            None  # EKF state [p(3), q(4), v(3), omega(3), F_ext(3), M_ext(3)]
        )
        self.P: np.ndarray | None = (
            None  # covariance error-state [dp,dtheta,dv,domega,dF,dM]
        )

    def process_state(
        self,
        t: float,
        dt: float,
        input_data: RobotState3D | OdomInput3D,
        actual_actuation: ValveOpenings,
    ) -> tuple[RobotState3D, WrenchEstimate3D]:
        """Run one EKF predict/update step and return estimated state plus external wrench."""
        del t
        odom = self._as_odom_input(input_data)  # typed visual/inertial odometry input
        if self.z is None or self.P is None:
            if odom.visual is None:
                q = quat_normalize(
                    odom.inertial.q
                )  # inertial attitude before visual EKF initialization
                R_body_to_world = quat_to_rot_matrix(q)  # body->world rotation matrix
                observed_state = RobotState3D(
                    q=q,
                    omega=odom.inertial.omega.copy(),
                    a=R_body_to_world @ odom.inertial.a_body,
                )
                return observed_state, WrenchEstimate3D.from_array(
                    self._initial_external_wrench()
                )
            self._initialize(odom)

        u = valve_openings_to_forces(
            actual_actuation, self.params.robot_params
        )  # фактические силы по controller-visible actuator model
        dt = max(float(dt), 0.0)
        if dt > 0.0:
            self._predict(dt, u)
        self._update(odom, u)

        q = quat_normalize(self.z[3:7])  # оцененный unit quaternion [w,x,y,z]
        R_body_to_world = quat_to_rot_matrix(q)  # body->world rotation matrix
        estimated_alpha = self._angular_acceleration(
            self.z, u
        )  # body-frame angular acceleration
        observed_state = RobotState3D(
            p=self.z[0:3].copy(),
            q=q,
            v=self.z[7:10].copy(),
            omega=self.z[10:13].copy(),
            a=R_body_to_world @ odom.inertial.a_body,
            alpha=estimated_alpha,
        )
        estimation = WrenchEstimate3D.from_array(self.z[13:19])
        return observed_state, estimation

    def initialize(self, state: RobotState3D) -> None:
        """Initialize the EKF from a control-visible state estimate.

        This is used at controller handoff, where an inertial-only tick may arrive
        before the next visual sample. The external wrench still comes from the
        configured preflight estimate.
        """
        p = np.asarray(state.p, dtype=float)  # initial world-frame position
        q = quat_normalize(
            np.asarray(state.q, dtype=float)
        )  # initial attitude quaternion
        v = np.asarray(state.v, dtype=float)  # initial world-frame velocity
        omega = np.asarray(
            state.omega, dtype=float
        )  # initial body-frame angular velocity
        external_wrench0 = (
            self._initial_external_wrench()
        )  # initial world-frame external wrench estimate
        if external_wrench0.shape != (6,):
            raise ValueError(
                f"initial_external_wrench must have shape (6,), got {external_wrench0.shape}"
            )
        self.z = np.concatenate([p, q, v, omega, external_wrench0])
        self.P = self._initial_covariance()

    def _initialize(self, odom: OdomInput3D) -> None:
        """Initialize EKF state from the first visual pose and inertial rate sample."""
        if odom.visual is None:
            raise ValueError(
                "KalmanObserver3D requires visual odometry for initialization"
            )
        self.initialize(
            RobotState3D(
                p=np.asarray(odom.visual.p, dtype=float),
                q=quat_normalize(np.asarray(odom.visual.q, dtype=float)),
                v=np.zeros(3, dtype=float),
                omega=np.asarray(odom.inertial.omega, dtype=float),
            )
        )

    def _initial_covariance(self) -> np.ndarray:
        """Return the configured initial covariance in the 18D error state."""
        p = self.params  # EKF covariance parameters
        std = np.array(
            [
                p.initial_position_std,
                p.initial_position_std,
                p.initial_position_std,
                p.initial_orientation_std,
                p.initial_orientation_std,
                p.initial_orientation_std,
                p.initial_velocity_std,
                p.initial_velocity_std,
                p.initial_velocity_std,
                p.initial_omega_std,
                p.initial_omega_std,
                p.initial_omega_std,
                p.initial_force_std,
                p.initial_force_std,
                p.initial_force_std,
                p.initial_moment_std,
                p.initial_moment_std,
                p.initial_moment_std,
            ],
            dtype=float,
        )
        return np.diag(std**2)

    def _predict(self, dt: float, u: ThrusterForces) -> None:
        """Predict augmented robot/disturbance state through the nonlinear 6DoF dynamics model."""
        F = self._process_jacobian(self.z, dt, u)  # error-state transition matrix
        Q = self._process_noise(dt)  # covariance process noise
        self.z = self._process_model(self.z, dt, u)
        self.P = F @ self.P @ F.T + Q
        self.z[3:7] = quat_normalize(self.z[3:7])

    def _update(self, odom: OdomInput3D, u: ThrusterForces) -> None:
        """Correct predicted state with optional visual pose and high-rate IMU measurements."""
        residual = self._measurement_residual(
            odom, u
        )  # innovation vector for available measurements
        H = self._measurement_jacobian(
            self.z, u, has_visual=odom.visual is not None
        )  # dh/d(error-state)
        R = self._measurement_noise(
            has_visual=odom.visual is not None
        )  # covariance measurement noise
        S = H @ self.P @ H.T + R  # covariance innovation-а
        K = np.linalg.solve(S.T, H @ self.P.T).T  # Kalman gain
        dx = K @ residual  # correction error-state [dp,dtheta,dv,domega,dF,dM]
        I = np.eye(
            self.P.shape[0], dtype=float
        )  # identity для Joseph covariance update
        P_corrected = (
            (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        )  # covariance до reset error-state
        self.z = self._apply_error_state(self.z, dx)
        reference_q = (
            odom.visual.q if odom.visual is not None else odom.inertial.q
        )  # latest measured attitude
        self.z[3:7] = self._align_quaternion(self.z[3:7], reference_q)
        G = self._covariance_reset_jacobian(dx[3:6])  # reset error-state Jacobian
        self.P = G @ P_corrected @ G.T
        self.P = 0.5 * (self.P + self.P.T)

    def _process_model(self, z: np.ndarray, dt: float, u: ThrusterForces) -> np.ndarray:
        """Propagate z = [robot state, external wrench] by one Euler step."""
        z = np.asarray(z, dtype=float)  # EKF state [p, q, v, omega, F_ext, M_ext]
        params = self.params.robot_params
        q = quat_normalize(z[3:7])  # unit quaternion текущей ориентации
        v = z[7:10]  # world-frame линейная скорость
        omega = z[10:13]  # body-frame угловая скорость
        wrench_body = u.to_wrench(params)  # body-frame wrench тяг [F_body, M_body]
        external_wrench = z[13:19]  # world-frame external wrench [F_ext, M_ext]
        a_world, alpha = body_origin_acceleration_3d(
            q, omega, wrench_body, external_wrench, params
        )

        next_z = np.array(z, dtype=float, copy=True)
        next_z[0:3] = z[0:3] + v * dt + 0.5 * a_world * dt**2
        next_z[3:7] = quat_normalize(q + quat_derivative(q, omega) * dt)
        next_z[7:10] = v + a_world * dt
        next_z[10:13] = omega + alpha * dt
        return next_z

    def _process_jacobian(
        self, z: np.ndarray, dt: float, u: ThrusterForces
    ) -> np.ndarray:
        """Build analytic 18×18 right-error-state Jacobian for one Euler prediction step.

        Error state is [dp, dtheta, dv, domega, dF_ext, dM_ext], while nominal
        orientation remains a unit quaternion. The active RobotParams3D allocation
        matrix is used through u.to_wrench(params), so geometry changes affect
        the body-origin wrench at the linearization point.
        """
        z = np.asarray(z, dtype=float)  # nominal EKF state [p,q,v,omega,F_ext,M_ext]
        q = quat_normalize(z[3:7])  # unit quaternion текущей ориентации
        omega = z[10:13]  # body-frame angular velocity
        (
            da_dtheta,
            da_domega,
            da_dforce,
            da_dmoment,
            dalpha_dtheta,
            dalpha_domega,
            dalpha_dmoment,
        ) = self._acceleration_jacobians(q, omega, z[13:19], u)

        E = np.eye(3, dtype=float)  # 3×3 identity
        F = np.eye(18, dtype=float)  # error-state transition matrix
        F[0:3, 3:6] = 0.5 * dt**2 * da_dtheta
        F[0:3, 6:9] = E * dt
        F[0:3, 9:12] = 0.5 * dt**2 * da_domega
        F[0:3, 12:15] = 0.5 * dt**2 * da_dforce
        F[0:3, 15:18] = 0.5 * dt**2 * da_dmoment
        F[3:6, 3:6] = E - self._skew(omega) * dt
        F[3:6, 9:12] = E * dt
        F[6:9, 3:6] = dt * da_dtheta
        F[6:9, 9:12] = dt * da_domega
        F[6:9, 12:15] = dt * da_dforce
        F[6:9, 15:18] = dt * da_dmoment
        F[9:12, 3:6] = dt * dalpha_dtheta
        F[9:12, 9:12] = E + dt * dalpha_domega
        F[9:12, 15:18] = dt * dalpha_dmoment
        return F

    def _measurement_residual(self, odom: OdomInput3D, u: ThrusterForces) -> np.ndarray:
        """Compute innovation for visual pose plus inertial attitude/omega/body acceleration."""
        q = quat_normalize(self.z[3:7])  # unit quaternion nominal state
        residuals = []  # stacked measurement residual blocks
        if odom.visual is not None:
            dtheta_visual = self._orientation_error(
                odom.visual.q, q
            )  # rotation vector q_nom⁻¹⊗q_visual
            residuals.extend(np.asarray(odom.visual.p, dtype=float) - self.z[0:3])
            residuals.extend(dtheta_visual)
        dtheta_inertial = self._orientation_error(
            odom.inertial.q, q
        )  # rotation vector q_nom⁻¹⊗q_imu
        a_world = self._world_acceleration(
            self.z, u
        )  # predicted world-frame acceleration
        R_body_to_world = quat_to_rot_matrix(q)  # body->world rotation matrix
        a_body = (
            R_body_to_world.T @ a_world
        )  # predicted body-frame accelerometer reading
        residuals.extend(dtheta_inertial)
        residuals.extend(np.asarray(odom.inertial.a_body, dtype=float) - a_body)
        residuals.extend(np.asarray(odom.inertial.omega, dtype=float) - self.z[10:13])
        return np.asarray(residuals, dtype=float)

    def _measurement_jacobian(
        self, z: np.ndarray, u: ThrusterForces, *, has_visual: bool
    ) -> np.ndarray:
        """Build analytic measurement Jacobian for visual pose and IMU body-frame channels."""
        params = self.params.robot_params
        q = quat_normalize(z[3:7])  # unit quaternion nominal state
        omega = z[10:13]  # body-frame angular velocity
        (
            da_dtheta,
            da_domega,
            da_dforce,
            da_dmoment,
            _dalpha_dtheta,
            _dalpha_domega,
            _dalpha_dmoment,
        ) = self._acceleration_jacobians(q, omega, z[13:19], u)

        E = np.eye(3, dtype=float)  # 3×3 identity
        rows = 15 if has_visual else 9  # measurement dimension
        H = np.zeros((rows, 18), dtype=float)  # measurement matrix по error-state
        row = 0  # current residual row offset
        if has_visual:
            H[row : row + 3, 0:3] = E
            row += 3
            H[row : row + 3, 3:6] = E
            row += 3
        H[row : row + 3, 3:6] = E
        row += 3

        a_world = self._world_acceleration(
            z, u
        )  # predicted world-frame origin acceleration
        R_body_to_world = quat_to_rot_matrix(q)  # body->world rotation matrix
        a_body = R_body_to_world.T @ a_world  # predicted body-frame acceleration
        H[row : row + 3, 3:6] = self._skew(a_body) + R_body_to_world.T @ da_dtheta
        H[row : row + 3, 9:12] = R_body_to_world.T @ da_domega
        H[row : row + 3, 12:15] = R_body_to_world.T @ (E / params.m)
        H[row : row + 3, 15:18] = R_body_to_world.T @ da_dmoment
        row += 3
        H[row : row + 3, 9:12] = E
        return H

    def _apply_error_state(self, z: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Inject additive translational errors and right-multiply quaternion by Exp(dtheta)."""
        next_z = np.array(z, dtype=float, copy=True)
        dtheta = dx[3:6]  # small body-frame rotation vector
        dq = self._rotation_vector_to_quat(dtheta)  # quaternion Exp(dtheta)
        next_z[0:3] += dx[0:3]
        next_z[3:7] = quat_normalize(quat_multiply(next_z[3:7], dq))
        next_z[7:10] += dx[6:9]
        next_z[10:13] += dx[9:12]
        next_z[13:16] += dx[12:15]
        next_z[16:19] += dx[15:18]
        return next_z

    @classmethod
    def _covariance_reset_jacobian(cls, dtheta: np.ndarray) -> np.ndarray:
        """Return the first-order covariance reset Jacobian for right attitude injection."""
        dtheta = np.asarray(dtheta, dtype=float)  # injected body-frame rotation error
        if dtheta.shape != (3,):
            raise ValueError(f"dtheta must have shape (3,), got {dtheta.shape}")
        G = np.eye(18, dtype=float)  # full error-state reset Jacobian
        G[3:6, 3:6] -= 0.5 * cls._skew(dtheta)
        return G

    def _acceleration_jacobians(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        external_wrench: np.ndarray,
        u: ThrusterForces,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return analytic acceleration Jacobian blocks for fixed-origin 3D rigid-body dynamics."""
        params = self.params.robot_params
        c = params.r_cm_body  # body-frame вектор от origin к CM
        R_body_to_world = quat_to_rot_matrix(q)  # матрица поворота body→world
        wrench_body = u.to_wrench(params)  # body-origin wrench тяг [F_body, M_origin]
        F_body = wrench_body[0:3]  # body-frame сила тяг
        M_ext_body = (
            R_body_to_world.T @ external_wrench[3:6]
        )  # body-frame внешний момент около CM
        I_omega = self.I_body @ omega  # I_body·omega
        gyro_jacobian = self._skew(omega) @ self.I_body - self._skew(
            I_omega
        )  # ∂(omega×Iomega)/∂omega

        E = np.eye(3, dtype=float)  # 3×3 identity
        dalpha_dtheta = self.I_inv @ self._skew(M_ext_body)  # ∂alpha/∂dtheta
        dalpha_domega = -self.I_inv @ gyro_jacobian  # ∂alpha/∂omega
        dalpha_dmoment = self.I_inv @ R_body_to_world.T  # ∂alpha/∂M_ext_world

        _a_world, alpha = body_origin_acceleration_3d(
            q, omega, wrench_body, external_wrench, params
        )
        offset_body = self._cross3(alpha, c) + self._cross3(
            omega, self._cross3(omega, c)
        )  # a_CM-origin body
        centripetal_jacobian = (
            (float(omega @ c) * E) + np.outer(omega, c) - 2.0 * np.outer(c, omega)
        )
        offset_domega = (
            -self._skew(c) @ dalpha_domega + centripetal_jacobian
        )  # ∂offset_body/∂omega

        da_dtheta = (
            -R_body_to_world @ self._skew(F_body) / params.m
            + R_body_to_world @ self._skew(offset_body)
            + R_body_to_world @ self._skew(c) @ dalpha_dtheta
        )  # ∂a_origin/∂dtheta
        da_domega = -R_body_to_world @ offset_domega  # ∂a_origin/∂omega
        da_dforce = E / params.m  # ∂a_origin/∂F_ext_world
        da_dmoment = (
            R_body_to_world @ self._skew(c) @ dalpha_dmoment
        )  # ∂a_origin/∂M_ext_world
        return (
            da_dtheta,
            da_domega,
            da_dforce,
            da_dmoment,
            dalpha_dtheta,
            dalpha_domega,
            dalpha_dmoment,
        )

    def _world_acceleration(self, z: np.ndarray, u: ThrusterForces) -> np.ndarray:
        """Compute world-frame linear acceleration from thrusters and external force estimate."""
        params = self.params.robot_params
        q = quat_normalize(z[3:7])  # unit quaternion body→world
        omega = z[10:13]  # body-frame angular velocity
        wrench_body = u.to_wrench(params)  # body-frame wrench тяг [F_body, M_body]
        external_wrench = z[13:19]  # world-frame external wrench [F_ext, M_ext]
        a_world, _alpha = body_origin_acceleration_3d(
            q, omega, wrench_body, external_wrench, params
        )
        return a_world

    def _angular_acceleration(self, z: np.ndarray, u: ThrusterForces) -> np.ndarray:
        """Compute body-frame angular acceleration from thrust and external moment estimate."""
        params = self.params.robot_params
        q = quat_normalize(z[3:7])  # unit quaternion body→world
        omega = z[10:13]  # body-frame угловая скорость
        wrench_body = u.to_wrench(params)  # body-frame wrench тяг [F_body, M_body]
        external_wrench = z[13:19]  # world-frame external wrench [F_ext, M_ext]
        _a_world, alpha = body_origin_acceleration_3d(
            q, omega, wrench_body, external_wrench, params
        )
        return alpha

    def _process_noise(self, dt: float) -> np.ndarray:
        """Build diagonal process covariance for one prediction step."""
        p = self.params  # параметры EKF
        scale = max(dt, 1e-9)
        std = np.array(
            [
                p.process_position_std,
                p.process_position_std,
                p.process_position_std,
                p.process_orientation_std,
                p.process_orientation_std,
                p.process_orientation_std,
                p.process_velocity_std,
                p.process_velocity_std,
                p.process_velocity_std,
                p.process_omega_std,
                p.process_omega_std,
                p.process_omega_std,
                p.process_force_std,
                p.process_force_std,
                p.process_force_std,
                p.process_moment_std,
                p.process_moment_std,
                p.process_moment_std,
            ],
            dtype=float,
        )
        return np.diag((std * scale) ** 2)

    def _measurement_noise(self, *, has_visual: bool) -> np.ndarray:
        """Build diagonal measurement covariance for available visual and inertial channels."""
        p = self.params  # параметры EKF
        std_values = []  # standard deviations in residual order
        if has_visual:
            std_values.extend([p.measurement_visual_position_std] * 3)
            std_values.extend([p.measurement_visual_orientation_std] * 3)
        std_values.extend([p.measurement_inertial_orientation_std] * 3)
        std_values.extend([p.measurement_accel_std] * 3)
        std_values.extend([p.measurement_omega_std] * 3)
        std = np.asarray(std_values, dtype=float)  # measurement std vector
        return np.diag(std**2)

    def _initial_external_wrench(self) -> np.ndarray:
        """Return configured initial external wrench estimate as a 6D vector."""
        p = self.params  # параметры EKF
        return (
            np.zeros(6, dtype=float)
            if p.initial_external_wrench is None
            else np.asarray(p.initial_external_wrench, dtype=float)
        )

    @staticmethod
    def _as_odom_input(input_data: RobotState3D | OdomInput3D) -> OdomInput3D:
        """Convert full 3D state samples into typed visual+inertial odometry for EKF tests/replay."""
        if isinstance(input_data, OdomInput3D):
            return input_data
        R_body_to_world = quat_to_rot_matrix(
            input_data.q
        )  # body->world rotation matrix
        return OdomInput3D(
            visual=VisualOdom3D(p=input_data.p.copy(), q=input_data.q.copy()),
            inertial=InertialOdom3D(
                q=input_data.q.copy(),
                omega=input_data.omega.copy(),
                a_body=R_body_to_world.T @ input_data.a,
            ),
        )

    @staticmethod
    def _orientation_error(q_est: np.ndarray, q_meas: np.ndarray) -> np.ndarray:
        """Return shortest rotation-vector error q_meas⁻¹ ⊗ q_est."""
        q_est = quat_normalize(
            np.asarray(q_est, dtype=float)
        )  # estimated unit quaternion
        q_meas = quat_normalize(
            np.asarray(q_meas, dtype=float)
        )  # measured unit quaternion
        q_err = quat_multiply(
            quat_conjugate(q_meas), q_est
        )  # relative quaternion q_meas⁻¹⊗q_est
        if q_err[0] < 0.0:
            q_err = -q_err
        vector_norm = float(
            np.linalg.norm(q_err[1:4])
        )  # norm векторной части quaternion error
        if vector_norm < 1e-12:
            return np.zeros(3, dtype=float)
        angle = 2.0 * np.arctan2(
            vector_norm, np.clip(float(q_err[0]), -1.0, 1.0)
        )  # угол rotation-vector
        return q_err[1:4] * (angle / vector_norm)

    @staticmethod
    def _align_quaternion(q: np.ndarray, reference_q: np.ndarray) -> np.ndarray:
        """Normalize q and choose the same S³ hemisphere as reference_q."""
        q = quat_normalize(np.asarray(q, dtype=float))  # unit quaternion после update
        reference_q = quat_normalize(
            np.asarray(reference_q, dtype=float)
        )  # reference unit quaternion
        if float(np.dot(q, reference_q)) < 0.0:
            q = -q
        return q

    @staticmethod
    def _rotation_vector_to_quat(theta: np.ndarray) -> np.ndarray:
        """Convert rotation vector theta to quaternion Exp(theta)."""
        theta = np.asarray(theta, dtype=float)  # rotation vector
        angle = float(np.linalg.norm(theta))  # rotation angle
        if angle < 1e-12:
            return quat_normalize(
                np.array(
                    [1.0, 0.5 * theta[0], 0.5 * theta[1], 0.5 * theta[2]], dtype=float
                )
            )
        axis = theta / angle  # unit rotation axis
        half_angle = 0.5 * angle  # half rotation angle
        return np.array(
            [
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle),
            ],
            dtype=float,
        )

    @staticmethod
    def _skew(vector: np.ndarray) -> np.ndarray:
        """Return skew-symmetric matrix [vector]x such that [vector]x b = vector × b."""
        x, y, z = vector  # компоненты 3D-вектора
        return np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=float,
        )

    @staticmethod
    def _cross3(vector_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
        """Return 3D cross product without numpy dispatch overhead in EKF inner loops."""
        return np.array(
            [
                vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1],
                vector_a[2] * vector_b[0] - vector_a[0] * vector_b[2],
                vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0],
            ],
            dtype=float,
        )
