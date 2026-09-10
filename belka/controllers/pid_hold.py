"""Mass-agnostic 3D PID endpoint hold controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from belka.common import (
    RobotParams3D,
    RobotState3D,
    ValveOpenings,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_rot_matrix,
)
from belka.actuation import (
    active_thruster_mask,
    allocate_wrench_to_openings,
    valve_openings_to_forces,
)
from belka.shared.controllers import ControlHistory, Controller
from belka.shared.observers import WrenchEstimate3D
from belka.shared.planners import TrajectoryPlanner


@dataclass(frozen=True)
class PIDHoldConfig3D:
    """Gains and integral limits for 3D pose hold without body-property knowledge."""

    robot_params: RobotParams3D
    kp_p: float = 0.75  # translational proportional gain [N/m]
    ki_p: float = 0.25  # translational integral gain [N/(m*s)]
    kd_p: float = 3.0  # translational velocity damping gain [N*s/m]
    kp_q: float = 0.18  # attitude proportional gain [N*m/rad]
    ki_q: float = 0.015  # attitude integral gain [N*m/(rad*s)]
    kd_q: float = 0.10  # angular-rate damping gain [N*m*s/rad]
    integral_limit_p: float = 0.80  # componentwise translational integral limit [m*s]
    integral_limit_q: float = 0.30  # componentwise attitude integral limit [rad*s]
    max_position_force_ratio: float = (
        1.2  # translation-force cap relative to one thruster max
    )
    active_thrusters: tuple[int, ...] | None = None


class PIDHoldController3D(Controller):
    """3D PID hold controller that maps pose/rate errors to constrained thruster forces."""

    def __init__(
        self,
        planner: TrajectoryPlanner,
        config: PIDHoldConfig3D,
    ) -> None:
        """Initialize endpoint hold controller, integral state and debug/history buffers."""
        self.history = ControlHistory()
        self.set_planner(planner)
        self.config = config
        self.last_t: float | None = None
        self._integral_p = np.zeros(3, dtype=float)
        self._integral_q = np.zeros(3, dtype=float)
        self.last_requested_wrench_body = np.zeros(6, dtype=float)

    def __call__(
        self,
        t: float,
        current_forces: ValveOpenings,
        estimated_state: RobotState3D,
        estimation: WrenchEstimate3D | None,
    ) -> ValveOpenings:
        """Run one PID endpoint-hold step and record controller history."""
        dt = (
            0.0 if self.last_t is None else max(0.0, float(t) - self.last_t)
        )  # elapsed control time
        self.last_t = float(t)

        target_state = self.planner.next_target(t, estimated_state).state
        position_error_world = (
            target_state.p - estimated_state.p
        )  # world-frame position error
        attitude_error_body = _rotation_error_body(
            estimated_state.q, target_state.q
        )  # body-frame rotation error

        candidate_integral_p = self._integral_p
        if dt > 0.0:
            candidate_integral_p = np.clip(
                self._integral_p + position_error_world * dt,
                -self.config.integral_limit_p,
                self.config.integral_limit_p,
            )
            self._integral_q = np.clip(
                self._integral_q + attitude_error_body * dt,
                -self.config.integral_limit_q,
                self.config.integral_limit_q,
            )

        force_world = (
            self.config.kp_p * position_error_world
            + self.config.ki_p * candidate_integral_p
            - self.config.kd_p * estimated_state.v
        )  # requested world-frame corrective force-like command
        force_limit = (
            self.config.max_position_force_ratio
            * self.config.robot_params.max_thruster_force_n
        )  # robust PID force limit
        force_norm = float(
            np.linalg.norm(force_world)
        )  # norm of requested translational force
        if force_norm > force_limit:
            force_world = (
                self.config.kp_p * position_error_world
                + self.config.ki_p * self._integral_p
                - self.config.kd_p * estimated_state.v
            )  # saturated request without the rejected integral update
            force_norm = float(
                np.linalg.norm(force_world)
            )  # norm before robust saturation
            if force_norm > force_limit:
                force_world *= force_limit / force_norm
        else:
            self._integral_p = candidate_integral_p
        moment_body = (
            self.config.kp_q * attitude_error_body
            + self.config.ki_q * self._integral_q
            - self.config.kd_q * estimated_state.omega
        )  # requested body-frame corrective moment-like command
        R_body_to_world = quat_to_rot_matrix(estimated_state.q)  # body->world rotation
        force_body = (
            R_body_to_world.T @ force_world
        )  # body-frame corrective force-like command
        requested_wrench = np.concatenate([force_body, moment_body])  # [F_body, M_body]
        mask = active_thruster_mask(
            self.config.robot_params.nu,
            self.config.active_thrusters,
        )
        control = allocate_wrench_to_openings(
            requested_wrench,
            self.config.robot_params,
            active_mask=mask,
        )

        error_state = estimated_state - target_state
        self.last_requested_wrench_body = requested_wrench
        self._record_history(
            ts=t,
            state=estimated_state,
            control=control,
            reference_state=target_state,
            real_force=valve_openings_to_forces(
                current_forces,
                self.config.robot_params,
                active_mask=mask,
            ),
            error=error_state,
            estimation=estimation,
        )
        return control


def _rotation_error_body(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Return body-frame rotation vector that moves current attitude toward target attitude."""
    q_current = quat_normalize(
        np.asarray(q_current, dtype=float)
    )  # current attitude quaternion
    q_target = quat_normalize(
        np.asarray(q_target, dtype=float)
    )  # target attitude quaternion
    q_err = quat_multiply(quat_conjugate(q_current), q_target)  # current^-1 * target
    if q_err[0] < 0.0:
        q_err = -q_err
    vector_norm = float(np.linalg.norm(q_err[1:4]))  # norm of quaternion vector part
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * np.arctan2(
        vector_norm, np.clip(float(q_err[0]), -1.0, 1.0)
    )  # shortest angle
    return q_err[1:4] * (angle / vector_norm)
