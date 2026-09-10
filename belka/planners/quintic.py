"""SE(3) quintic trajectory planner: position quintic + SLERP orientation."""

import math
from dataclasses import dataclass, field

import numpy as np

from belka.common import (
    RobotState3D,
    StateTolerances3D,
    quat_conjugate,
    quat_from_rot_matrix,
    quat_multiply,
    quat_normalize,
    quat_to_rot_matrix,
)
from belka.shared.planners import PlanSample, PlanningParameters, TrajectoryPlanner


def _quat_log(q):
    """Логарифмическое отображение S³→R³: log(q) = axis * angle."""
    q = quat_normalize(q)  # unit quaternion
    if q[0] < 0.0:
        q = -q
    w = float(q[0])  # скалярная часть
    v = q[1:4]  # векторная часть
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(v_norm, w)  # θ = 2·atan2(|v|, w)
    return angle * v / v_norm  # axis * θ


def _quat_exp(v):
    """Экспоненциальное отображение R³→S³: exp(axis·angle) = кватернион."""
    v = np.asarray(v, dtype=float)  # rotation vector axis·angle
    angle = float(np.linalg.norm(v))  # θ = |v|
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = v / angle  # единичная ось
    half = angle / 2.0
    return np.array(
        [
            math.cos(half),
            math.sin(half) * axis[0],
            math.sin(half) * axis[1],
            math.sin(half) * axis[2],
        ],
        dtype=float,
    )


def _quintic_coeffs(x0, dx0, ddx0, xT, dxT, ddxT, T):
    """Коэффициенты quintic-полинома x(t) = Σ aᵢtⁱ по граничным условиям.

    Система для [a₃,a₄,a₅]:
      A·[a₃,a₄,a₅]ᵀ = b
    где A образована условиями на t=T: x(T)=xT, ẋ(T)=dxT, ẍ(T)=ddxT,
    за вычетом вклада известных a₀,a₁,a₂ от начальных условий.
    """
    T2 = T * T  # T²
    T3 = T2 * T  # T³
    T4 = T3 * T  # T⁴
    T5 = T4 * T  # T⁵

    A = np.array(
        [  # матрица условий для старших коэффициентов
            [T3, T4, T5],
            [3 * T2, 4 * T3, 5 * T4],
            [6 * T, 12 * T2, 20 * T3],
        ],
        dtype=float,
    )
    b = np.array(
        [  # правая часть граничных условий
            xT - x0 - dx0 * T - 0.5 * ddx0 * T2,  # x(T) − (a₀ + a₁T + a₂T²)
            dxT - dx0 - ddx0 * T,  # ẋ(T) − (a₁ + 2a₂T)
            ddxT - ddx0,  # ẍ(T) − 2a₂
        ],
        dtype=float,
    )
    a345 = np.linalg.solve(A, b)  # коэффициенты [a3, a4, a5]
    return np.array([x0, dx0, 0.5 * ddx0, a345[0], a345[1], a345[2]], dtype=float)


def _quintic_eval(coeffs, t, deriv=0):
    """Вычислить значение quintic-полинома или его производной в t."""
    c = coeffs  # [a₀, a₁, a₂, a₃, a₄, a₅]
    if deriv == 0:
        return c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5
    elif deriv == 1:
        return c[1] + 2 * c[2] * t + 3 * c[3] * t**2 + 4 * c[4] * t**3 + 5 * c[5] * t**4
    elif deriv == 2:
        return 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t**2 + 20 * c[5] * t**3
    else:
        raise ValueError(f"Deriv {deriv} not supported")


@dataclass(frozen=True, kw_only=True)
class QuinticPlanningParameters3D(PlanningParameters):
    """Параметры SE(3) quintic-плана с quaternion orientation target."""

    initial_state: RobotState3D
    final_state: RobotState3D
    replan_tolerances: StateTolerances3D = field(default_factory=StateTolerances3D)
    finish_tolerances: StateTolerances3D = field(default_factory=StateTolerances3D)

    @property
    def finished_target(self) -> RobotState3D:
        """Вернуть состояние, достижение которого завершает 3D план."""
        return self.final_state

    @property
    def average_velocity(self) -> float:
        """Средняя линейная скорость исходного плана по длине 3D перемещения."""
        return float(np.linalg.norm(self.final_state.p - self.initial_state.p)) / self.T

    def recreate_from_state(
        self, current_t: float, new_T: float = None, **param_map
    ) -> "QuinticPlanningParameters3D":
        """Recreate params for replanning from current 3D state, preserving nominal average speed."""
        if new_T is None:
            initial_state = param_map.get("initial_state")
            if initial_state is not None and self.average_velocity > 1e-12:
                new_T = (
                    float(np.linalg.norm(self.final_state.p - initial_state.p))
                    / self.average_velocity
                )
            else:
                new_T = self.T
        return super().recreate_from_state(current_t, new_T, **param_map)


class QuinticPlanner3D(TrajectoryPlanner):
    """Планировщик quintic-траектории в SE(3).

    Позиция: покоординатный quintic от start.p до target.p.
    Ориентация: SLERP в PresentedFrame с quintic-профилем угла от start.q до target.q.
    """

    def __init__(
        self,
        params: QuinticPlanningParameters3D,
        *,
        presented_frame: np.ndarray | None = None,
    ):
        """Инициализировать 3D planner из явных planning parameters."""
        self._R_base_to_presented = self._validated_presented_frame(presented_frame)
        super().__init__(params)
        self.recalculate(params)

    def recalculate(self, params: QuinticPlanningParameters3D):
        """Recompute quintic coefficients from planning parameters."""
        super().recalculate(params)
        start = self._to_presented_state(params.initial_state)
        target = self._to_presented_state(params.final_state)
        T = params.T  # длительность траектории
        start_v = np.asarray(
            start.v, dtype=float
        )  # начальная линейная скорость world-frame

        # Покоординатные quintic для позиции
        px_c = _quintic_coeffs(
            start.p[0], start_v[0], start.a[0], target.p[0], 0.0, 0.0, T
        )
        py_c = _quintic_coeffs(
            start.p[1], start_v[1], start.a[1], target.p[1], 0.0, 0.0, T
        )
        pz_c = _quintic_coeffs(
            start.p[2], start_v[2], start.a[2], target.p[2], 0.0, 0.0, T
        )
        self._pos_coeffs = np.array([px_c, py_c, pz_c], dtype=float)  # 3×6

        # SLERP: q(t) = q_start ⊗ exp(s(t)·log(q_start⁻¹⊗q_target)).
        q_start = quat_normalize(start.q)  # начальная ориентация
        q_target = quat_normalize(target.q)  # целевая ориентация
        if float(np.dot(q_start, q_target)) < 0.0:
            q_target = -q_target
        q_rel = quat_multiply(quat_conjugate(q_start), q_target)  # q_start⁻¹⊗q_target
        self._omega_log = _quat_log(q_rel)  # axis·θ — логарифм относительного поворота
        self._omega_coeffs = _quintic_coeffs(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, T
        )  # s(t) от 0 до 1
        self._q_start = q_start  # q(0)

    def target_at_time(self, t: float) -> PlanSample:
        """Целевое состояние траектории в absolute-время t."""
        tau = self.planner_time(t)  # локальное время траектории от 0 до T
        if tau <= 0.0:
            tau = 0.0
        if tau >= self.params.T:
            return PlanSample(self.params.final_state)

        # Позиция и её производные
        p = np.array(
            [_quintic_eval(self._pos_coeffs[i], tau, 0) for i in range(3)]
        )  # p(tau)
        v = np.array(
            [_quintic_eval(self._pos_coeffs[i], tau, 1) for i in range(3)]
        )  # ṗ(tau)
        a = np.array(
            [_quintic_eval(self._pos_coeffs[i], tau, 2) for i in range(3)]
        )  # p̈(tau)

        # Ориентация: s(t) ∈ [0,1] — прогресс SLERP
        s = float(_quintic_eval(self._omega_coeffs, tau, 0))  # s(tau)
        ds = float(_quintic_eval(self._omega_coeffs, tau, 1))  # ṡ(tau)
        dds = float(_quintic_eval(self._omega_coeffs, tau, 2))  # s̈(tau)

        s = np.clip(s, 0.0, 1.0)
        delta = s * self._omega_log  # s(t)·axis·θ
        q = quat_multiply(self._q_start, _quat_exp(delta))  # q(t) = q₀⊗exp(δ)

        omega = ds * self._omega_log  # ω(t) = ṡ·axis·θ — угловая скорость body-frame
        alpha = dds * self._omega_log  # α(t) = s̈·axis·θ — угловое ускорение body-frame

        state = RobotState3D(
            p=p,
            q=quat_normalize(q),
            v=v,
            omega=omega,
            a=a,
            alpha=alpha,
        )
        return PlanSample(self._from_presented_state(state))

    @staticmethod
    def _validated_presented_frame(presented_frame: np.ndarray | None) -> np.ndarray:
        """Validate and return R_base_to_presented for planner frame conversion."""
        if presented_frame is None:
            return np.eye(3, dtype=float)
        R = np.asarray(presented_frame, dtype=float)  # R_base_to_presented
        if R.shape != (3, 3):
            raise ValueError(f"presented_frame must have shape (3, 3), got {R.shape}")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-10):
            raise ValueError("presented_frame must be orthonormal")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-10):
            raise ValueError("presented_frame must be a proper rotation")
        return R

    def _to_presented_state(self, state: RobotState3D) -> RobotState3D:
        """Represent base-frame attitude and angular vectors in PresentedFrame coordinates."""
        R = self._R_base_to_presented  # R_base_to_presented
        R_world_base = quat_to_rot_matrix(state.q)  # base body frame → world
        R_world_presented = R_world_base @ R.T  # presented body frame → world
        return RobotState3D(
            p=np.asarray(state.p, dtype=float).copy(),
            q=quat_from_rot_matrix(R_world_presented),
            v=np.asarray(state.v, dtype=float).copy(),
            omega=R @ np.asarray(state.omega, dtype=float),
            a=np.asarray(state.a, dtype=float).copy(),
            alpha=R @ np.asarray(state.alpha, dtype=float),
        )

    def _from_presented_state(self, state: RobotState3D) -> RobotState3D:
        """Convert a PresentedFrame planner sample back into the base body frame."""
        R = self._R_base_to_presented  # R_base_to_presented
        R_world_presented = quat_to_rot_matrix(state.q)  # presented body frame → world
        R_world_base = R_world_presented @ R  # base body frame → world
        return RobotState3D(
            p=np.asarray(state.p, dtype=float).copy(),
            q=quat_from_rot_matrix(R_world_base),
            v=np.asarray(state.v, dtype=float).copy(),
            omega=R.T @ np.asarray(state.omega, dtype=float),
            a=np.asarray(state.a, dtype=float).copy(),
            alpha=R.T @ np.asarray(state.alpha, dtype=float),
        )
