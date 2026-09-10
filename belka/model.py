"""6DoF rigid-body dynamics with quaternion kinematics."""

import numpy as np

from belka.common import RobotParams3D, quat_derivative, quat_to_rot_matrix


def body_origin_acceleration_3d(
    q: np.ndarray,
    omega: np.ndarray,
    wrench_body: np.ndarray,
    external_wrench: np.ndarray,
    params: RobotParams3D,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute body-origin linear acceleration and body angular acceleration.

    The body frame origin is fixed at the symmetric robot frame, while the center
    of mass can be offset by params.r_cm_body. Wrenches from thrusters are about
    body origin; external wrench moment is interpreted about center of mass.
    """
    return body_origin_acceleration_from_properties(
        q,
        omega,
        wrench_body,
        external_wrench,
        params.m,
        params.r_cm_body,
        params.I_body,
    )


def body_origin_acceleration_from_properties(
    q: np.ndarray,
    omega: np.ndarray,
    wrench_body: np.ndarray,
    external_wrench: np.ndarray,
    m: float,
    r_cm_body: np.ndarray,
    I_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fixed-origin acceleration for explicit mass, CM offset and inertia tensor."""
    R = quat_to_rot_matrix(q)  # матрица поворота body→world
    c = np.asarray(r_cm_body, dtype=float)  # body-frame вектор от origin к CM
    I = np.asarray(I_body, dtype=float)  # тензор инерции около CM
    m = float(m)  # масса rigid body

    F_body = wrench_body[0:3]  # body-frame сила тяг
    M_origin_body = wrench_body[3:6]  # body-frame момент тяг около origin
    M_ext_body = R.T @ external_wrench[3:6]  # внешний момент около CM в body-frame
    M_cm_body = (
        M_origin_body - np.cross(c, F_body) + M_ext_body
    )  # момент тяг+внешний около CM
    I_omega = I @ omega  # I·ω
    alpha = np.linalg.solve(
        I, M_cm_body - np.cross(omega, I_omega)
    )  # body-frame угловое ускорение

    a_cm_world = (R @ F_body + external_wrench[0:3]) / m  # world-frame ускорение CM
    a_offset_body = np.cross(alpha, c) + np.cross(
        omega, np.cross(omega, c)
    )  # a_CM относительно origin
    a_origin_world = a_cm_world - R @ a_offset_body  # world-frame ускорение body origin
    return a_origin_world, alpha


def robot_derivative_3d(
    state_array: np.ndarray,  # x = [p(3), q(4), v(3), ω(3)] — 13-вектор состояния
    u: np.ndarray,  # N-вектор тяг движителей
    alloc: np.ndarray,  # 6×N allocation matrix: тяги → [F_body; M_body]
    external_wrench: np.ndarray,  # [F_world(3), M_world(3)] — внешний wrench
    params: RobotParams3D,
) -> np.ndarray:
    """Вычислить ẋ = f(x,u) — производную 13-мерного состояния.

    Уравнения:
      ṗ = v
      q̇ = ½ q ⊗ [0, ω]
      v̇_origin = (R(q)·F_body + F_ext) / m − R(q)(ω̇×c + ω×(ω×c))
      ω̇ = I⁻¹·(M_origin − c×F_body + Rᵀ·M_ext_world − ω×Iω)
    """
    q = state_array[3:7]  # кватернион ориентации [w,x,y,z]
    v = state_array[7:10]  # линейная скорость world-frame
    omega = state_array[10:13]  # угловая скорость body-frame ω

    wrench_body = alloc @ u  # w = B·u — [Fx,Fy,Fz,Mx,My,Mz] в body-frame

    p_dot = v  # ṗ
    q_dot = quat_derivative(q, omega)  # q̇
    v_dot, omega_dot = body_origin_acceleration_3d(
        q, omega, wrench_body, external_wrench, params
    )

    return np.concatenate([p_dot, q_dot, v_dot, omega_dot])
