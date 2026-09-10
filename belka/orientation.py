"""3D orientation helpers for presented-frame scenario endpoints."""

from __future__ import annotations

import numpy as np

from belka.common import (
    RobotState3D,
    get_euler,
    quat_from_rot_matrix,
    quat_to_rot_matrix,
)


def rot_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """Build body-to-world rotation matrix from roll/pitch/yaw as Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = np.asarray(
        rpy, dtype=float
    )  # Euler angles [rad] in presented frame
    cr, sr = np.cos(roll), np.sin(roll)  # cos/sin roll
    cp, sp = np.cos(pitch), np.sin(pitch)  # cos/sin pitch
    cy, sy = np.cos(yaw), np.sin(yaw)  # cos/sin yaw
    R_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float
    )  # roll rotation
    R_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float
    )  # pitch rotation
    R_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )  # yaw rotation
    return R_z @ R_y @ R_x


def state_from_presented_position_angles(
    position: np.ndarray,
    angles: np.ndarray,
    presented_frame: np.ndarray,
) -> RobotState3D:
    """Create base-frame RobotState3D from world position and presented-frame roll/pitch/yaw."""
    p = np.asarray(position, dtype=float)  # world-frame position
    if p.shape != (3,):
        raise ValueError(f"3D scenario position must have shape (3,), got {p.shape}")
    rpy = np.asarray(angles, dtype=float)  # presented-frame [roll,pitch,yaw]
    if rpy.shape != (3,):
        raise ValueError(f"3D scenario angles must have shape (3,), got {rpy.shape}")
    R_bp = np.asarray(presented_frame, dtype=float)  # R_base_to_presented
    if R_bp.shape != (3, 3):
        raise ValueError(f"presented_frame must have shape (3, 3), got {R_bp.shape}")
    R_wp = rot_matrix_from_rpy(rpy)  # presented body frame → world
    R_wb = R_wp @ R_bp  # base body frame → world
    return RobotState3D(p=p, q=quat_from_rot_matrix(R_wb))


def presented_angles_from_state(
    state: RobotState3D, presented_frame: np.ndarray
) -> np.ndarray:
    """Return roll/pitch/yaw of the presented body frame in world coordinates."""
    R_wb = quat_to_rot_matrix(state.q)  # base body frame → world
    R_bp = np.asarray(presented_frame, dtype=float)  # R_base_to_presented
    R_wp = R_wb @ R_bp.T  # presented body frame → world
    return get_euler(quat_from_rot_matrix(R_wp))
