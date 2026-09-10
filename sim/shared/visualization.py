import mujoco
import numpy as np


def body_arrow_geom_vars(
    name: str,
    direction_body: np.ndarray,
    *,
    length: float,
    radius: float,
    rgba: str,
) -> list[dict]:
    """Return MJCF geom variables for a body-fixed capsule+sphere direction arrow."""
    direction = np.asarray(direction_body, dtype=float)  # body-frame arrow direction
    norm = float(np.linalg.norm(direction))  # direction norm
    if norm < 1e-12:
        raise ValueError(f"{name} arrow direction must be non-zero")
    axis = direction / norm  # unit body-frame arrow axis
    length = float(length)  # full visual arrow length
    radius = float(radius)  # capsule/sphere radius
    shaft_end = axis * max(0.0, length - 2.0 * radius)  # endpoint before round tip
    tip = axis * length  # arrow tip center
    return [
        {
            "name": f"{name}_shaft",
            "type": "capsule",
            "fromto": [
                0.0,
                0.0,
                0.0,
                float(shaft_end[0]),
                float(shaft_end[1]),
                float(shaft_end[2]),
            ],
            "size": radius,
            "rgba": rgba,
        },
        {
            "name": f"{name}_tip",
            "type": "sphere",
            "pos": [float(tip[0]), float(tip[1]), float(tip[2])],
            "size": radius * 1.8,
            "rgba": rgba,
        },
    ]


def thrust_arrow_rgba(intensity: float) -> np.ndarray:
    """Return thrust-arrow color from cool blue at low thrust to hot orange at high thrust."""
    t = float(np.sqrt(np.clip(intensity, 0.0, 1.0)))  # визуальная интенсивность тяги
    low = np.array([0.05, 0.55, 1.0, 0.82], dtype=np.float32)
    high = np.array([1.0, 0.24, 0.04, 0.98], dtype=np.float32)
    return ((1.0 - t) * low + t * high).astype(np.float32)


def draw_mujoco_arrow(
    viewer, start: np.ndarray, end: np.ndarray, radius: float, rgba: np.ndarray
) -> bool:
    """Append one arrow geom to viewer.user_scn if there is room."""
    if viewer.user_scn.ngeom >= len(viewer.user_scn.geoms):
        return False
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3, dtype=float),
        pos=np.zeros(3, dtype=float),
        mat=np.zeros(9, dtype=float),
        rgba=rgba,
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, radius, start, end)
    geom.rgba = rgba
    viewer.user_scn.ngeom += 1
    return True


def draw_mujoco_sphere(viewer, pos: np.ndarray, size: float, rgba: np.ndarray) -> bool:
    """Append one sphere geom to viewer.user_scn if there is room."""
    if viewer.user_scn.ngeom >= len(viewer.user_scn.geoms):
        return False
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, size, size], dtype=float),
        pos=pos,
        mat=np.eye(3, dtype=float).reshape(-1),
        rgba=rgba,
    )
    viewer.user_scn.ngeom += 1
    return True
