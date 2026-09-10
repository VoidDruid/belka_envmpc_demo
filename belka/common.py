import dataclasses
import json as _json

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from belka.numeric_config import (
    parse_int_group,
    parse_numeric,
    parse_numeric_vector,
    require_exact_keys,
)


PLANAR_THRUSTER_NAMES = tuple(f"F{index}" for index in range(1, 9))
PLANAR_THRUSTER_DIRECTIONS_XY = (
    (1.0, 0.0),  # F1: +Fx, +Mz
    (1.0, 0.0),  # F2: +Fx, -Mz
    (-1.0, 0.0),  # F3: -Fx, +Mz
    (-1.0, 0.0),  # F4: -Fx, -Mz
    (0.0, 1.0),  # F5: +Fy, +Mz
    (0.0, 1.0),  # F6: +Fy, -Mz
    (0.0, -1.0),  # F7: -Fy, +Mz
    (0.0, -1.0),  # F8: -Fy, -Mz
)
PLANAR_AXIS_PAIRS = {
    "+X": (0, 1),
    "-X": (2, 3),
    "+Y": (4, 5),
    "-Y": (6, 7),
}
PLANAR_SIDE_FORCE_GROUPS = ((1, 2, 6, 7), (0, 3, 4, 5))


def quat_multiply(q1, q2):
    """Hamilton product q1 ⊗ q2."""
    w1, x1, y1, z1 = q1  # компоненты первого кватерниона [w, x, y, z]
    w2, x2, y2, z2 = q2  # компоненты второго кватерниона [w, x, y, z]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_conjugate(q):
    """Conjugate q* = [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_normalize(q):
    """Normalize quaternion to unit norm."""
    n = np.linalg.norm(q)  # норма кватерниона
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_to_rot_matrix(q):
    """Convert unit quaternion to 3×3 rotation matrix."""
    q = quat_normalize(q)  # единичный кватернион [w, x, y, z]
    w, x, y, z = q  # компоненты единичного кватерниона
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=float,
    )


def quat_from_rot_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a proper 3×3 rotation matrix to a unit quaternion [w,x,y,z]."""
    R = np.asarray(R, dtype=float)  # rotation matrix
    if R.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {R.shape}")
    trace = float(np.trace(R))  # matrix trace
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0  # four times quaternion scalar
        q = np.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ],
            dtype=float,
        )
    else:
        i = int(np.argmax(np.diag(R)))  # dominant diagonal axis
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # four times qx
            q = np.array(
                [
                    (R[2, 1] - R[1, 2]) / s,
                    0.25 * s,
                    (R[0, 1] + R[1, 0]) / s,
                    (R[0, 2] + R[2, 0]) / s,
                ],
                dtype=float,
            )
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # four times qy
            q = np.array(
                [
                    (R[0, 2] - R[2, 0]) / s,
                    (R[0, 1] + R[1, 0]) / s,
                    0.25 * s,
                    (R[1, 2] + R[2, 1]) / s,
                ],
                dtype=float,
            )
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # four times qz
            q = np.array(
                [
                    (R[1, 0] - R[0, 1]) / s,
                    (R[0, 2] + R[2, 0]) / s,
                    (R[1, 2] + R[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=float,
            )
    return quat_normalize(q)


def get_euler(q) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to roll, pitch, yaw Euler angles for logs/plots only."""
    q = quat_normalize(q)  # единичный кватернион [w, x, y, z]
    w, x, y, z = q  # компоненты кватерниона
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))  # roll
    sin_pitch = 2.0 * (w * y - z * x)  # sin(pitch)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))  # pitch
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # yaw
    return np.array([roll, pitch, yaw], dtype=float)


def quat_derivative(q, omega):
    """Quaternion kinematics: q_dot = 0.5 * q ⊗ [0, omega]."""
    omega_quat = np.array(
        [0.0, omega[0], omega[1], omega[2]], dtype=float
    )  # pure quaternion угловой скорости
    return 0.5 * quat_multiply(q, omega_quat)


def quat_to_yaw(q):
    """Extract yaw angle from quaternion."""
    return float(get_euler(q)[2])


def quat_angle_error(q1, q2) -> float:
    """Return shortest angular distance between two orientations in radians."""
    q1 = quat_normalize(q1)  # первый unit quaternion [w,x,y,z]
    q2 = quat_normalize(q2)  # второй unit quaternion [w,x,y,z]
    dot = abs(float(np.dot(q1, q2)))  # |q1·q2| учитывает q и -q как одну ориентацию
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def quat_distance(q1, q2) -> float:
    """Return the shortest quaternion geodesic distance between two orientations."""
    return quat_angle_error(q1, q2)


@dataclass(frozen=True)
class StateTolerances3D:
    """3D state tolerances for finish and replanning checks."""

    position_tolerance: float = 0.1
    velocity_tolerance: float = 0.05
    orientation_tolerance: float = 0.2
    omega_tolerance: float = 0.1


@dataclass(frozen=True)
class RobotState3D:
    """Spatial robot state with position, quaternion attitude, velocities and sensor accelerations."""

    p: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )  # позиция world-frame [x,y,z]
    q: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    )  # кватернион [w,x,y,z]
    v: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )  # линейная скорость world-frame
    omega: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )  # угловая скорость body-frame
    a: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )  # ускорение world-frame (measurement)
    alpha: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )  # угловое ускорение body-frame (measurement)

    def is_close(
        self, other: "RobotState3D", tolerances: StateTolerances3D | None = None
    ) -> bool:
        """Compare 6DoF states using position, velocity, orientation and angular-rate tolerances."""
        if tolerances is None:
            tolerances = StateTolerances3D()
        return (
            float(np.linalg.norm(self.p - other.p)) < tolerances.position_tolerance
            and float(np.linalg.norm(self.v - other.v)) < tolerances.velocity_tolerance
            and quat_distance(self.q, other.q) < tolerances.orientation_tolerance
            and float(np.linalg.norm(self.omega - other.omega))
            < tolerances.omega_tolerance
        )

    def __sub__(self, other: "RobotState3D") -> "RobotState3D":
        """Return a lightweight difference state for history/debug error records."""
        return dataclasses.replace(
            self,
            p=self.p - other.p,
            q=np.array([quat_distance(self.q, other.q), 0.0, 0.0, 0.0], dtype=float),
            v=self.v - other.v,
            omega=self.omega - other.omega,
            a=np.zeros(3, dtype=float),
            alpha=np.zeros(3, dtype=float),
        )

    @property
    def array(self) -> NDArray:
        """13-элементный вектор состояния [p, q, v, omega]."""
        return np.concatenate([self.p, self.q, self.v, self.omega]).astype(np.float32)


@dataclass(frozen=True)
class VisualOdom3D:
    """Low-rate visual odometry sample: world position and orientation only."""

    p: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )  # world-frame position
    q: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    )  # quaternion [w,x,y,z]


@dataclass(frozen=True)
class InertialOdom3D:
    """High-rate inertial odometry sample from IMU-like measurements."""

    q: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    )  # quaternion [w,x,y,z]
    omega: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )  # body-frame angular velocity
    a_body: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )  # body-frame linear acceleration


@dataclass(frozen=True)
class OdomInput3D:
    """3D observer input: optional visual odometry plus mandatory inertial sample."""

    visual: VisualOdom3D | None
    inertial: InertialOdom3D


@dataclass(frozen=True)
class _BoxComponent:
    """Axis-aligned rigid box component used for composite 3D mass properties."""

    name: str
    mass: float
    size: np.ndarray
    center: np.ndarray


def _box_inertia_about_center(mass: float, size: np.ndarray) -> np.ndarray:
    """Return inertia tensor of an axis-aligned box about its own center."""
    sx, sy, sz = size  # полные размеры box-а по body-frame осям
    return np.diag(
        [
            mass * (sy**2 + sz**2) / 12.0,
            mass * (sx**2 + sz**2) / 12.0,
            mass * (sx**2 + sy**2) / 12.0,
        ]
    )


def _composite_mass_properties(
    components: list[_BoxComponent],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute total mass, CM position and inertia about CM for axis-aligned boxes."""
    active = [
        component
        for component in components
        if component.mass > 0.0 and np.all(component.size > 0.0)
    ]
    if not active:
        raise ValueError("At least one positive-mass 3D body component is required")

    m = float(sum(component.mass for component in active))  # total mass
    r_cm = (
        sum(component.mass * component.center for component in active) / m
    )  # body-frame CM
    I = np.zeros((3, 3), dtype=float)  # inertia tensor about total CM
    E = np.eye(3, dtype=float)  # 3×3 identity
    for component in active:
        d = component.center - r_cm  # offset from total CM to component CM
        I_center = _box_inertia_about_center(
            component.mass, component.size
        )  # component inertia about own CM
        I += I_center + component.mass * ((float(d @ d) * E) - np.outer(d, d))
    return m, r_cm.astype(float), 0.5 * (I + I.T)


def _validate_3d_inertia(I_body: np.ndarray) -> None:
    """Fail fast when a computed 3D inertia tensor is not symmetric positive definite."""
    if I_body.shape != (3, 3):
        raise ValueError(f"I_body must have shape (3, 3), got {I_body.shape}")
    if not np.allclose(I_body, I_body.T, atol=1e-10):
        raise ValueError("I_body must be symmetric")
    if np.linalg.eigvalsh(I_body)[0] <= 0.0:
        raise ValueError("I_body must be positive definite")


@dataclass(frozen=True)
class CargoParams:
    """Unknown payload geometry and mass used by the simulator, not by the controller."""

    cargo_x: float = 0.0
    cargo_y: float = 0.0
    cargo_z: float = 0.0
    cargo_m: float = 0.0
    cargo_pos_x: float = 0.0
    cargo_pos_y: float = 0.0
    cargo_pos_z: float = 0.0

    def __post_init__(self) -> None:
        """Validate direct CargoParams construction as well as JSON construction."""
        self._validate()

    @classmethod
    def from_json(cls, json_path: str | Path) -> "CargoParams":
        """Load CargoParams from a top-level JSON 'cargo' block."""
        json_path = Path(json_path)
        with open(json_path) as f:
            cfg = _json.load(f)
        cargo_cfg = dict(cfg["cargo"])
        return cls.from_mapping(cargo_cfg)

    @classmethod
    def from_mapping(cls, cfg: dict | None) -> "CargoParams":
        """Create CargoParams from mapping values, defaulting to zero cargo."""
        if cfg is None:
            return cls()
        cfg = dict(cfg)
        require_exact_keys(
            cfg,
            {
                "cargo_x",
                "cargo_y",
                "cargo_z",
                "cargo_m",
                "cargo_pos_x",
                "cargo_pos_y",
                "cargo_pos_z",
            },
            field_name="cargo",
        )
        cargo = cls(
            cargo_x=parse_numeric(cfg["cargo_x"], field_name="CargoParams.cargo_x"),
            cargo_y=parse_numeric(cfg["cargo_y"], field_name="CargoParams.cargo_y"),
            cargo_z=parse_numeric(cfg["cargo_z"], field_name="CargoParams.cargo_z"),
            cargo_m=parse_numeric(cfg["cargo_m"], field_name="CargoParams.cargo_m"),
            cargo_pos_x=parse_numeric(
                cfg["cargo_pos_x"], field_name="CargoParams.cargo_pos_x"
            ),
            cargo_pos_y=parse_numeric(
                cfg["cargo_pos_y"], field_name="CargoParams.cargo_pos_y"
            ),
            cargo_pos_z=parse_numeric(
                cfg["cargo_pos_z"], field_name="CargoParams.cargo_pos_z"
            ),
        )
        cargo._validate()
        return cargo

    def _validate(self) -> None:
        """Validate payload dimensions, mass and position."""
        non_negative = [self.cargo_x, self.cargo_y, self.cargo_z, self.cargo_m]
        if any(np.isnan(value) or value < 0.0 for value in non_negative):
            raise ValueError("Cargo dimensions and mass must be non-negative")
        positions = [self.cargo_pos_x, self.cargo_pos_y, self.cargo_pos_z]
        if any(np.isnan(value) for value in positions):
            raise ValueError("Cargo position must not contain NaN")

    @property
    def enabled(self) -> bool:
        """Return whether cargo contributes mass/inertia and visualization geometry."""
        return (
            self.cargo_m > 0.0
            and self.cargo_x > 0.0
            and self.cargo_y > 0.0
            and self.cargo_z > 0.0
        )

    def component(self) -> _BoxComponent | None:
        """Return payload box component or None for zero cargo."""
        if not self.enabled:
            return None
        return _BoxComponent(
            name="cargo",
            mass=self.cargo_m,
            size=np.array([self.cargo_x, self.cargo_y, self.cargo_z], dtype=float),
            center=np.array(
                [self.cargo_pos_x, self.cargo_pos_y, self.cargo_pos_z], dtype=float
            ),
        )

    def to_mapping(self) -> dict[str, float]:
        """Return JSON-serializable cargo config values."""
        return {
            "cargo_x": self.cargo_x,
            "cargo_y": self.cargo_y,
            "cargo_z": self.cargo_z,
            "cargo_m": self.cargo_m,
            "cargo_pos_x": self.cargo_pos_x,
            "cargo_pos_y": self.cargo_pos_y,
            "cargo_pos_z": self.cargo_pos_z,
        }

    def inertia_about_cm(self) -> np.ndarray:
        """Return payload box inertia about payload CM."""
        if not self.enabled:
            return np.zeros((3, 3), dtype=float)
        return _box_inertia_about_center(
            self.cargo_m,
            np.array([self.cargo_x, self.cargo_y, self.cargo_z], dtype=float),
        )


@dataclass(frozen=True)
class PiecewiseLinearCurve:
    """Validated one-dimensional piecewise-linear calibration curve."""

    x: np.ndarray
    y: np.ndarray

    @classmethod
    def from_mapping(cls, cfg: dict, *, field_name: str) -> "PiecewiseLinearCurve":
        """Parse strictly increasing knots and finite monotone values."""
        if set(cfg) != {"x", "y"}:
            raise ValueError(f"{field_name} must contain exactly x and y")
        x = np.asarray(
            parse_numeric_vector(cfg["x"], field_name=f"{field_name}.x"),
            dtype=float,
        )
        y = np.asarray(
            parse_numeric_vector(cfg["y"], field_name=f"{field_name}.y"),
            dtype=float,
        )
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
            raise ValueError(
                f"{field_name} x/y must be equal-length vectors with at least two knots"
            )
        if np.any(np.diff(x) <= 0.0):
            raise ValueError(f"{field_name}.x must be strictly increasing")
        if np.any(np.diff(y) < 0.0):
            raise ValueError(f"{field_name}.y must be nondecreasing")
        x.setflags(write=False)
        y.setflags(write=False)
        return cls(x=x, y=y)

    def evaluate(self, value: float | np.ndarray) -> float | np.ndarray:
        """Evaluate with constant endpoint extrapolation."""
        result = np.interp(value, self.x, self.y)
        return float(result) if np.ndim(result) == 0 else result


@dataclass(frozen=True)
class ValveOpeningCurve:
    """Smooth normalized logistic opening-to-force curve."""

    midpoint: float
    steepness: float

    @classmethod
    def from_mapping(cls, cfg: dict, *, field_name: str) -> "ValveOpeningCurve":
        """Parse the two shape coefficients of the normalized logistic."""
        if set(cfg) != {"midpoint", "steepness"}:
            raise ValueError(
                f"{field_name} must contain exactly midpoint and steepness"
            )
        midpoint = parse_numeric(cfg["midpoint"], field_name=f"{field_name}.midpoint")
        steepness = parse_numeric(
            cfg["steepness"], field_name=f"{field_name}.steepness"
        )
        if not 0.0 < midpoint < 1.0:
            raise ValueError(f"{field_name}.midpoint must be in (0, 1)")
        if not np.isfinite(steepness) or steepness <= 0.0:
            raise ValueError(f"{field_name}.steepness must be finite and positive")
        return cls(midpoint=midpoint, steepness=steepness)

    def evaluate(self, opening: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the endpoint-normalized logistic on clipped openings."""
        opening_array = np.clip(np.asarray(opening, dtype=float), 0.0, 1.0)

        def sigmoid(value):
            return 1.0 / (1.0 + np.exp(-value))

        lower = sigmoid(-self.steepness * self.midpoint)
        upper = sigmoid(self.steepness * (1.0 - self.midpoint))
        raw = sigmoid(self.steepness * (opening_array - self.midpoint))
        result = (raw - lower) / (upper - lower)
        result = np.clip(result, 0.0, 1.0)
        return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class ThrusterModelParams3D:
    """Static valve/impeller calibration used by D3 control and simulation."""

    impeller_power_to_single_max_fraction: PiecewiseLinearCurve
    impeller_power_to_side_limit_n: PiecewiseLinearCurve
    valve_opening_to_fraction: ValveOpeningCurve
    per_thruster_max_force_n: np.ndarray
    side_force_groups: tuple[tuple[int, ...], tuple[int, ...]]
    valve_response_per_s: float

    @classmethod
    def from_mapping(
        cls,
        cfg: dict,
        *,
        nu: int,
        field_name: str = "RobotParams3D.thruster_model",
    ) -> "ThrusterModelParams3D":
        """Parse and validate the D3 static actuator model."""
        expected = {
            "impeller_power_to_single_max_fraction",
            "impeller_power_to_side_limit_n",
            "valve_opening_to_fraction",
            "per_thruster_max_force_n",
            "side_force_groups",
            "valve_response_per_s",
        }
        if set(cfg) != expected:
            raise ValueError(
                f"{field_name} must contain exactly {', '.join(sorted(expected))}"
            )
        per_thruster = np.asarray(
            parse_numeric_vector(
                cfg["per_thruster_max_force_n"],
                field_name=f"{field_name}.per_thruster_max_force_n",
            ),
            dtype=float,
        )
        if per_thruster.shape != (nu,):
            raise ValueError(
                f"{field_name}.per_thruster_max_force_n must have shape ({nu},), "
                f"got {per_thruster.shape}"
            )
        if np.any(~np.isfinite(per_thruster)) or np.any(per_thruster <= 0.0):
            raise ValueError(
                f"{field_name}.per_thruster_max_force_n must be finite and positive"
            )

        groups_raw = cfg["side_force_groups"]
        if len(groups_raw) != 2:
            raise ValueError(
                f"{field_name}.side_force_groups must contain left and right groups"
            )
        groups = tuple(
            parse_int_group(group, field_name=f"{field_name}.side_force_groups[{idx}]")
            for idx, group in enumerate(groups_raw)
        )
        flattened = [index for group in groups for index in group]
        if sorted(flattened) != list(range(nu)):
            raise ValueError(
                f"{field_name}.side_force_groups must partition all thrusters exactly once"
            )

        valve_response = parse_numeric(
            cfg["valve_response_per_s"],
            field_name=f"{field_name}.valve_response_per_s",
        )
        if not np.isfinite(valve_response) or valve_response <= 0.0:
            raise ValueError(
                f"{field_name}.valve_response_per_s must be finite and positive"
            )

        single_curve = PiecewiseLinearCurve.from_mapping(
            cfg["impeller_power_to_single_max_fraction"],
            field_name=f"{field_name}.impeller_power_to_single_max_fraction",
        )
        side_curve = PiecewiseLinearCurve.from_mapping(
            cfg["impeller_power_to_side_limit_n"],
            field_name=f"{field_name}.impeller_power_to_side_limit_n",
        )
        for curve_name, curve in (
            ("impeller_power_to_single_max_fraction", single_curve),
            ("impeller_power_to_side_limit_n", side_curve),
        ):
            if not np.isclose(curve.x[0], 0.0) or not np.isclose(curve.x[-1], 1.0):
                raise ValueError(
                    f"{field_name}.{curve_name}.x must start at 0 and end at 1"
                )
        if (
            np.any(single_curve.y < 0.0)
            or np.any(single_curve.y > 1.0)
            or not np.isclose(single_curve.y[0], 0.0)
            or not np.isclose(single_curve.y[-1], 1.0)
        ):
            raise ValueError(
                f"{field_name}.impeller_power_to_single_max_fraction.y "
                "must span normalized fractions from 0 to 1"
            )
        if not np.isclose(side_curve.y[0], 0.0) or side_curve.y[-1] <= 0.0:
            raise ValueError(
                f"{field_name}.impeller_power_to_side_limit_n.y must start at 0 "
                "and end at a positive force"
            )

        per_thruster.setflags(write=False)
        return cls(
            impeller_power_to_single_max_fraction=single_curve,
            impeller_power_to_side_limit_n=side_curve,
            valve_opening_to_fraction=ValveOpeningCurve.from_mapping(
                cfg["valve_opening_to_fraction"],
                field_name=f"{field_name}.valve_opening_to_fraction",
            ),
            per_thruster_max_force_n=per_thruster,
            side_force_groups=groups,  # type: ignore[arg-type]
            valve_response_per_s=valve_response,
        )


@dataclass(frozen=True)
class RobotParams3D:
    """Spatial telescopic robot physical parameters loaded from JSON."""

    side_L: float = 0.33  # длина боковины по X/Z
    side_w: float = 0.165  # толщина одной боковины по Y
    side_m: float = 2.0  # масса одной боковины
    base_L: float = 0.0  # ширина посадочного места между бортами в сложенном состоянии
    base_w: float = 0.0  # толщина дна по Z
    base_m: float = 0.0  # масса дна
    extension_L: float = 0.0  # раздвижение каждого борта по Y
    l: float = 0.05  # отступ движителя от ребра
    m: float = 4.0  # known robot mass without cargo
    r_cm_body: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )  # CM offset from body origin
    I_body: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=float)
    )  # inertia about CM in body frame
    allocation_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((6, 0), dtype=float)
    )  # тяги → [F_body; M_origin]
    thruster_positions_body: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )
    thruster_directions_body: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )
    impeller_power_left: float = 1.0
    impeller_power_right: float = 1.0
    thruster_model: ThrusterModelParams3D | None = None

    def __init__(self, *args, **kwargs):
        """Prevent direct construction so JSON remains the single source of model params."""
        raise TypeError(
            "RobotParams3D cannot be instantiated directly. Use RobotParams3D.from_json(path) instead."
        )

    @classmethod
    def from_json(cls, json_path: str | Path) -> "RobotParams3D":
        """Загрузить параметры 3D робота из top-level JSON block 'robot'."""
        json_path = Path(json_path)
        with open(json_path) as f:
            cfg = _json.load(f)
        cfg = dict(cfg["robot"])
        required = {
            "side_L",
            "side_w",
            "base_L",
            "base_w",
            "base_m",
            "extension_L",
            "l",
            "impeller_power_left",
            "impeller_power_right",
            "thruster_model",
        }
        allowed = required | {"m", "side_m", "mass_properties"}
        if not required <= set(cfg) or not set(cfg) <= allowed:
            raise ValueError(
                "RobotParams3D.robot has unexpected or missing fields; expected "
                "the fixed geometry/impeller/thruster_model fields, exactly one of "
                "m or side_m, and optional mass_properties"
            )
        if ("m" in cfg) == ("side_m" in cfg):
            raise ValueError(
                "RobotParams3D JSON must define exactly one of m or side_m"
            )

        side_L = parse_numeric(
            cfg["side_L"], field_name="RobotParams3D.side_L"
        )  # длина боковины по X/Z
        side_w = parse_numeric(
            cfg["side_w"], field_name="RobotParams3D.side_w"
        )  # толщина боковины по Y
        base_L = parse_numeric(
            cfg["base_L"], field_name="RobotParams3D.base_L"
        )  # ширина посадочного места
        base_w = parse_numeric(
            cfg["base_w"], field_name="RobotParams3D.base_w"
        )  # толщина дна
        base_m = parse_numeric(
            cfg["base_m"], field_name="RobotParams3D.base_m"
        )  # масса дна
        if "m" in cfg:
            total_m = parse_numeric(cfg["m"], field_name="RobotParams3D.m")
            side_m = (total_m - base_m) / 2.0
        else:
            side_m = parse_numeric(
                cfg["side_m"], field_name="RobotParams3D.side_m"
            )  # масса одной боковины
        extension_L = parse_numeric(
            cfg["extension_L"], field_name="RobotParams3D.extension_L"
        )  # раздвижение борта
        l = parse_numeric(cfg["l"], field_name="RobotParams3D.l")  # отступ движителя
        cls._validate_geometry(
            side_L, side_w, side_m, base_L, base_w, base_m, extension_L, l
        )

        positions, directions = _make_3d_thruster_layout(
            side_L, side_w, base_L, extension_L, l
        )
        allocation_matrix = _make_3d_allocation_matrix_from_layout(
            positions, directions
        )
        components = cls._known_components(
            side_L, side_w, side_m, base_L, base_w, base_m, extension_L
        )
        m, r_cm_body, I_body = _composite_mass_properties(components)
        if "mass_properties" in cfg:
            mass_properties = dict(cfg["mass_properties"])
            if set(mass_properties) != {
                "center_of_mass_body_m",
                "inertia_body_kg_m2",
            }:
                raise ValueError(
                    "RobotParams3D.mass_properties must contain exactly "
                    "center_of_mass_body_m and inertia_body_kg_m2"
                )
            r_cm_body = np.asarray(
                parse_numeric_vector(
                    mass_properties["center_of_mass_body_m"],
                    field_name="RobotParams3D.mass_properties.center_of_mass_body_m",
                ),
                dtype=float,
            )  # body-frame center of mass
            I_body = np.asarray(
                [
                    parse_numeric_vector(
                        row,
                        field_name=(
                            "RobotParams3D.mass_properties.inertia_body_kg_m2"
                            f"[{row_index}]"
                        ),
                    )
                    for row_index, row in enumerate(
                        mass_properties["inertia_body_kg_m2"]
                    )
                ],
                dtype=float,
            )  # inertia tensor about CM in body frame
            if r_cm_body.shape != (3,):
                raise ValueError(
                    "RobotParams3D.mass_properties.center_of_mass_body_m "
                    "must contain 3 values"
                )
        _validate_3d_inertia(I_body)

        impeller_power_left = parse_numeric(
            cfg["impeller_power_left"],
            field_name="RobotParams3D.impeller_power_left",
        )
        impeller_power_right = parse_numeric(
            cfg["impeller_power_right"],
            field_name="RobotParams3D.impeller_power_right",
        )
        for name, value in (
            ("impeller_power_left", impeller_power_left),
            ("impeller_power_right", impeller_power_right),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"RobotParams3D.{name} must be finite and in [0, 1]")
        thruster_model = ThrusterModelParams3D.from_mapping(
            cfg["thruster_model"],
            nu=allocation_matrix.shape[1],
        )
        instance = object.__new__(cls)
        for name, val in [
            ("side_L", side_L),
            ("side_w", side_w),
            ("side_m", side_m),
            ("base_L", base_L),
            ("base_w", base_w),
            ("base_m", base_m),
            ("extension_L", extension_L),
            ("l", l),
            ("m", m),
            ("r_cm_body", r_cm_body),
            ("I_body", I_body),
            ("allocation_matrix", allocation_matrix),
            ("thruster_positions_body", positions),
            ("thruster_directions_body", directions),
            ("impeller_power_left", impeller_power_left),
            ("impeller_power_right", impeller_power_right),
            ("thruster_model", thruster_model),
        ]:
            object.__setattr__(instance, name, val)
        return instance

    @staticmethod
    def _validate_geometry(
        side_L: float,
        side_w: float,
        side_m: float,
        base_L: float,
        base_w: float,
        base_m: float,
        extension_L: float,
        l: float,
    ) -> None:
        """Validate telescopic robot geometry before computing mass properties."""
        required_positive = {
            "side_L": side_L,
            "side_w": side_w,
            "side_m": side_m,
        }
        for name, value in required_positive.items():
            if np.isnan(value) or value <= 0.0:
                raise ValueError(f"RobotParams3D.{name} must be positive")
        non_negative = {
            "base_L": base_L,
            "base_w": base_w,
            "base_m": base_m,
            "extension_L": extension_L,
            "l": l,
        }
        for name, value in non_negative.items():
            if np.isnan(value) or value < 0.0:
                raise ValueError(f"RobotParams3D.{name} must be non-negative")
        inner_width = (
            base_L + 2.0 * extension_L
        )  # ширина посадочного места между внутренними стенками
        outer_half_y = inner_width / 2.0 + side_w  # половина внешней ширины робота
        if l >= min(side_L / 2.0, outer_half_y):
            raise ValueError(
                "RobotParams3D.l must be smaller than side_L/2 and outer_width/2"
            )
        if base_m > 0.0 and (base_w <= 0.0 or inner_width <= 0.0):
            raise ValueError("Positive base_m requires positive base_w and inner_width")

    @staticmethod
    def _known_components(
        side_L: float,
        side_w: float,
        side_m: float,
        base_L: float,
        base_w: float,
        base_m: float,
        extension_L: float,
    ) -> list[_BoxComponent]:
        """Return known robot box components, excluding unknown cargo."""
        inner_width = base_L + 2.0 * extension_L  # ширина посадочного места по Y
        side_y = inner_width / 2.0 + side_w / 2.0  # |Y| центра боковины
        components = [
            _BoxComponent(
                name="left_side",
                mass=side_m,
                size=np.array([side_L, side_w, side_L], dtype=float),
                center=np.array([0.0, -side_y, 0.0], dtype=float),
            ),
            _BoxComponent(
                name="right_side",
                mass=side_m,
                size=np.array([side_L, side_w, side_L], dtype=float),
                center=np.array([0.0, side_y, 0.0], dtype=float),
            ),
        ]
        if base_m > 0.0 and base_w > 0.0 and inner_width > 0.0:
            components.append(
                _BoxComponent(
                    name="base",
                    mass=base_m,
                    size=np.array([side_L, inner_width, base_w], dtype=float),
                    center=np.array(
                        [0.0, 0.0, -side_L / 2.0 + base_w / 2.0], dtype=float
                    ),
                )
            )
        return components

    @property
    def nu(self) -> int:
        """Количество движителей."""
        return self.allocation_matrix.shape[1]

    @property
    def side_force_groups(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return left/right impeller groups from the actuator model."""
        if self.thruster_model is None:
            return ((), ())
        return self.thruster_model.side_force_groups

    @property
    def impeller_powers(self) -> tuple[float, float]:
        """Return constant left/right impeller powers for this experiment."""
        return (self.impeller_power_left, self.impeller_power_right)

    @property
    def valve_response_per_s(self) -> float:
        """Return maximum valve-opening change per second."""
        if self.thruster_model is None:
            raise RuntimeError("RobotParams3D has no thruster model")
        return self.thruster_model.valve_response_per_s

    @property
    def per_thruster_force_caps_n(self) -> np.ndarray:
        """Return unconstrained full-opening force caps at current impeller powers."""
        if self.thruster_model is None:
            raise RuntimeError("RobotParams3D has no thruster model")
        caps = np.zeros(self.nu, dtype=float)
        for group, power in zip(
            self.side_force_groups, self.impeller_powers, strict=True
        ):
            power_fraction = (
                self.thruster_model.impeller_power_to_single_max_fraction.evaluate(
                    power
                )
            )
            caps[list(group)] = self.thruster_model.per_thruster_max_force_n[
                list(group)
            ] * float(power_fraction)
        return caps

    @property
    def side_force_limits_n(self) -> np.ndarray:
        """Return left/right summed-force limits at current impeller powers."""
        if self.thruster_model is None:
            raise RuntimeError("RobotParams3D has no thruster model")
        return np.asarray(
            [
                self.thruster_model.impeller_power_to_side_limit_n.evaluate(power)
                for power in self.impeller_powers
            ],
            dtype=float,
        )

    @property
    def max_thruster_force_n(self) -> float:
        """Return the largest current per-thruster force cap in newtons."""
        return float(np.max(self.per_thruster_force_caps_n))

    @property
    def inner_width(self) -> float:
        """Return payload bay width between inner side-wall faces."""
        return self.base_L + 2.0 * self.extension_L

    @property
    def outer_half_width_y(self) -> float:
        """Return half-width along body-frame Y including side walls."""
        return self.inner_width / 2.0 + self.side_w

    def known_components(self) -> list[_BoxComponent]:
        """Return known robot box components with no cargo."""
        return self._known_components(
            self.side_L,
            self.side_w,
            self.side_m,
            self.base_L,
            self.base_w,
            self.base_m,
            self.extension_L,
        )

    def composite_body_properties(
        self, cargo: CargoParams | None = None
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Combine measured robot body properties with an optional box payload."""
        if cargo is None or not cargo.enabled:
            return self.m, self.r_cm_body.copy(), self.I_body.copy()

        cargo_center = np.array(
            [cargo.cargo_pos_x, cargo.cargo_pos_y, cargo.cargo_pos_z], dtype=float
        )  # payload CM in the robot body frame
        total_mass = self.m + cargo.cargo_m  # robot plus payload mass
        total_cm = (
            self.m * self.r_cm_body + cargo.cargo_m * cargo_center
        ) / total_mass  # combined CM in the body frame
        E = np.eye(3, dtype=float)  # identity used by the parallel-axis theorem
        robot_offset = self.r_cm_body - total_cm  # robot CM relative to combined CM
        cargo_offset = cargo_center - total_cm  # payload CM relative to combined CM
        total_inertia = (
            self.I_body
            + self.m
            * (
                float(robot_offset @ robot_offset) * E
                - np.outer(robot_offset, robot_offset)
            )
            + cargo.inertia_about_cm()
            + cargo.cargo_m
            * (
                float(cargo_offset @ cargo_offset) * E
                - np.outer(cargo_offset, cargo_offset)
            )
        )  # inertia about the combined CM in the body frame
        total_inertia = 0.5 * (total_inertia + total_inertia.T)
        _validate_3d_inertia(total_inertia)
        return total_mass, total_cm, total_inertia

    def get_presented_frame(self) -> np.ndarray:
        """Return R_base_to_presented chosen from known robot geometry only."""
        outer_y = (
            self.inner_width + 2.0 * self.side_w
        )  # full body width along base-frame Y
        if outer_y > self.side_L + 1e-12:
            return np.array(
                [
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        return np.eye(3, dtype=float)

    def presented_front_axis_body(self) -> np.ndarray:
        """Return the presented +X front direction expressed in base body coordinates."""
        R_bp = self.get_presented_frame()  # R_base_to_presented
        return R_bp.T @ np.array([1.0, 0.0, 0.0], dtype=float)

    def with_body_properties(
        self, m: float, r_cm_body: np.ndarray, I_body: np.ndarray
    ) -> "RobotParams3D":
        """Return params with the same geometry/limits and replaced identified rigid-body properties."""
        m = float(m)  # total identified mass
        r_cm_body = np.asarray(r_cm_body, dtype=float)  # body-frame CM offset
        I_body = np.asarray(I_body, dtype=float)  # inertia tensor about identified CM
        if np.isnan(m) or m <= 0.0:
            raise ValueError(
                "RobotParams3D.with_body_properties requires positive mass"
            )
        if r_cm_body.shape != (3,):
            raise ValueError(f"r_cm_body must have shape (3,), got {r_cm_body.shape}")
        _validate_3d_inertia(I_body)

        instance = object.__new__(type(self))
        for name in (
            "side_L",
            "side_w",
            "side_m",
            "base_L",
            "base_w",
            "base_m",
            "extension_L",
            "l",
            "allocation_matrix",
            "thruster_positions_body",
            "thruster_directions_body",
            "impeller_power_left",
            "impeller_power_right",
            "thruster_model",
        ):
            object.__setattr__(instance, name, getattr(self, name))
        object.__setattr__(instance, "m", m)
        object.__setattr__(instance, "r_cm_body", r_cm_body.copy())
        object.__setattr__(instance, "I_body", I_body.copy())
        return instance

    def with_thruster_model(
        self,
        thruster_model: ThrusterModelParams3D,
        *,
        impeller_power_left: float | None = None,
        impeller_power_right: float | None = None,
    ) -> "RobotParams3D":
        """Return params with an alternate identified actuator model."""
        left = (
            self.impeller_power_left
            if impeller_power_left is None
            else float(impeller_power_left)
        )
        right = (
            self.impeller_power_right
            if impeller_power_right is None
            else float(impeller_power_right)
        )
        for name, value in (
            ("impeller_power_left", left),
            ("impeller_power_right", right),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"RobotParams3D.{name} must be finite and in [0, 1]")
        if thruster_model.per_thruster_max_force_n.shape != (self.nu,):
            raise ValueError(
                f"thruster_model per-thruster vector must have shape ({self.nu},)"
            )

        instance = object.__new__(type(self))
        for name in (
            "side_L",
            "side_w",
            "side_m",
            "base_L",
            "base_w",
            "base_m",
            "extension_L",
            "l",
            "m",
            "r_cm_body",
            "I_body",
            "allocation_matrix",
            "thruster_positions_body",
            "thruster_directions_body",
        ):
            value = getattr(self, name)
            object.__setattr__(
                instance, name, value.copy() if isinstance(value, np.ndarray) else value
            )
        object.__setattr__(instance, "impeller_power_left", left)
        object.__setattr__(instance, "impeller_power_right", right)
        object.__setattr__(instance, "thruster_model", thruster_model)
        return instance


@dataclass
class ThrusterForces:
    """Canonical vector of per-thruster physical forces."""

    values: np.ndarray

    @classmethod
    def zeros(cls, nu: int) -> "ThrusterForces":
        """Return zero thruster vector of length nu."""
        return cls(values=np.zeros(nu, dtype=float))

    @classmethod
    def from_array(cls, array) -> "ThrusterForces":
        """Create thruster vector from array-like values."""
        return cls(values=np.asarray(array, dtype=float).copy())

    def __post_init__(self) -> None:
        """Copy the vector and reject non-finite or non-vector values."""
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or np.any(~np.isfinite(values)):
            raise ValueError("thruster forces must be a finite one-dimensional vector")
        self.values = values.copy()

    def to_wrench(self, params_or_alloc) -> np.ndarray:
        """Вычислить body-frame wrench через allocation matrix или spatial robot params."""
        alloc = (
            params_or_alloc.allocation_matrix
            if hasattr(params_or_alloc, "allocation_matrix")
            else params_or_alloc
        )
        alloc = np.asarray(alloc, dtype=float)  # allocation matrix: тяги → wrench
        return alloc @ self.to_array()

    @property
    def nu(self) -> int:
        """Количество движителей в векторе тяг."""
        return int(self.values.size)

    def to_array(self) -> np.ndarray:
        """Вернуть массив тяг движителей."""
        return self.values.copy()


class ValveOpenings(ThrusterForces):
    """Vector of normalized valve openings used as controller commands."""

    def to_wrench(self, params_or_alloc) -> np.ndarray:
        """Prevent accidental use of openings as forces."""
        del params_or_alloc
        raise TypeError(
            "ValveOpenings must be converted through the spatial thruster model "
            "before computing a wrench"
        )


def _make_3d_thruster_layout(
    side_L: float,
    side_w: float,
    base_L: float,
    extension_L: float,
    l: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create 12 thruster positions and force directions for telescopic 3D geometry."""
    hx = side_L / 2.0  # half-size along body X
    hz = side_L / 2.0  # half-size along body Z
    hy = (base_L + 2.0 * extension_L) / 2.0 + side_w  # half outer width along body Y
    ax = hx - l  # tangent offset along X
    ay = hy - l  # tangent offset along Y
    az = hz - l  # tangent offset along Z

    positions = np.array(  # body-frame thruster application points
        [
            [-hx, -ay, az],  # F1: rear-right, +Fx, +Mz
            [-hx, ay, -az],  # F2: rear-left, +Fx, -Mz
            [hx, ay, az],  # F3: front-left, backward
            [hx, -ay, -az],  # F4: front-right, backward
            [ax, -hy, az],  # F5: front-right, +Fy, +Mz
            [-ax, -hy, -az],  # F6: rear-right, +Fy, -Mz
            [-ax, hy, az],  # F7: rear-left, -Fy, +Mz
            [ax, hy, -az],  # F8: front-left, -Fy, -Mz
            [-ax, -ay, -hz],
            [ax, ay, -hz],
            [-ax, -ay, hz],
            [ax, ay, hz],
        ],
        dtype=float,
    )

    directions = np.array(  # body-frame force directions from positive thrust commands
        [
            [*PLANAR_THRUSTER_DIRECTIONS_XY[0], 0],
            [*PLANAR_THRUSTER_DIRECTIONS_XY[1], 0],
            [*PLANAR_THRUSTER_DIRECTIONS_XY[2], 0],
            [*PLANAR_THRUSTER_DIRECTIONS_XY[3], 0],
            [*PLANAR_THRUSTER_DIRECTIONS_XY[4], 0],
            [*PLANAR_THRUSTER_DIRECTIONS_XY[5], 0],
            [*PLANAR_THRUSTER_DIRECTIONS_XY[6], 0],
            [*PLANAR_THRUSTER_DIRECTIONS_XY[7], 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, -1],
            [0, 0, -1],
        ],
        dtype=float,
    )
    return positions, directions


def _make_3d_allocation_matrix_from_layout(
    positions: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Create allocation matrix mapping thruster forces to body-origin wrench."""
    nu = positions.shape[0]  # количество движителей
    alloc = np.zeros(
        (6, nu), dtype=float
    )  # allocation matrix: тяги → [F_body, M_origin]
    for i in range(nu):  # индекс движителя
        alloc[0:3, i] = directions[i]
        alloc[3:6, i] = np.cross(positions[i], directions[i])
    return alloc
