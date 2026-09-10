"""Active 3D rigid-body identification for unknown payload mass properties."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from belka.common import ThrusterForces, RobotParams3D, RobotState3D, quat_to_rot_matrix
from belka.model import body_origin_acceleration_from_properties
from belka.shared.derivatives import least_squares_derivative


@dataclass(frozen=True)
class IdentificationSample3D:
    """One sample used by the active rigid-body identifier."""

    t: float
    state: RobotState3D
    forces: ThrusterForces
    segment_id: int


@dataclass(frozen=True)
class RigidBodyEstimate3D:
    """Estimated effective rigid-body parameters and fit diagnostics."""

    total_m: float
    r_cm_body: np.ndarray
    I_body: np.ndarray
    segment_external_wrenches: dict[int, np.ndarray]
    residual_rms: float
    quality_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CargoEstimate3D:
    """Derived payload mass properties for diagnostics only."""

    cargo_m: float
    cargo_cm_body: np.ndarray
    cargo_I_body: np.ndarray


@dataclass(frozen=True)
class RigidBodyIdentifier3DConfig:
    """Numerical weights and bounds for batch rigid-body identification."""

    accel_weight: float = 1.0
    alpha_weight: float = 0.35
    cm_bound_margin: float = 0.35
    min_interval_dt: float = 1e-4


class RigidBodyIdentifier3D:
    """Batch least-squares estimator for effective mass, CM offset and inertia tensor."""

    def __init__(
        self,
        nominal_params: RobotParams3D,
        config: RigidBodyIdentifier3DConfig | None = None,
    ) -> None:
        """Store known geometry/allocation and identifier numerical settings."""
        self.nominal_params = nominal_params
        self.config = config or RigidBodyIdentifier3DConfig()

    def fit(
        self,
        samples: list[IdentificationSample3D],
        initial_external_wrench: np.ndarray | None = None,
    ) -> RigidBodyEstimate3D:
        """Estimate body properties from segment-centered active-maneuver dynamics."""
        intervals = self._intervals(
            samples
        )  # adjacent sample pairs within the same segment
        if not intervals:
            raise ValueError(
                "RigidBodyIdentifier3D.fit requires at least one valid same-segment interval"
            )
        interval_measurements = self._interval_measurements(
            samples, intervals
        )  # intervals plus smoothed alpha
        segment_ids = sorted(
            {sample.segment_id for sample in samples}
        )  # active maneuver segments
        x0 = self._initial_vector()
        lower, upper = self._bounds()
        fixed_external_wrench = self._validated_external_wrench(initial_external_wrench)

        def residual(x: np.ndarray) -> np.ndarray:
            """Return weighted dynamics residuals for scipy least_squares."""
            m, c, I, external_by_segment = self._unpack(
                x,
                segment_ids,
                fixed_external_wrench,
            )  # candidate body properties and the hold-stage external-wrench baseline
            values = []  # scalar residual values
            acceleration_by_segment: dict[int, list[np.ndarray]] = {}
            angular_acceleration_by_segment: dict[int, list[np.ndarray]] = {}

            # Dynamics residuals from integrated origin and angular acceleration.
            for s0, s1, alpha_meas in interval_measurements:
                u = 0.5 * (
                    s0.forces.to_array() + s1.forces.to_array()
                )  # average applied thruster vector
                wrench_body = (
                    self.nominal_params.allocation_matrix @ u
                )  # body-frame thrust wrench about origin
                omega = 0.5 * (
                    s0.state.omega + s1.state.omega
                )  # average body-frame angular rate
                ext = external_by_segment[
                    s0.segment_id
                ]  # fixed world-frame wrench estimated during holding
                a_pred, alpha_pred = body_origin_acceleration_from_properties(
                    s0.state.q,
                    omega,
                    wrench_body,
                    ext,
                    m,
                    c,
                    I,
                )
                a_meas = 0.5 * (
                    s0.state.a + s1.state.a
                )  # measured world-frame origin acceleration
                acceleration_by_segment.setdefault(s0.segment_id, []).append(
                    a_pred - a_meas
                )
                if alpha_meas is not None:
                    angular_acceleration_by_segment.setdefault(
                        s0.segment_id, []
                    ).append(alpha_pred - alpha_meas)

            for segment_id in segment_ids:
                acceleration_residuals = np.asarray(
                    acceleration_by_segment.get(segment_id, []), dtype=float
                )
                angular_residuals = np.asarray(
                    angular_acceleration_by_segment.get(segment_id, []), dtype=float
                )
                # Centering preserves the pulse response while cancelling a segment-constant disturbance and bias.
                if acceleration_residuals.size:
                    acceleration_residuals = acceleration_residuals - np.mean(
                        acceleration_residuals, axis=0
                    )
                if angular_residuals.size:
                    angular_residuals = angular_residuals - np.mean(
                        angular_residuals, axis=0
                    )
                values.extend(
                    (self.config.accel_weight * acceleration_residuals).reshape(-1)
                )
                values.extend(
                    (self.config.alpha_weight * angular_residuals).reshape(-1)
                )
            return np.asarray(values, dtype=float)

        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.5,
            max_nfev=250,
        )
        m, c, I, external_by_segment = self._unpack(
            result.x,
            segment_ids,
            fixed_external_wrench,
        )  # fitted body properties and the fixed hold-stage external wrench
        r = residual(result.x)  # final residual vector
        flags = []  # fit quality flags
        if not result.success:
            flags.append(f"least_squares_status_{result.status}")
        if len(intervals) < 12:
            flags.append("few_intervals")
        if result.cost > 0.5 * len(r):
            flags.append("large_cost")
        return RigidBodyEstimate3D(
            total_m=m,
            r_cm_body=c,
            I_body=I,
            segment_external_wrenches=external_by_segment,
            residual_rms=float(np.sqrt(np.mean(r * r))) if r.size else 0.0,
            quality_flags=tuple(flags),
        )

    def residual_rms(
        self,
        samples: list[IdentificationSample3D],
        m: float,
        r_cm_body: np.ndarray,
        I_body: np.ndarray,
        segment_external_wrenches: dict[int, np.ndarray] | None = None,
    ) -> float:
        """Compute unoptimized dynamics residual RMS for validation and comparisons."""
        intervals = self._intervals(samples)  # adjacent same-segment intervals
        if not intervals:
            return float("nan")
        interval_measurements = self._interval_measurements(
            samples, intervals
        )  # intervals plus smoothed alpha
        c = np.asarray(r_cm_body, dtype=float)  # body-frame CM offset
        I = np.asarray(I_body, dtype=float)  # inertia about CM
        residuals = []  # unweighted acceleration residuals
        for s0, s1, alpha_meas in interval_measurements:
            u = 0.5 * (
                s0.forces.to_array() + s1.forces.to_array()
            )  # average applied thruster vector
            wrench_body = self.nominal_params.allocation_matrix @ u  # body-frame wrench
            ext = (
                np.zeros(6, dtype=float)
                if segment_external_wrenches is None
                else np.asarray(
                    segment_external_wrenches.get(
                        s0.segment_id, np.zeros(6, dtype=float)
                    ),
                    dtype=float,
                )
            )  # world-frame external wrench for this segment
            omega = 0.5 * (
                s0.state.omega + s1.state.omega
            )  # average body-frame angular velocity
            a_pred, alpha_pred = body_origin_acceleration_from_properties(
                s0.state.q,
                omega,
                wrench_body,
                ext,
                m,
                c,
                I,
            )
            residuals.extend(a_pred - 0.5 * (s0.state.a + s1.state.a))
            if alpha_meas is not None:
                residuals.extend(alpha_pred - alpha_meas)
        r = np.asarray(residuals, dtype=float)  # unweighted residual vector
        return float(np.sqrt(np.mean(r * r)))

    def _intervals(
        self, samples: list[IdentificationSample3D]
    ) -> list[tuple[IdentificationSample3D, IdentificationSample3D]]:
        """Return adjacent positive-dt pairs that belong to the same maneuver segment."""
        ordered = sorted(
            samples, key=lambda sample: sample.t
        )  # samples sorted by simulation time
        intervals = []  # adjacent intervals used by the dynamics residual
        for s0, s1 in zip(ordered[:-1], ordered[1:], strict=True):
            dt = s1.t - s0.t  # interval duration
            same_segment = (
                s0.segment_id == s1.segment_id
            )  # interval stays inside one nuisance-wrench segment
            if same_segment and dt >= self.config.min_interval_dt:
                intervals.append((s0, s1))
        return intervals

    def _interval_measurements(
        self,
        samples: list[IdentificationSample3D],
        intervals: list[tuple[IdentificationSample3D, IdentificationSample3D]],
    ) -> list[tuple[IdentificationSample3D, IdentificationSample3D, np.ndarray | None]]:
        """Attach LS-smoothed angular acceleration estimates to identification intervals."""
        by_segment: dict[int, list[tuple[float, np.ndarray]]] = {}
        for sample in samples:
            by_segment.setdefault(sample.segment_id, []).append(
                (sample.t, sample.state.omega)
            )
        measurements = []  # interval records with optional alpha measurement
        for s0, s1 in intervals:
            segment_samples = by_segment.get(
                s0.segment_id, []
            )  # omega samples for this maneuver segment
            mid_t = 0.5 * (
                s0.t + s1.t
            )  # interval midpoint for local derivative evaluation
            local_samples = sorted(
                segment_samples, key=lambda item: abs(item[0] - mid_t)
            )[:7]  # nearest omega window
            alpha = least_squares_derivative(
                local_samples,
                derivative_order=1,
                polynomial_degree=1,
                at_t=mid_t,
                min_samples=3,
            )  # body-frame angular acceleration from LS slope of omega(t)
            measurements.append((s0, s1, alpha))
        return measurements

    def _initial_vector(self) -> np.ndarray:
        """Build the ten-parameter optimizer vector from nominal body properties."""
        L = np.linalg.cholesky(
            self.nominal_params.I_body
        )  # Cholesky factor of nominal inertia
        chol = np.array(
            [
                np.log(L[0, 0]),
                L[1, 0],
                np.log(L[1, 1]),
                L[2, 0],
                L[2, 1],
                np.log(L[2, 2]),
            ],
            dtype=float,
        )
        body = np.concatenate(
            [
                np.array([np.log(self.nominal_params.m)], dtype=float),
                self.nominal_params.r_cm_body,
                chol,
            ]
        )
        return body

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return optimizer bounds for mass, CM and Cholesky inertia."""
        side = self.nominal_params.side_L  # robot side length along X/Z
        c_margin = (
            self.config.cm_bound_margin
        )  # allowed cargo/CM margin around robot body
        c_abs = np.array(
            [
                side / 2.0 + c_margin,
                self.nominal_params.outer_half_width_y + c_margin,
                side / 2.0 + c_margin,
            ],
            dtype=float,
        )  # component-wise CM absolute bounds
        lower_one = np.concatenate(
            [
                np.array([np.log(self.nominal_params.m)], dtype=float),
                -c_abs,
                np.array(
                    [np.log(1e-4), -5.0, np.log(1e-4), -5.0, -5.0, np.log(1e-4)],
                    dtype=float,
                ),
            ]
        )
        upper_one = np.concatenate(
            [
                np.array([np.log(3.0 * self.nominal_params.m)], dtype=float),
                c_abs,
                np.array(
                    [np.log(5.0), 5.0, np.log(5.0), 5.0, 5.0, np.log(5.0)], dtype=float
                ),
            ]
        )
        return lower_one, upper_one

    def _unpack(
        self,
        x: np.ndarray,
        segment_ids: list[int],
        fixed_external_wrench: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        """Decode body properties and attach the fixed hold-stage wrench to each segment."""
        m = float(np.exp(x[0]))  # total mass
        c = np.asarray(x[1:4], dtype=float)  # body-frame CM offset
        chol = x[4:10]  # compact Cholesky parameters
        L = np.array(  # lower-triangular Cholesky factor
            [
                [np.exp(chol[0]), 0.0, 0.0],
                [chol[1], np.exp(chol[2]), 0.0],
                [chol[3], chol[4], np.exp(chol[5])],
            ],
            dtype=float,
        )
        I = L @ L.T  # SPD inertia tensor
        ext_values = np.tile(fixed_external_wrench, (len(segment_ids), 1))
        external_by_segment = {
            segment_id: ext_values[i].copy() for i, segment_id in enumerate(segment_ids)
        }
        return m, c.copy(), 0.5 * (I + I.T), external_by_segment

    @staticmethod
    def _validated_external_wrench(
        initial_external_wrench: np.ndarray | None,
    ) -> np.ndarray:
        """Return the fixed steady-state nuisance wrench with a validated shape."""
        value = (
            np.zeros(6, dtype=float)
            if initial_external_wrench is None
            else np.asarray(initial_external_wrench, dtype=float)
        )
        if value.shape != (6,):
            raise ValueError(
                f"initial_external_wrench must have shape (6,), got {value.shape}"
            )
        return value.copy()


def estimate_steady_external_wrench_3d(
    samples: list[IdentificationSample3D], params: RobotParams3D
) -> np.ndarray:
    """Estimate steady force baseline from hover equilibrium without using true wind."""
    if not samples:
        return np.zeros(6, dtype=float)
    forces_world = []  # force balance estimates per sample
    for sample in samples:
        R = quat_to_rot_matrix(sample.state.q)  # body->world rotation
        F_body = sample.forces.to_wrench(params)[0:3]  # body-frame thrust force
        forces_world.append(-(R @ F_body))
    F_ext = np.mean(
        np.asarray(forces_world, dtype=float), axis=0
    )  # steady world-frame external force
    return np.concatenate([F_ext, np.zeros(3, dtype=float)])


def derive_cargo_estimate_3d(
    nominal_params: RobotParams3D, estimate: RigidBodyEstimate3D
) -> CargoEstimate3D:
    """Derive payload-only mass properties from nominal robot and total identified body properties."""
    cargo_m = float(estimate.total_m - nominal_params.m)  # payload mass estimate
    if cargo_m <= 1e-9:
        return CargoEstimate3D(
            cargo_m=0.0,
            cargo_cm_body=np.zeros(3, dtype=float),
            cargo_I_body=np.zeros((3, 3), dtype=float),
        )

    E = np.eye(3, dtype=float)  # 3x3 identity matrix
    total_cm = np.asarray(estimate.r_cm_body, dtype=float)  # identified total CM
    robot_cm = np.asarray(nominal_params.r_cm_body, dtype=float)  # known robot-only CM
    cargo_cm = (
        estimate.total_m * total_cm - nominal_params.m * robot_cm
    ) / cargo_m  # payload CM
    robot_offset = robot_cm - total_cm  # robot CM offset from total CM
    cargo_offset = cargo_cm - total_cm  # payload CM offset from total CM
    robot_shift = nominal_params.m * (
        (float(robot_offset @ robot_offset) * E) - np.outer(robot_offset, robot_offset)
    )
    cargo_shift = cargo_m * (
        (float(cargo_offset @ cargo_offset) * E) - np.outer(cargo_offset, cargo_offset)
    )
    cargo_I = estimate.I_body - nominal_params.I_body - robot_shift - cargo_shift
    return CargoEstimate3D(
        cargo_m=cargo_m,
        cargo_cm_body=cargo_cm,
        cargo_I_body=0.5 * (cargo_I + cargo_I.T),
    )
