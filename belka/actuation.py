"""Spatial valve-opening to force conversion shared by control and simulation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from belka.common import RobotParams3D, ThrusterForces, ValveOpenings


def active_thruster_mask(
    nu: int,
    active_thrusters: Sequence[int] | None,
) -> np.ndarray:
    """Build a validated 0/1 mask; None means every thruster is active."""
    if active_thrusters is None:
        return np.ones(nu, dtype=float)
    indices = tuple(int(index) for index in active_thrusters)
    if len(set(indices)) != len(indices):
        raise ValueError("active_thrusters must not contain duplicates")
    if any(index < 0 or index >= nu for index in indices):
        raise ValueError(f"active_thrusters indices must be in [0, {nu})")
    mask = np.zeros(nu, dtype=float)
    mask[list(indices)] = 1.0
    return mask


def desired_thruster_forces_n(
    openings: ValveOpenings | np.ndarray,
    params: RobotParams3D,
    *,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert openings to individual forces before the shared side limit."""
    if params.thruster_model is None:
        raise RuntimeError("RobotParams3D has no thruster model")
    values = (
        openings.to_array()
        if hasattr(openings, "to_array")
        else np.asarray(openings, dtype=float)
    )
    values = np.asarray(values, dtype=float)
    if values.shape != (params.nu,):
        raise ValueError(f"openings must have shape ({params.nu},), got {values.shape}")
    values = np.clip(values, 0.0, 1.0)
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=float)
        if mask.shape != (params.nu,):
            raise ValueError(
                f"active_mask must have shape ({params.nu},), got {mask.shape}"
            )
        values = values * mask
    fractions = params.thruster_model.valve_opening_to_fraction.evaluate(values)
    return params.per_thruster_force_caps_n * np.asarray(fractions, dtype=float)


def apply_side_force_limits_n(
    desired_forces: np.ndarray,
    params: RobotParams3D,
) -> np.ndarray:
    """Apply the measured per-impeller summed-force saturation proportionally."""
    actual = np.maximum(np.asarray(desired_forces, dtype=float), 0.0).copy()
    if actual.shape != (params.nu,):
        raise ValueError(
            f"desired_forces must have shape ({params.nu},), got {actual.shape}"
        )
    for group, limit in zip(
        params.side_force_groups, params.side_force_limits_n, strict=True
    ):
        total = float(np.sum(actual[list(group)]))
        if total > limit and total > 0.0:
            actual[list(group)] *= float(limit) / total
    return actual


def valve_openings_to_forces(
    openings: ValveOpenings | np.ndarray,
    params: RobotParams3D,
    *,
    active_mask: np.ndarray | None = None,
) -> ThrusterForces:
    """Convert valve openings into physical per-thruster forces in newtons."""
    desired = desired_thruster_forces_n(openings, params, active_mask=active_mask)
    return ThrusterForces.from_array(apply_side_force_limits_n(desired, params))


def force_vector_to_openings(
    forces: ThrusterForces | np.ndarray,
    params: RobotParams3D,
    *,
    active_mask: np.ndarray | None = None,
) -> ValveOpenings:
    """Invert the static individual-force map after enforcing plant limits."""
    if params.thruster_model is None:
        raise RuntimeError("RobotParams3D has no thruster model")
    values = (
        forces.to_array()
        if hasattr(forces, "to_array")
        else np.asarray(forces, dtype=float)
    )
    values = np.minimum(
        np.maximum(np.asarray(values, dtype=float), 0.0),
        params.per_thruster_force_caps_n,
    )
    values = apply_side_force_limits_n(values, params)
    caps = params.per_thruster_force_caps_n
    fractions = np.divide(values, caps, out=np.zeros_like(values), where=caps > 0.0)

    curve = params.thruster_model.valve_opening_to_fraction
    lower = 1.0 / (1.0 + np.exp(curve.steepness * curve.midpoint))
    upper = 1.0 / (1.0 + np.exp(-curve.steepness * (1.0 - curve.midpoint)))
    raw = np.clip(lower + fractions * (upper - lower), 1e-12, 1.0 - 1e-12)
    openings = curve.midpoint + np.log(raw / (1.0 - raw)) / curve.steepness
    openings = np.clip(openings, 0.0, 1.0)
    if active_mask is not None:
        openings *= np.asarray(active_mask, dtype=float)
    return ValveOpenings.from_array(openings)


def allocate_wrench_to_openings(
    wrench_body: np.ndarray,
    params: RobotParams3D,
    *,
    active_mask: np.ndarray | None = None,
) -> ValveOpenings:
    """Allocate a requested body wrench in force space, then invert to openings."""
    from scipy.optimize import linprog

    wrench = np.asarray(wrench_body, dtype=float)
    if wrench.shape != (6,):
        raise ValueError(f"wrench_body must have shape (6,), got {wrench.shape}")
    allocation = np.asarray(params.allocation_matrix, dtype=float)
    caps = params.per_thruster_force_caps_n.copy()
    if active_mask is not None:
        caps *= np.asarray(active_mask, dtype=float)

    row_authority = np.max(np.abs(allocation) * caps[None, :], axis=1)
    row_scale = np.ones(6, dtype=float)
    controllable = row_authority > 1e-12
    row_scale[controllable] = 1.0 / row_authority[controllable]
    scaled_allocation = row_scale[:, None] * allocation
    scaled_wrench = row_scale * wrench

    k = len(wrench)  # wrench dimension and number of L1 residual slacks
    error_constraints = np.block(
        [
            [scaled_allocation, -np.eye(k, dtype=float)],
            [-scaled_allocation, -np.eye(k, dtype=float)],
        ]
    )
    error_upper = np.concatenate((scaled_wrench, -scaled_wrench))
    side_constraints = []
    side_upper = []
    for group, limit in zip(
        params.side_force_groups, params.side_force_limits_n, strict=True
    ):
        row = np.zeros(params.nu + k, dtype=float)  # summed force for one impeller side
        row[list(group)] = 1.0
        side_constraints.append(row)
        side_upper.append(float(limit))
    A_ub = np.vstack([*side_constraints, *error_constraints])
    b_ub = np.concatenate((np.asarray(side_upper, dtype=float), error_upper))
    result = linprog(
        c=np.concatenate(
            (np.full(params.nu, 1e-9, dtype=float), np.ones(k, dtype=float))
        ),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0.0, float(cap)) for cap in caps] + [(0.0, None)] * k,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"wrench allocation failed: {result.message}")
    forces = np.asarray(result.x[: params.nu], dtype=float)
    return force_vector_to_openings(forces, params, active_mask=active_mask)


def symbolic_desired_thruster_forces(
    ca,
    openings,
    force_caps_n,
    midpoint,
    steepness,
    active_mask: np.ndarray,
):
    """Build differentiable desired forces from runtime actuator parameters."""
    lower = 1.0 / (1.0 + ca.exp(steepness * midpoint))
    upper = 1.0 / (1.0 + ca.exp(-steepness * (1.0 - midpoint)))
    clipped = ca.fmin(1.0, ca.fmax(0.0, openings))
    raw = 1.0 / (1.0 + ca.exp(-steepness * (clipped - midpoint)))
    fraction = (raw - lower) / (upper - lower)
    caps = ca.times(
        force_caps_n,
        ca.DM(np.asarray(active_mask, dtype=float)),
    )
    return ca.times(caps, fraction)


def symbolic_applied_thruster_forces(
    ca,
    desired_forces,
    side_force_groups: tuple[tuple[int, ...], tuple[int, ...]],
    side_limits_n,
):
    """Apply runtime proportional side saturation to a CasADi force vector."""
    actual = desired_forces
    nu = int(desired_forces.shape[0])
    for group_index, group in enumerate(side_force_groups):
        indices = list(group)
        total = ca.sum1(actual[indices])
        scale = ca.fmin(1.0, side_limits_n[group_index] / (total + 1e-12))
        updated = [
            actual[index] * scale if index in group else actual[index]
            for index in range(nu)
        ]
        actual = ca.vertcat(*updated)
    return actual
