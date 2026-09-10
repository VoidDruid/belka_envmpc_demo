from pathlib import Path

import numpy as np
import pytest

from belka.common import RobotParams3D, ValveOpenings
from belka.actuation import (
    active_thruster_mask,
    desired_thruster_forces_n,
    force_vector_to_openings,
    valve_openings_to_forces,
)


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
JSON_3D = ROOT / "sim" / "models" / "belka3d.json"


def test_opening_curve_endpoints_and_inverse_round_trip():
    params = RobotParams3D.from_json(JSON_3D)
    openings = ValveOpenings.from_array(np.linspace(0.0, 0.3, params.nu))

    desired = desired_thruster_forces_n(openings, params)
    recovered = force_vector_to_openings(desired, params)

    np.testing.assert_allclose(recovered.to_array(), openings.to_array(), atol=1e-10)
    assert desired[0] == pytest.approx(0.0)
    assert desired[-1] < params.per_thruster_force_caps_n[-1]


def test_side_limit_is_applied_per_impeller_group():
    params = RobotParams3D.from_json(JSON_3D)
    forces = valve_openings_to_forces(
        ValveOpenings.from_array(np.ones(params.nu)), params
    )
    values = forces.to_array()

    for group, limit in zip(
        params.side_force_groups,
        params.side_force_limits_n,
        strict=True,
    ):
        assert np.sum(values[list(group)]) == pytest.approx(limit)


def test_planar_d3_mask_zeros_vertical_channels_before_force_conversion():
    params = RobotParams3D.from_json(JSON_3D)
    mask = active_thruster_mask(params.nu, tuple(range(8)))
    forces = valve_openings_to_forces(
        ValveOpenings.from_array(np.ones(params.nu)),
        params,
        active_mask=mask,
    )

    np.testing.assert_allclose(forces.to_array()[8:12], 0.0, atol=0.0)
    assert np.all(forces.to_array()[:8] > 0.0)
