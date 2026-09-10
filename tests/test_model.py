import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from belka.common import (
    CargoParams,
    PLANAR_THRUSTER_NAMES,
    RobotParams3D,
    RobotState3D,
    quat_to_rot_matrix,
)
from belka.model import body_origin_acceleration_3d, robot_derivative_3d
from sim import SimParams
from sim.core import SimIO


pytestmark = pytest.mark.unit
JSON_3D = Path(__file__).resolve().parents[1] / "sim" / "models" / "belka3d.json"


def _params_json(tmp_path: Path, **robot_updates) -> Path:
    """Create a test-local 3D robot JSON with explicit robot-block edits."""
    cfg = json.loads(JSON_3D.read_text())
    for name, value in robot_updates.items():
        if value is None:
            cfg["robot"].pop(name, None)
        else:
            cfg["robot"][name] = value
    json_path = tmp_path / "belka3d.json"
    json_path.write_text(json.dumps(cfg))
    return json_path


def test_model_3d_dynamics_zero_state_is_stationary():
    params = RobotParams3D.from_json(JSON_3D)
    deriv = robot_derivative_3d(
        RobotState3D().array,
        np.zeros(params.nu, dtype=float),
        params.allocation_matrix,
        np.zeros(6, dtype=float),
        params,
    )

    assert deriv.shape == (13,)
    np.testing.assert_allclose(deriv, np.zeros(13), atol=1e-10)


def test_thruster_sites_match_telescopic_offsets_and_allocation():
    sim = SimIO(
        sim_params=SimParams(
            t_max=0.01,
            full_odom_hz=20.0,
            highfreq_odom_hz=20.0,
        ),
        initial_state=RobotState3D(),
    )
    params = sim.robot_params
    hx = params.side_L / 2.0  # half-size along body X
    hz = params.side_L / 2.0  # half-size along body Z
    hy = params.outer_half_width_y  # half-size along body Y
    ax = hx - params.l  # tangent offset along body X
    ay = hy - params.l  # tangent offset along body Y
    az = hz - params.l  # tangent offset along body Z
    expected_abs_tangents = {
        0: np.array([ay, az], dtype=float),
        1: np.array([ax, az], dtype=float),
        2: np.array([ax, ay], dtype=float),
    }
    expected_abs_normal = np.array([hx, hy, hz], dtype=float)
    alloc = params.allocation_matrix

    for i, site_id in enumerate(sim._site_ids):
        pos = sim.model.site_pos[site_id]  # body-frame позиция thruster site
        direction = alloc[0:3, i]  # body-frame направление силы движителя
        normal_axis = int(np.argmax(np.abs(direction)))
        tangent_axes = [axis for axis in range(3) if axis != normal_axis]

        np.testing.assert_allclose(
            abs(pos[normal_axis]), expected_abs_normal[normal_axis], atol=1e-12
        )
        np.testing.assert_allclose(
            np.abs(pos[tangent_axes]), expected_abs_tangents[normal_axis], atol=1e-12
        )
        np.testing.assert_allclose(sim.model.site_size[site_id, 0], 0.012, atol=1e-12)
        np.testing.assert_allclose(alloc[3:6, i], np.cross(pos, direction), atol=1e-12)


def test_mujoco_actuators_render_from_canonical_d3_layout():
    sim = SimIO(
        sim_params=SimParams(
            t_max=0.01,
            full_odom_hz=20.0,
            highfreq_odom_hz=20.0,
        ),
        initial_state=RobotState3D(),
    )

    np.testing.assert_allclose(
        sim.model.actuator_gear[:, :3],
        sim.robot_params.thruster_directions_body,
        atol=1e-12,
    )
    actuator_names = [
        mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(sim.robot_params.nu)
    ]
    assert actuator_names[:8] == list(PLANAR_THRUSTER_NAMES)
    assert actuator_names[8:] == ["F9", "F10", "F11", "F12"]


def test_degenerate_two_half_sides_reproduce_solid_cube_mass_properties():
    params = RobotParams3D.from_json(JSON_3D)
    expected_m = 2.0 * params.side_m  # total mass of two side boxes
    expected_I = (
        np.eye(3, dtype=float) * expected_m * params.side_L**2 / 6.0
    )  # solid cube inertia

    assert params.m == pytest.approx(expected_m)
    np.testing.assert_allclose(params.r_cm_body, np.zeros(3, dtype=float), atol=1e-12)
    np.testing.assert_allclose(params.I_body, expected_I, atol=1e-12)


def test_explicit_measured_mass_properties_are_preserved_without_payload(tmp_path):
    measured_cm = np.array([0.012, -0.018, 0.025], dtype=float)
    measured_inertia = np.array(
        [
            [0.31, 0.012, -0.006],
            [0.012, 0.42, 0.009],
            [-0.006, 0.009, 0.53],
        ],
        dtype=float,
    )
    params = RobotParams3D.from_json(
        _params_json(
            tmp_path,
            m=7.4,
            side_m=None,
            mass_properties={
                "center_of_mass_body_m": measured_cm.tolist(),
                "inertia_body_kg_m2": measured_inertia.tolist(),
            },
        )
    )

    mass, center, inertia = params.composite_body_properties()

    assert mass == pytest.approx(7.4)
    np.testing.assert_allclose(center, measured_cm, atol=1e-12)
    np.testing.assert_allclose(inertia, measured_inertia, atol=1e-12)


def test_asymmetric_cargo_changes_true_mass_cm_and_inertia_without_changing_robot_params(
    tmp_path,
):
    params = RobotParams3D.from_json(
        _params_json(
            tmp_path,
            base_L=0.16,
            base_w=0.04,
            base_m=0.7,
            extension_L=0.08,
        )
    )
    cargo = CargoParams.from_mapping(
        {
            "cargo_x": 0.18,
            "cargo_y": 0.11,
            "cargo_z": 0.07,
            "cargo_m": 1.2,
            "cargo_pos_x": 0.09,
            "cargo_pos_y": -0.06,
            "cargo_pos_z": -0.20,
        }
    )

    true_m, true_cm, true_I = params.composite_body_properties(cargo)

    cargo_pos = np.array(
        [cargo.cargo_pos_x, cargo.cargo_pos_y, cargo.cargo_pos_z], dtype=float
    )
    expected_m = params.m + cargo.cargo_m  # true robot + cargo mass
    expected_cm = (params.m * params.r_cm_body + cargo.cargo_m * cargo_pos) / expected_m
    np.testing.assert_allclose(true_m, expected_m, atol=1e-12)
    np.testing.assert_allclose(true_cm, expected_cm, atol=1e-12)

    E = np.eye(3, dtype=float)  # 3×3 identity
    robot_offset = params.r_cm_body - expected_cm  # robot-only CM offset from true CM
    cargo_offset = cargo_pos - expected_cm  # cargo CM offset from true CM
    cargo_I_center = cargo.inertia_about_cm()  # cargo box inertia about own center
    expected_I = (
        params.I_body
        + params.m
        * (
            (float(robot_offset @ robot_offset) * E)
            - np.outer(robot_offset, robot_offset)
        )
        + cargo_I_center
        + cargo.cargo_m
        * (
            (float(cargo_offset @ cargo_offset) * E)
            - np.outer(cargo_offset, cargo_offset)
        )
    )
    np.testing.assert_allclose(true_I, expected_I, atol=1e-12)
    assert abs(true_I[0, 1]) > 1e-5
    assert abs(true_I[0, 2]) > 1e-5


def test_body_origin_acceleration_accounts_for_nonzero_cm_offset(tmp_path):
    params = RobotParams3D.from_json(
        _params_json(
            tmp_path,
            base_L=0.20,
            base_w=0.05,
            base_m=0.9,
            extension_L=0.05,
        )
    )
    q = np.array([0.96, 0.12, -0.17, 0.19], dtype=float)
    q = q / np.linalg.norm(q)
    omega = np.array([0.2, -0.1, 0.15], dtype=float)
    wrench_body = np.array([0.4, -0.2, 0.3, 0.03, -0.04, 0.05], dtype=float)
    external_wrench = np.array([0.08, -0.03, 0.04, -0.02, 0.01, 0.03], dtype=float)

    a_origin, alpha = body_origin_acceleration_3d(
        q, omega, wrench_body, external_wrench, params
    )

    R = quat_to_rot_matrix(q)  # body→world rotation matrix
    c = params.r_cm_body  # body-frame vector from origin to CM
    I_body = params.I_body  # inertia tensor about CM
    F_body = wrench_body[0:3]  # body-frame thrust force
    M_origin = wrench_body[3:6]  # body-frame thrust moment about origin
    M_ext_body = R.T @ external_wrench[3:6]  # external moment about CM in body frame
    expected_alpha = np.linalg.solve(
        I_body,
        M_origin - np.cross(c, F_body) + M_ext_body - np.cross(omega, I_body @ omega),
    )
    expected_a = (R @ F_body + external_wrench[0:3]) / params.m - R @ (
        np.cross(expected_alpha, c) + np.cross(omega, np.cross(omega, c))
    )

    np.testing.assert_allclose(alpha, expected_alpha, atol=1e-12)
    np.testing.assert_allclose(a_origin, expected_a, atol=1e-12)
