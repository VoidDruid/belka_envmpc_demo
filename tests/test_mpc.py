from pathlib import Path

import numpy as np
import pytest

from belka.common import RobotParams3D, RobotState3D, ValveOpenings
from belka.controllers.mpc import MPCController3D
from belka.planners.simple import EndpointPlanner3D
from belka.shared.observers import WrenchEstimate3D


pytestmark = [pytest.mark.unit, pytest.mark.slow]
JSON_3D = Path(__file__).resolve().parents[1] / "sim" / "models" / "belka3d.json"


def test_mpc3d_setup_one_step_and_history_estimation_integration():
    params = RobotParams3D.from_json(JSON_3D)
    mpc = MPCController3D(
        params,
        state_weight=np.eye(13, dtype=np.float64),
        thruster_weight=np.eye(params.nu, dtype=np.float64) * 0.1,
        thruster_delta_weight=np.eye(params.nu, dtype=np.float64) * 0.2,
        horizon=4,
        dt=0.05,
        planner=EndpointPlanner3D(
            RobotState3D(p=np.array([0.2, 0.0, 0.0], dtype=float))
        ),
        active_thrusters=tuple(range(8)),
    )

    current = ValveOpenings.zeros(params.nu)
    estimation = WrenchEstimate3D(Fx=0.1)
    control = mpc(0.0, current, RobotState3D(), estimation)

    assert control.nu == params.nu
    assert len(mpc.history.ts) == 1
    assert mpc.history.reference_state[-1] is not None
    np.testing.assert_allclose(mpc.history.reference_state[-1].p, [0.2, 0.0, 0.0])
    assert mpc.history.estimations[-1] == estimation
    assert np.all(control.to_array() >= -1e-9)
    assert np.all(control.to_array() <= 1.0 + 1e-9)
    np.testing.assert_allclose(control.to_array()[8:12], 0.0, atol=0.0)
    assert mpc._runtime_model_parameters(None).shape == (params.nu + 4,)
    assert mpc.effective_solver_options["nlp_solver_type"] == "SQP"
    assert mpc.effective_solver_options["hessian_approx"] == "GAUSS_NEWTON"
    assert mpc.effective_solver_options["qp_solver"] == "PARTIAL_CONDENSING_HPIPM"
    assert mpc.effective_solver_options["nlp_solver_max_iter"] == 10
    assert mpc.effective_solver_options["qp_solver_warm_start"] == 1
    for name in (
        "nlp_solver_tol_stat",
        "nlp_solver_tol_eq",
        "nlp_solver_tol_ineq",
        "nlp_solver_tol_comp",
    ):
        assert mpc.effective_solver_options[name] == pytest.approx(1e-6)


def test_mpc3d_reference_matrix_aligns_quaternion_sign_to_current_state():
    params = RobotParams3D.from_json(JSON_3D)
    mpc = MPCController3D(
        params,
        state_weight=np.eye(13, dtype=np.float64),
        thruster_weight=np.eye(params.nu, dtype=np.float64),
        thruster_delta_weight=np.eye(params.nu, dtype=np.float64),
        horizon=2,
        dt=0.05,
        planner=EndpointPlanner3D(RobotState3D()),
    )

    refs = mpc._reference_state_matrix(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        [RobotState3D(q=np.array([-1.0, 0.0, 0.0, 0.0], dtype=float))],
    )

    np.testing.assert_allclose(refs[3:7, 0], np.array([1.0, -0.0, -0.0, -0.0]))
