import numpy as np
import pytest
from types import SimpleNamespace

from experiments.research.common import aggregate, flight_metrics, write_run_log
from experiments.research.paper_1_external_wrench.run import _cases as paper1_cases
from experiments.research.paper_1_external_wrench.run import (
    _base_mapping as paper1_base_mapping,
)
from experiments.research.paper_1_external_wrench.run import (
    _configure_experiment_mapping,
)
from experiments.research.paper_1_external_wrench.run import _profile
from experiments.research.paper_1_external_wrench.run import controller_wrench_input
from experiments.research.paper_1_external_wrench.applicability import (
    COMPOSITIONS,
    EvaluationCase,
    bandwidth_cases,
    compensable_wrench_limit,
    confirmation_cases,
    screening_cases,
)
from belka.common import RobotParams3D, RobotState3D, ValveOpenings
from belka.shared.controllers import ControlHistory
from belka.shared.observers import WrenchEstimate3D

pytestmark = pytest.mark.unit


def test_paper1_case_matrix_has_expected_size_and_modes():
    assert len(paper1_cases("exp1_estimation")) == 15
    assert len(paper1_cases("exp2_control_comparison")) == 15
    assert len(paper1_cases("exp3_robustness")) == 45
    assert {case["mode"] for case in paper1_cases("exp2_control_comparison")} == {
        "ignore",
        "estimate",
        "oracle",
    }


def test_envmpc_applicability_case_matrices_have_preregistered_sizes():
    assert len(screening_cases()) == 36
    selected = {
        composition: {
            "force_direction": [1.0, 0.0, 0.0]
            if composition != "moment_only"
            else [0.0, 0.0, 0.0],
            "moment_direction": [0.0, 0.0, 1.0]
            if composition != "force_only"
            else [0.0, 0.0, 0.0],
            "relative_scale": 0.35,
            "modulation_frequency_hz": 0.015,
        }
        for composition in COMPOSITIONS
    }
    assert len(bandwidth_cases(selected)) == 18
    confirmation = confirmation_cases(selected)
    assert len(confirmation) == 45
    assert {case.seed for case in confirmation} == set(range(101, 106))


def test_static_compensable_limit_balances_wrench_and_respects_limits():
    params = RobotParams3D.from_json("experiments/3D/cube.json")
    case = EvaluationCase(
        stage="screening",
        candidate_id="force_x_s35",
        composition="force_only",
        mode="ignore",
        force_direction=(1.0, 0.0, 0.0),
        moment_direction=(0.0, 0.0, 0.0),
        relative_scale=0.35,
        modulation_frequency_hz=0.0,
        modulation_amplitude=0.0,
        seed=0,
    )

    result = compensable_wrench_limit(params, case)

    assert result["lambda_max"] > 0.0
    assert result["balance_residual_norm"] < 1e-8
    assert np.all(np.asarray(result["thruster_forces_n"]) <= params.per_thruster_force_caps_n + 1e-9)
    assert np.all(np.asarray(result["group_totals_n"]) <= params.side_force_limits_n + 1e-9)


def test_applicability_case_rejects_nonunit_active_direction():
    with pytest.raises(ValueError, match="unit length"):
        EvaluationCase(
            stage="screening",
            candidate_id="bad",
            composition="force_only",
            mode="ignore",
            force_direction=(2.0, 0.0, 0.0),
            moment_direction=(0.0, 0.0, 0.0),
            relative_scale=0.35,
            modulation_frequency_hz=0.0,
            modulation_amplitude=0.0,
            seed=0,
        )


def test_paper1_step_profile_is_zero_before_step_and_nonzero_after():
    params = RobotParams3D.from_json("experiments/3D/cube.json")
    profile = _profile(params, "step", force_ratio=0.25, seed=0)
    np.testing.assert_allclose(profile(11.9, None), np.zeros(6), atol=1e-12)
    assert np.linalg.norm(profile(12.1, None)) > 0.0


def test_only_oracle_mode_receives_true_wrench():
    estimate = WrenchEstimate3D(Fx=0.1)
    truth = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    np.testing.assert_allclose(controller_wrench_input("ignore", estimate, truth).to_array(), 0.0)
    assert controller_wrench_input("estimate", estimate, truth) is estimate
    np.testing.assert_allclose(
        controller_wrench_input("oracle", estimate, truth).to_array(), truth
    )


def test_paper1_estimation_series_holds_one_pose_for_sixty_seconds():
    mapping = _configure_experiment_mapping(paper1_base_mapping(seed=0, noise_multiplier=1.0), "exp1_estimation")
    assert mapping["run"]["target_p"] == mapping["run"]["initial_p"]
    assert mapping["run"]["target_angles"] == mapping["run"]["initial_angles"]
    assert mapping["run"]["t_max"] == 60.0


def test_metric_aggregation_reports_mean_and_sample_deviation():
    rows = [
        {"experiment": "a", "mode": "x", "seed": 0, "metric": 1.0},
        {"experiment": "a", "mode": "x", "seed": 1, "metric": 3.0},
    ]
    result = aggregate(rows, ["experiment", "mode"])[0]
    assert result["n"] == 2
    assert result["metric_mean"] == 2.0
    assert result["metric_std"] == np.sqrt(2.0)


def test_run_log_records_revision_case_and_artifacts(tmp_path):
    path = tmp_path / "run.log"
    write_run_log(
        path,
        experiment="exp",
        case={"seed": 3, "mode": "estimate"},
        revision="abc123",
        artifacts=["snapshot.pkl", "metrics.csv"],
    )
    value = path.read_text(encoding="utf-8")
    assert "status=completed" in value
    assert "simulation_revision=abc123" in value
    assert 'case={"mode": "estimate", "seed": 3}' in value
    assert "artifacts=snapshot.pkl,metrics.csv" in value


def test_flight_metrics_separates_truth_tracking_estimated_tracking_and_observer():
    params = RobotParams3D.from_json("experiments/3D/cube.json")
    reference = RobotState3D()
    true_state = RobotState3D(
        p=np.array([1.0, 0.0, 0.0]),
        q=np.array([np.cos(0.1), 0.0, 0.0, np.sin(0.1)]),
    )
    estimated_state = RobotState3D(
        p=np.array([3.0, 0.0, 0.0]),
        q=np.array([np.cos(0.25), 0.0, 0.0, np.sin(0.25)]),
    )
    history = ControlHistory()
    history.update(
        ts=0.0,
        state=estimated_state,
        reference_state=reference,
        control=ValveOpenings.zeros(params.nu),
        estimation=WrenchEstimate3D.zeros(),
    )
    snapshot = {
        "control_history": history,
        "sim_history": SimpleNamespace(
            ts=[0.0],
            real_state=[true_state],
            external_wrench=[np.zeros(6)],
        ),
        "effective_robot_params": params,
    }

    metrics = flight_metrics(snapshot)

    assert metrics["tracking_position_rmse_m"] == pytest.approx(1.0)
    assert metrics["estimated_tracking_position_rmse_m"] == pytest.approx(3.0)
    assert metrics["observer_position_rmse_m"] == pytest.approx(2.0)
    assert metrics["tracking_orientation_rmse_rad"] == pytest.approx(0.2)
    assert metrics["estimated_tracking_orientation_rmse_rad"] == pytest.approx(0.5)
    assert metrics["observer_orientation_rmse_rad"] == pytest.approx(0.3)
    assert "control_effort_n2s" not in metrics
    assert metrics["control_effort_normalized2_s"] == pytest.approx(0.0)


def test_flight_metrics_rejects_missing_exact_reference():
    params = RobotParams3D.from_json("experiments/3D/cube.json")
    history = ControlHistory()
    history.update(
        ts=0.0,
        state=RobotState3D(),
        control=ValveOpenings.zeros(params.nu),
    )
    snapshot = {
        "control_history": history,
        "sim_history": SimpleNamespace(
            ts=[0.0], real_state=[RobotState3D()], external_wrench=[np.zeros(6)]
        ),
        "effective_robot_params": params,
    }

    with pytest.raises(ValueError, match="reference_state is missing"):
        flight_metrics(snapshot)
