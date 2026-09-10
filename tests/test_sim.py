import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from belka.common import (
    CargoParams,
    InertialOdom3D,
    OdomInput3D,
    RobotState3D,
    ThrusterForces,
    ValveOpenings,
    VisualOdom3D,
)
from belka.shared.logic import HighlevelLogic
from belka.shared.observers import WrenchEstimate3D
from sim import SimParams
from sim.core import SimIO
from sim.shared import DummyViewer


pytestmark = pytest.mark.integration


class LegendScene:
    def __init__(self):
        self.ngeom = 0
        self.geoms = []


class LegendViewer(DummyViewer):
    def __init__(self):
        self.user_scn = LegendScene()
        self.texts = None
        self.sync_count = 0
        self.set_texts_count = 0

    def sync(self):
        self.sync_count += 1

    def set_texts(self, texts):
        self.texts = texts
        self.set_texts_count += 1


class SceneViewer(DummyViewer):
    def __init__(self, model, maxgeom: int = 64):
        self.user_scn = mujoco.MjvScene(model, maxgeom=maxgeom)
        self.texts = None

    def set_texts(self, texts):
        self.texts = texts


class ConstantLogic3D(HighlevelLogic):
    def __init__(self, command: ValveOpenings, events: list[str] | None = None):
        self.command = command
        self.events = events

    def configure(self, initial_state: RobotState3D) -> None:
        self._estimated_state = initial_state

    def observation_input_type(self) -> type:
        return RobotState3D

    def think(self, t: float, actual_actuation: ValveOpenings) -> ValveOpenings:
        del t, actual_actuation
        if self.events is not None:
            self.events.append("think")
        return self.command

    def control_hz(self) -> float:
        return 20.0


class CaptureConfigLogic3D(ConstantLogic3D):
    def configure(self, initial_state: RobotState3D) -> None:
        self.initial_state = initial_state


class DynamicHzLogic3D(ConstantLogic3D):
    def __init__(self, control_hz: float = 20.0):
        super().__init__(ValveOpenings.zeros(12))
        self._control_hz = float(control_hz)
        self.think_ts: list[float] = []

    def think(self, t: float, actual_actuation: ValveOpenings) -> ValveOpenings:
        del actual_actuation
        self.think_ts.append(float(t))
        return self.command

    def control_hz(self) -> float:
        return self._control_hz


class OdomCaptureLogic3D(HighlevelLogic):
    def __init__(self):
        self.events: list[tuple[str, float]] = []
        self.observations: list[OdomInput3D] = []

    def configure(self, initial_state: RobotState3D) -> None:
        self.initial_state = initial_state
        self._estimated_state = initial_state

    def observation_input_type(self) -> type:
        return OdomInput3D

    def observe(
        self,
        t: float,
        input_data: OdomInput3D,
        actual_actuation: ValveOpenings,
    ) -> RobotState3D:
        del actual_actuation
        self.events.append(("observe", float(t)))
        self.observations.append(input_data)
        p = (
            np.zeros(3, dtype=float)
            if input_data.visual is None
            else input_data.visual.p.copy()
        )  # observed position
        state = RobotState3D(
            p=p, q=input_data.inertial.q.copy(), omega=input_data.inertial.omega.copy()
        )
        self._estimated_state = state
        return state

    def think(self, t: float, actual_actuation: ValveOpenings) -> ValveOpenings:
        del actual_actuation
        self.events.append(("think", float(t)))
        return ValveOpenings.zeros(12)

    def control_hz(self) -> float:
        return 20.0


class ActuationCaptureLogic3D(ConstantLogic3D):
    def __init__(self):
        super().__init__(ValveOpenings.from_array(np.ones(12)))
        self.feedback: list[np.ndarray] = []

    def observe(self, t, input_data, actual_actuation):
        del t
        self.feedback.append(actual_actuation.to_array())
        return input_data


def _params(t_max: float = 0.01) -> SimParams:
    return SimParams(
        t_max=t_max,
        full_odom_hz=20.0,
        highfreq_odom_hz=20.0,
    )


def test_simulator_3d_open_loop_zero_motion():
    sim = SimIO(_params(), RobotState3D())
    sim.viewer = DummyViewer()
    sim.visualize = lambda: None

    for _ in range(5):
        sim.step()

    np.testing.assert_allclose(sim.state.p, np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(sim.state.v, np.zeros(3), atol=1e-8)


def test_simulator_3d_keeps_full_state_and_6d_external_wrench_history():
    wrench = WrenchEstimate3D(Fx=1.0, Fy=2.0, Fz=3.0, Mx=4.0, My=5.0, Mz=6.0)
    sim = SimIO(
        _params(),
        RobotState3D(),
        highlevel_control=ConstantLogic3D(ValveOpenings.zeros(12)),
        external_wrench=lambda t, state: wrench,
    )
    sim.viewer = DummyViewer()
    sim.visualize = lambda: None

    sim.step()

    assert isinstance(sim.history.real_state[-1], RobotState3D)
    np.testing.assert_allclose(
        sim.history.external_wrench[-1].to_array(), wrench.to_array()
    )


def test_simulator_3d_visual_thruster_forces_are_smoothed_without_changing_real_forces():
    sim = SimIO(_params(), RobotState3D())
    target_forces = np.ones(sim.robot_params.nu, dtype=float)
    sim._applied_forces = ThrusterForces.from_array(target_forces)

    first_visual = sim._visualized_thruster_forces()
    second_visual = sim._visualized_thruster_forces()

    assert np.all(first_visual > 0.0)
    assert np.all(second_visual > first_visual)
    assert np.all(second_visual < target_forces)
    np.testing.assert_allclose(sim._applied_forces.to_array(), target_forces)

    sim._applied_forces = ThrusterForces.zeros(sim.robot_params.nu)
    fading_visual = sim._visualized_thruster_forces()

    assert np.all(fading_visual > 0.0)
    assert np.all(fading_visual < second_visual)
    np.testing.assert_allclose(
        sim._applied_forces.to_array(),
        np.zeros(sim.robot_params.nu),
    )


def test_simulator_3d_visualize_sets_marker_legend_overlay():
    sim = SimIO(_params(), RobotState3D())
    viewer = LegendViewer()
    sim.viewer = viewer

    sim.visualize()

    assert viewer.texts is not None
    state_overlay, legend_overlay = viewer.texts
    state_font, state_gridpos, state_left_text, state_right_text = state_overlay
    font, gridpos, left_text, right_text = legend_overlay
    assert state_font == mujoco.mjtFontScale.mjFONTSCALE_150
    assert state_gridpos == mujoco.mjtGridPos.mjGRID_TOP
    assert state_left_text.startswith("State:")
    assert state_right_text == ""
    assert font == mujoco.mjtFontScale.mjFONTSCALE_100
    assert gridpos == mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
    assert left_text
    assert right_text


def test_simulator_3d_owns_perceived_position_marker_toggle():
    perceived = RobotState3D(p=np.array([0.2, -0.1, 0.3], dtype=float))

    shown = SimIO(_params(), RobotState3D(), sensor_noise=lambda t, state: perceived)
    shown.viewer = SceneViewer(shown.model)
    shown.step()
    shown.visualize()

    hidden = SimIO(
        _params(),
        RobotState3D(),
        sensor_noise=lambda t, state: perceived,
        show_perceived_position=False,
    )
    hidden.viewer = SceneViewer(hidden.model)
    hidden.step()
    hidden.visualize()

    assert shown._last_perceived_state is perceived
    assert shown.viewer.user_scn.ngeom == 1
    assert hidden._last_perceived_state is perceived
    assert hidden.viewer.user_scn.ngeom == 0


def test_simulator_3d_renders_static_front_arrows_and_can_disable_them(tmp_path):
    with_arrows = SimIO(_params(), RobotState3D(), output_dir=tmp_path / "with_arrows")
    without_arrows = SimIO(
        _params(),
        RobotState3D(),
        output_dir=tmp_path / "without_arrows",
        show_front_arrows=False,
    )

    xml_with = (with_arrows.output_dir / "model.xml").read_text()
    xml_without = (without_arrows.output_dir / "model.xml").read_text()

    assert "front_marker" not in xml_with
    assert 'geom name="real_front_shaft"' in xml_with
    assert 'geom name="presented_front_shaft"' in xml_with
    assert "front_marker" not in xml_without
    assert "real_front_shaft" not in xml_without
    assert "presented_front_shaft" not in xml_without


def test_simulator_3d_viewer_sync_is_throttled_to_display_rate():
    sim = SimIO(_params(t_max=0.01), RobotState3D())
    viewer = LegendViewer()
    sim.viewer = viewer

    while sim.is_running():
        sim.step()

    assert len(sim.history.ts) > 1
    assert viewer.sync_count == 1
    assert viewer.set_texts_count == 1


def test_simulator_3d_sensor_engine_external_hooks_order():
    events: list[str] = []

    def sensor_noise(t, state):
        events.append("sensor")
        return state

    def engine_noise(t, state):
        events.append("engine")
        return ThrusterForces.zeros(12)

    def external_wrench(t, state):
        events.append("external")
        return WrenchEstimate3D(Fz=0.1)

    sim = SimIO(
        _params(),
        RobotState3D(),
        highlevel_control=ConstantLogic3D(ValveOpenings.zeros(12), events),
        sensor_noise=sensor_noise,
        engine_noise=engine_noise,
        external_wrench=external_wrench,
    )
    sim.viewer = DummyViewer()
    sim.visualize = lambda: None

    sim.step()

    assert events == ["sensor", "think", "engine", "external"]
    assert sim.history.external_wrench[-1].Fz == pytest.approx(0.1)


def test_simulator_3d_uses_experiment_control_frequency():
    logic = DynamicHzLogic3D()
    sim = SimIO(_params(t_max=0.27), RobotState3D(), highlevel_control=logic)
    sim.viewer = DummyViewer()
    sim.visualize = lambda: None

    while sim.is_running():
        sim.step()

    assert logic.think_ts == pytest.approx([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])


def test_simulator_3d_multi_rate_observe_precedes_control_and_visual_is_sparse():
    logic = OdomCaptureLogic3D()
    sim = SimIO(
        SimParams(
            t_max=0.21,
            full_odom_hz=5.0,
            highfreq_odom_hz=20.0,
        ),
        RobotState3D(),
        highlevel_control=logic,
    )
    sim.viewer = DummyViewer()
    sim.visualize = lambda: None

    while sim.is_running():
        sim.step()

    assert logic.events[0][0] == "observe"
    assert logic.events[1][0] == "think"
    observe_times = [t for kind, t in logic.events if kind == "observe"]
    think_times = [t for kind, t in logic.events if kind == "think"]
    assert observe_times[:5] == pytest.approx([0.0, 0.05, 0.10, 0.15, 0.20], abs=2e-9)
    assert think_times[:5] == pytest.approx([0.0, 0.05, 0.10, 0.15, 0.20], abs=2e-9)
    assert [sample.visual is not None for sample in logic.observations[:5]] == [
        True,
        False,
        False,
        False,
        True,
    ]


def test_simulator_3d_control_can_tick_between_odom_updates():
    logic = OdomCaptureLogic3D()
    sim = SimIO(
        SimParams(
            t_max=0.16,
            full_odom_hz=5.0,
            highfreq_odom_hz=5.0,
        ),
        RobotState3D(),
        highlevel_control=logic,
    )
    sim.viewer = DummyViewer()
    sim.visualize = lambda: None

    while sim.is_running():
        sim.step()

    observe_times = [t for kind, t in logic.events if kind == "observe"]
    think_times = [t for kind, t in logic.events if kind == "think"]
    assert observe_times == pytest.approx([0.0], abs=2e-9)
    assert think_times[:4] == pytest.approx([0.0, 0.05, 0.10, 0.15], abs=2e-9)


def test_simulator_3d_reports_slewed_and_masked_actual_valve_feedback():
    logic = ActuationCaptureLogic3D()
    sim = SimIO(
        SimParams(t_max=0.06, full_odom_hz=20.0, highfreq_odom_hz=20.0),
        RobotState3D(),
        highlevel_control=logic,
        active_thrusters=tuple(range(8)),
    )
    sim.viewer = DummyViewer()
    sim.visualize = lambda: None

    while sim.is_running():
        sim.step()

    np.testing.assert_allclose(logic.feedback[0], np.zeros(12), atol=1e-12)
    assert np.all(logic.feedback[1][:8] > 0.0)
    assert np.all(logic.feedback[1][:8] < 1.0)
    np.testing.assert_allclose(logic.feedback[1][8:], np.zeros(4), atol=1e-12)


def test_simulator_3d_rejects_non_multiple_highfreq_odom_rate():
    with pytest.raises(ValueError, match="integer multiple"):
        SimIO(
            SimParams(
                t_max=0.0,
                full_odom_hz=6.0,
                highfreq_odom_hz=20.0,
            ),
            RobotState3D(),
        )


def test_simulator_3d_applies_visual_and_inertial_noise_independently():
    logic = OdomCaptureLogic3D()
    visual_calls = 0
    inertial_calls = 0

    def visual_noise(t: float, odom: VisualOdom3D) -> VisualOdom3D:
        nonlocal visual_calls
        del t
        visual_calls += 1
        return VisualOdom3D(
            p=odom.p + np.array([1.0, 2.0, 3.0], dtype=float), q=odom.q.copy()
        )

    def inertial_noise(t: float, odom: InertialOdom3D) -> InertialOdom3D:
        nonlocal inertial_calls
        del t
        inertial_calls += 1
        return InertialOdom3D(
            q=odom.q.copy(),
            omega=odom.omega + np.array([4.0, 5.0, 6.0], dtype=float),
            a_body=odom.a_body.copy(),
        )

    sim = SimIO(
        _params(t_max=0.0),
        RobotState3D(),
        highlevel_control=logic,
        visual_noise=visual_noise,
        inertial_noise=inertial_noise,
    )
    sim.step()

    sample = logic.observations[0]
    assert visual_calls == 1
    assert inertial_calls == 1
    np.testing.assert_allclose(logic.initial_state.p, sample.visual.p, atol=1e-12)
    np.testing.assert_allclose(
        logic.initial_state.omega, sample.inertial.omega, atol=1e-12
    )
    np.testing.assert_allclose(sample.visual.p, np.array([1.0, 2.0, 3.0], dtype=float))
    np.testing.assert_allclose(
        sample.inertial.omega, np.array([4.0, 5.0, 6.0], dtype=float)
    )


def test_simulator_3d_uses_robot_params_for_control_and_robot_cargo_inertia_for_mujoco(
    tmp_path,
):
    custom_json = tmp_path / "with_cargo.json"
    cfg = json.loads(
        (
            Path(__file__).resolve().parents[1] / "sim" / "models" / "belka3d.json"
        ).read_text()
    )
    cfg["cargo"] = {
        "cargo_x": 0.18,
        "cargo_y": 0.12,
        "cargo_z": 0.08,
        "cargo_m": 1.1,
        "cargo_pos_x": 0.07,
        "cargo_pos_y": -0.03,
        "cargo_pos_z": -0.18,
    }
    custom_json.write_text(json.dumps(cfg))
    cargo = CargoParams.from_json(custom_json)
    logic = CaptureConfigLogic3D(ValveOpenings.zeros(12))
    controller_visible = RobotState3D(
        p=np.array([0.4, -0.2, 0.1], dtype=float),
        q=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
    )

    sim = SimIO(
        _params(),
        RobotState3D(),
        highlevel_control=logic,
        json_path=custom_json,
        output_dir=tmp_path,
        sensor_noise=lambda t, state: controller_visible,
    )
    true_m, true_cm, true_I = sim.robot_params.composite_body_properties(cargo)

    assert not hasattr(logic, "initial_state")
    sim.step()
    assert logic.initial_state is controller_visible
    assert logic.initial_state is not sim.history.real_state[0]
    assert sim.robot_params.m < true_m
    np.testing.assert_allclose(
        sim.model.body_mass[sim.belka_body_id], true_m, atol=1e-12
    )
    np.testing.assert_allclose(
        sim.model.body_ipos[sim.belka_body_id], true_cm, atol=1e-12
    )
    np.testing.assert_allclose(
        np.sort(sim.model.body_inertia[sim.belka_body_id]),
        np.sort(np.linalg.eigvalsh(true_I)),
        atol=1e-12,
    )

    xml = (tmp_path / "model.xml").read_text()
    assert 'geom name="left_side"' in xml
    assert 'geom name="right_side"' in xml
    assert 'geom name="cargo"' in xml

    copied_config = json.loads((tmp_path / "with_cargo.json").read_text())
    assert copied_config["cargo"]["cargo_m"] == pytest.approx(cargo.cargo_m)
    assert copied_config["cargo"]["cargo_pos_z"] == pytest.approx(cargo.cargo_pos_z)


def test_simulator_3d_json_name_does_not_control_template_lookup(tmp_path):
    custom_json = tmp_path / "custom_robot_params.json"
    model_json = Path(__file__).resolve().parents[1] / "sim" / "models" / "belka3d.json"
    model_config = json.loads(model_json.read_text())
    custom_json.write_text(
        json.dumps(
            {
                "robot": model_config["robot"],
                "cargo": {
                    "cargo_x": 0.0,
                    "cargo_y": 0.0,
                    "cargo_z": 0.0,
                    "cargo_m": 0.0,
                    "cargo_pos_x": 0.0,
                    "cargo_pos_y": 0.0,
                    "cargo_pos_z": 0.0,
                },
            }
        )
    )

    sim = SimIO(
        _params(),
        RobotState3D(),
        json_path=custom_json,
        output_dir=tmp_path / "rendered",
    )

    assert sim.robot_params.nu == 12
    assert (tmp_path / "rendered" / "model.xml").exists()
    assert (tmp_path / "rendered" / "custom_robot_params.json").exists()
