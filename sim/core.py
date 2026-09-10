from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import jinja2
import mujoco
import numpy as np

from sim import SimHistory, SimParams
from sim.shared.visualization import (
    body_arrow_geom_vars,
    draw_mujoco_arrow,
    draw_mujoco_sphere,
    thrust_arrow_rgba,
)

from belka.common import (
    CargoParams,
    InertialOdom3D,
    OdomInput3D,
    RobotParams3D,
    RobotState3D,
    ThrusterForces,
    ValveOpenings,
    VisualOdom3D,
    quat_to_rot_matrix,
)
from belka.actuation import active_thruster_mask, valve_openings_to_forces
from belka.shared.logic import HighlevelLogic
from belka.shared.observers import WrenchEstimate3D

if TYPE_CHECKING:
    from mujoco.viewer import Handle

MODELS_DIR = Path(__file__).resolve().absolute().parent / "models"

SensorNoise3D = Callable[[float, RobotState3D], RobotState3D]
VisualNoise3D = Callable[[float, VisualOdom3D], VisualOdom3D]
InertialNoise3D = Callable[[float, InertialOdom3D], InertialOdom3D]
EngineNoise3D = Callable[[float, RobotState3D], ThrusterForces]
ExtForce3D = Callable[
    [float, RobotState3D], WrenchEstimate3D | np.ndarray | None
]  # [F_world(3), M_world(3)]


class ZeroLogic(HighlevelLogic):
    """Default open-loop 3D logic that commands zero thrust."""

    def configure(self, initial_state: RobotState3D):
        """Accept simulator configuration without internal state."""
        self._estimated_state = initial_state

    def think(self, t: float, actual_actuation: ValveOpenings) -> ValveOpenings:
        """Return zero 3D thruster command."""
        del t
        return ValveOpenings.zeros(actual_actuation.nu)

    def control_hz(self) -> float:
        """Return the default open-loop scheduler rate."""
        return 20.0


class SimIO:
    """Spatial MuJoCo IO with observation, control and visualization scheduling."""

    thrust_visual_smoothing_tau = 0.08  # visual-only EMA time constant, seconds
    viewer_hz = 30.0  # visual sync rate; physics/control use their own rates

    def __init__(
        self,
        sim_params: SimParams,
        initial_state: RobotState3D,
        *,
        json_path: Path = MODELS_DIR / "belka3d.json",
        template_path: Path = MODELS_DIR / "belka3d.xml.j2",
        output_dir: Path | None = None,
        highlevel_control: HighlevelLogic | None = None,
        sensor_noise: SensorNoise3D | None = None,
        visual_noise: VisualNoise3D | None = None,
        inertial_noise: InertialNoise3D | None = None,
        engine_noise: EngineNoise3D | None = None,
        external_wrench: ExtForce3D | None = None,
        show_perceived_position: bool = True,
        show_front_arrows: bool = True,
        active_thrusters: tuple[int, ...] | None = None,
        robot_params: RobotParams3D | None = None,
    ):
        """Create the spatial simulator from explicit model and experiment inputs."""
        self.show_front_arrows = bool(show_front_arrows)
        self._active_thrusters = active_thrusters
        config = json.loads(Path(json_path).read_text())
        self.cargo_params = CargoParams.from_mapping(config.get("cargo"))
        self.visual_noise = visual_noise
        self.inertial_noise = inertial_noise
        self.robot_params = robot_params or RobotParams3D.from_json(json_path)
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="habitat_sim_"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        template_dir = template_path.parent
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)))
        xml_str = env.get_template(template_path.name).render(**self.template_vars())
        model_xml_path = self.output_dir / "model.xml"
        model_xml_path.write_text(xml_str)
        (self.output_dir / Path(json_path).name).write_text(
            json.dumps(config, indent=2)
        )

        self.model = mujoco.MjModel.from_xml_path(str(model_xml_path))
        self.data = mujoco.MjData(self.model)
        self.write_initial_state(initial_state)
        mujoco.mj_forward(self.model, self.data)
        self.belka_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "belka"
        )
        self._site_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"thr{i}")
                for i in range(self.robot_params.nu)
            ],
            dtype=int,
        )
        self.viewer: Handle | None = None
        self.dt = self.model.opt.timestep
        self.highlevel_control = highlevel_control or ZeroLogic()
        self.full_odom_hz = float(sim_params.full_odom_hz)
        self.highfreq_odom_hz = float(sim_params.highfreq_odom_hz)
        self.control_hz = float(self.highlevel_control.control_hz())
        self._validate_timing_rates(
            self.full_odom_hz,
            self.highfreq_odom_hz,
            self.control_hz,
        )
        self.full_odom_dt = 1.0 / self.full_odom_hz
        self.highfreq_odom_dt = 1.0 / self.highfreq_odom_hz
        self.control_dt = 1.0 / self.control_hz
        self.sim_params = sim_params
        self.sensor_noise = sensor_noise
        self.engine_noise = engine_noise
        self.external_wrench = external_wrench
        self.show_perceived_position = bool(show_perceived_position)
        self.state = initial_state
        self._last_perceived_state: RobotState3D | None = None
        self._logic_configured = False
        self.history = SimHistory()
        self._actual_actuation = ValveOpenings.zeros(self.robot_params.nu)
        self._commanded_actuation = ValveOpenings.zeros(self.robot_params.nu)
        self._applied_forces = self.command_to_forces(self._actual_actuation)
        self._visualized_force_values = self._applied_forces.to_array()
        self._next_full_odom_time = 0.0
        self._next_highfreq_odom_time = 0.0
        self._next_control_time = 0.0
        self._next_viewer_time = 0.0
        self._viewer_text_overlay_signature = None

    def __enter__(self):
        """Open the passive MuJoCo viewer and configure its spatial camera."""
        from mujoco import viewer

        self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 6.0
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 45
        return self

    def __exit__(self, exc_type, exc, tb):
        """Close the MuJoCo viewer on context exit."""
        if self.viewer is not None:
            self.viewer.close()

    def is_running(self) -> bool:
        """Return whether viewer is alive and simulation time is within t_max."""
        if self.viewer is not None and not self.viewer.is_running():
            return False
        return self.data.time <= self.sim_params.t_max + 1e-12

    def step(self) -> None:
        """Run one MuJoCo step with multi-rate observation/control scheduling."""
        if not self.is_running():
            return

        t = float(self.data.time)  # current simulation time
        self.state = self.read_state()
        full_due = t >= self._next_full_odom_time - 1e-12
        highfreq_due = t >= self._next_highfreq_odom_time - 1e-12
        control_due = t >= self._next_control_time - 1e-12

        input_type = self.highlevel_control.observation_input_type()
        if input_type is OdomInput3D:
            if highfreq_due or full_due:
                input_data = self.odom_input(t, self.state, include_visual=full_due)
                self._configure_logic_from(input_data)
                self._last_perceived_state = self.highlevel_control.observe(
                    t, input_data, self._actual_actuation
                )
                if full_due:
                    self._next_full_odom_time = self._next_tick(
                        self._next_full_odom_time, t, self.full_odom_dt
                    )
                if highfreq_due:
                    self._next_highfreq_odom_time = self._next_tick(
                        self._next_highfreq_odom_time, t, self.highfreq_odom_dt
                    )
        elif full_due:
            input_data = self.full_observation_input(t, self.state)
            self._configure_logic_from(input_data)
            self._last_perceived_state = self.highlevel_control.observe(
                t, input_data, self._actual_actuation
            )
            self._next_full_odom_time = self._next_tick(
                self._next_full_odom_time, t, self.full_odom_dt
            )

        if control_due:
            if not self._logic_configured:
                raise RuntimeError("control tick occurred before the first observation")
            self._commanded_actuation = self.highlevel_control.think(
                t, self._actual_actuation
            )
            self._next_control_time = self._next_tick(
                self._next_control_time, t, self.control_dt
            )

        self._update_actual_actuation(self._commanded_actuation, self.dt)
        real_forces = self.command_to_forces(self._actual_actuation)
        if self.engine_noise is not None:
            real_forces = ThrusterForces.from_array(
                real_forces.to_array() + self.engine_noise(t, self.state).to_array()
            )
        self._applied_forces = real_forces
        self.data.ctrl[:] = real_forces.to_array()

        external_wrench = (
            self.external_wrench(t, self.state)
            if self.external_wrench is not None
            else WrenchEstimate3D.zeros()
        )
        if external_wrench is None:
            external_wrench = WrenchEstimate3D.zeros()
        elif not isinstance(external_wrench, WrenchEstimate3D):
            values = (
                external_wrench.to_array()
                if hasattr(external_wrench, "to_array")
                else np.asarray(external_wrench, dtype=float)
            )
            external_wrench = WrenchEstimate3D.from_array(values)
        self.data.xfrc_applied[self.belka_body_id, :] = (
            external_wrench.to_array().astype(np.float32)
        )

        if self._last_perceived_state is None:
            raise RuntimeError(
                "simulation step completed without a controller-visible state"
            )
        self.history.update(
            ts=self.data.time,
            real_state=self.state,
            engine_forces=real_forces,
            perceived_state=self._last_perceived_state,
            external_wrench=external_wrench,
        )

        should_sync_viewer = self.viewer is not None and (
            self.viewer_hz <= 0.0 or self.data.time >= self._next_viewer_time - 1e-12
        )
        if should_sync_viewer and hasattr(self.viewer, "user_scn"):
            self.visualize()
        mujoco.mj_step(self.model, self.data)
        if should_sync_viewer:
            self.viewer.sync()
            viewer_dt = self.dt if self.viewer_hz <= 0.0 else 1.0 / self.viewer_hz
            self._next_viewer_time = self._next_tick(
                self._next_viewer_time, t, viewer_dt
            )

    @staticmethod
    def _next_tick(current: float, t: float, period: float) -> float:
        """Advance one periodic scheduler deadline beyond current time t."""
        current = max(current, t)
        while current <= t + 1e-12:
            current += period
        return current

    def full_observation_input(self, t: float, state: RobotState3D) -> RobotState3D:
        """Return one full-state sample through the configured sensor-noise hook."""
        return self.sensor_noise(t, state) if self.sensor_noise is not None else state

    def _configure_logic_from(self, input_data: RobotState3D | OdomInput3D) -> None:
        """Configure logic once from the first sample visible through the IO boundary."""
        if self._logic_configured:
            return
        self.highlevel_control.configure(self.logic_initial_state(input_data))
        self._logic_configured = True

    @classmethod
    def _validate_timing_rates(
        cls, full_odom_hz: float, highfreq_odom_hz: float, control_hz: float
    ) -> None:
        """Validate simulator odometry/control rates at the timing trust boundary."""
        for name, rate_hz in (
            ("full_odom_hz", full_odom_hz),
            ("highfreq_odom_hz", highfreq_odom_hz),
            ("control_hz", control_hz),
        ):
            if not np.isfinite(rate_hz) or rate_hz <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {rate_hz}")
        if highfreq_odom_hz + 1e-12 < full_odom_hz:
            raise ValueError(
                "highfreq_odom_hz must be greater than or equal to full_odom_hz"
            )
        ratio = highfreq_odom_hz / full_odom_hz  # high-rate samples per full sample
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                "highfreq_odom_hz must be an integer multiple of full_odom_hz, "
                f"got {highfreq_odom_hz}/{full_odom_hz}"
            )

    def set_viewer_legend(self, left_text: str, right_text: str) -> None:
        """Set MuJoCo viewer state label and marker legend overlays."""
        if self.viewer is None or not hasattr(self.viewer, "set_texts"):
            return
        state_label = str(self.highlevel_control.state_label()).strip()
        state_label = state_label or self.highlevel_control.__class__.__name__
        signature = (id(self.viewer), state_label, left_text, right_text)
        if self._viewer_text_overlay_signature == signature:
            return
        self.viewer.set_texts(
            [
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOP,
                    f"State: {state_label}",
                    "",
                ),
                (
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                    left_text,
                    right_text,
                ),
            ]
        )
        self._viewer_text_overlay_signature = signature

    def _visualized_thruster_forces(self) -> np.ndarray:
        """Return visually smoothed physical thruster forces for viewer arrows."""
        target = self._applied_forces.to_array()  # physical thruster-force vector
        if self._visualized_force_values.shape != target.shape:
            self._visualized_force_values = np.zeros_like(target)
        tau = float(self.thrust_visual_smoothing_tau)  # EMA time constant
        if tau <= 0.0:
            self._visualized_force_values = target.copy()
        else:
            alpha = 1.0 - np.exp(-self.dt / tau)  # per-step EMA coefficient
            self._visualized_force_values += alpha * (
                target - self._visualized_force_values
            )
        return self._visualized_force_values.copy()

    def template_vars(self) -> dict:
        """Return 3D MuJoCo template variables."""
        true_m, true_r_cm_body, true_I_body = (
            self.robot_params.composite_body_properties(self.cargo_params)
        )
        geoms = self._body_geom_template_vars()
        sites = [
            {
                "name": f"thr{i}",
                "pos": self.robot_params.thruster_positions_body[i].tolist(),
            }
            for i in range(self.robot_params.nu)
        ]
        actuators = [
            {
                "name": f"F{i + 1}",
                "site": f"thr{i}",
                "direction": self.robot_params.thruster_directions_body[i].tolist(),
            }
            for i in range(self.robot_params.nu)
        ]
        return {
            "m": true_m,
            "r_cm_body": true_r_cm_body.tolist(),
            "I_body": true_I_body.tolist(),
            "geoms": geoms,
            "front_arrow_geoms": self._front_arrow_template_vars(),
            "sites": sites,
            "actuators": actuators,
            "max_force": self.robot_params.max_thruster_force_n,
        }

    def _body_geom_template_vars(self) -> list[dict]:
        """Return visual body/cargo box geometry variables for the MuJoCo template."""
        geoms = []
        colors = {
            "left_side": "0.2 0.4 0.9 1",
            "right_side": "0.2 0.4 0.9 1",
            "base": "0.15 0.28 0.65 1",
            "cargo": "0.85 0.55 0.15 1",
        }
        for component in self.robot_params.known_components():
            geoms.append(
                {
                    "name": component.name,
                    "pos": component.center.tolist(),
                    "size": (component.size / 2.0).tolist(),
                    "rgba": colors.get(component.name, "0.2 0.4 0.9 1"),
                }
            )
        cargo_component = self.cargo_params.component()
        if cargo_component is not None:
            geoms.append(
                {
                    "name": cargo_component.name,
                    "pos": cargo_component.center.tolist(),
                    "size": (cargo_component.size / 2.0).tolist(),
                    "rgba": colors["cargo"],
                }
            )
        return geoms

    def _front_arrow_template_vars(self) -> list[dict]:
        """Return static body-front arrow visual geoms for the MuJoCo template."""
        if not self.show_front_arrows:
            return []
        real_front = np.array([1.0, 0.0, 0.0], dtype=float)  # real base-frame +X front
        presented_front = (
            self.robot_params.presented_front_axis_body()
        )  # presented +X in base-frame coordinates
        half_extents = np.array(
            [
                self.robot_params.side_L / 2.0,
                self.robot_params.outer_half_width_y,
                self.robot_params.side_L / 2.0,
            ],
            dtype=float,
        )  # approximate body half-extents for arrow length
        radius = max(0.006, 0.018 * self.robot_params.side_L)  # static arrow radius

        def arrow(name: str, axis: np.ndarray, rgba: str) -> list[dict]:
            """Return shaft/tip geom vars for one body-frame front arrow."""
            axis = np.asarray(axis, dtype=float)  # body-frame arrow axis
            length = float(
                np.dot(np.abs(axis), half_extents) + 0.07
            )  # center-origin arrow length
            return body_arrow_geom_vars(
                name, axis, length=length, radius=radius, rgba=rgba
            )

        return [
            *arrow("real_front", real_front, "0.9 0.1 0.1 0.95"),
            *arrow("presented_front", presented_front, "0.95 0.8 0.05 0.95"),
        ]

    def write_initial_state(self, initial_state: RobotState3D) -> None:
        """Write 3D state into MuJoCo freejoint qpos/qvel."""
        self.data.qpos[0:3] = initial_state.p
        self.data.qpos[3:7] = initial_state.q
        self.data.qvel[0:3] = initial_state.v
        self.data.qvel[3:6] = initial_state.omega

    def read_state(self) -> RobotState3D:
        """Read 3D MuJoCo freejoint state and acceleration."""
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        return RobotState3D(
            p=qpos[0:3].copy(),
            q=qpos[3:7].copy(),
            v=qvel[0:3].copy(),
            omega=qvel[3:6].copy(),
            a=qacc[0:3].copy(),
            alpha=qacc[3:6].copy(),
        )

    def odom_input(
        self, t: float, state: RobotState3D, *, include_visual: bool
    ) -> OdomInput3D:
        """Build one typed 3D odometry input with high-rate inertial and optional visual data."""
        R_body_to_world = quat_to_rot_matrix(state.q)  # body->world rotation matrix
        a_body = (
            R_body_to_world.T @ state.a
        )  # body-frame linear acceleration of body origin
        visual = None
        if include_visual:
            visual = VisualOdom3D(p=state.p.copy(), q=state.q.copy())
            if self.visual_noise is not None:
                visual = self.visual_noise(t, visual)
        inertial = InertialOdom3D(
            q=state.q.copy(), omega=state.omega.copy(), a_body=a_body
        )
        if self.inertial_noise is not None:
            inertial = self.inertial_noise(t, inertial)
        return OdomInput3D(visual=visual, inertial=inertial)

    def logic_initial_state(
        self, input_data: RobotState3D | OdomInput3D
    ) -> RobotState3D:
        """Build controller-visible initialization from the first delivered sample."""
        if isinstance(input_data, RobotState3D):
            return input_data
        if input_data.visual is None:
            raise RuntimeError("first OdomInput3D sample must include visual odometry")
        q = input_data.visual.q.copy()  # first controller-visible orientation
        return RobotState3D(
            p=input_data.visual.p.copy(),
            q=q,
            v=np.zeros(3, dtype=float),
            omega=input_data.inertial.omega.copy(),
            a=quat_to_rot_matrix(q) @ input_data.inertial.a_body,
        )

    @property
    def active_thruster_mask(self) -> np.ndarray:
        """Return the experiment actuator mask in D3 channel order."""
        return active_thruster_mask(self.robot_params.nu, self._active_thrusters)

    def _update_actual_actuation(
        self, target_actuation: ValveOpenings, dt: float
    ) -> None:
        """Apply valve-opening slew, box bounds and the experiment actuator mask."""
        current = self._actual_actuation.to_array()
        target = np.asarray(target_actuation.to_array(), dtype=float)
        if target.shape != (self.robot_params.nu,):
            raise ValueError(
                f"D3 valve command must have shape ({self.robot_params.nu},), got {target.shape}"
            )
        max_change = float(self.robot_params.valve_response_per_s) * float(dt)
        actual = current + np.clip(target - current, -max_change, max_change)
        actual = np.clip(actual, 0.0, 1.0) * self.active_thruster_mask
        self._actual_actuation = ValveOpenings.from_array(actual)

    def command_to_forces(self, command: ValveOpenings) -> ThrusterForces:
        """Convert realized valve openings to side-limited physical forces."""
        return valve_openings_to_forces(
            ValveOpenings.from_array(command.to_array()),
            self.robot_params,
            active_mask=self.active_thruster_mask,
        )

    def visualize(self):
        """Draw 3D thruster arrows and high-level markers."""
        THRUST_ARROW_MIN_INTENSITY = 0.015
        THRUST_ARROW_GAP = 0.018
        THRUST_ARROW_MIN_LENGTH = 0.06
        THRUST_ARROW_MAX_EXTRA_LENGTH = 0.16
        THRUST_ARROW_MIN_RADIUS = 0.007
        THRUST_ARROW_MAX_EXTRA_RADIUS = 0.009
        THRUST_GLOW_MIN_SIZE = 0.010
        THRUST_GLOW_MAX_EXTRA_SIZE = 0.010

        if not hasattr(self.viewer, "user_scn"):
            return
        self.set_viewer_legend(
            "Markers\nblue sphere\nred sphere\ngreen spheres\nbrown sphere\nred front arrow\nyellow front arrow\nthrust arrows",
            "target\nstart\nactive trajectory\nperceived state\nreal body +X\npresented front\nthruster force",
        )
        self.viewer.user_scn.ngeom = 0
        max_geoms = len(self.viewer.user_scn.geoms)

        _colors = {
            "red": (0.9, 0.2, 0.2, 0.95),
            "green": (0.2, 0.8, 0.2, 0.95),
            "blue": (0.2, 0.4, 0.9, 0.95),
            "yellow": (0.9, 0.8, 0.2, 0.95),
            "orange": (0.95, 0.5, 0.15, 0.95),
            "brown": (0.50, 0.28, 0.12, 0.95),
            "white": (0.95, 0.95, 0.95, 0.95),
        }

        for marker in self.highlevel_control.get_markers():
            if self.viewer.user_scn.ngeom >= max_geoms:
                return
            rgba = np.array(
                _colors.get(marker.color.lower(), _colors["blue"]), dtype=np.float32
            )
            marker_pos = np.array([marker.x, marker.y, marker.z], dtype=float)
            if not draw_mujoco_sphere(self.viewer, marker_pos, marker.size, rgba):
                return
            if marker.direction is not None:
                direction = np.asarray(
                    marker.direction, dtype=float
                )  # world-frame marker arrow direction
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm > 1e-9:
                    unit_dir = (
                        direction / direction_norm
                    )  # unit world-frame arrow direction
                    arrow_rgba = np.array(
                        _colors.get(
                            (marker.arrow_color or marker.color).lower(),
                            _colors["yellow"],
                        ),
                        dtype=np.float32,
                    )
                    arrow_start = (
                        marker_pos + unit_dir * marker.size * 1.25
                    )  # marker direction arrow start
                    arrow_end = arrow_start + unit_dir * max(
                        0.08, marker.size * 3.5
                    )  # marker direction arrow end
                    if not draw_mujoco_arrow(
                        self.viewer,
                        arrow_start,
                        arrow_end,
                        max(0.003, marker.size * 0.12),
                        arrow_rgba,
                    ):
                        return

        if self.show_perceived_position and self._last_perceived_state is not None:
            if self.viewer.user_scn.ngeom >= max_geoms:
                return
            marker_pos = np.asarray(
                self._last_perceived_state.p, dtype=float
            )  # latest perceived world position
            rgba = np.array(_colors["brown"], dtype=np.float32)
            if not draw_mujoco_sphere(self.viewer, marker_pos, 0.07, rgba):
                return

        body_pos = self.data.xpos[self.belka_body_id].copy()
        R_body_to_world = quat_to_rot_matrix(
            self.state.q
        )  # R_body_to_world orientation matrix
        thrusts = (
            self._visualized_thruster_forces()
        )  # smoothed thruster force magnitudes
        nu = len(thrusts)  # number of thruster channels
        max_f = self.robot_params.max_thruster_force_n  # display force scale

        for i in range(nu):
            if self.viewer.user_scn.ngeom >= max_geoms:
                return
            site_pos_body = self.model.site_pos[self._site_ids[i]]
            world_pos = body_pos + R_body_to_world @ site_pos_body

            alloc = (
                self.robot_params.allocation_matrix
            )  # thruster-to-wrench allocation matrix
            dir_body = alloc[0:3, i].copy()
            dir_norm = float(np.linalg.norm(dir_body))  # thruster direction norm
            if dir_norm < 1e-9:
                continue
            world_dir = R_body_to_world @ (dir_body / dir_norm)

            intensity = (
                float(thrusts[i]) / max_f if max_f > 0 else 0.0
            )  # normalized thrust magnitude
            intensity = float(np.clip(intensity, 0.0, 1.0))
            if intensity < THRUST_ARROW_MIN_INTENSITY:
                continue
            visual_intensity = float(
                np.sqrt(intensity)
            )  # perceptual scaling for small thrusts
            arrow_len = (
                THRUST_ARROW_MIN_LENGTH
                + THRUST_ARROW_MAX_EXTRA_LENGTH * visual_intensity
            )  # visual arrow length
            radius = (
                THRUST_ARROW_MIN_RADIUS
                + THRUST_ARROW_MAX_EXTRA_RADIUS * visual_intensity
            )  # visual arrow radius
            start = world_pos - world_dir * (
                THRUST_ARROW_GAP + arrow_len
            )  # visual thrust arrow start
            end = world_pos - world_dir * THRUST_ARROW_GAP  # visual thrust arrow end

            rgba = thrust_arrow_rgba(intensity)
            if not draw_mujoco_arrow(self.viewer, start, end, radius, rgba):
                return

            glow_size = (
                THRUST_GLOW_MIN_SIZE + THRUST_GLOW_MAX_EXTRA_SIZE * visual_intensity
            )
            if not draw_mujoco_sphere(self.viewer, end, glow_size, rgba):
                return
