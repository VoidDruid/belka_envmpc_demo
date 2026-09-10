"""Minimal 6-DoF MuJoCo runner used by the EnvMPC paper experiments."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from belka.common import (
    OdomInput3D,
    RobotParams3D,
    RobotState3D,
    StateTolerances3D,
    quat_to_rot_matrix,
)
from belka.controllers.mpc import EnvMPCController3D, MPCController3D
from belka.observers import KalmanEstimator3DParams, KalmanObserver3D
from belka.planners import QuinticPlanner3D, QuinticPlanningParameters3D
from belka.shared.controllers import ControlHistory
from belka.shared.logic import VisualizationMarker
from belka.shared.planners import TrajectoryPlanner
from belka_scenarios.tracking import TrackingLogic, TrackingLogicConfig
from experiments.config import Scenario3DConfig
from sim.core import SimIO
from sim.environment import make_random_wind_field_3d
from sim.noise import make_inertial_noise, make_visual_noise


STANDARD_FLIGHT_CONTROL_HZ = 20.0


class FlightLogic3D(TrackingLogic):
    """Track a spatial reference with MPC or disturbance-aware EnvMPC."""

    def __init__(
        self,
        target: RobotState3D,
        T: float,
        hold_time: float,
        controller_params: RobotParams3D,
        initial_external_wrench: np.ndarray | None = None,
        home_state: RobotState3D | None = None,
        show_front_arrows: bool = True,
        active_thrusters: tuple[int, ...] | None = None,
        environment_aware: bool = True,
    ) -> None:
        """Configure the paper's 20 Hz trajectory-tracking control stack."""
        self.T = float(T)
        self.initial_external_wrench = (
            None
            if initial_external_wrench is None
            else np.asarray(initial_external_wrench, dtype=float)
        )
        self.home_state = home_state
        self.show_front_arrows = bool(show_front_arrows)
        self.active_thrusters = active_thrusters
        self.controller_params = controller_params
        self.environment_aware = bool(environment_aware)
        self.control_dt_s = 1.0 / STANDARD_FLIGHT_CONTROL_HZ
        self.presented_frame = np.eye(3, dtype=float)
        self.presented_front_axis_body = np.array([1.0, 0.0, 0.0], dtype=float)
        self.finish_tolerances = StateTolerances3D(
            position_tolerance=0.08,
            velocity_tolerance=0.05,
            orientation_tolerance=0.20,
            omega_tolerance=0.08,
        )
        self.replan_tolerances = StateTolerances3D(
            position_tolerance=0.35,
            velocity_tolerance=np.inf,
            orientation_tolerance=0.45,
            omega_tolerance=np.inf,
        )
        self.target = target
        super().__init__(
            TrackingLogicConfig(
                target_state=target,
                stabilization_time=hold_time,
                control_hz=STANDARD_FLIGHT_CONTROL_HZ,
                trajectory_planner_builder=self._build_quintic_planner,
                controller_builder=self._build_controller,
                observer_builder=self._build_observer,
                marker_projector=self._marker_from_state,
                trajectory_marker_count=21,
                trajectory_marker_size=0.03,
                target_marker_size=0.07,
            )
        )

    def configure(self, initial_state: RobotState3D) -> None:
        """Build the controller and observer from the exact initial state."""
        self.presented_frame = self.controller_params.get_presented_frame()
        self.presented_front_axis_body = (
            self.controller_params.presented_front_axis_body()
        )
        super().configure(initial_state)
        if self.home_state is not None:
            self.initial_state = self.home_state

    def observation_input_type(self) -> type:
        """Use visual-plus-inertial odometry as the observer input."""
        return OdomInput3D

    def _marker_from_state(
        self, state: RobotState3D, color: str, size: float
    ) -> VisualizationMarker:
        """Project one state to a MuJoCo viewer marker."""
        direction = None
        arrow_color = None
        if self.show_front_arrows and color.lower() == "green":
            R_world_base = quat_to_rot_matrix(state.q)  # body-to-world rotation
            direction = R_world_base @ self.presented_front_axis_body
            arrow_color = "yellow"
        return VisualizationMarker(
            x=float(state.p[0]),
            y=float(state.p[1]),
            z=float(state.p[2]),
            color=color,
            size=size,
            direction=direction,
            arrow_color=arrow_color,
        )

    def _build_quintic_planner(
        self,
        current_t: float,
        initial_state: RobotState3D,
        final_state: RobotState3D,
    ) -> QuinticPlanner3D:
        """Build the fifth-order reference trajectory used in the paper."""
        return QuinticPlanner3D(
            QuinticPlanningParameters3D(
                initial_state=initial_state,
                final_state=final_state,
                T=self.T,
                start_t=current_t,
                replan_tolerances=self.replan_tolerances,
                finish_tolerances=self.finish_tolerances,
                replan_threshold_steps=20,
            ),
            presented_frame=self.presented_frame,
        )

    def _build_controller(
        self,
        initial_state: RobotState3D,
        planner: TrajectoryPlanner,
    ) -> MPCController3D:
        """Build MPC with the published weights, horizon and constraints."""
        del initial_state
        params = self.controller_params
        Q = np.diag(  # state penalty matrix
            [64, 64, 64, 32, 32, 32, 32, 64, 64, 64, 16, 16, 16]
        ).astype(np.float64)
        R = np.eye(params.nu, dtype=np.float64) * 0.1  # control penalty
        S = np.eye(params.nu, dtype=np.float64) * 0.2  # control-rate penalty
        controller_type = (
            EnvMPCController3D if self.environment_aware else MPCController3D
        )
        return controller_type(
            params,
            state_weight=Q,
            thruster_weight=R,
            thruster_delta_weight=S,
            terminal_weight=Q * 2.0,
            horizon=16,
            dt=self.control_dt_s,
            planner=planner,
            active_thrusters=self.active_thrusters,
        )

    def _build_observer(self, initial_state: RobotState3D) -> KalmanObserver3D:
        """Build the published 18-dimensional error-state EKF."""
        observer = KalmanObserver3D(
            KalmanEstimator3DParams(
                robot_params=self.controller_params,
                initial_external_wrench=self.initial_external_wrench,
                measurement_visual_position_std=0.02,
                measurement_visual_orientation_std=0.03,
                measurement_inertial_orientation_std=0.02,
                measurement_accel_std=0.04,
                measurement_omega_std=0.03,
            )
        )
        observer.initialize(initial_state)
        return observer


def save_3d_run_artifacts(
    sim: SimIO,
    control_history: ControlHistory | None,
    scenario: dict,
    scenario_json_path: Path,
    target: RobotState3D,
    save_plots: bool,
    effective_robot_params: RobotParams3D | None = None,
) -> Path:
    """Save the truth, estimate, reference and configuration of one run."""
    if save_plots:
        raise ValueError(
            "The minimal runner writes data; use the paper analysis command for plots"
        )
    output_dir = sim.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "scenario": scenario,
        "scenario_json_path": str(scenario_json_path),
        "sim_history": sim.history,
        "control_history": control_history or ControlHistory(),
        "final_state": sim.state,
        "target_state": target,
        "robot_params": sim.robot_params,
        "effective_robot_params": effective_robot_params or sim.robot_params,
        "cargo_params": sim.cargo_params,
        "sim_params": sim.sim_params,
        "output_dir": str(output_dir),
    }
    with (output_dir / "snapshot.pkl").open("wb") as handle:
        pickle.dump(snapshot, handle)
    return output_dir


def _build_wind(config: Scenario3DConfig, robot_params: RobotParams3D):
    """Build the simulator-only external-wrench generator."""
    wind = config.wind
    return make_random_wind_field_3d(
        robot_params,
        mean_freq=wind.mean_freq,
        jitter_freq=wind.jitter_freq,
        force_ratio=wind.force_ratio,
        secondary_ratio=wind.secondary_ratio,
        tertiary_ratio=wind.tertiary_ratio,
        moment_lever_scale=wind.moment_lever_scale,
        wave_amplitude_ratio=wind.wave_amplitude_ratio,
        jitter_tube_ratio=wind.jitter_tube_ratio,
    )


def _build_visual_noise(config: Scenario3DConfig):
    """Build the visual pose noise used by the numerical experiments."""
    noise = config.sensor_noise
    return make_visual_noise(
        position_bias_m=noise.position_bias_m,
        position_slow_amplitude_m=noise.position_slow_amplitude_m,
        position_fast_amplitude_m=noise.position_fast_amplitude_m,
        attitude_bias_rad=noise.attitude_bias_rad,
        attitude_slow_amplitude_rad=noise.attitude_slow_amplitude_rad,
        attitude_fast_amplitude_rad=noise.attitude_fast_amplitude_rad,
        position_slow_freq=noise.position_slow_freq,
        position_fast_freq=noise.position_fast_freq,
        attitude_slow_freq=noise.attitude_slow_freq,
        attitude_fast_freq=noise.attitude_fast_freq,
    )


def _build_inertial_noise(config: Scenario3DConfig):
    """Build the IMU orientation, acceleration and rate noise."""
    noise = config.sensor_noise
    return make_inertial_noise(
        attitude_bias_rad=0.2 * noise.attitude_bias_rad,
        attitude_slow_amplitude_rad=0.2 * noise.attitude_slow_amplitude_rad,
        attitude_fast_amplitude_rad=0.2 * noise.attitude_fast_amplitude_rad,
        attitude_slow_freq=noise.attitude_slow_freq,
        attitude_fast_freq=noise.attitude_fast_freq,
        omega_std=noise.omega_std,
        accel_std=noise.accel_std,
    )
