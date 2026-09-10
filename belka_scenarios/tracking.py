from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

from belka.common import OdomInput3D, RobotState3D, ValveOpenings
from belka.observers import KalmanObserver3D
from belka.planners import EndpointPlanner3D
from belka.shared.controllers import Controller
from belka.shared.logic import HighlevelLogic, VisualizationMarker
from belka.shared.observers import WrenchEstimate3D
from belka.shared.planners import TrajectoryPlanner

TrajectoryPlannerBuilder = Callable[
    [float, RobotState3D, RobotState3D], TrajectoryPlanner
]
ControllerBuilder = Callable[[RobotState3D, TrajectoryPlanner], Controller]
MarkerProjector = Callable[[RobotState3D, str, float], VisualizationMarker]


@dataclass(frozen=True)
class TrackingLogicConfig:
    """Builders and constants for the spatial tracking state machine."""

    target_state: RobotState3D
    stabilization_time: float
    control_hz: float
    trajectory_planner_builder: TrajectoryPlannerBuilder
    controller_builder: ControllerBuilder
    marker_projector: MarkerProjector
    observer_builder: Callable[[RobotState3D], KalmanObserver3D | None] | None = None
    trajectory_marker_count: int = 16
    trajectory_marker_size: float = 0.02
    target_marker_size: float = 0.05
    hold_initial_endpoint: bool = False


class TrackingLogic(HighlevelLogic):
    """Target-tracking state machine for the spatial control stack."""

    def __init__(self, config: TrackingLogicConfig):
        """Store tracking configuration and initialize lifecycle state."""
        self.config = config
        self.initial_state: RobotState3D | None = None
        self.controller: Controller | None = None
        self.observer: KalmanObserver3D | None = None
        self._initial_hold: TrajectoryPlanner | None = None
        self._movement_planner: TrajectoryPlanner | None = None
        self._active_destination: RobotState3D | None = None
        self._hold_until: float | None = None
        self._hold_next_state: RobotState3D | None = None
        self._last_observer_t: float | None = None
        self._estimation: WrenchEstimate3D | None = None
        if not 0.0 < float(config.control_hz) < float("inf"):
            raise ValueError(
                "TrackingLogicConfig.control_hz must be finite and positive"
            )

    def configure(self, initial_state: RobotState3D) -> None:
        """Build initial endpoint hold and attach the configured low-level controller."""
        self.initial_state = initial_state
        self._initial_hold = EndpointPlanner3D(initial_state)
        self.controller = self.config.controller_builder(
            initial_state, self._initial_hold
        )
        self.observer = (
            None
            if self.config.observer_builder is None
            else self.config.observer_builder(initial_state)
        )
        self._movement_planner = None
        self._active_destination = None
        self._hold_until = 0.0
        self._hold_next_state = self.config.target_state
        self._estimated_state = None
        self._last_observer_t = None
        self._estimation = None

    def observe(
        self,
        t: float,
        input_data: RobotState3D | OdomInput3D,
        actual_actuation: ValveOpenings,
    ) -> RobotState3D:
        """Update the logic-owned observer and store the latest tracking state estimate."""
        if self.controller is None:
            raise RuntimeError("TrackingLogic must be configured before observe()")
        last_t = self._last_observer_t  # timestamp of previous observer update
        dt = (
            0.0 if last_t is None else max(0.0, float(t) - float(last_t))
        )  # observer update period
        self._last_observer_t = float(t)
        if self.observer is None:
            if not isinstance(input_data, RobotState3D):
                raise TypeError("OdomInput3D tracking requires an observer")
            observed_state = dataclasses.replace(input_data)
            estimation = None
        else:
            observed_state, estimation = self.observer.process_state(
                t, dt, input_data, actual_actuation
            )
        self._estimated_state = observed_state
        self._estimation = estimation
        return observed_state

    def think(self, t: float, actual_actuation: ValveOpenings) -> ValveOpenings:
        """Advance tracking lifecycle and delegate the control step to the configured controller."""
        if self.controller is None or self.initial_state is None:
            raise RuntimeError("TrackingLogic must be configured before think()")
        state = self._last_state_or_noop(t, actual_actuation)
        if state is None:
            return self._zero_like_actuation(actual_actuation)

        planner = self.controller.planner
        if not self.config.hold_initial_endpoint:
            if planner is self._initial_hold:
                self._start_movement(t, state, self.config.target_state)
            elif planner is self._movement_planner and planner.is_finished(t, state):
                self._start_hold(t, self._active_destination)
            elif (
                self._hold_until is not None
                and self._hold_next_state is not None
                and t >= self._hold_until
            ):
                self._start_movement(t, state, self._hold_next_state)

        control = self.controller(t, actual_actuation, state, self._estimation)
        return control

    def control_hz(self) -> float:
        """Return the command frequency owned by this tracking configuration."""
        return float(self.config.control_hz)

    @property
    def estimation(self) -> WrenchEstimate3D | None:
        """Return the latest observer-owned environment estimate."""
        return self._estimation

    @property
    def history(self):
        """Expose controller history live so recorders can stream every control step."""
        return super().history if self.controller is None else self.controller.history

    def finish(self) -> None:
        """Keep the live controller history in place without copying it on shutdown."""

    def get_markers(self) -> list[VisualizationMarker]:
        """Return endpoint markers and active-trajectory markers only while trajectory tracking is active."""
        target = (
            self.initial_state
            if self.config.hold_initial_endpoint and self.initial_state is not None
            else self.config.target_state
        )
        markers: list[VisualizationMarker] = [
            self.config.marker_projector(target, "blue", self.config.target_marker_size)
        ]
        if self.initial_state is not None:
            markers.append(
                self.config.marker_projector(
                    self.initial_state, "red", self.config.target_marker_size
                )
            )
        if (
            self._movement_planner is not None
            and self.controller is not None
            and self.controller.planner is self._movement_planner
        ):
            markers.extend(self._trajectory_markers(self._movement_planner))
        return markers

    def state_label(self) -> str:
        """Return the current tracking lifecycle phase for viewer overlays."""
        if self.controller is None:
            return "UNCONFIGURED"
        if self.config.hold_initial_endpoint:
            return "POINT_HOLD"
        planner = self.controller.planner
        if planner is self._initial_hold:
            return "INITIAL_HOLD"
        if planner is self._movement_planner:
            return "TRACKING"
        if self._hold_until is not None:
            return "ENDPOINT_HOLD"
        return "TRACKING"

    def _start_movement(
        self, now: float, state: RobotState3D, destination: RobotState3D
    ) -> None:
        """Switch from endpoint hold to an active trajectory planner."""
        if self.controller is None:
            raise RuntimeError("TrackingLogic controller is not configured")
        self._movement_planner = self.config.trajectory_planner_builder(
            now, state, destination
        )
        self._active_destination = destination
        self._hold_until = None
        self._hold_next_state = None
        self.controller.set_planner(self._movement_planner, ts=now)

    def _start_hold(self, now: float, destination: RobotState3D | None) -> None:
        """Switch from an active trajectory to endpoint stabilization hold."""
        if self.controller is None or self.initial_state is None or destination is None:
            raise RuntimeError(
                "TrackingLogic cannot start hold before movement is configured"
            )
        self.controller.set_planner(
            EndpointPlanner3D(destination), ts=now
        )
        self._movement_planner = None
        self._active_destination = None
        self._hold_until = now + self.config.stabilization_time
        self._hold_next_state = (
            self.initial_state
            if destination is self.config.target_state
            else self.config.target_state
        )

    def _trajectory_markers(
        self, planner: TrajectoryPlanner
    ) -> list[VisualizationMarker]:
        """Sample the active trajectory planner at absolute simulation times for visualization."""
        params = planner.params
        if float("inf") == params.T:
            return []
        count = max(
            2, int(self.config.trajectory_marker_count)
        )  # количество markers вдоль траектории
        return [
            self.config.marker_projector(
                planner.target_at_time(
                    params.start_t + params.T * i / (count - 1)
                ).state,
                "green",
                self.config.trajectory_marker_size,
            )
            for i in range(count)
        ]
