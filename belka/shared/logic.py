from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from belka.common import OdomInput3D, RobotState3D, ValveOpenings
from belka.shared.controllers import ControlHistory


@dataclass(frozen=True)
class VisualizationMarker:
    """Simple marker descriptor consumed by simulator visualization code."""

    x: float
    y: float
    z: float = 0.1
    color: str = "blue"
    size: float = 0.1
    direction: Sequence[float] | None = None
    arrow_color: str | None = None


class HighlevelLogic(metaclass=ABCMeta):
    """Portable spatial behavior API shared by simulator and future RobotIO."""

    @abstractmethod
    def configure(self, initial_state: RobotState3D) -> None:
        """Initialize runtime behavior from the first observed state."""
        pass

    def observation_input_type(self) -> type:
        """Return the runtime observation input type expected by observe()."""
        return RobotState3D

    def observe(
        self,
        t: float,
        input_data: RobotState3D | OdomInput3D,
        actual_actuation: ValveOpenings,
    ) -> RobotState3D:
        """Update the stored state estimate from one odometry input sample."""
        del t, actual_actuation
        if not isinstance(input_data, RobotState3D):
            raise TypeError(
                f"{self.__class__.__name__} must implement OdomInput3D observe()"
            )
        self._estimated_state = input_data
        return input_data

    @abstractmethod
    def think(self, t: float, actual_actuation: ValveOpenings) -> ValveOpenings:
        """Compute the next command from state estimate and actual actuator state."""
        pass

    def get_markers(self) -> list[VisualizationMarker]:
        """Return optional visualization markers for the current behavior state."""
        return []

    @abstractmethod
    def control_hz(self) -> float:
        """Return the measured nominal rate used by controller models and SimIO."""
        pass

    def _last_state_or_noop(
        self, t: float, actual_actuation: ValveOpenings
    ) -> RobotState3D | None:
        """Return stored observed state or log and leave control as no-op."""
        state = getattr(
            self, "_estimated_state", None
        )  # latest state estimate from observe()
        if state is None and not getattr(self, "_warned_no_observation", False):
            print(
                f"[{self.__class__.__name__} t={t:.3f}] think before observe; returning zero command",
                flush=True,
            )
            self._warned_no_observation = True
            if hasattr(self, "_history"):
                self._history.mark(t, "<warning> think_before_observe")
        return state

    def _zero_like_actuation(self, actual_actuation: ValveOpenings) -> ValveOpenings:
        """Return a zero valve command with the current actuator count."""
        return ValveOpenings.zeros(actual_actuation.nu)

    def state_label(self) -> str:
        """Return a short human-readable state-machine label for viewer overlays."""
        return self.__class__.__name__

    @property
    def history(self) -> ControlHistory:
        """Return accumulated high-level history."""
        if not hasattr(self, "_history"):
            self._history = ControlHistory()
        return self._history

    def finish(self):
        """Collect controller history when the behavior finishes."""
        if hasattr(self, "controller") and self.controller is not None:
            self.history.concatenate(self.controller.history)
