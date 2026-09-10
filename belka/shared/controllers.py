from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from belka.common import RobotState3D, ThrusterForces, ValveOpenings
from belka.shared.observers import WrenchEstimate3D
from belka.shared.planners import TrajectoryPlanner


@dataclass
class ControlHistory:
    """Controller history for the spatial control stack."""

    ts: list[float] = field(default_factory=list)
    observed_state: list[RobotState3D] = field(default_factory=list)
    reference_state: list[RobotState3D | None] = field(default_factory=list)
    control_forces: list[ValveOpenings] = field(default_factory=list)
    real_forces: list[ThrusterForces] = field(default_factory=list)
    error: list[RobotState3D | None] = field(default_factory=list)
    estimations: list[WrenchEstimate3D | None] = field(default_factory=list)
    events: list[tuple[float, str]] = field(default_factory=list)

    def mark(self, ts: float, event: str):
        """Append a lifecycle event marker to the history."""
        self.events.append((ts, event))

    def update(
        self,
        ts: float,
        state: RobotState3D,
        control: ValveOpenings,
        reference_state: RobotState3D | None = None,
        real_force: ThrusterForces | None = None,
        error: RobotState3D | None = None,
        estimation: WrenchEstimate3D | None = None,
    ):
        """Append one controller sample to history."""
        if real_force is None:
            real_force = control
        self.ts.append(ts)
        self.observed_state.append(state)
        self.reference_state.append(reference_state)
        self.control_forces.append(control)
        self.real_forces.append(real_force)
        self.error.append(error)
        self.estimations.append(estimation)

    def concatenate(self, other: "ControlHistory"):
        """Append another control history in-place."""
        self.ts.extend(other.ts)
        self.observed_state.extend(other.observed_state)
        self.reference_state.extend(other.reference_state)
        self.control_forces.extend(other.control_forces)
        self.real_forces.extend(other.real_forces)
        self.error.extend(other.error)
        self.estimations.extend(other.estimations)
        self.events.extend(other.events)

    def __len__(self):
        """Return number of controller samples."""
        return len(self.ts)


class Controller(metaclass=ABCMeta):
    """Spatial low-level controller API used by tracking logic."""

    planner: TrajectoryPlanner
    history: ControlHistory

    def set_planner(self, new_planner: TrajectoryPlanner, ts: float | None = None):
        """Attach a new planner and record the planner switch."""
        if ts is not None:
            new_planner.recalculate(new_planner.params.recreate_from_state(ts))
        self.planner = new_planner
        self.history.mark(ts or 0.0, f"<planner> {type(new_planner).__name__}")

    def _record_history(
        self,
        ts: float,
        state: RobotState3D,
        control: ValveOpenings,
        reference_state: RobotState3D | None = None,
        real_force: ThrusterForces | None = None,
        error: RobotState3D | None = None,
        estimation: WrenchEstimate3D | None = None,
    ) -> None:
        """Record one controller output sample."""
        self.history.update(
            ts=ts,
            state=state,
            control=control,
            reference_state=reference_state,
            real_force=real_force,
            error=error,
            estimation=estimation,
        )

    @abstractmethod
    def __call__(
        self,
        t: float,
        current_forces: ValveOpenings,
        estimated_state: RobotState3D,
        estimation: WrenchEstimate3D | None,
    ) -> ValveOpenings:
        """Compute a force command from the current state estimate and external-wrench estimate."""
        pass
