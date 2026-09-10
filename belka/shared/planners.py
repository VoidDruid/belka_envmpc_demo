import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, replace

from belka.common import RobotState3D, StateTolerances3D


@dataclass(frozen=True)
class PlanSample:
    """One trajectory-planner target state."""

    state: RobotState3D


@dataclass(frozen=True, init=False, kw_only=True)
class PlanningParameters(metaclass=ABCMeta):
    """Lifecycle parameters shared by spatial trajectory planners."""

    T: float = math.inf
    start_t: float = 0.0
    replan_tolerances: StateTolerances3D
    finish_tolerances: StateTolerances3D
    replan_threshold_steps: int = 20

    def is_almost_finished(self, state: RobotState3D) -> bool:
        """Check whether state reaches the configured finished target."""
        target = self.finished_target
        return target is not None and state.is_close(target, self.finish_tolerances)

    def is_state_achievable(
        self, t: float, current_state: RobotState3D, next_state: RobotState3D
    ) -> bool:
        """Check whether planned state is still close enough for tracking without replanning."""
        return not (
            t > self.T or not current_state.is_close(next_state, self.replan_tolerances)
        )

    @property
    @abstractmethod
    def finished_target(self) -> RobotState3D | None:
        """Return final target that defines planner completion."""
        pass

    def recreate_from_state(
        self, current_t: float, new_T: float | None = None, **param_map
    ) -> "PlanningParameters":
        """Create updated parameters for replanning from the current simulation state."""
        return replace(
            self,
            start_t=current_t,
            T=new_T if new_T is not None else self.T,
            **param_map,
        )


class PlanningError(Exception):
    """Planner exception carrying the current state and failed target sample."""

    def __init__(
        self,
        message,
        current_state: RobotState3D | None = None,
        next_planned: PlanSample | None = None,
        **planner_params,
    ):
        """Create a planning error with debug context."""
        super().__init__(message)
        self.next_planned = next_planned
        self.current_state = current_state
        self.planner_params = planner_params


class TrajectoryPlanner(metaclass=ABCMeta):
    """Spatial trajectory planner lifecycle with tolerance-based replanning and finish."""

    def __init__(self, params: PlanningParameters) -> None:
        """Initialize planner params and replanning streak state."""
        self.params = params
        self._unachievable_streak = 0

    def planner_time(self, current_t: float) -> float:
        """Convert absolute simulation time to local planner time."""
        return current_t - self.params.start_t

    def recalculate(self, params: PlanningParameters):
        """Replace planner parameters and recompute concrete trajectory internals."""
        self.params = params

    @abstractmethod
    def target_at_time(self, t: float) -> PlanSample:
        """Return target sample at absolute simulation time t."""
        pass

    def is_state_achievable(
        self, t: float, current_state: RobotState3D, next_state: RobotState3D
    ) -> bool:
        """Delegate achievable-state check to planning parameters."""
        return self.params.is_state_achievable(t, current_state, next_state)

    def is_done(self, current_state: RobotState3D):
        """Check whether current state satisfies finish tolerances."""
        return self.params.is_almost_finished(current_state)

    def is_finished(self, current_t: float, current_state: RobotState3D) -> bool:
        """Return whether trajectory time elapsed and its final state was reached."""
        return self.planner_time(current_t) >= self.params.T and self.is_done(
            current_state
        )

    def next_target(
        self,
        current_t: float,
        state: RobotState3D,
        fail_on_unachievable: bool = False,
    ) -> PlanSample:
        """Follow the timed trajectory, then stabilize or replan its final target.

        Before T, reaching the final state does not truncate the trajectory. At
        or after T, a completed plan stabilizes at its target; an incomplete
        plan is immediately rebuilt from the current state.
        """
        t = self.planner_time(current_t)
        if t >= self.params.T:
            if self.is_done(state):
                self._unachievable_streak = 0
                target = self.params.finished_target
                if target is not None:
                    return PlanSample(target)
            self._unachievable_streak = 0
            self.recalculate(
                self.params.recreate_from_state(current_t, initial_state=state)
            )
            return self.next_target(current_t, state, fail_on_unachievable=True)

        planned = self.target_at_time(current_t)

        if not self.is_state_achievable(t, state, planned.state):
            self._unachievable_streak += 1
            if (
                not fail_on_unachievable
                and self._unachievable_streak < self.params.replan_threshold_steps
            ):
                return planned
            if fail_on_unachievable:
                raise PlanningError(
                    "Next planned state is unachievable",
                    next_planned=planned,
                    current_state=state,
                    params=self.params,
                )
            self._unachievable_streak = 0
            self.recalculate(
                self.params.recreate_from_state(current_t, initial_state=state)
            )
            return self.next_target(current_t, state, fail_on_unachievable=True)
        self._unachievable_streak = 0
        return planned
