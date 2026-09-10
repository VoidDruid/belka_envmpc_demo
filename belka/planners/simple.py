from dataclasses import dataclass, field

from belka.common import RobotState3D, StateTolerances3D
from belka.shared.planners import PlanSample, PlanningParameters, TrajectoryPlanner


@dataclass(frozen=True, kw_only=True)
class _EndpointPlanningParameters3D(PlanningParameters):
    """Параметры 3D endpoint planner-а для удержания 6DoF состояния."""

    target_state: RobotState3D
    replan_tolerances: StateTolerances3D = field(default_factory=StateTolerances3D)
    finish_tolerances: StateTolerances3D = field(default_factory=StateTolerances3D)

    def is_state_achievable(
        self, t: float, current_state: RobotState3D, next_state: RobotState3D
    ) -> bool:
        """Endpoint hold всегда достижим как reference для стабилизации."""
        return True

    @property
    def finished_target(self) -> RobotState3D:
        """Вернуть удерживаемое 3D состояние."""
        return self.target_state


class EndpointPlanner3D(TrajectoryPlanner):
    """Planner, который всегда возвращает фиксированное 3D состояние."""

    def __init__(self, target_state: RobotState3D) -> None:
        """Создать 3D endpoint planner для фиксированного состояния."""
        super().__init__(_EndpointPlanningParameters3D(target_state=target_state))

    def target_at_time(self, current_t: float) -> PlanSample:
        """Вернуть endpoint sample; absolute time не влияет на hold target."""
        return PlanSample(self.params.target_state)
