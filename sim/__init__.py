from dataclasses import dataclass, field

from belka.common import RobotState3D, ThrusterForces
from belka.shared.observers import WrenchEstimate3D


@dataclass(frozen=True)
class SimParams:
    """Numerical experiment scheduler parameters."""

    t_max: float
    full_odom_hz: float
    highfreq_odom_hz: float


@dataclass
class SimHistory:
    """History recorded by a numerical simulator IO."""

    ts: list[float] = field(default_factory=list)
    real_state: list[RobotState3D] = field(default_factory=list)
    perceived_state: list[RobotState3D] = field(default_factory=list)

    external_wrench: list[WrenchEstimate3D] = field(default_factory=list)
    engine_forces: list[ThrusterForces] = field(default_factory=list)

    def update(
        self,
        ts: float,
        real_state: RobotState3D,
        engine_forces: ThrusterForces,
        perceived_state: RobotState3D,
        external_wrench: WrenchEstimate3D,
    ):
        """Append one simulator sample to history."""
        self.real_state.append(real_state)
        self.perceived_state.append(perceived_state)
        self.engine_forces.append(engine_forces)
        self.ts.append(ts)
        self.external_wrench.append(external_wrench)


__all__ = [
    "SimHistory",
    "SimParams",
]
