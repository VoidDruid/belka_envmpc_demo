from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WrenchEstimate3D:
    """World-frame 3D external wrench estimate [Fx, Fy, Fz, Mx, My, Mz]."""

    Fx: float = 0.0
    Fy: float = 0.0
    Fz: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mz: float = 0.0

    @classmethod
    def zeros(cls) -> "WrenchEstimate3D":
        """Return zero 6D external wrench estimate."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_array(cls, values: np.ndarray) -> "WrenchEstimate3D":
        """Create 3D estimate from [F_world, M_world]."""
        vector = np.asarray(values, dtype=float)  # 6D world-frame wrench
        if vector.shape != (6,):
            raise ValueError(f"WrenchEstimate3D expects shape (6,), got {vector.shape}")
        return cls(
            Fx=float(vector[0]),
            Fy=float(vector[1]),
            Fz=float(vector[2]),
            Mx=float(vector[3]),
            My=float(vector[4]),
            Mz=float(vector[5]),
        )

    def to_array(self) -> np.ndarray:
        """Return [F_world, M_world]."""
        return np.array(
            [self.Fx, self.Fy, self.Fz, self.Mx, self.My, self.Mz], dtype=float
        )
