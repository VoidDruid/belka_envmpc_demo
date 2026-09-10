"""Least-squares derivative helpers for noisy sampled signals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


def least_squares_derivative(
    samples: Iterable[tuple[float, np.ndarray]],
    *,
    derivative_order: int = 1,
    polynomial_degree: int = 1,
    at_t: float | None = None,
    min_samples: int | None = None,
) -> np.ndarray | None:
    """Estimate a signal derivative by fitting a local polynomial in least squares.

    The input samples are `(t, value)` pairs. Values can be scalar or vector-shaped;
    each component is fitted independently with the same Vandermonde matrix.
    """
    ordered = sorted((float(t), np.asarray(value, dtype=float)) for t, value in samples)
    if not ordered:
        return None
    n = len(ordered)  # количество samples в локальном окне
    degree = int(polynomial_degree)  # степень polynomial fit-а
    order = int(derivative_order)  # порядок нужной производной
    required = max(degree + 1, order + 1) if min_samples is None else int(min_samples)
    if degree < order or n < required:
        return None

    t0 = float(
        ordered[-1][0] if at_t is None else at_t
    )  # центр локальной аппроксимации
    ts = np.array([t - t0 for t, _value in ordered], dtype=float)  # centered times
    y = np.stack([value for _t, value in ordered], axis=0)  # measured values
    powers = np.arange(degree + 1, dtype=int)  # polynomial powers 0..degree
    V = ts[:, None] ** powers[None, :]  # Vandermonde matrix
    coeffs, *_ = np.linalg.lstsq(V, y.reshape((n, -1)), rcond=None)
    factor = 1.0
    for multiplier in range(order):
        factor *= float(order - multiplier)
    derivative = factor * coeffs[order]  # derivative at centered time t=0
    return derivative.reshape(y.shape[1:])


@dataclass
class RollingDerivative:
    """Fixed-size sample buffer that returns least-squares derivatives."""

    maxlen: int = 8
    samples: deque[tuple[float, np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        """Create the bounded sample deque."""
        self.samples = deque(maxlen=int(self.maxlen))

    def add(self, t: float, value: np.ndarray) -> None:
        """Append one timestamped signal sample."""
        self.samples.append((float(t), np.asarray(value, dtype=float).copy()))

    def derivative(
        self,
        *,
        derivative_order: int = 1,
        polynomial_degree: int = 1,
        at_t: float | None = None,
        min_samples: int | None = None,
    ) -> np.ndarray | None:
        """Return the requested derivative from the current rolling window."""
        return least_squares_derivative(
            self.samples,
            derivative_order=derivative_order,
            polynomial_degree=polynomial_degree,
            at_t=at_t,
            min_samples=min_samples,
        )
