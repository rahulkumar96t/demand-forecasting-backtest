"""Sequential split-conformal prediction intervals from earlier backtest origins.

For each model and horizon step ``h`` we keep the signed errors ``actual - forecast`` of
previous origins, divided by each series' MASE scale so all series can share one pool. At
origin ``o`` an error from origin ``o_j`` at step ``h`` is only used if it was observable by
then, i.e. ``o_j + h <= o``. The interval is::

    [forecast + q_lo * scale, forecast + q_hi * scale]

with ``q_lo`` / ``q_hi`` the ``alpha/2`` and ``1 - alpha/2`` empirical quantiles (with the usual
finite-sample correction). Separate tails keep the interval asymmetric, which suits skewed
demand. Bounds are clipped at zero; since demand is non-negative that never changes coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConformalCalibrator:
    alpha: float
    horizon: int
    min_samples: int = 20
    _origins: list[int] = field(default_factory=list)
    _errors: list[np.ndarray] = field(default_factory=list)  # each (horizon, n_series)

    def update(self, origin: int, errors: np.ndarray, scale: np.ndarray) -> None:
        """Store the scaled errors of the forecast made at ``origin``."""
        if errors.shape[0] != self.horizon:
            raise ValueError("errors must have one row per horizon step")
        self._origins.append(origin)
        self._errors.append(errors / scale[None, :])

    def pool(self, origin: int, step: int) -> np.ndarray:
        """Scaled errors at horizon ``step`` (1-based) that were observed by ``origin``."""
        usable = [
            e[step - 1]
            for o, e in zip(self._origins, self._errors, strict=True)
            if o + step <= origin
        ]
        return np.concatenate(usable) if usable else np.empty(0)

    def bounds(
        self, origin: int, mean: np.ndarray, scale: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interval for a forecast made at ``origin``; NaN where the pool is too small."""
        lower = np.full_like(mean, np.nan, dtype=float)
        upper = np.full_like(mean, np.nan, dtype=float)
        for step in range(1, self.horizon + 1):
            pool = self.pool(origin, step)
            n = pool.size
            if n < self.min_samples:
                continue
            lo_level = max(math.floor((n + 1) * self.alpha / 2) / n, 0.0)
            hi_level = min(math.ceil((n + 1) * (1 - self.alpha / 2)) / n, 1.0)
            q_lo = np.quantile(pool, lo_level, method="lower")
            q_hi = np.quantile(pool, hi_level, method="higher")
            lower[step - 1] = np.clip(mean[step - 1] + q_lo * scale, 0.0, None)
            upper[step - 1] = np.clip(mean[step - 1] + q_hi * scale, 0.0, None)
        return lower, upper
