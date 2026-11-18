"""The contract every forecaster implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from demand_forecast.panel import ForecastContext


@dataclass(frozen=True)
class Forecast:
    """Forecasts for one origin: arrays of shape (horizon, n_series).

    ``lower`` / ``upper`` are the model's own central interval when it has one (the
    gradient-boosting quantile models); every model also gets a conformal interval from the
    backtest.
    """

    mean: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    info: dict[str, float] = field(default_factory=dict)


class Forecaster(Protocol):
    name: str

    def forecast(self, ctx: ForecastContext) -> Forecast:
        """Fit on ``ctx.history`` and forecast ``ctx.horizon`` steps for every series."""
        ...
