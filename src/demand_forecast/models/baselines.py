"""Benchmarks every other model has to beat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from demand_forecast.models.base import Forecast
from demand_forecast.panel import ForecastContext


@dataclass(frozen=True)
class Naive:
    """Repeat the last observed value."""

    name: str = "naive"

    def forecast(self, ctx: ForecastContext) -> Forecast:
        return Forecast(mean=np.repeat(ctx.history[-1:], ctx.horizon, axis=0))


def seasonal_naive_mean(history: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
    """Repeat the last full season: step h uses the value ``season_length`` days earlier."""
    n = history.shape[0]
    if n < season_length:
        raise ValueError("history is shorter than one season")
    idx = n - season_length + (np.arange(horizon) % season_length)
    return history[idx].astype(float)


@dataclass(frozen=True)
class SeasonalNaive:
    """Same weekday last week."""

    name: str = "seasonal_naive"

    def forecast(self, ctx: ForecastContext) -> Forecast:
        return Forecast(mean=seasonal_naive_mean(ctx.history, ctx.horizon, ctx.season_length))


@dataclass(frozen=True)
class MovingAverage:
    """Flat forecast at the mean of the last ``window`` days."""

    window: int = 28
    name: str = "moving_average"

    def forecast(self, ctx: ForecastContext) -> Forecast:
        level = ctx.history[-self.window :].mean(axis=0, keepdims=True)
        return Forecast(mean=np.repeat(level, ctx.horizon, axis=0))
