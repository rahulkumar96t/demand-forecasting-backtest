"""Holt-Winters exponential smoothing, one model per series (statsmodels)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from demand_forecast.models.base import Forecast
from demand_forecast.models.baselines import seasonal_naive_mean
from demand_forecast.panel import ForecastContext


def holt_winters(series: np.ndarray, horizon: int, season_length: int, damped: bool) -> np.ndarray:
    """Additive (damped) trend + additive seasonality; raises if the fit is unusable."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter on flat or sparse series
        fit = ExponentialSmoothing(
            series,
            trend="add",
            damped_trend=damped,
            seasonal="add",
            seasonal_periods=season_length,
            initialization_method="estimated",
        ).fit()
        out = np.asarray(fit.forecast(horizon), dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite forecast")
    return out


def _one(
    series: np.ndarray, horizon: int, season_length: int, damped: bool
) -> tuple[np.ndarray, bool]:
    try:
        return holt_winters(series, horizon, season_length, damped), False
    except (ValueError, np.linalg.LinAlgError):
        fallback = seasonal_naive_mean(series[:, None], horizon, season_length)[:, 0]
        return fallback, True


@dataclass(frozen=True)
class HoltWinters:
    """Damped additive Holt-Winters with weekly seasonality, fitted per series.

    Only the last ``fit_window`` days are used (the yearly cycle is out of reach for a daily
    seasonal period of 7 anyway), forecasts are clipped at zero and a failed fit falls back
    to the seasonal naive forecast (counted in ``info['fallbacks']``).
    """

    fit_window: int = 365
    damped: bool = True
    n_jobs: int = 1
    name: str = "holt_winters"

    def forecast(self, ctx: ForecastContext) -> Forecast:
        window = ctx.history[-self.fit_window :]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_one)(window[:, j], ctx.horizon, ctx.season_length, self.damped)
            for j in range(ctx.n_series)
        )
        mean = np.column_stack([r[0] for r in results])
        fallbacks = float(sum(r[1] for r in results))
        return Forecast(mean=np.clip(mean, 0.0, None), info={"fallbacks": fallbacks})
