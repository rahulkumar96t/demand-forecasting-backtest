"""Rolling-origin backtest.

Origins are spaced ``step`` days apart and end ``horizon`` days before the last observation::

    |----- history -----|o1 -- h --|
    |------- history -------|o2 -- h --|
    |----------- history -----------|o3 -- h --|        ... up to the end of the data

The first ``calibration_origins`` origins are not scored. Their errors only seed the conformal
intervals, so every scored origin gets an interval built from errors observed before it. At
each origin a model receives a :class:`~demand_forecast.panel.ForecastContext` whose history
stops at the origin, so it cannot read the actuals it is scored on.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from demand_forecast.intervals import ConformalCalibrator
from demand_forecast.metrics import seasonal_scale
from demand_forecast.models.base import Forecast, Forecaster
from demand_forecast.panel import ForecastContext, Panel

CALIBRATION = "calibration"
EVALUATION = "evaluation"


@dataclass(frozen=True)
class BacktestConfig:
    horizon: int = 28
    step: int = 28
    n_origins: int = 8
    calibration_origins: int = 4
    season_length: int = 7
    alpha: float = 0.2
    min_train: int = 365
    min_conformal_samples: int = 20

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.step < 1 or self.n_origins < 1:
            raise ValueError("horizon, step and n_origins must be positive")
        if self.calibration_origins < 0:
            raise ValueError("calibration_origins must be >= 0")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    points: pd.DataFrame  # one row per model, origin, horizon step and series
    runs: pd.DataFrame  # one row per model and origin: fit + predict time, diagnostics
    config: BacktestConfig


def origin_indices(n_obs: int, cfg: BacktestConfig) -> np.ndarray:
    """Indices of the last observed day at each origin, oldest first."""
    total = cfg.calibration_origins + cfg.n_origins
    last = n_obs - 1 - cfg.horizon
    origins = last - cfg.step * np.arange(total)[::-1]
    if origins[0] + 1 < cfg.min_train:
        raise ValueError(
            f"first origin has {origins[0] + 1} days of history, min_train is {cfg.min_train}; "
            "use fewer origins, a smaller step or a smaller min_train"
        )
    return origins


def _check(fc: Forecast, ctx: ForecastContext, name: str) -> None:
    shape = (ctx.horizon, ctx.n_series)
    for label, arr in (("mean", fc.mean), ("lower", fc.lower), ("upper", fc.upper)):
        if arr is not None and arr.shape != shape:
            raise ValueError(f"{name}: {label} has shape {arr.shape}, expected {shape}")
    if not np.all(np.isfinite(fc.mean)):
        raise ValueError(f"{name}: non-finite forecast")


def _frame(
    panel: Panel,
    model: str,
    origin: int,
    role: str,
    ctx: ForecastContext,
    fc: Forecast,
    actual: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    scale: np.ndarray,
) -> pd.DataFrame:
    h, n = ctx.horizon, ctx.n_series
    nan = np.full((h, n), np.nan)
    return pd.DataFrame(
        {
            "model": model,
            "series_id": np.tile(np.array(panel.series_ids, dtype=object), h),
            "origin": panel.dates[origin],
            "role": role,
            "h": np.repeat(np.arange(1, h + 1), n),
            "date": np.repeat(ctx.target_dates.values, n),
            "actual": actual.ravel(),
            "forecast": fc.mean.ravel(),
            "lower": bounds[0].ravel(),
            "upper": bounds[1].ravel(),
            "lower_native": (fc.lower if fc.lower is not None else nan).ravel(),
            "upper_native": (fc.upper if fc.upper is not None else nan).ravel(),
            "scale": np.tile(scale, h),
        }
    )


def run_backtest(
    panel: Panel,
    models: Mapping[str, Forecaster],
    cfg: BacktestConfig,
    log: Callable[[str], None] | None = None,
) -> BacktestResult:
    origins = origin_indices(panel.n_obs, cfg)
    roles = [CALIBRATION] * cfg.calibration_origins + [EVALUATION] * cfg.n_origins
    frames: list[pd.DataFrame] = []
    runs: list[dict[str, object]] = []
    for name, model in models.items():
        calibrator = ConformalCalibrator(cfg.alpha, cfg.horizon, cfg.min_conformal_samples)
        started = time.process_time()
        for origin, role in zip(origins, roles, strict=True):
            ctx = panel.context(int(origin), cfg.horizon, cfg.season_length)
            t0, c0 = time.perf_counter(), time.process_time()
            fc = model.forecast(ctx)
            seconds, cpu_seconds = time.perf_counter() - t0, time.process_time() - c0
            _check(fc, ctx, name)
            actual = panel.y[origin + 1 : origin + 1 + cfg.horizon]
            scale = seasonal_scale(ctx.history, cfg.season_length)
            bounds = calibrator.bounds(int(origin), fc.mean, scale)
            calibrator.update(int(origin), actual - fc.mean, scale)
            frames.append(_frame(panel, name, int(origin), role, ctx, fc, actual, bounds, scale))
            runs.append(
                {
                    "model": name,
                    "origin": panel.dates[origin],
                    "role": role,
                    "seconds": seconds,
                    "cpu_seconds": cpu_seconds,
                    **fc.info,
                }
            )
        if log:
            log(f"{name}: {len(origins)} origins, {time.process_time() - started:.1f} CPU s")
    return BacktestResult(
        points=pd.concat(frames, ignore_index=True), runs=pd.DataFrame(runs), config=cfg
    )


def forecast_future(panel: Panel, model: Forecaster, cfg: BacktestConfig) -> pd.DataFrame:
    """Forecast ``horizon`` days past the end of the data with conformal intervals.

    The intervals come from the ``calibration_origins`` most recent backtest origins.
    """
    calibrator = ConformalCalibrator(cfg.alpha, cfg.horizon, cfg.min_conformal_samples)
    last = panel.n_obs - 1
    for k in range(cfg.calibration_origins, 0, -1):
        origin = last - cfg.horizon - cfg.step * (k - 1)
        ctx = panel.context(origin, cfg.horizon, cfg.season_length)
        fc = model.forecast(ctx)
        actual = panel.y[origin + 1 : origin + 1 + cfg.horizon]
        calibrator.update(origin, actual - fc.mean, seasonal_scale(ctx.history, cfg.season_length))
    ctx = panel.context(last, cfg.horizon, cfg.season_length)
    fc = model.forecast(ctx)
    _check(fc, ctx, model.name)
    scale = seasonal_scale(ctx.history, cfg.season_length)
    bounds = calibrator.bounds(last, fc.mean, scale)
    out = _frame(panel, model.name, last, "forecast", ctx, fc, np.full_like(fc.mean, np.nan),
                 bounds, scale)  # fmt: skip
    return out.drop(columns=["actual", "role"])
