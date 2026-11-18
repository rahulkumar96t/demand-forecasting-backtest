"""Origin-anchored features for the global gradient-boosting model.

One row describes "series ``j``, standing at origin ``o``, forecasting ``h`` days ahead". The
target is ``y[o + h, j]``. Every target-derived feature reads ``y[t]`` only for ``t <= o``:

* level: mean of the last 28 days (plus one unit, so slow movers do not divide by zero); all
  other target features are divided by it and the global model learns shapes, not volumes;
* last value, 7 / 91-day means, 28-day standard deviation and share of zero days;
* seasonal lags: the most recent same weekday at or before ``o`` and the mean of the last four;
  the same weekday 52 weeks before the target and the week around it, cut at ``o``;
* days since the last sale;
* horizon ``h`` and the calendar of the target date;
* covariates known in advance (promotion plan, holidays) on the target date and their mean
  over the 7 days before it;
* static attributes (store, item, category) as categoricals.

Because one row only needs data up to its own origin, the same function builds the training
set (many past origins) and the prediction rows (the current origin) without recursion.
``tests/test_leakage.py`` checks that rewriting the future leaves every feature unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

LEVEL_WINDOW = 28
YEAR_LAG = 364  # 52 weeks, so the lag lands on the same weekday


def training_pairs(
    last_obs: int, horizon: int, lookback: int, stride: int, min_history: int
) -> tuple[np.ndarray, np.ndarray]:
    """(origin, horizon) pairs for training, all with ``origin + h <= last_obs``.

    Origins step back from ``last_obs - 1`` every ``stride`` days for ``lookback`` days, so the
    most recent targets are used for short horizons too.
    """
    origins: list[int] = []
    steps: list[int] = []
    first = max(min_history - 1, last_obs - lookback)
    for o in range(last_obs - 1, first - 1, -stride):
        for h in range(1, min(horizon, last_obs - o) + 1):
            origins.append(o)
            steps.append(h)
    return np.asarray(origins, dtype=int), np.asarray(steps, dtype=int)


def _window_mean(cum: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Mean of rows ``start..end`` (inclusive) from a cumulative sum with a leading zero row."""
    valid = (start >= 0) & (end >= start)
    s = np.where(valid, start, 0)
    e = np.where(valid, end, 0)
    count = np.where(valid, e - s + 1, 1)[:, None]
    out = (cum[e + 1] - cum[s]) / count
    out[~valid] = np.nan
    return out


def _take(y: np.ndarray, idx: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """``y[idx]`` per row, NaN where ``idx`` is before the start or after ``upper``."""
    valid = (idx >= 0) & (idx <= upper)
    out = y[np.where(valid, idx, 0)].astype(float)
    out[~valid] = np.nan
    return out


def make_features(
    history: np.ndarray,
    origins: np.ndarray,
    horizons: np.ndarray,
    dates: pd.DatetimeIndex,
    known: Mapping[str, np.ndarray],
    static: pd.DataFrame | None,
    season_length: int = 7,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature rows for every (pair, series); rows are ordered pair-major.

    Returns the feature frame and the per-row normaliser (level + 1). ``history`` must not
    contain anything after the largest origin a caller wants to be leakage-free for; the
    function itself only indexes ``history`` at or before each row's origin.
    """
    y = np.asarray(history, dtype=float)
    o = np.asarray(origins, dtype=int)
    h = np.asarray(horizons, dtype=int)
    n, n_series = y.shape
    if o.shape != h.shape:
        raise ValueError("origins and horizons must have the same length")
    if o.size and (o.min() < 0 or o.max() >= n):
        raise ValueError("origins must index into history")
    if h.size and h.min() < 1:
        raise ValueError("horizons start at 1")
    m = season_length
    target_idx = o + h
    if target_idx.size and target_idx.max() >= len(dates):
        raise ValueError("dates do not cover the target dates")

    zero = np.zeros((1, n_series))
    cum = np.vstack([zero, np.cumsum(y, axis=0)])
    cum_sq = np.vstack([zero, np.cumsum(y**2, axis=0)])
    cum_zero = np.vstack([zero, np.cumsum(y == 0, axis=0)])

    def trailing(cumulative: np.ndarray, window: int) -> np.ndarray:
        return _window_mean(cumulative, np.maximum(o - window + 1, 0), o)

    level = trailing(cum, LEVEL_WINDOW)
    norm = level + 1.0
    std = np.sqrt(np.clip(trailing(cum_sq, LEVEL_WINDOW) - level**2, 0.0, None))

    # most recent same weekday as the target, at or before the origin
    same_dow = target_idx - m * np.ceil(h / m).astype(int)
    recent = np.stack([_take(y, same_dow - k * m, o) for k in range(4)])
    n_valid = np.sum(~np.isnan(recent), axis=0)
    same_dow_mean = np.where(
        n_valid > 0, np.nansum(recent, axis=0) / np.maximum(n_valid, 1), np.nan
    )
    year_ago = target_idx - YEAR_LAG
    year_lag = _take(y, year_ago, o)
    year_week = _window_mean(cum, year_ago - 3, np.minimum(year_ago + 3, o))

    # days since the last non-zero value, up to each origin
    idx = np.arange(n)[:, None]
    last_sale = np.maximum.accumulate(np.where(y > 0, idx, -1), axis=0)
    days_since = (o[:, None] - last_sale[o]).astype(float)
    days_since[last_sale[o] < 0] = np.nan

    cols: dict[str, np.ndarray] = {
        "h": np.repeat(h[:, None], n_series, axis=1).astype(float),
        "log_level": np.log(norm),
        "last": y[o] / norm,
        "mean_7": trailing(cum, 7) / norm,
        "mean_91": trailing(cum, 91) / norm,
        "std_28": std / norm,
        "zero_share_28": trailing(cum_zero, LEVEL_WINDOW),
        "same_weekday": _take(y, same_dow, o) / norm,
        "same_weekday_mean_4": same_dow_mean / norm,
        "year_lag": year_lag / norm,
        "year_week_mean": year_week / norm,
        "days_since_sale": days_since,
    }
    target_dates = dates[target_idx]
    for name, values in {
        "dow": target_dates.day_of_week,
        "day_of_month": target_dates.day,
        "month": target_dates.month,
        "day_of_year": target_dates.day_of_year,
    }.items():
        cols[name] = np.repeat(np.asarray(values, dtype=float)[:, None], n_series, axis=1)

    for name, values in known.items():
        v = np.asarray(values, dtype=float)
        if target_idx.size and v.shape[0] <= target_idx.max():
            raise ValueError(f"known covariate {name!r} does not cover the target dates")
        cum_v = np.vstack([zero, np.cumsum(v, axis=0)])
        cols[name] = v[target_idx]
        cols[f"{name}_prev_7"] = _window_mean(cum_v, target_idx - 7, target_idx - 1)

    frame = pd.DataFrame({k: v.ravel() for k, v in cols.items()})
    if static is not None and not static.empty:
        for col in static.columns:
            levels = sorted(static[col].astype(str).unique())
            values = np.tile(static[col].astype(str).to_numpy(), len(o))
            frame[col] = pd.Categorical(values, categories=levels)
    return frame, norm.ravel()
