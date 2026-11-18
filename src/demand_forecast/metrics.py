"""Point and interval forecast metrics.

Conventions:

* errors are ``forecast - actual``, so a positive bias means over-forecasting;
* sMAPE is in percent (0..200) and a term where actual and forecast are both zero counts as 0,
  which matters for intermittent demand;
* MASE divides by the in-sample mean absolute seasonal difference of the training history
  (Hyndman and Koehler, 2006), computed per series and per forecast origin;
* the interval score is the Winkler / Gneiting-Raftery score of a central (1 - alpha) interval;
  scaled by the MASE denominator and averaged it is MSIS (as used in the M4 competition).
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray | list[float]


def _arrays(*values: ArrayLike) -> list[np.ndarray]:
    arrays = [np.asarray(v, dtype=float) for v in values]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("inputs must have the same shape")
    return arrays


def mae(actual: ArrayLike, forecast: ArrayLike) -> float:
    a, f = _arrays(actual, forecast)
    return float(np.mean(np.abs(f - a)))


def rmse(actual: ArrayLike, forecast: ArrayLike) -> float:
    a, f = _arrays(actual, forecast)
    return float(np.sqrt(np.mean((f - a) ** 2)))


def smape_terms(actual: ArrayLike, forecast: ArrayLike) -> np.ndarray:
    a, f = _arrays(actual, forecast)
    denom = np.abs(a) + np.abs(f)
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = np.where(denom == 0, 0.0, 2.0 * np.abs(f - a) / denom)
    return 100.0 * terms


def smape(actual: ArrayLike, forecast: ArrayLike) -> float:
    return float(np.mean(smape_terms(actual, forecast)))


def mase(actual: ArrayLike, forecast: ArrayLike, scale: ArrayLike) -> float:
    a, f, s = _arrays(actual, forecast, np.broadcast_to(np.asarray(scale, float), np.shape(actual)))
    return float(np.mean(np.abs(f - a) / s))


def bias(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Mean error, forecast minus actual."""
    a, f = _arrays(actual, forecast)
    return float(np.mean(f - a))


def bias_pct(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Total over- (+) or under-forecast (-) as a percentage of total actual demand."""
    a, f = _arrays(actual, forecast)
    total = float(np.sum(a))
    return float("nan") if total == 0 else 100.0 * float(np.sum(f - a)) / total


def coverage(actual: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    a, lo, hi = _arrays(actual, lower, upper)
    return float(np.mean((a >= lo) & (a <= hi)))


def interval_score(
    actual: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float
) -> np.ndarray:
    """Per-point interval score: width plus 2/alpha times the distance outside the interval."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    a, lo, hi = _arrays(actual, lower, upper)
    below = np.clip(lo - a, 0.0, None)
    above = np.clip(a - hi, 0.0, None)
    return (hi - lo) + (2.0 / alpha) * (below + above)


def msis(
    actual: ArrayLike, lower: ArrayLike, upper: ArrayLike, scale: ArrayLike, alpha: float
) -> float:
    s = np.broadcast_to(np.asarray(scale, float), np.shape(actual))
    return float(np.mean(interval_score(actual, lower, upper, alpha) / s))


def seasonal_scale(history: np.ndarray, season_length: int) -> np.ndarray:
    """MASE denominator for each column of ``history`` (time on axis 0).

    Mean absolute seasonal difference; falls back to the lag-1 difference when the series is
    shorter than two seasons or perfectly seasonal, and to 1.0 for a constant series, so the
    result is always positive.
    """
    h = np.asarray(history, dtype=float)
    if h.ndim == 1:
        h = h[:, None]
    out = np.full(h.shape[1], np.nan)
    if h.shape[0] > season_length:
        out = np.mean(np.abs(h[season_length:] - h[:-season_length]), axis=0)
    if h.shape[0] > 1:
        lag1 = np.mean(np.abs(np.diff(h, axis=0)), axis=0)
        out = np.where(np.isnan(out) | (out == 0), lag1, out)
    return np.where(np.isnan(out) | (out == 0), 1.0, out)
