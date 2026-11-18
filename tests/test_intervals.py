import numpy as np

from demand_forecast.intervals import ConformalCalibrator


def _simulate(errors: np.ndarray, alpha: float = 0.2, warmup: int = 10):
    """errors: (origins, horizon, series). Mean forecast 100, scale 1."""
    n_origins, horizon, n_series = errors.shape
    cal = ConformalCalibrator(alpha=alpha, horizon=horizon, min_samples=20)
    scale = np.ones(n_series)
    mean = np.full((horizon, n_series), 100.0)
    covered, lows, highs = [], [], []
    for k in range(n_origins):
        origin = k * horizon
        lo, hi = cal.bounds(origin, mean, scale)
        actual = mean + errors[k]
        if k >= warmup:
            covered.append((actual >= lo) & (actual <= hi))
            lows.append(lo)
            highs.append(hi)
        cal.update(origin, actual - mean, scale)
    return np.mean(covered), np.mean(lows), np.mean(highs)


def test_coverage_is_close_to_nominal_on_exchangeable_errors() -> None:
    rng = np.random.default_rng(0)
    coverage, _, _ = _simulate(rng.normal(0, 5, size=(80, 7, 30)))
    assert abs(coverage - 0.8) < 0.03


def test_skewed_errors_give_an_asymmetric_interval() -> None:
    rng = np.random.default_rng(1)
    coverage, low, high = _simulate(rng.exponential(5, size=(60, 7, 30)) - 5)
    assert abs(coverage - 0.8) < 0.04
    # exponential errors: the 90 % quantile is about 1.45x as far from the mean as the 10 %
    assert (high - 100) > 1.3 * (100 - low)


def test_no_interval_until_enough_errors() -> None:
    cal = ConformalCalibrator(alpha=0.2, horizon=3, min_samples=10)
    mean = np.full((3, 2), 5.0)
    lo, _ = cal.bounds(0, mean, np.ones(2))
    assert np.isnan(lo).all()
    cal.update(0, np.zeros((3, 2)), np.ones(2))
    lo, _ = cal.bounds(10, mean, np.ones(2))
    assert np.isnan(lo).all()  # 2 errors per step < 10


def test_bounds_scale_per_series_and_clip_at_zero() -> None:
    cal = ConformalCalibrator(alpha=0.5, horizon=1, min_samples=4)
    for origin in range(10):
        cal.update(origin, np.array([[-3.0, 3.0]]), np.array([1.0, 1.0]))
    lo, hi = cal.bounds(20, np.array([[1.0, 1.0]]), np.array([1.0, 10.0]))
    assert lo[0, 0] == 0.0  # 1 - 3 clipped
    assert hi[0, 1] == 1.0 + 3.0 * 10.0
