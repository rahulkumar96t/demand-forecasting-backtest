"""Explicit no-look-ahead checks: rewriting the future must not change anything computed
for an earlier origin."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from demand_forecast.backtest import origin_indices, run_backtest
from demand_forecast.features import make_features, training_pairs
from demand_forecast.intervals import ConformalCalibrator
from demand_forecast.models.base import Forecast
from demand_forecast.panel import ForecastContext


def _scramble_after(y: np.ndarray, cut: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = y.copy()
    out[cut + 1 :] = rng.uniform(0, 1e6, size=out[cut + 1 :].shape)
    return out


@pytest.mark.parametrize("cut", [60, 150, 249])
def test_features_ignore_everything_after_each_rows_origin(cut: int) -> None:
    rng = np.random.default_rng(1)
    y = rng.poisson(8.0, size=(400, 3)).astype(float)
    dates = pd.date_range("2023-01-01", periods=430, freq="D")
    promo = rng.integers(0, 2, size=(430, 3)).astype(float)
    origins, steps = training_pairs(cut, horizon=14, lookback=364, stride=3, min_history=28)
    origins = np.concatenate([origins, np.full(14, cut)])  # plus the prediction rows
    steps = np.concatenate([steps, np.arange(1, 15)])
    before, norm_before = make_features(y, origins, steps, dates, {"promo": promo}, None)
    after, norm_after = make_features(
        _scramble_after(y, cut), origins, steps, dates, {"promo": promo}, None
    )
    pd.testing.assert_frame_equal(before, after)
    np.testing.assert_array_equal(norm_before, norm_after)


def test_training_targets_are_observed_at_the_origin() -> None:
    for last in (40, 200, 731):
        for horizon in (1, 7, 28, 90):
            origins, steps = training_pairs(last, horizon, 364, 7, 28)
            assert (origins + steps <= last).all()


def test_no_model_can_read_actuals_after_its_origin(tiny_panel, fast_models) -> None:
    origin = 160
    future_changed = replace(tiny_panel, y=_scramble_after(tiny_panel.y, origin))
    for name, model in fast_models.items():
        a = model.forecast(tiny_panel.context(origin, 14))
        b = model.forecast(future_changed.context(origin, 14))
        np.testing.assert_array_equal(a.mean, b.mean, err_msg=name)
        if a.lower is not None:
            np.testing.assert_array_equal(a.lower, b.lower, err_msg=name)


def test_context_physically_drops_future_actuals(tiny_panel) -> None:
    ctx = tiny_panel.context(100, 14)
    assert ctx.history.shape[0] == 101
    np.testing.assert_array_equal(ctx.history, tiny_panel.y[:101])


def test_backtest_forecasts_and_intervals_do_not_depend_on_scored_actuals(
    tiny_panel, fast_models, tiny_config
) -> None:
    models = {k: fast_models[k] for k in ("seasonal_naive", "gbm")}
    last_origin = int(origin_indices(tiny_panel.n_obs, tiny_config)[-1])
    changed = replace(tiny_panel, y=_scramble_after(tiny_panel.y, last_origin))
    changed.y[tiny_panel.n_obs :] = np.nan
    a = run_backtest(tiny_panel, models, tiny_config).points
    b = run_backtest(changed, models, tiny_config).points
    cols = ["forecast", "lower", "upper", "lower_native", "upper_native"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    last = a["origin"] == a["origin"].max()
    assert not np.allclose(a.loc[last, "actual"], b.loc[last, "actual"])


def test_conformal_pool_only_contains_errors_observed_by_the_origin() -> None:
    cal = ConformalCalibrator(alpha=0.2, horizon=5, min_samples=1)
    scale = np.ones(2)
    cal.update(0, np.full((5, 2), 1.0), scale)
    cal.update(2, np.full((5, 2), 2.0), scale)
    # at origin 4: step 1 errors of origins 0 and 2 are known (0+1, 2+1 <= 4) ...
    assert sorted(cal.pool(4, 1).tolist()) == [1.0, 1.0, 2.0, 2.0]
    # ... step 3 only from origin 0 (2 + 3 > 4), step 5 from none yet
    assert cal.pool(4, 3).tolist() == [1.0, 1.0]
    assert cal.pool(4, 5).size == 0


class _Peeker:
    """A deliberately cheating model: it would need data beyond the origin."""

    name = "peeker"

    def forecast(self, ctx: ForecastContext) -> Forecast:
        future = ctx.history[ctx.n_hist : ctx.n_hist + ctx.horizon]
        return Forecast(mean=future)


def test_a_model_that_reaches_for_the_future_gets_nothing(tiny_panel, tiny_config) -> None:
    with pytest.raises(ValueError, match="shape"):
        run_backtest(tiny_panel, {"peeker": _Peeker()}, tiny_config)
