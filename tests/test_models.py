import numpy as np
import pytest

from demand_forecast.models import (
    GlobalGBM,
    HoltWinters,
    MovingAverage,
    Naive,
    SeasonalNaive,
    build_models,
    ets,
)
from demand_forecast.models.baselines import seasonal_naive_mean


def test_naive_and_moving_average(panel_of) -> None:
    y = np.column_stack([np.arange(1.0, 31.0), np.full(30, 4.0)])
    ctx = panel_of(y).context(origin=29, horizon=3)
    np.testing.assert_array_equal(Naive().forecast(ctx).mean, [[30, 4]] * 3)
    ma = MovingAverage(window=4).forecast(ctx).mean
    np.testing.assert_allclose(ma, [[28.5, 4.0]] * 3)


def test_seasonal_naive_repeats_last_season() -> None:
    history = np.arange(1.0, 15.0)[:, None]  # last week: 8..14
    np.testing.assert_array_equal(
        seasonal_naive_mean(history, 10, 7)[:, 0], [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]
    )
    with pytest.raises(ValueError):
        seasonal_naive_mean(history[:3], 5, 7)


def test_seasonal_naive_model_uses_context_season(panel_of) -> None:
    y = np.tile([1.0, 5.0, 3.0], 10)[:, None]
    ctx = panel_of(y).context(origin=29, horizon=4, season_length=3)
    np.testing.assert_array_equal(SeasonalNaive().forecast(ctx).mean[:, 0], [1, 5, 3, 1])


def test_holt_winters_recovers_clean_trend_and_weekly_pattern(panel_of) -> None:
    t = np.arange(200.0)
    pattern = np.array([0, 2, 4, 6, 4, 2, 0], dtype=float)
    y = (20 + 0.05 * t + pattern[t.astype(int) % 7])[:, None]
    panel = panel_of(y)
    ctx = panel.context(origin=185, horizon=14)
    fc = HoltWinters(fit_window=180).forecast(ctx)
    assert np.max(np.abs(fc.mean[:, 0] - y[186:200, 0])) < 0.5
    assert fc.info["fallbacks"] == 0


def test_holt_winters_falls_back_to_seasonal_naive(monkeypatch, panel_of) -> None:
    def broken(*args, **kwargs):
        raise ValueError("no fit")

    monkeypatch.setattr(ets, "holt_winters", broken)
    y = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 6)[:, None]
    ctx = panel_of(y).context(origin=41, horizon=7)
    fc = HoltWinters().forecast(ctx)
    assert fc.info["fallbacks"] == 1
    np.testing.assert_array_equal(fc.mean[:, 0], [1, 2, 3, 4, 5, 6, 7])


def _promo_panel(panel_of, days: int = 260, n: int = 4, seed: int = 3):
    rng = np.random.default_rng(seed)
    promo = np.zeros((days, n))
    for j in range(n):
        for start in range(rng.integers(0, 20), days, 30):
            promo[start : start + 6, j] = 1.0
    weekly = np.array([1.0, 1.0, 1.0, 1.1, 1.3, 1.6, 0.7])[np.arange(days) % 7]
    mean = 20.0 * weekly[:, None] * np.where(promo > 0, 2.5, 1.0)
    y = rng.poisson(mean).astype(float)
    return panel_of(y, promo=promo)


def test_gbm_uses_the_known_promotion_plan(panel_of) -> None:
    panel = _promo_panel(panel_of)
    origin, horizon = 230, 28
    ctx = panel.context(origin, horizon)
    actual = panel.y[origin + 1 : origin + 1 + horizon]
    gbm = GlobalGBM(max_iter=60, lookback=200, min_samples_leaf=10).forecast(ctx)
    snaive = SeasonalNaive().forecast(ctx)
    gbm_mae = np.mean(np.abs(gbm.mean - actual))
    snaive_mae = np.mean(np.abs(snaive.mean - actual))
    assert gbm_mae < 0.7 * snaive_mae
    on_promo = ctx.known["promo"][origin + 1 :] > 0
    assert gbm.mean[on_promo].mean() > 1.8 * gbm.mean[~on_promo].mean()


def test_gbm_output_shapes_and_quantile_order(tiny_panel) -> None:
    ctx = tiny_panel.context(180, 14)
    fc = GlobalGBM(max_iter=30, lookback=150, min_samples_leaf=10).forecast(ctx)
    assert fc.mean.shape == fc.lower.shape == fc.upper.shape == (14, 4)
    assert (fc.mean >= 0).all()
    assert (fc.lower <= fc.upper).all()
    assert fc.info["train_rows"] > 0
    assert GlobalGBM(max_iter=10, alpha=None).forecast(ctx).lower is None


def test_gbm_needs_covariates_for_the_whole_window(tiny_panel) -> None:
    ctx = tiny_panel.context(209, 21)  # the fixture has only 14 future rows
    with pytest.raises(ValueError, match="ends before the forecast window"):
        GlobalGBM(max_iter=5).forecast(ctx)


def test_registry() -> None:
    models = build_models(["naive", "gbm"], alpha=0.1, gbm_max_iter=7)
    assert list(models) == ["naive", "gbm"]
    assert models["gbm"].alpha == 0.1 and models["gbm"].max_iter == 7
    with pytest.raises(ValueError, match="unknown models"):
        build_models(["prophet"])
