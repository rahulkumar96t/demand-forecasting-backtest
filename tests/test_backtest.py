import json

import numpy as np
import pandas as pd
import pytest

from demand_forecast import metrics
from demand_forecast.backtest import (
    BacktestConfig,
    forecast_future,
    origin_indices,
    run_backtest,
)
from demand_forecast.leaderboard import by_horizon, interval_table, overall, per_series, point_terms
from demand_forecast.models import GlobalGBM, HoltWinters, MovingAverage, Naive, SeasonalNaive
from demand_forecast.report import write_report


def test_origins_are_evenly_spaced_and_end_one_horizon_before_the_data() -> None:
    cfg = BacktestConfig(horizon=28, step=14, n_origins=5, calibration_origins=2, min_train=100)
    origins = origin_indices(500, cfg)
    assert len(origins) == 7
    assert origins[-1] == 500 - 1 - 28
    assert set(np.diff(origins)) == {14}
    with pytest.raises(ValueError, match="min_train"):
        origin_indices(200, BacktestConfig(n_origins=10, min_train=150))


@pytest.mark.parametrize(
    "kwargs", [{"horizon": 0}, {"alpha": 1.0}, {"calibration_origins": -1}, {"step": 0}]
)
def test_config_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


@pytest.fixture(scope="module")
def tiny_result(request):
    tiny_panel = request.getfixturevalue("tiny_panel")
    models = request.getfixturevalue("fast_models")
    cfg = request.getfixturevalue("tiny_config")
    return run_backtest(tiny_panel, models, cfg)


@pytest.fixture(scope="module")
def fast_models():  # module-scoped copy so the end-to-end run happens once
    return {
        "naive": Naive(),
        "seasonal_naive": SeasonalNaive(),
        "moving_average": MovingAverage(window=14),
        "holt_winters": HoltWinters(fit_window=120),
        "gbm": GlobalGBM(max_iter=40, lookback=120, min_samples_leaf=10),
    }


@pytest.fixture(scope="module")
def tiny_config():
    return BacktestConfig(
        horizon=7, step=7, n_origins=3, calibration_origins=3, min_train=120,
        min_conformal_samples=8,
    )  # fmt: skip


def test_end_to_end_backtest_on_fixture(tiny_result, tiny_panel) -> None:
    points = tiny_result.points
    assert len(points) == 5 * 6 * 7 * 4  # models x origins x horizon x series
    scored_per_model = points[points["role"] == "evaluation"].groupby("model").size()
    assert (scored_per_model == 3 * 7 * 4).all()
    assert points["forecast"].notna().all()
    scored = points[points["role"] == "evaluation"]
    assert scored["lower"].notna().all()  # every scored origin has a conformal interval
    assert (scored["lower"] <= scored["upper"]).all()
    # the forecast dates are the days right after each origin
    first = points[(points["h"] == 1)]
    assert (first["date"] - first["origin"] == pd.Timedelta(days=1)).all()


def test_leaderboards(tiny_result) -> None:
    terms = point_terms(tiny_result.points, alpha=0.2)
    series = per_series(terms)
    board = overall(terms, series, tiny_result.runs)
    assert len(series) == 4 * 5
    assert set(board["model"]) == {
        "naive",
        "seasonal_naive",
        "moving_average",
        "holt_winters",
        "gbm",
    }
    assert board["mase"].is_monotonic_increasing
    assert board["wins"].sum() >= 4  # one winner per series, ties count twice
    assert ((board["coverage"] >= 0) & (board["coverage"] <= 1)).all()
    assert np.isfinite(board[["mase", "smape", "mae", "rmse", "msis"]].to_numpy()).all()
    # pooled numbers equal a direct computation with the metric functions
    gbm = terms[terms["model"] == "gbm"]
    row = board.set_index("model").loc["gbm"]
    assert row["mae"] == pytest.approx(metrics.mae(gbm["actual"], gbm["forecast"]))
    assert row["rmse"] == pytest.approx(metrics.rmse(gbm["actual"], gbm["forecast"]))
    per = series[series["model"] == "gbm"].set_index("series_id")["mase"]
    one = gbm[gbm["series_id"] == per.index[0]]
    assert per.iloc[0] == pytest.approx(metrics.mase(one["actual"], one["forecast"], one["scale"]))
    methods = interval_table(terms)
    assert {"conformal", "quantile"} <= set(methods["method"])
    assert set(methods[methods["method"] == "quantile"]["model"]) == {"gbm"}
    weekly = by_horizon(terms, width=7)
    assert list(weekly.index) == ["1-7"]


def test_report_files(tiny_result, tiny_panel, tmp_path) -> None:
    summary = write_report(tiny_result, tiny_panel, "tiny", tmp_path, wall_seconds=1.0)
    for name in ("leaderboard.md", "leaderboard.csv", "per_series.csv", "metrics.json"):
        assert (tmp_path / name).stat().st_size > 0
    png = tmp_path / "results.png"
    assert png.read_bytes()[:4] == b"\x89PNG"
    assert png.stat().st_size < 300_000
    loaded = json.loads((tmp_path / "metrics.json").read_text())
    assert (
        loaded["series"] == 4
        and loaded["leaderboard"][0]["model"] == summary["leaderboard"][0]["model"]
    )
    assert "| model |" in (tmp_path / "leaderboard.md").read_text()


def test_forecast_future_starts_after_the_data(tiny_panel) -> None:
    cfg = BacktestConfig(horizon=7, step=7, calibration_origins=4, min_conformal_samples=8)
    out = forecast_future(tiny_panel, MovingAverage(window=14), cfg)
    assert len(out) == 7 * 4
    assert out["date"].min() == tiny_panel.dates[tiny_panel.n_obs]
    assert out["lower"].notna().all() and (out["lower"] <= out["upper"]).all()
