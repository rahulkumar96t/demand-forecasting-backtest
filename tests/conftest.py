from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from demand_forecast.backtest import BacktestConfig
from demand_forecast.models import GlobalGBM, HoltWinters, MovingAverage, Naive, SeasonalNaive
from demand_forecast.models.base import Forecaster
from demand_forecast.panel import Panel, panel_from_long, read_panel

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_panel.csv"


@pytest.fixture(scope="session", autouse=True)
def _single_openmp_thread():
    """Tiny fits are faster on one thread, and many OpenMP threads crawl on a busy machine."""
    with threadpool_limits(limits=1, user_api="openmp"):
        yield


@pytest.fixture(scope="session")
def tiny_panel() -> Panel:
    """4 series x 210 observed days (+14 future rows), written by the seeded generator."""
    return read_panel(str(FIXTURE))


@pytest.fixture
def fast_models() -> dict[str, Forecaster]:
    return {
        "naive": Naive(),
        "seasonal_naive": SeasonalNaive(),
        "moving_average": MovingAverage(window=14),
        "holt_winters": HoltWinters(fit_window=120),
        "gbm": GlobalGBM(max_iter=40, lookback=120, min_samples_leaf=10),
    }


@pytest.fixture
def tiny_config() -> BacktestConfig:
    return BacktestConfig(
        horizon=7,
        step=7,
        n_origins=3,
        calibration_origins=3,
        min_train=120,
        min_conformal_samples=8,
    )


@pytest.fixture
def panel_of():
    """Factory: ``panel_of(y, promo=...)`` builds a panel from a (days, series) array."""
    return make_panel


def make_panel(y: np.ndarray, start: str = "2024-01-01", **known: np.ndarray) -> Panel:
    """Panel from a (days, series) array; NaN rows at the end become future rows."""
    n, k = y.shape
    dates = pd.date_range(start, periods=n, freq="D")
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.values, k),
            "series_id": np.tile([f"s{j}" for j in range(k)], n),
            "sales": y.ravel(),
        }
    )
    for name, values in known.items():
        frame[name] = values.ravel()
    return panel_from_long(frame)
