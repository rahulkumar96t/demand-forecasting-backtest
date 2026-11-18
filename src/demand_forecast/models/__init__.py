"""Model registry."""

from __future__ import annotations

from demand_forecast.models.base import Forecast, Forecaster
from demand_forecast.models.baselines import MovingAverage, Naive, SeasonalNaive
from demand_forecast.models.ets import HoltWinters
from demand_forecast.models.gbm import GlobalGBM

MODEL_NAMES = ("naive", "seasonal_naive", "moving_average", "holt_winters", "gbm")


def build_models(
    names: list[str] | tuple[str, ...] = MODEL_NAMES,
    alpha: float = 0.2,
    n_jobs: int = 1,
    gbm_max_iter: int = 150,
    ma_window: int = 28,
    gbm_threads: int | None = None,
) -> dict[str, Forecaster]:
    """Instantiate models by name, in the given order."""
    factories = {
        "naive": lambda: Naive(),
        "seasonal_naive": lambda: SeasonalNaive(),
        "moving_average": lambda: MovingAverage(window=ma_window),
        "holt_winters": lambda: HoltWinters(n_jobs=n_jobs),
        "gbm": lambda: GlobalGBM(alpha=alpha, max_iter=gbm_max_iter, threads=gbm_threads),
    }
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise ValueError(f"unknown models {unknown}; choose from {list(factories)}")
    return {n: factories[n]() for n in names}


__all__ = [
    "MODEL_NAMES",
    "Forecast",
    "Forecaster",
    "GlobalGBM",
    "HoltWinters",
    "MovingAverage",
    "Naive",
    "SeasonalNaive",
    "build_models",
]
