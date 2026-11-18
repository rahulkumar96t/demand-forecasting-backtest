"""Global gradient-boosting model: one model for all series and all horizons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from demand_forecast.features import make_features, training_pairs
from demand_forecast.models.base import Forecast
from demand_forecast.panel import ForecastContext


@dataclass(frozen=True)
class GlobalGBM:
    """Direct multi-horizon ``HistGradientBoostingRegressor`` on origin-anchored features.

    At each origin it is retrained on (past origin, horizon) pairs from the last ``lookback``
    days whose targets are already observed. The point model uses a Poisson loss on demand
    divided by the series level; with ``alpha`` set, two quantile models give the central
    (1 - alpha) interval.
    """

    max_iter: int = 150
    learning_rate: float = 0.1
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    lookback: int = 364
    stride: int = 14
    min_history: int = 28
    alpha: float | None = 0.2
    random_state: int = 0
    threads: int | None = None  # OpenMP threads for fitting; None lets the library decide
    name: str = "gbm"

    def _regressor(self, loss: str, quantile: float | None = None) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss=loss,
            quantile=quantile,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            categorical_features="from_dtype",
            early_stopping=False,
            random_state=self.random_state,
        )

    def forecast(self, ctx: ForecastContext) -> Forecast:
        last = ctx.n_hist - 1
        known = {name: ctx.require_known(name) for name in ctx.known}
        origins, steps = training_pairs(
            last, ctx.horizon, self.lookback, self.stride, self.min_history
        )
        if origins.size == 0:
            raise ValueError("not enough history to build a training set")
        x_train, norm_train = make_features(
            ctx.history, origins, steps, ctx.dates, known, ctx.static, ctx.season_length
        )
        y_train = ctx.history[origins + steps].ravel() / norm_train

        horizon = np.arange(1, ctx.horizon + 1)
        x_pred, norm_pred = make_features(
            ctx.history,
            np.full(ctx.horizon, last),
            horizon,
            ctx.dates,
            known,
            ctx.static,
            ctx.season_length,
        )
        # columns that are empty for this history (e.g. the year lag early on) carry no signal
        usable = [c for c in x_train.columns if x_train[c].notna().any()]
        x_train, x_pred = x_train[usable], x_pred[usable]
        shape = (ctx.horizon, ctx.n_series)

        def fit_predict(loss: str, quantile: float | None = None) -> np.ndarray:
            with threadpool_limits(limits=self.threads, user_api="openmp"):
                model = self._regressor(loss, quantile).fit(x_train, y_train)
                pred = model.predict(x_pred)
            return np.clip(pred * norm_pred, 0.0, None).reshape(shape)

        mean = fit_predict("poisson")
        info = {"train_rows": float(len(x_train))}
        if self.alpha is None:
            return Forecast(mean=mean, info=info)
        lo = fit_predict("quantile", self.alpha / 2)
        hi = fit_predict("quantile", 1 - self.alpha / 2)
        return Forecast(mean=mean, lower=np.minimum(lo, hi), upper=np.maximum(lo, hi), info=info)
