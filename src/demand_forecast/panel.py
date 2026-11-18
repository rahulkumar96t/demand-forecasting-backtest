"""Daily multi-series panel and the truncated view a model is allowed to see.

The long CSV contract (one row per date and series):

* ``date``       ISO date, daily and gap-free for every series
* ``series_id``  series key
* ``sales``      non-negative target; empty on trailing *future* rows
* numeric columns  covariates known in advance (promotion plan, holiday calendar)
* text columns     static series attributes (store, item, category)

Trailing rows with an empty ``sales`` value carry known covariates for dates after the
last observation, so a model can forecast past the end of the data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATE = "date"
SERIES = "series_id"
TARGET = "sales"


@dataclass(frozen=True)
class ForecastContext:
    """Everything a model may use at one forecast origin.

    ``history`` stops at the origin, so future actuals are not just unused but absent.
    Known covariates and dates extend ``horizon`` steps past it.
    """

    history: np.ndarray  # (n_hist, n_series)
    dates: pd.DatetimeIndex  # n_hist + horizon dates
    known: Mapping[str, np.ndarray]  # each (<= n_hist + horizon, n_series)
    static: pd.DataFrame  # one row per series
    horizon: int
    season_length: int

    @property
    def n_hist(self) -> int:
        return int(self.history.shape[0])

    @property
    def n_series(self) -> int:
        return int(self.history.shape[1])

    @property
    def target_dates(self) -> pd.DatetimeIndex:
        return self.dates[self.n_hist :]

    def require_known(self, name: str) -> np.ndarray:
        """Return a covariate that covers the whole forecast window, or fail clearly."""
        values = self.known[name]
        if values.shape[0] < self.n_hist + self.horizon:
            raise ValueError(
                f"known covariate {name!r} ends before the forecast window; "
                "add future rows (empty sales) to the data"
            )
        return values


@dataclass(frozen=True)
class Panel:
    """Wide, validated panel: ``y[t, j]`` is the target of series ``j`` on ``dates[t]``."""

    dates: pd.DatetimeIndex
    series_ids: tuple[str, ...]
    y: np.ndarray  # (n_dates, n_series); NaN on future rows
    known: dict[str, np.ndarray]
    static: pd.DataFrame
    n_obs: int

    @property
    def n_series(self) -> int:
        return len(self.series_ids)

    @property
    def observed(self) -> np.ndarray:
        return self.y[: self.n_obs]

    def context(self, origin: int, horizon: int, season_length: int = 7) -> ForecastContext:
        """View at ``origin`` (index of the last observed day a model may use)."""
        if not 0 <= origin < self.n_obs:
            raise IndexError(f"origin {origin} outside observed range [0, {self.n_obs})")
        end = origin + 1 + horizon
        dates = self.dates[:end]
        if len(dates) < end:  # forecasting past the panel: extend the calendar
            extra = pd.date_range(self.dates[-1], periods=end - len(dates) + 1, freq="D")[1:]
            dates = dates.append(extra)
        return ForecastContext(
            history=self.y[: origin + 1].copy(),
            dates=dates,
            known={k: v[:end].copy() for k, v in self.known.items()},
            static=self.static,
            horizon=horizon,
            season_length=season_length,
        )

    def to_long(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                DATE: np.repeat(self.dates.values, self.n_series),
                SERIES: np.tile(np.array(self.series_ids, dtype=object), len(self.dates)),
                TARGET: self.y.ravel(),
            }
        )
        for name, values in self.known.items():
            frame[name] = values.ravel()
        if not self.static.empty:
            frame = frame.merge(self.static, left_on=SERIES, right_index=True, how="left")
        return frame.sort_values([SERIES, DATE], kind="stable").reset_index(drop=True)


def panel_from_long(frame: pd.DataFrame) -> Panel:
    """Validate a long frame (see module docstring) and pivot it to a :class:`Panel`."""
    missing = {DATE, SERIES, TARGET} - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = frame.copy()
    df[DATE] = pd.to_datetime(df[DATE])
    df[SERIES] = df[SERIES].astype(str)
    if df.duplicated([DATE, SERIES]).any():
        raise ValueError("duplicate (date, series_id) rows")

    extra = [c for c in df.columns if c not in (DATE, SERIES, TARGET)]
    known_cols = [c for c in extra if pd.api.types.is_numeric_dtype(df[c])]
    static_cols = [c for c in extra if c not in known_cols]

    dates = pd.date_range(df[DATE].min(), df[DATE].max(), freq="D")
    series_ids = tuple(sorted(df[SERIES].unique()))
    if len(df) != len(dates) * len(series_ids):
        raise ValueError("every series needs one row per calendar day (no gaps)")

    df = df.sort_values([DATE, SERIES], kind="stable")
    on_grid = df[DATE].to_numpy() == np.repeat(dates.values, len(series_ids))
    if not on_grid.all():
        raise ValueError("every series needs one row per calendar day (no gaps)")

    def wide(col: str) -> np.ndarray:
        return df[col].to_numpy(dtype=float).reshape(len(dates), len(series_ids))

    y = wide(TARGET)
    observed_rows = ~np.isnan(y).all(axis=1)
    n_obs = int(observed_rows.sum())
    if n_obs == 0 or not observed_rows[:n_obs].all():
        raise ValueError("sales may only be empty on trailing future rows")
    if np.isnan(y[:n_obs]).any():
        raise ValueError("sales has gaps inside the observed period; fill them first")
    if (y[:n_obs] < 0).any():
        raise ValueError("sales must be non-negative")

    known = {c: wide(c) for c in known_cols}
    for name, values in known.items():
        if np.isnan(values).any():
            raise ValueError(f"known covariate {name!r} has missing values")

    static = pd.DataFrame(index=pd.Index(series_ids, name=SERIES))
    for col in static_cols:
        per_series = df.groupby(SERIES)[col].nunique(dropna=False)
        if (per_series > 1).any():
            raise ValueError(f"static column {col!r} varies within a series")
        static[col] = df.groupby(SERIES)[col].first().reindex(series_ids).astype(str)
    return Panel(
        dates=dates,
        series_ids=series_ids,
        y=y,
        known=known,
        static=static,
        n_obs=n_obs,
    )


def read_panel(path: str) -> Panel:
    return panel_from_long(pd.read_csv(path))
