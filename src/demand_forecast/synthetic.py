"""Seeded synthetic retail demand: stores x items, daily.

Every effect below is planted on purpose, so a backtest can check that a model picks it up:

* **level**: item base demand (log-normal) times a store size factor; a share of items are slow
  movers whose mean is below about 1.5 units a day, which makes them intermittent;
* **trend**: a per-series growth rate (on average +3 % a year, sd 10 %);
* **weekly seasonality**: a day-of-week profile per category;
* **yearly seasonality**: a cosine with a category-specific peak (summer drinks, December gifts);
* **holidays**: a fixed calendar; stores close on 1 Jan and 25 Dec, trade at 60 % on the others;
* **promotions**: per-series episodes of 5-10 days every 4-10 weeks, with an item-specific
  uplift (x1.3 to x2.2) and a 7-day dip to 85 % afterwards (customers stocked up);
* **noise**: gamma-Poisson (negative binomial) counts, so variance grows with the mean.

The promotion plan and the holiday calendar are written as known-in-advance covariates and
also cover ``future_days`` after the last observed day (with empty sales).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand_forecast.panel import DATE, SERIES, TARGET

CATEGORIES: dict[str, dict[str, object]] = {
    # weekly profile Mon..Sun, yearly amplitude, day of year of the yearly peak
    "staples": {"weekly": [0.95, 0.9, 0.95, 1.0, 1.1, 1.3, 0.8], "amp": 0.05, "peak": 15},
    "fresh": {"weekly": [0.85, 0.85, 0.9, 1.0, 1.25, 1.45, 0.7], "amp": 0.10, "peak": 180},
    "drinks": {"weekly": [0.8, 0.8, 0.85, 0.95, 1.2, 1.5, 0.9], "amp": 0.35, "peak": 200},
    "gifts": {"weekly": [0.9, 0.9, 0.9, 1.0, 1.1, 1.4, 0.8], "amp": 0.60, "peak": 350},
}
CLOSED_DAYS = ("01-01", "12-25")
REDUCED_DAYS = ("05-01", "08-01", "12-26")
REDUCED_FACTOR = 0.6
POST_PROMO_DIP = 0.85
POST_PROMO_DAYS = 7


@dataclass(frozen=True)
class SyntheticConfig:
    n_stores: int = 5
    n_items: int = 20
    start: str = "2022-01-01"
    n_days: int = 1096
    future_days: int = 28
    intermittent_share: float = 0.25
    dispersion: float = 4.0  # gamma shape of the count noise; lower means noisier
    seed: int = 7


def holiday_calendar(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return (holiday flag, demand multiplier) for each date."""
    md = dates.strftime("%m-%d")
    closed = np.isin(md, CLOSED_DAYS)
    reduced = np.isin(md, REDUCED_DAYS)
    factor = np.where(closed, 0.0, np.where(reduced, REDUCED_FACTOR, 1.0))
    return (closed | reduced).astype(float), factor


def promotion_plan(
    n_days: int, rng: np.random.Generator, gap: tuple[int, int] = (28, 70)
) -> np.ndarray:
    """0/1 promotion flags for one series."""
    promo = np.zeros(n_days)
    t = int(rng.integers(0, gap[1]))
    while t < n_days:
        length = int(rng.integers(5, 11))
        promo[t : t + length] = 1.0
        t += length + int(rng.integers(*gap))
    return promo


def promo_multiplier(promo: np.ndarray, uplift: float) -> np.ndarray:
    """Uplift while on promotion, a dip for ``POST_PROMO_DAYS`` days after each episode."""
    mult = np.where(promo > 0, uplift, 1.0)
    ends = np.flatnonzero((promo[:-1] > 0) & (promo[1:] == 0)) + 1
    for end in ends:
        window = slice(end, min(end + POST_PROMO_DAYS, len(promo)))
        mult[window] = np.where(promo[window] > 0, mult[window], POST_PROMO_DIP)
    return mult


def generate(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Long frame: date, series_id, store_id, item_id, category, promo, holiday, sales."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    total_days = cfg.n_days + cfg.future_days
    dates = pd.date_range(cfg.start, periods=total_days, freq="D")
    t = np.arange(total_days)
    dow = dates.day_of_week.to_numpy()
    doy = dates.day_of_year.to_numpy()
    holiday, holiday_factor = holiday_calendar(dates)

    names = list(CATEGORIES)
    item_category = [names[i % len(names)] for i in rng.permutation(cfg.n_items)]
    slow = rng.random(cfg.n_items) < cfg.intermittent_share
    item_level = np.where(
        slow, rng.uniform(0.3, 1.5, cfg.n_items), rng.lognormal(np.log(12.0), 0.7, cfg.n_items)
    )
    item_uplift = rng.uniform(1.3, 2.2, cfg.n_items)
    store_size = rng.lognormal(0.0, 0.3, cfg.n_stores)

    frames = []
    for s in range(cfg.n_stores):
        for i in range(cfg.n_items):
            cat = CATEGORIES[item_category[i]]
            weekly = np.asarray(cat["weekly"], dtype=float)[dow]
            yearly = 1.0 + float(cat["amp"]) * np.cos(
                2 * np.pi * (doy - float(cat["peak"])) / 365.25
            )
            growth = rng.normal(0.03, 0.10)
            trend = np.exp(growth * t / 365.25)
            promo = promotion_plan(total_days, rng)
            uplift = item_uplift[i] * rng.uniform(0.9, 1.1)
            mean = (
                item_level[i]
                * store_size[s]
                * trend
                * weekly
                * yearly
                * holiday_factor
                * promo_multiplier(promo, uplift)
            )
            lam = mean * rng.gamma(cfg.dispersion, 1.0 / cfg.dispersion, total_days)
            sales = rng.poisson(lam).astype(float)
            sales[cfg.n_days :] = np.nan
            frames.append(
                pd.DataFrame(
                    {
                        DATE: dates,
                        SERIES: f"S{s + 1:02d}-I{i + 1:02d}",
                        "store_id": f"S{s + 1:02d}",
                        "item_id": f"I{i + 1:02d}",
                        "category": item_category[i],
                        "promo": promo,
                        "holiday": holiday,
                        TARGET: sales,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)
