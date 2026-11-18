import numpy as np
import pandas as pd

from demand_forecast.panel import panel_from_long
from demand_forecast.synthetic import (
    POST_PROMO_DIP,
    SyntheticConfig,
    generate,
    holiday_calendar,
    promo_multiplier,
)

SMALL = SyntheticConfig(n_stores=2, n_items=8, n_days=730, future_days=14, seed=5)


def test_same_seed_same_data_and_different_seed_differs() -> None:
    a = generate(SMALL)
    pd.testing.assert_frame_equal(a, generate(SMALL))
    other = generate(SyntheticConfig(**{**SMALL.__dict__, "seed": 6}))
    assert not a["sales"].equals(other["sales"])


def test_schema_and_future_rows() -> None:
    df = generate(SMALL)
    assert list(df.columns) == [
        "date", "series_id", "store_id", "item_id", "category", "promo", "holiday", "sales",
    ]  # fmt: skip
    assert df["series_id"].nunique() == 16
    assert len(df) == 16 * (730 + 14)
    future = df["date"] >= pd.Timestamp("2022-01-01") + pd.Timedelta(days=730)
    assert df.loc[future, "sales"].isna().all()
    assert df.loc[~future, "sales"].notna().all() and (df.loc[~future, "sales"] >= 0).all()
    panel = panel_from_long(df)
    assert panel.n_obs == 730 and set(panel.known) == {"promo", "holiday"}


def test_planted_effects_are_visible() -> None:
    df = generate(SMALL).dropna(subset=["sales"])
    md = df["date"].dt.strftime("%m-%d")
    assert df.loc[md.isin(["01-01", "12-25"]), "sales"].sum() == 0  # stores closed
    normal = df[(df["holiday"] == 0) & (df["promo"] == 0)]
    by_dow = normal.groupby(normal["date"].dt.day_of_week)["sales"].mean()
    assert by_dow[5] > 1.2 * by_dow[1]  # Saturday peak
    uplift = df[df["promo"] == 1]["sales"].mean() / normal["sales"].mean()
    assert uplift > 1.3
    zero_share = df.groupby("series_id")["sales"].apply(lambda s: (s == 0).mean())
    assert (zero_share > 0.3).any() and (zero_share < 0.05).any()  # slow and fast movers
    gifts = normal[normal["category"] == "gifts"]
    month = gifts.groupby(gifts["date"].dt.month)["sales"].mean()
    assert month[12] > 1.5 * month[6]  # December peak


def test_promo_multiplier_uplift_then_dip() -> None:
    promo = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    mult = promo_multiplier(promo, 2.0)
    np.testing.assert_allclose(mult[:3], [1.0, 2.0, 2.0])
    np.testing.assert_allclose(mult[3:10], POST_PROMO_DIP)
    assert mult[10] == 1.0


def test_holiday_calendar() -> None:
    dates = pd.to_datetime(["2023-12-24", "2023-12-25", "2023-12-26", "2024-05-01"])
    flag, factor = holiday_calendar(pd.DatetimeIndex(dates))
    np.testing.assert_array_equal(flag, [0, 1, 1, 1])
    np.testing.assert_allclose(factor, [1.0, 0.0, 0.6, 0.6])
