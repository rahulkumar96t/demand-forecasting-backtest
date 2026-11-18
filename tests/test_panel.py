import numpy as np
import pandas as pd
import pytest

from demand_forecast.panel import panel_from_long


def test_round_trip_keeps_values_and_detects_future_rows(tiny_panel) -> None:
    assert tiny_panel.n_series == 4
    assert tiny_panel.n_obs == 210
    assert len(tiny_panel.dates) == 224
    assert np.isnan(tiny_panel.y[210:]).all()
    assert set(tiny_panel.known) == {"promo", "holiday"}
    assert list(tiny_panel.static.columns) == ["store_id", "item_id", "category"]
    back = panel_from_long(tiny_panel.to_long())
    np.testing.assert_array_equal(back.y, tiny_panel.y)
    np.testing.assert_array_equal(back.known["promo"], tiny_panel.known["promo"])


def test_context_truncates_history_but_not_known_covariates(tiny_panel) -> None:
    ctx = tiny_panel.context(origin=99, horizon=7)
    assert ctx.history.shape == (100, 4)
    assert ctx.known["promo"].shape == (107, 4)
    assert len(ctx.target_dates) == 7
    assert ctx.target_dates[0] == tiny_panel.dates[100]


def test_context_past_the_end_extends_dates(tiny_panel) -> None:
    ctx = tiny_panel.context(origin=209, horizon=21)  # only 14 future rows exist
    assert len(ctx.dates) == 231
    assert ctx.dates[-1] == tiny_panel.dates[209] + pd.Timedelta(days=21)
    with pytest.raises(ValueError, match="ends before the forecast window"):
        ctx.require_known("promo")


def test_context_rejects_future_origin(tiny_panel) -> None:
    with pytest.raises(IndexError):
        tiny_panel.context(origin=210, horizon=7)


def test_validation_errors(panel_of) -> None:
    y = np.arange(20.0).reshape(10, 2)
    good = panel_of(y).to_long()
    with pytest.raises(ValueError, match="no gaps"):
        panel_from_long(good.drop(index=3))
    negative = good.copy()
    negative.loc[0, "sales"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        panel_from_long(negative)
    hole = good.copy()
    hole.loc[(hole["series_id"] == "s0") & (hole["date"] == hole["date"].min()), "sales"] = np.nan
    with pytest.raises(ValueError, match="gaps inside"):
        panel_from_long(hole)
    with pytest.raises(ValueError, match="duplicate"):
        panel_from_long(pd.concat([good, good.iloc[:1]]))
    varying = good.assign(store=np.where(good.index % 2 == 0, "a", "b"))
    with pytest.raises(ValueError, match="varies within a series"):
        panel_from_long(varying)
