import numpy as np
import pandas as pd
import pytest

from demand_forecast.features import make_features, training_pairs


def _one_series(values: np.ndarray, extra_days: int = 10):
    dates = pd.date_range("2024-01-01", periods=len(values) + extra_days, freq="D")
    return values.reshape(-1, 1).astype(float), dates


def test_hand_computed_features_for_one_row() -> None:
    y, dates = _one_series(np.arange(1, 61))  # y[t] = t + 1
    frame, norm = make_features(y, np.array([40]), np.array([3]), dates, {}, None)
    row = frame.iloc[0]
    level = np.mean(np.arange(14, 42))  # y[13..40]
    assert norm[0] == pytest.approx(level + 1)
    assert row["h"] == 3
    assert row["last"] == pytest.approx(41 / norm[0])
    assert row["mean_7"] == pytest.approx(np.mean(np.arange(35, 42)) / norm[0])
    # target index 43; most recent same weekday at or before the origin is index 36
    assert row["same_weekday"] == pytest.approx(37 / norm[0])
    assert row["same_weekday_mean_4"] == pytest.approx(np.mean([37, 30, 23, 16]) / norm[0])
    assert np.isnan(row["year_lag"])  # not enough history
    assert row["days_since_sale"] == 0
    assert row["zero_share_28"] == 0
    assert row["dow"] == dates[43].day_of_week
    assert row["day_of_year"] == dates[43].day_of_year


def test_year_lag_lands_on_same_weekday_last_year() -> None:
    y, dates = _one_series(np.arange(1, 401))
    frame, norm = make_features(y, np.array([380]), np.array([5]), dates, {}, None)
    target = 385
    assert dates[target - 364].day_of_week == dates[target].day_of_week
    assert frame.loc[0, "year_lag"] == pytest.approx(y[target - 364, 0] / norm[0])
    window = y[target - 364 - 3 : target - 364 + 4, 0]
    assert frame.loc[0, "year_week_mean"] == pytest.approx(window.mean() / norm[0])


def test_days_since_sale_and_zero_share() -> None:
    values = np.ones(40)
    values[37:] = 0.0  # last sale at index 36
    y, dates = _one_series(values)
    frame, _ = make_features(y, np.array([39]), np.array([1]), dates, {}, None)
    assert frame.loc[0, "days_since_sale"] == 3
    assert frame.loc[0, "zero_share_28"] == pytest.approx(3 / 28)


def test_known_covariates_on_target_date_and_week_before() -> None:
    y, dates = _one_series(np.full(50, 5.0))
    promo = np.zeros((60, 1))
    promo[[40, 41, 43]] = 1.0
    frame, _ = make_features(y, np.array([39]), np.array([4]), dates, {"promo": promo}, None)
    assert frame.loc[0, "promo"] == 1.0  # target index 43
    assert frame.loc[0, "promo_prev_7"] == pytest.approx(2 / 7)  # indices 36..42


def test_static_columns_become_categoricals_and_rows_are_pair_major() -> None:
    y = np.column_stack([np.full(40, 1.0), np.full(40, 9.0)])
    dates = pd.date_range("2024-01-01", periods=45, freq="D")
    static = pd.DataFrame({"store_id": ["a", "b"]}, index=["s0", "s1"])
    frame, norm = make_features(y, np.array([30, 31]), np.array([1, 2]), dates, {}, static)
    assert isinstance(frame["store_id"].dtype, pd.CategoricalDtype)
    assert frame["store_id"].tolist() == ["a", "b", "a", "b"]
    np.testing.assert_allclose(norm, [2.0, 10.0, 2.0, 10.0])
    assert frame["h"].tolist() == [1, 1, 2, 2]


def test_training_pairs_use_fresh_targets_and_never_pass_last_observation() -> None:
    origins, steps = training_pairs(last_obs=100, horizon=7, lookback=30, stride=5, min_history=28)
    assert (origins + steps <= 100).all()
    assert (99, 1) in set(zip(origins.tolist(), steps.tolist(), strict=True))
    assert set(origins.tolist()) == {99, 94, 89, 84, 79, 74}
    assert steps[origins == 94].tolist() == [1, 2, 3, 4, 5, 6]
    assert steps[origins == 74].max() == 7


def test_training_pairs_respect_min_history() -> None:
    origins, _ = training_pairs(last_obs=40, horizon=7, lookback=364, stride=1, min_history=28)
    assert origins.min() == 27


def test_invalid_inputs_raise() -> None:
    y, dates = _one_series(np.ones(30), extra_days=2)
    with pytest.raises(ValueError, match="index into history"):
        make_features(y, np.array([30]), np.array([1]), dates, {}, None)
    with pytest.raises(ValueError, match="cover the target dates"):
        make_features(y, np.array([29]), np.array([5]), dates, {}, None)
    with pytest.raises(ValueError, match="does not cover"):
        make_features(y, np.array([29]), np.array([1]), dates, {"promo": np.zeros((30, 1))}, None)
