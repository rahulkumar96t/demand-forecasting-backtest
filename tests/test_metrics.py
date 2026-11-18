import numpy as np
import pytest

from demand_forecast import metrics as m


def test_point_metrics_on_hand_computed_example() -> None:
    actual = [1.0, 2.0, 3.0, 4.0]
    forecast = [2.0, 2.0, 1.0, 8.0]  # errors +1, 0, -2, +4
    assert m.mae(actual, forecast) == pytest.approx(7 / 4)
    assert m.rmse(actual, forecast) == pytest.approx(np.sqrt(21 / 4))
    assert m.bias(actual, forecast) == pytest.approx(3 / 4)
    assert m.bias_pct(actual, forecast) == pytest.approx(30.0)


def test_bias_sign_means_over_forecast() -> None:
    assert m.bias([10.0], [12.0]) > 0
    assert m.bias([10.0], [8.0]) < 0
    assert np.isnan(m.bias_pct([0.0, 0.0], [1.0, 1.0]))


def test_smape_handles_zeros() -> None:
    # both zero -> 0; actual zero, forecast positive -> 200
    assert m.smape([0.0], [0.0]) == 0.0
    assert m.smape([0.0], [5.0]) == pytest.approx(200.0)
    assert m.smape([0.0, 10.0], [0.0, 30.0]) == pytest.approx(50.0)  # (0 + 2*20/40) / 2
    assert m.smape([4.0], [4.0]) == 0.0


def test_seasonal_scale_hand_example_and_fallbacks() -> None:
    y = np.array([1.0, 3.0, 2.0, 5.0])
    assert m.seasonal_scale(y, 1)[0] == pytest.approx((2 + 1 + 3) / 3)
    periodic = np.tile([1.0, 2.0, 3.0], 5)  # perfectly seasonal: seasonal diff is zero
    assert m.seasonal_scale(periodic, 3)[0] == pytest.approx(np.mean(np.abs(np.diff(periodic))))
    assert m.seasonal_scale(np.zeros(20), 7)[0] == 1.0  # constant series
    short = np.array([2.0, 4.0, 3.0])  # shorter than a season -> lag-1 difference
    assert m.seasonal_scale(short, 7)[0] == pytest.approx(1.5)


def test_seasonal_scale_is_per_column() -> None:
    y = np.column_stack([np.arange(10.0), 2 * np.arange(10.0)])
    np.testing.assert_allclose(m.seasonal_scale(y, 2), [2.0, 4.0])


def test_in_sample_seasonal_naive_has_mase_one() -> None:
    rng = np.random.default_rng(0)
    y = rng.poisson(5.0, 200).astype(float)
    scale = m.seasonal_scale(y, 7)[0]
    assert m.mase(y[7:], y[:-7], scale) == pytest.approx(1.0)


def test_mase_divides_by_scale() -> None:
    assert m.mase([1.0, 2.0], [2.0, 4.0], 3.0) == pytest.approx(0.5)
    assert m.mase([1.0, 2.0], [2.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_coverage_and_interval_score() -> None:
    actual = np.array([3.0, 1.0, 6.0, 2.0])
    lower = np.full(4, 2.0)
    upper = np.full(4, 4.0)
    assert m.coverage(actual, lower, upper) == pytest.approx(0.5)  # bounds are inclusive
    # alpha 0.2: width 2, plus 2/0.2 = 10 times the miss distance
    np.testing.assert_allclose(m.interval_score(actual, lower, upper, 0.2), [2.0, 12.0, 22.0, 2.0])
    assert m.msis(actual, lower, upper, 2.0, 0.2) == pytest.approx((2 + 12 + 22 + 2) / 4 / 2)


def test_interval_score_rewards_narrow_correct_intervals() -> None:
    actual = np.array([5.0])
    narrow = m.interval_score(actual, np.array([4.0]), np.array([6.0]), 0.2)
    wide = m.interval_score(actual, np.array([0.0]), np.array([10.0]), 0.2)
    assert narrow[0] < wide[0]


def test_shape_mismatch_and_bad_alpha_raise() -> None:
    with pytest.raises(ValueError):
        m.mae([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        m.interval_score([1.0], [0.0], [2.0], alpha=1.5)
