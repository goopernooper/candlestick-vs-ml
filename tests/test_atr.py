"""Tests for ATR: correctness against TA-Lib, the one-day lag, and no look-ahead."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import talib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import atr, lagged_atr, true_range  # noqa: E402


def random_ohlc(n=300, seed=0):
    """A fake but valid price series: High >= Open, Close >= Low, with gaps."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    idx = pd.bdate_range("2002-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def test_true_range_by_hand():
    df = pd.DataFrame({"High": [10, 12, 11], "Low": [9, 11, 8], "Close": [9.5, 11.5, 10]},
                      index=pd.bdate_range("2002-01-02", periods=3))
    tr = true_range(df)
    assert np.isnan(tr.iloc[0])          # no previous close
    assert tr.iloc[1] == pytest.approx(2.5)  # gap up: 12 - 9.5
    assert tr.iloc[2] == pytest.approx(3.5)  # gap down: |8 - 11.5|


def test_atr_matches_talib():
    df = random_ohlc()
    ours = atr(df).to_numpy()
    ref = talib.ATR(df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy(), 14)
    assert np.array_equal(np.isnan(ours), np.isnan(ref))
    np.testing.assert_allclose(ours[~np.isnan(ours)], ref[~np.isnan(ref)], rtol=1e-12)


def test_lag_is_exactly_one_day():
    df = random_ohlc()
    a, lag = atr(df), lagged_atr(df)
    pd.testing.assert_series_equal(lag.iloc[1:], a.iloc[:-1].set_axis(lag.index[1:]),
                                   check_names=False)
    assert lag.first_valid_index() == df.index[15]


@pytest.mark.parametrize("cut", [20, 100, 250])
def test_lagged_atr_cannot_see_the_future(cut):
    """Scramble every price from day `cut` onward; lagged ATR up to and including
    day `cut` must not change, because it only uses closes through day cut-1."""
    df = random_ohlc()
    future = df.copy()
    future.iloc[cut:] *= np.random.default_rng(1).uniform(0.5, 2.0, size=future.iloc[cut:].shape)
    before, after = lagged_atr(df), lagged_atr(future)
    pd.testing.assert_series_equal(before.iloc[:cut + 1], after.iloc[:cut + 1])
    assert not before.iloc[cut + 1:].equals(after.iloc[cut + 1:])  # the test can fail
