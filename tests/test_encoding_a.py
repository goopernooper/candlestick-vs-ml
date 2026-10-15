"""Tests for encoding A: shape, values, two hand-built patterns, and no look-ahead."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import PATTERNS, encoding_a, pattern_column  # noqa: E402


def random_ohlc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    idx = pd.bdate_range("2002-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def quiet_then(last_candles):
    """30 small, ordinary candles followed by the given (open, high, low, close) candles."""
    base = [(100.0 + 0.1 * i, 100.6 + 0.1 * i, 99.4 + 0.1 * i, 100.2 + 0.1 * i) for i in range(30)]
    rows = base + list(last_candles)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"],
                        index=pd.bdate_range("2002-01-02", periods=len(rows)))


def test_one_column_per_pattern_with_signed_values():
    enc = encoding_a(random_ohlc())
    assert list(enc.columns) == [pattern_column(p) for p in PATTERNS]
    assert len(enc.columns) == 61
    assert set(np.unique(enc.to_numpy())) <= {-1, 0, 1}


def test_raw_values_are_talib_scale():
    raw = encoding_a(random_ohlc(), raw=True)
    assert set(np.unique(raw.to_numpy())) <= {-200, -100, 0, 100, 200}


def test_bullish_engulfing_detected():
    # A down day, then an up day whose body completely covers it.
    df = quiet_then([(103.5, 103.7, 102.4, 102.6), (102.3, 104.4, 102.2, 104.2)])
    assert encoding_a(df)["a_engulfing"].iloc[-1] == 1


def test_bearish_engulfing_detected():
    df = quiet_then([(102.6, 103.7, 102.4, 103.5), (103.8, 103.9, 101.8, 102.0)])
    assert encoding_a(df)["a_engulfing"].iloc[-1] == -1


def test_doji_detected():
    # Open equals close, with wicks on both sides.
    df = quiet_then([(103.0, 104.0, 102.0, 103.0)])
    assert encoding_a(df)["a_doji"].iloc[-1] == 1


@pytest.mark.parametrize("cut", [50, 200, 350])
def test_encoding_a_cannot_see_the_future(cut):
    """Scramble every price from day `cut` onward; every pattern on days before
    `cut` must be unchanged, since each uses only its own day and earlier days."""
    df = random_ohlc()
    future = df.copy()
    future.iloc[cut:] *= np.random.default_rng(1).uniform(0.5, 2.0, size=future.iloc[cut:].shape)
    # Keep the scrambled candles valid (High highest, Low lowest).
    future["High"] = future[["Open", "High", "Low", "Close"]].max(axis=1)
    future["Low"] = future[["Open", "High", "Low", "Close"]].min(axis=1)
    before, after = encoding_a(df), encoding_a(future)
    pd.testing.assert_frame_equal(before.iloc[:cut], after.iloc[:cut])
    assert not before.iloc[cut:].equals(after.iloc[cut:])  # the test can fail
