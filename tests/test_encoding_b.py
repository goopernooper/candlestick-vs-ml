"""Tests for continuous encoding B: hand-checked values, scale invariance, no look-ahead."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import B_CONTINUOUS, encoding_b_continuous, lagged_atr  # noqa: E402


def random_ohlc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    idx = pd.bdate_range("2002-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def test_columns_and_warmup():
    enc = encoding_b_continuous(random_ohlc())
    assert list(enc.columns) == B_CONTINUOUS
    assert enc.iloc[:15].isna().all().all()      # ATR not ready yet
    assert enc.iloc[15:].notna().all().all()     # every later day has values


def test_values_by_hand():
    df = random_ohlc()
    enc, atr = encoding_b_continuous(df), lagged_atr(df)
    day, prev = df.index[100], df.index[99]
    o, h, l, c = df.loc[day, ["Open", "High", "Low", "Close"]]
    a = atr[day]
    assert enc.at[day, "b_body"] == pytest.approx((c - o) / a)
    assert enc.at[day, "b_upper"] == pytest.approx((h - max(o, c)) / a)
    assert enc.at[day, "b_lower"] == pytest.approx((min(o, c) - l) / a)
    assert enc.at[day, "b_gap"] == pytest.approx((o - df.at[prev, "Close"]) / a)


def test_signs_and_pieces_add_up():
    df = random_ohlc()
    enc = encoding_b_continuous(df).dropna()
    atr = lagged_atr(df).loc[enc.index]
    up = df.loc[enc.index, "Close"] > df.loc[enc.index, "Open"]
    assert (enc.loc[up, "b_body"] > 0).all() and (enc.loc[~up, "b_body"] <= 0).all()
    assert (enc[["b_upper", "b_lower"]] >= 0).all().all()
    # |body| + upper wick + lower wick = the whole high-low range
    rng = (df["High"] - df["Low"]).loc[enc.index] / atr
    np.testing.assert_allclose(enc["b_body"].abs() + enc["b_upper"] + enc["b_lower"], rng)


def test_same_values_at_any_price_level():
    """A stock at $5 and the same stock at $500 give identical features. This is what
    makes pooling stocks legitimate, and why split/dividend adjustment can't distort it."""
    df = random_ohlc()
    pd.testing.assert_frame_equal(encoding_b_continuous(df), encoding_b_continuous(df * 100))


def test_wick_clipped_for_rounding_error():
    df = random_ohlc()
    df.iloc[200, df.columns.get_loc("High")] = max(df.iloc[200]["Open"], df.iloc[200]["Close"]) * (1 - 1e-9)
    assert encoding_b_continuous(df).iloc[200]["b_upper"] == 0


@pytest.mark.parametrize("cut", [50, 200, 350])
def test_encoding_b_cannot_see_the_future(cut):
    df = random_ohlc()
    future = df.copy()
    future.iloc[cut:] *= np.random.default_rng(1).uniform(0.5, 2.0, size=future.iloc[cut:].shape)
    future["High"] = future.max(axis=1)
    future["Low"] = future[["Open", "High", "Low", "Close"]].min(axis=1)
    before, after = encoding_b_continuous(df), encoding_b_continuous(future)
    pd.testing.assert_frame_equal(before.iloc[:cut], after.iloc[:cut])
    assert not before.iloc[cut:].equals(after.iloc[cut:])  # the test can fail
