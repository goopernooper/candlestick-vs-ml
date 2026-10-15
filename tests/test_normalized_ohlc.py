"""Tests for normalized OHLC: hand-checked values, link to encoding B, scale, no look-ahead."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import (NORMALIZED_OHLC, encoding_b_continuous, lagged_atr,  # noqa: E402
                      normalized_ohlc)


def random_ohlc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    idx = pd.bdate_range("2002-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def test_columns_and_warmup():
    n = normalized_ohlc(random_ohlc())
    assert list(n.columns) == NORMALIZED_OHLC
    assert n.iloc[:15].isna().all().all() and n.iloc[15:].notna().all().all()


def test_values_by_hand():
    df = random_ohlc()
    n, atr = normalized_ohlc(df), lagged_atr(df)
    day, prev = df.index[100], df.index[99]
    pc = df.at[prev, "Close"]
    for col, src in zip(NORMALIZED_OHLC, ["Open", "High", "Low", "Close"]):
        assert n.at[day, col] == pytest.approx((df.at[day, src] - pc) / atr[day])


def test_high_and_low_bracket_open_and_close():
    n = normalized_ohlc(random_ohlc()).dropna()
    assert (n["n_high"] >= n[["n_open", "n_close"]].max(axis=1)).all()
    assert (n["n_low"] <= n[["n_open", "n_close"]].min(axis=1)).all()


def test_same_information_as_encoding_b():
    """The ablation depends on this: both feature sets describe the same candle."""
    df = random_ohlc()
    n, b = normalized_ohlc(df).dropna(), encoding_b_continuous(df).dropna()
    np.testing.assert_allclose(b["b_gap"], n["n_open"])
    np.testing.assert_allclose(b["b_body"], n["n_close"] - n["n_open"])
    np.testing.assert_allclose(b["b_upper"], n["n_high"] - n[["n_open", "n_close"]].max(axis=1))
    np.testing.assert_allclose(b["b_lower"], n[["n_open", "n_close"]].min(axis=1) - n["n_low"])


def test_never_uses_price_levels():
    df = random_ohlc()
    pd.testing.assert_frame_equal(normalized_ohlc(df), normalized_ohlc(df * 37.5))


@pytest.mark.parametrize("cut", [50, 200, 350])
def test_normalized_ohlc_cannot_see_the_future(cut):
    df = random_ohlc()
    future = df.copy()
    future.iloc[cut:] *= np.random.default_rng(1).uniform(0.5, 2.0, size=future.iloc[cut:].shape)
    future["High"] = future.max(axis=1)
    future["Low"] = future[["Open", "High", "Low", "Close"]].min(axis=1)
    before, after = normalized_ohlc(df), normalized_ohlc(future)
    pd.testing.assert_frame_equal(before.iloc[:cut], after.iloc[:cut])
    assert not before.iloc[cut:].equals(after.iloc[cut:])
