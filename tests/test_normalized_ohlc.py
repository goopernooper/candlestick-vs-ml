"""Tests for normalized OHLC: hand-checked values, link to encoding B, scale, no look-ahead."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import (NORMALIZED_OHLC, encoding_b_continuous, encoding_b_window,  # noqa: E402
                      lagged_atr, normalized_ohlc, normalized_ohlc_window, ohlc_window_columns,
                      window_columns)


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


# ------------------------------------------------- the 3-day window (D-024)
def test_window_columns_and_lags():
    df = random_ohlc()
    w, today = normalized_ohlc_window(df), normalized_ohlc(df)
    assert list(w.columns) == ohlc_window_columns() and len(ohlc_window_columns()) == 12
    for k in [1, 2]:
        for c in NORMALIZED_OHLC:
            pd.testing.assert_series_equal(w[f"{c}_lag{k}"], today[c].shift(k), check_names=False)
    assert w.iloc[:17].isna().any(axis=1).all() and w.iloc[17:].notna().all().all()


def test_window_has_the_same_information_as_the_b_window():
    """The grid's "ohlc" and "b" models see the same 3 days, described differently:
    each window can be computed exactly from the other."""
    df = random_ohlc()
    n, b = normalized_ohlc_window(df).dropna(), encoding_b_window(df, days=3).dropna()
    assert list(b.columns) == window_columns(3)
    for suffix in ["", "_lag1", "_lag2"]:
        o, c = n[f"n_open{suffix}"], n[f"n_close{suffix}"]
        hi, lo = n[f"n_high{suffix}"], n[f"n_low{suffix}"]
        # ohlc -> b
        np.testing.assert_allclose(b[f"b_gap{suffix}"], o)
        np.testing.assert_allclose(b[f"b_body{suffix}"], c - o)
        np.testing.assert_allclose(b[f"b_upper{suffix}"], hi - np.maximum(o, c))
        np.testing.assert_allclose(b[f"b_lower{suffix}"], np.minimum(o, c) - lo)
        # b -> ohlc
        g, body = b[f"b_gap{suffix}"], b[f"b_body{suffix}"]
        np.testing.assert_allclose(o, g)
        np.testing.assert_allclose(c, g + body)
        np.testing.assert_allclose(hi, np.maximum(g, g + body) + b[f"b_upper{suffix}"])
        np.testing.assert_allclose(lo, np.minimum(g, g + body) - b[f"b_lower{suffix}"])


def test_table_with_both_ohlc_sets_has_each_column_once():
    """Day t's normalized OHLC is in both feature sets; the table keeps one copy."""
    from dataset import build_table, merge_parts
    table = build_table(["AAA", "BBB"], ["normalized_ohlc", "normalized_ohlc_window"],
                        lambda t: random_ohlc(n=1000, seed=len(t)))
    assert not table.columns.duplicated().any()
    assert list(table[["n_close"]].columns) == ["n_close"]
    assert set(ohlc_window_columns()) <= set(table.columns)
    a = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        merge_parts([a, pd.DataFrame({"x": [1.0, 3.0]})])
