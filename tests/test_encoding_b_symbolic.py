"""Tests for symbolic encoding B: bin edges, symbol numbering, config, no look-ahead."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import (N_SYMBOLS, encoding_b_continuous, encoding_b_symbolic,  # noqa: E402
                      load_config, symbol_label)

EDGES, WICK = [-0.6, -0.1, 0.1, 0.6], 0.25


def random_ohlc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    idx = pd.bdate_range("2002-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def test_config_has_the_planned_edges():
    cfg = load_config()["encoding_b"]
    assert cfg["body_edges"] == EDGES
    assert cfg["wick_edge"] == WICK


def test_symbols_range_and_warmup():
    sym = encoding_b_symbolic(random_ohlc())
    assert sym["b_symbol"].iloc[:15].isna().all()
    s = sym["b_symbol"].dropna()
    assert s.between(0, N_SYMBOLS - 1).all()


def test_levels_match_continuous_values():
    df = random_ohlc()
    cont, sym = encoding_b_continuous(df).dropna(), encoding_b_symbolic(df).dropna()
    body = cont["b_body"]
    expected = np.select([body < -0.6, body < -0.1, body < 0.1, body < 0.6], [0, 1, 2, 3], 4)
    np.testing.assert_array_equal(sym["b_body_level"].to_numpy(int), expected)
    np.testing.assert_array_equal(sym["b_upper_long"].to_numpy(int), (cont["b_upper"] >= WICK).to_numpy(int))
    np.testing.assert_array_equal(sym["b_lower_long"].to_numpy(int), (cont["b_lower"] >= WICK).to_numpy(int))
    np.testing.assert_array_equal(
        sym["b_symbol"].to_numpy(int),
        (sym["b_body_level"] * 4 + sym["b_upper_long"] * 2 + sym["b_lower_long"]).to_numpy(int))


def test_values_on_an_edge_go_to_the_level_above():
    assert np.digitize([-0.6, -0.1, 0.1, 0.6], EDGES).tolist() == [1, 2, 3, 4]


def test_labels_are_unique_and_readable():
    labels = [symbol_label(s) for s in range(N_SYMBOLS)]
    assert len(set(labels)) == N_SYMBOLS
    assert labels[0] == "-- u l" and labels[19] == "++ U L" and labels[14] == "+ U l"


def test_passing_edges_overrides_config():
    df = random_ohlc()
    wide = encoding_b_symbolic(df, body_edges=[-9, -8, 8, 9], wick_edge=99)
    assert (wide["b_symbol"].dropna() == 8).all()   # every day "doji-like, short wicks"


@pytest.mark.parametrize("cut", [50, 200, 350])
def test_symbols_cannot_see_the_future(cut):
    df = random_ohlc()
    future = df.copy()
    future.iloc[cut:] *= np.random.default_rng(1).uniform(0.5, 2.0, size=future.iloc[cut:].shape)
    future["High"] = future.max(axis=1)
    future["Low"] = future[["Open", "High", "Low", "Close"]].min(axis=1)
    before, after = encoding_b_symbolic(df), encoding_b_symbolic(future)
    pd.testing.assert_frame_equal(before.iloc[:cut], after.iloc[:cut])
    assert not before.iloc[cut:].equals(after.iloc[cut:])
