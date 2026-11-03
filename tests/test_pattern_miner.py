"""Tests for the pattern miner. This part: counting contexts of consecutive symbols."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import build_table  # noqa: E402
from features import encoding_b_symbol_window, symbol_window_columns  # noqa: E402
from models.pattern import context_columns, context_table, count_contexts  # noqa: E402


def symbols_frame(seq, ups):
    """One stock's symbol sequence turned into the b_symbol window columns."""
    s = pd.Series(pd.array(seq, dtype="Int8"))
    X = pd.DataFrame({"b_symbol": s, "b_symbol_lag1": s.shift(1), "b_symbol_lag2": s.shift(2)})
    return X, np.asarray(ups)


def test_context_columns_are_oldest_first():
    assert context_columns(1) == ["b_symbol"]
    assert context_columns(3) == ["b_symbol_lag2", "b_symbol_lag1", "b_symbol"]
    assert symbol_window_columns() == ["b_symbol", "b_symbol_lag1", "b_symbol_lag2"]


def test_counts_by_hand():
    # day:      0  1  2  3  4  5
    # symbol:   4  8  4  8  4  8
    # next up:  1  0  1  1  0  1
    X, y = symbols_frame([4, 8, 4, 8, 4, 8], [1, 0, 1, 1, 0, 1])
    c = count_contexts(X, y, max_len=3)
    one, two, three = c[1], c[2], c[3]
    assert one.loc[4].tolist() == [3, 2] and one.loc[8].tolist() == [3, 2]       # [n, ups]
    # length 2 starts on day 1: (4,8) on days 1,3,5 -> ups 0,1,1 ; (8,4) on days 2,4 -> ups 1,0
    assert two.loc[(4, 8)].tolist() == [3, 2] and two.loc[(8, 4)].tolist() == [2, 1]
    # length 3 starts on day 2: (4,8,4) on days 2,4 -> ups 1,0 ; (8,4,8) on days 3,5 -> ups 1,1
    assert three.loc[(4, 8, 4)].tolist() == [2, 1] and three.loc[(8, 4, 8)].tolist() == [2, 2]


def test_order_matters():
    """(yesterday 4, today 8) and (yesterday 8, today 4) are different contexts."""
    X, y = symbols_frame([4, 8, 4, 8], [1, 1, 0, 0])
    two = count_contexts(X, y, max_len=2)[2]
    assert two.loc[(4, 8)].tolist() == [2, 1] and two.loc[(8, 4)].tolist() == [1, 0]


def test_totals_and_missing_symbols():
    rng = np.random.default_rng(0)
    X, y = symbols_frame(rng.integers(0, 20, 500), rng.integers(0, 2, 500))
    X.loc[100, "b_symbol_lag2"] = pd.NA            # missing only for length 3
    c = count_contexts(X, y)
    assert c[1]["n"].sum() == 500
    assert c[2]["n"].sum() == 499                  # the first day has no yesterday
    assert c[3]["n"].sum() == 497                  # first two days, and day 100
    assert (c[1]["ups"].sum(), c[2]["ups"].sum()) == (y.sum(), y[1:].sum())
    assert len(c[1]) <= 20 and len(c[2]) <= 400 and len(c[3]) <= 8000


def test_mismatched_lengths_are_an_error():
    X, y = symbols_frame([1, 2, 3], [1, 0, 1])
    with pytest.raises(ValueError):
        count_contexts(X, y[:2])


def test_context_table_is_readable():
    X, y = symbols_frame([4, 8, 4, 8, 4, 8], [1, 0, 1, 1, 0, 1])
    t = context_table(count_contexts(X, y))
    row = t[(t["length"] == 2) & (t["symbols"] == "4 8")].iloc[0]
    assert row["context"] == "- u l  >  0 u l" and row["n"] == 3 and row["up_rate"] == pytest.approx(2 / 3)


def test_symbol_window_never_mixes_stocks():
    cal = pd.bdate_range("2001-10-01", "2004-12-31")
    def loader(t):
        rng = np.random.default_rng({"AAA": 0, "BBB": 1}[t])
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(cal))))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, len(cal)))
        return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.004,
                             "Low": np.minimum(o, c) * 0.996, "Close": c}, index=cal)
    table = build_table(["AAA", "BBB"], ["encoding_b_symbol_window"], loader)
    for t in ["AAA", "BBB"]:
        own = encoding_b_symbol_window(loader(t)).loc[table.index.min():]
        mine = table[table["ticker"] == t][symbol_window_columns()]
        pd.testing.assert_frame_equal(mine, own.loc[mine.index], check_names=False)
