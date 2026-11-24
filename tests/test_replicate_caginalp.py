"""Tests for the Caginalp & Laurent replication (replicate_caginalp.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replicate_caginalp import (DEFINITIONS, PATTERNS, build, evaluate, nw_t,  # noqa: E402
                                paper_shapes, stock_frame, trends)


def prices(n=300, seed=0, start="2002-01-02"):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.005, n))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.005,
                         "Low": np.minimum(o, c) * 0.995, "Close": c}, index=idx)


def candles(opens, closes):
    idx = pd.bdate_range("2002-01-02", periods=len(opens))
    o, c = pd.Series(opens, index=idx, dtype=float), pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.5, "Low": np.minimum(o, c) - 0.5, "Close": c})


FALL = (list(range(120, 100, -2)), list(range(119, 99, -2)))       # ten falling days: 120->119, ..., 102->101
SHAPES = {   # three days of (open, close) that complete each bullish pattern after FALL
    "three_white_soldiers": [(100, 102), (101, 104), (103, 106)],
    "three_inside_up": [(101, 97), (98, 100), (100, 103)],
    "three_outside_up": [(101, 99), (98, 102), (102, 105)],
    "morning_star": [(101, 97), (95, 96), (97, 100)],
}


def test_the_eight_patterns():
    assert len(PATTERNS) == 8 and sum(d > 0 for d, _ in PATTERNS.values()) == 4
    assert DEFINITIONS == ["paper", "talib", "talib_notrend"]


def test_trend_definition():
    """Six steps of the 3-day average, at most one against the trend."""
    c = pd.Series(np.arange(130.0, 100.0, -1))
    down, up = trends(c)
    assert down.iloc[8:].all() and not down.iloc[:8].any() and not up.any()   # needs 2 + 6 earlier days
    one_bad = c.copy()
    one_bad.iloc[20] += 4                         # the average rises once, then falls on
    assert trends(one_bad)[0].iloc[20]
    two_bad = one_bad.copy()
    two_bad.iloc[22] += 12
    assert not trends(two_bad)[0].iloc[23]
    assert trends(-c)[1].iloc[8:].all()


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_each_bullish_pattern_fires_on_its_last_day(name):
    o = FALL[0] + [a for a, _ in SHAPES[name]] + [105, 106]
    c = FALL[1] + [b for _, b in SHAPES[name]] + [106, 107]
    f = stock_frame(candles(o, c))
    last = len(FALL[0]) + 2
    fired = f[[f"paper__{n}" for n in PATTERNS]]
    assert fired.iloc[last][f"paper__{name}"]
    assert fired[f"paper__{name}"].sum() == 1                    # once, on its last day
    assert fired.iloc[last].sum() == 1                           # and no other pattern that day


def test_no_signal_without_the_downtrend():
    rise = (list(range(80, 100, 2)), list(range(81, 101, 2)))
    o = rise[0] + [100, 101, 103]
    c = rise[1] + [102, 104, 106]
    df = candles(o, c)
    assert paper_shapes(df["Open"], df["Close"])["three_white_soldiers"].iloc[-1]   # the shape is there
    assert not stock_frame(df)["paper__three_white_soldiers"].any()                 # but not the trend


def test_bearish_patterns_mirror_the_bullish_ones():
    df = prices(n=3000, seed=3)
    flipped = pd.DataFrame({"Open": 300 - df["Open"], "High": 300 - df["Low"],
                            "Low": 300 - df["High"], "Close": 300 - df["Close"]})
    a, b = stock_frame(df), stock_frame(flipped)
    pairs = [("three_white_soldiers", "three_black_crows"), ("three_inside_up", "three_inside_down"),
             ("three_outside_up", "three_outside_down"), ("morning_star", "evening_star")]
    for up, down in pairs:
        assert a[f"paper__{up}"].sum() > 0
        pd.testing.assert_series_equal(a[f"paper__{up}"], b[f"paper__{down}"], check_names=False)
        pd.testing.assert_series_equal(a[f"paper__{down}"], b[f"paper__{up}"], check_names=False)
    pd.testing.assert_series_equal(a["base_down"], b["base_up"], check_names=False)


def test_outcomes_and_exit_dates():
    df = prices()
    f, c, d, t = stock_frame(df), df["Close"], df.index, 100
    assert f["cl"].iloc[t] == pytest.approx(c.iloc[t + 1:t + 4].mean() / c.iloc[t] - 1)   # their r_b
    assert f["next_day"].iloc[t] == pytest.approx(c.iloc[t + 1] / c.iloc[t] - 1)
    assert f["exit_cl"].iloc[t] == d[t + 3] and f["exit_next_day"].iloc[t] == d[t + 1]
    assert np.isnan(f["cl"].iloc[-1])


def test_signals_only_look_backward():
    df = prices()
    later = df.copy()
    later.iloc[150:] *= 1.7
    a, b = stock_frame(df), stock_frame(later)
    cols = [c for c in a.columns if "__" in c or c.startswith("base_")]
    pd.testing.assert_frame_equal(a[cols].iloc[:150], b[cols].iloc[:150])


def test_talib_with_trend_is_a_subset_of_talib():
    f = stock_frame(prices(n=3000, seed=5))
    for name in PATTERNS:
        assert not (f[f"talib__{name}"] & ~f[f"talib_notrend__{name}"]).any()


def test_nw_t():
    x = pd.Series(np.r_[np.ones(50) * 0.01, -np.ones(50) * 0.005])
    e = x - x.mean()
    assert nw_t(x, 0) == pytest.approx(x.mean() / np.sqrt((e @ e / len(x)) / len(x)))
    assert np.isnan(nw_t(pd.Series([1.0, 2.0]), 0))


def test_build_stops_before_the_holdout():
    frame = build(["AAA", "BBB"], lambda t: prices(n=5600, seed=len(t), start="2001-10-01"))
    assert pd.Timestamp("2002-01-02") <= frame.index.min() and frame.index.max() < pd.Timestamp("2023-01-01")
    for outcome in ["cl", "next_day"]:
        used = frame[frame[outcome].notna()]
        assert (used[f"exit_{outcome}"] < pd.Timestamp("2023-01-01")).all()
    assert frame["cl"].isna().sum() > frame["next_day"].isna().sum()        # the last 3 days vs the last 1


def test_evaluate_reproduces_the_papers_z_and_finds_a_planted_edge():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2003-01-01", "2016-12-31")
    rows = []
    for t in range(10):
        r = rng.normal(0, 0.01, len(days))
        star = rng.random(len(days)) < 0.03
        r = np.where(star, np.abs(r) + 0.004, r)               # after a morning star: always up
        frame = pd.DataFrame(False, index=days,
                             columns=[f"{d}__{n}" for d in DEFINITIONS for n in PATTERNS])
        frame["paper__morning_star"] = star
        frame["paper__evening_star"] = rng.random(len(days)) < 0.03          # no edge
        frame["base_down"] = frame["base_up"] = True
        frame["cl"] = frame["next_day"] = r
        rows.append(frame.assign(ticker=f"S{t}"))
    out = evaluate(pd.concat(rows).sort_index(), cost_bps=5).set_index(
        ["definition", "pattern", "outcome", "period"])
    star = out.loc[("paper", "morning_star", "cl", "2002-2022")]
    assert star["success"] == 1.0 and star["t_dates"] > 10 and star["excess_bps"] > 40
    p0, n = star["base"], star["n"]
    assert star["z_binomial"] == pytest.approx(n * (1 - p0) / np.sqrt(n * p0 * (1 - p0)))   # their formula
    assert star["net_bps"] == pytest.approx(star["mean_bps"] - 10)
    evening = out.loc[("paper", "evening_star", "cl", "2002-2022")]
    assert abs(evening["t_dates"]) < 3 and abs(evening["success"] - evening["base"]) < 0.05
    assert out.loc[("talib", "morning_star", "cl", "2002-2022"), "n"] == 0
