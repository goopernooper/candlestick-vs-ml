"""Tests for the Caginalp & Laurent replication (replicate_caginalp.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replicate_caginalp import PATTERNS, build, evaluate, nw_t, stock_frame  # noqa: E402


def prices(n=300, seed=0, start="2002-01-02"):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.005, n))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.005,
                         "Low": np.minimum(o, c) * 0.995, "Close": c}, index=idx)


def test_patterns_are_caginalp_and_laurents_eight():
    assert len(PATTERNS) == 8
    assert sum(d > 0 for _, d in PATTERNS.values()) == 4
    assert {c for c, _ in PATTERNS.values()} == {"a_3whitesoldiers", "a_3blackcrows", "a_3inside",
                                                "a_3outside", "a_morningstar", "a_eveningstar"}


def test_forward_returns_and_exit_dates():
    df = prices()
    f = stock_frame(df)
    c, o, d = df["Close"], df["Open"], df.index
    t = 100
    assert f["next_day"].iloc[t] == pytest.approx(c.iloc[t + 1] / c.iloc[t] - 1)
    assert f["cl_style"].iloc[t] == pytest.approx(c.iloc[t + 1:t + 4].mean() / o.iloc[t + 1] - 1)
    assert f["exit_next_day"].iloc[t] == d[t + 1] and f["exit_cl_style"].iloc[t] == d[t + 3]
    assert np.isnan(f["cl_style"].iloc[-1])


def test_trend_rules():
    # a steady fall, then a steady rise
    c = np.r_[np.linspace(120, 100, 40), np.linspace(100, 120, 40)]
    idx = pd.bdate_range("2002-01-02", periods=len(c))
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c}, index=idx)
    f = stock_frame(df)
    assert f["down_strict"].iloc[15:40].all() and f["down_net"].iloc[15:40].all()
    assert f["up_strict"].iloc[55:].all() and not f["up_strict"].iloc[15:40].any()
    # the strict rule needs all six days falling; the net rule only the overall change
    assert f[["down_strict", "down_net"]].iloc[:9].sum().sum() == 0


def test_signals_and_trends_only_look_backward():
    df = prices()
    later = df.copy()
    later.iloc[150:] *= 1.7
    a, b = stock_frame(df), stock_frame(later)
    cols = list(PATTERNS) + ["down_net", "up_net", "down_strict", "up_strict"]
    pd.testing.assert_frame_equal(a[cols].iloc[:150], b[cols].iloc[:150])


def test_nw_t():
    x = pd.Series(np.r_[np.ones(50) * 0.01, -np.ones(50) * 0.005])
    e = x - x.mean()
    assert nw_t(x, 0) == pytest.approx(x.mean() / np.sqrt((e @ e / len(x)) / len(x)))
    assert np.isnan(nw_t(pd.Series([1.0, 2.0]), 0))


def test_build_stops_before_the_holdout():
    frame = build(["AAA", "BBB"], lambda t: prices(n=5400, seed=len(t), start="2001-10-01"))
    assert frame.index.max() < pd.Timestamp("2023-01-01")
    assert frame.index.min() >= pd.Timestamp("2002-01-02")
    # the last days of 2022 can't use exits in 2023
    for h in ["next_day", "cl_style"]:
        used = frame[frame[h].notna()]
        assert (used[f"exit_{h}"] < pd.Timestamp("2023-01-01")).all()


def test_evaluate_finds_a_planted_edge_and_nothing_in_noise():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2003-01-01", "2016-12-31")
    rows = []
    for t in range(10):
        r = rng.normal(0, 0.01, len(days))
        star = rng.random(len(days)) < 0.03
        r = np.where(star, np.abs(r) + 0.004, r)          # after a morning star: up, by more
        frame = pd.DataFrame({name: False for name in PATTERNS}, index=days)
        frame["morning_star"] = star
        frame["evening_star"] = rng.random(len(days)) < 0.03   # no edge
        for k in ["down_net", "up_net", "down_strict", "up_strict"]:
            frame[k] = True
        frame["next_day"] = r
        frame["cl_style"] = r
        rows.append(frame.assign(ticker=f"S{t}"))
    out = evaluate(pd.concat(rows).sort_index(), cost_bps=5).set_index(
        ["pattern", "trend_rule", "horizon", "period"])
    star = out.loc[("morning_star", "none", "next_day", "2002-2022")]
    assert star["success"] == 1.0 and star["excess_bps"] > 40 and star["t_dates"] > 10
    assert star["net_bps"] == pytest.approx(star["mean_bps"] - 10)
    evening = out.loc[("evening_star", "none", "next_day", "2002-2022")]
    assert abs(evening["t_dates"]) < 3 and abs(evening["success"] - evening["base"]) < 0.05
    assert out.loc[("three_white_soldiers", "none", "next_day", "2002-2022"), "n"] == 0
