"""Tests for the volatility regimes (regimes.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from regimes import (LABELS, analyse, episode, expanding_terciles, realized_volatility,  # noqa: E402
                     regime_days)

CAL = pd.bdate_range("2002-01-02", "2010-12-31")


def prices(vol_by_year, seed=0):
    rng = np.random.default_rng(seed)
    sd = np.array([vol_by_year.get(d.year, 0.01) for d in CAL])
    c = 100 * np.exp(np.cumsum(rng.normal(0, sd)))
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c}, index=CAL)


def test_realized_volatility():
    c = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 2000))))
    v = realized_volatility(c)
    assert v.iloc[:21].isna().all() and v.iloc[21:].notna().all()
    assert v.iloc[21:].mean() == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_terciles_use_only_earlier_days():
    x = pd.Series(np.random.default_rng(1).normal(size=800), index=pd.bdate_range("2002-01-02", periods=800))
    r = expanding_terciles(x)
    assert r.iloc[:252].isna().all() and set(r.iloc[252:]) == set(LABELS)
    changed = x.copy()
    changed.iloc[500:] = 99.0                                   # rewrite the future
    pd.testing.assert_series_equal(r.iloc[:500], expanding_terciles(changed).iloc[:500])
    i = 400                                                     # day 400 against days 0-399
    lo, hi = np.quantile(x.iloc[:i], [1 / 3, 2 / 3])
    expect = "low" if x.iloc[i] <= lo else ("high" if x.iloc[i] > hi else "medium")
    assert r.iloc[i] == expect


def test_regime_days_follow_volatility():
    spy = prices({2008: 0.04, 2009: 0.03})
    vix = spy.assign(Close=np.where(spy.index.year == 2008, 60.0, 15.0) + np.arange(len(spy)) * 1e-6)
    days = regime_days(spy, vix)
    in_2008 = days.loc["2008-03":"2008-12"]
    assert (in_2008["regime_spy"] == "high").mean() > 0.9 and (in_2008["regime_vix"] == "high").all()
    assert days.index.min() >= pd.Timestamp("2002-01-02")


def test_episodes():
    assert [episode(pd.Timestamp(d)) for d in ["2008-10-01", "2009-03-01", "2011-08-08", "2020-03-16",
                                                 "2022-06-13", "2015-08-24"]] == \
        ["2008-09", "2008-09", "2011", "2020", "2022", "other"]


def fake(seed=2):
    """Two stocks, 2005-2010. 'calm' is right on low-volatility days only,
    'storm' on high-volatility days only; reversal is a coin flip with a small edge."""
    rng = np.random.default_rng(seed)
    spy = prices({2008: 0.04, 2009: 0.03, 2010: 0.006}, seed=seed)
    days = regime_days(spy, spy.assign(Close=realized_volatility(spy["Close"]).bfill() * 100))
    test = days.index[days.index.year >= 2005]
    reg = days["regime_spy"].reindex(test).to_numpy()
    rows = []
    for t in ["AAA", "BBB"]:
        fwd = rng.normal(0, 0.01, len(test))
        up = (fwd > 0).astype(int)
        coin = rng.integers(0, 2, len(test))
        for model, pred in [("calm", np.where(reg == "low", up, coin)), ("storm", np.where(reg == "high", up, coin)),
                            ("reversal", np.where(rng.random(len(test)) < 0.55, up, 1 - up)),
                            ("always_long", np.ones(len(test), int)), ("random", rng.integers(0, 2, len(test)))]:
            rows.append(pd.DataFrame({"ticker": t, "model": model, "pred": pred,
                                      "up": pd.array(up, dtype="Int8"), "fwd_return": fwd,
                                      "test_year": test.year}, index=pd.Index(test, name="date")))
    return pd.concat(rows), days


def test_models_are_judged_within_each_regime():
    preds, days = fake()
    out = analyse(preds, days, reps=200)
    met = out["metrics"].set_index(["definition", "regime", "model"])["mcc"]
    assert met[("spy", "low", "calm")] > 0.9 and abs(met[("spy", "high", "calm")]) < 0.15
    assert met[("spy", "high", "storm")] > 0.9 and abs(met[("spy", "low", "storm")]) < 0.15
    t = out["tests"]
    assert set(t.loc[t["loss"] == "classification", "benchmark"]) == {"reversal"}   # one benchmark throughout
    assert set(t.loc[t["loss"] == "economic", "benchmark"]) == {"always_long"}
    p = t[(t["definition"] == "spy") & (t["loss"] == "classification")].set_index(["regime", "model"])
    assert p.loc[("low", "calm"), "dm_stat"] < -3 and p.loc[("high", "storm"), "dm_stat"] < -3
    m = out["mcs"]
    low = m[(m["definition"] == "spy") & (m["regime"] == "low") & (m["loss"] == "classification")]
    assert low.set_index("model").loc["calm", "in_mcs"] and not low.set_index("model").loc["storm", "in_mcs"]
    assert set(out["episodes"]["episode"]) <= {"2008-09", "2011", "2020", "2022", "other"}
    assert {"2008-09"} <= set(out["episodes"]["episode"])
