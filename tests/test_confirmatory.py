"""Tests for the confirmatory hypothesis code (confirmatory.py, ANALYSIS_PLAN.md 13.1)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from confirmatory import MODELS, confident_backtest, holm  # noqa: E402
from confirmatory import tests as run_tests  # noqa: E402


def test_models_are_the_six_in_the_plan():
    assert MODELS == ["svm_ohlc", "rf_b", "lightgbm_b", "knn_b3", "mlp_ohlc", "lightgbm_ohlc"]


def test_holm():
    assert holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}
    assert holm({"a": 0.2, "b": 0.001}) == {"a": False, "b": True}
    # stops at the first failure even if a later p would pass its own threshold
    assert holm({"a": 0.01, "b": 0.03, "c": 0.031})["c"] is False


def fake(edge, seed=0, days=500, stocks=10):
    """Validation (2005 fold) and test tables. Every model in MODELS is right
    with probability 0.5 + edge on its confident days; reversal is a coin flip."""
    rng = np.random.default_rng(seed)
    out = []
    for year in [2004, 2005]:
        idx = pd.Index(pd.bdate_range(f"{year}-01-03", periods=days), name="date")
        rows = []
        for t in range(stocks):
            up = rng.integers(0, 2, days)
            fwd = np.where(up == 1, 0.01, -0.01)
            rows.append(pd.DataFrame({"ticker": f"S{t}", "model": "reversal", "test_year": 2005,
                                      "pred": rng.integers(0, 2, days), "p_up": 0.5,
                                      "up": pd.array(up, dtype="Int8"), "fwd_return": fwd}, index=idx))
            for m in MODELS:
                conf = rng.uniform(0, 0.5, days)
                right = rng.random(days) < np.where(conf > 0.45, 0.5 + edge, 0.5)
                pred = np.where(right, up, 1 - up)
                rows.append(pd.DataFrame({"ticker": f"S{t}", "model": m, "test_year": 2005, "pred": pred,
                                          "p_up": np.where(pred == 1, 0.5 + conf, 0.5 - conf),
                                          "up": pd.array(up, dtype="Int8"), "fwd_return": fwd}, index=idx))
        out.append(pd.concat(rows))
    return out


def test_a_real_edge_is_confirmed():
    val, test = fake(edge=0.15)
    out = run_tests(val, test, reps=400, seed=1)
    assert out["h1"]["rejected"] and out["h1"]["p_one_sided"] < 0.01 and out["h1"]["delta"] > 0.2
    h2 = out["table"][out["table"]["test"] == "H2"]
    assert h2["holm_rejected"].all() and set(h2["model"]) == set(MODELS)
    assert (h2["stock_days"] / 5000).between(0.07, 0.13).all()          # about 10% of stock-days


def test_no_edge_is_not_confirmed():
    val, test = fake(edge=0.0, seed=2)
    out = run_tests(val, test, reps=400, seed=1)
    assert not out["h1"]["rejected"] and out["h1"]["p_one_sided"] > 0.05
    assert out["h1"]["ci_low"] < 0 < out["h1"]["ci_high"]


def test_missing_model_is_an_error():
    val, test = fake(edge=0.1)
    with pytest.raises(ValueError):
        run_tests(val, test[test["model"] != "rf_b"], reps=10)


def test_confident_backtest_by_hand():
    d = pd.bdate_range("2005-01-03", periods=3)
    sel = pd.DataFrame({"ticker": ["A", "B", "A"], "pred": [1, 0, 1], "fwd_return": [0.02, 0.01, -0.01]},
                       index=pd.Index([d[0], d[0], d[2]], name="date"))
    r = confident_backtest(sel, cost_bps=10, days_per_year=252, all_dates=d)
    # day 0: A long 1/2, B short 1/2 -> gross 0.01 - 0.005 = 0.005, turnover 1 -> cost 0.001
    # day 1: cash -> turnover 1 (closing both) -> -0.001
    # day 2: A long 1 -> gross -0.01, turnover 1 -> -0.011
    net = np.array([0.004, -0.001, -0.011])
    assert r["days_invested"] == 2 and r["days"] == 3
    assert r["ann_return"] == pytest.approx(np.expm1(np.log1p(net).sum() * 252 / 3))
    assert r["turnover_per_year"] == pytest.approx(252.0)
    assert np.isnan(r["break_even_bps"])                                 # loses even before costs
