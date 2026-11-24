"""Tests for the bootstrap confidence intervals (bootstrap_ci.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bootstrap_ci import bootstrap, stationary_indices  # noqa: E402
from evaluate import classification_metrics, performance, portfolio_returns, stock_returns  # noqa: E402


def test_stationary_bootstrap_blocks():
    idx = stationary_indices(1000, 200, block=20, rng=np.random.default_rng(0))
    assert idx.shape == (200, 1000) and idx.min() >= 0 and idx.max() < 1000
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % 1000).mean()
    assert continues == pytest.approx(1 - 1 / 20, abs=0.01)          # mean block length 20
    counts = np.bincount(idx.ravel(), minlength=1000) / 200
    assert counts.mean() == pytest.approx(1.0) and counts.min() > 0.5  # every date about equally likely


def fake_preds(seed=0, days=800, stocks=6, edge=0.0):
    """'always_long', 'reversal' (a coin flip) and 'model', which is right with
    probability 0.5 + edge. Stocks share a daily market move."""
    rng = np.random.default_rng(seed)
    idx = pd.Index(pd.bdate_range("2005-01-03", periods=days), name="date")
    market = rng.normal(0.0004, 0.01, days)
    rows = []
    for t in range(stocks):
        fwd = market + rng.normal(0, 0.01, days)
        up = pd.array(np.where(np.abs(fwd) < 0.0005, None, (fwd > 0).astype(int)), dtype="Int8")
        truth = (fwd > 0).astype(int)
        good = np.where(rng.random(days) < 0.5 + edge, truth, 1 - truth)
        for model, pred in [("always_long", np.ones(days, int)), ("reversal", rng.integers(0, 2, days)),
                            ("model", good)]:
            rows.append(pd.DataFrame({"ticker": f"S{t}", "model": model, "pred": pred, "up": up,
                                      "fwd_return": fwd}, index=idx))
    return pd.concat(rows)


def test_point_estimates_match_the_headline_metrics():
    preds = fake_preds()
    ci = bootstrap(preds, reps=50, block=20, strongest="reversal").set_index(["model", "metric"])
    for model, g in preds.groupby("model"):
        m = classification_metrics(g["up"], g["pred"])
        for k in ["mcc", "accuracy", "balanced_accuracy"]:
            assert ci.loc[(model, k), "estimate"] == pytest.approx(m[k], abs=1e-12)
        for strategy, k in [("long_short", "ls_sharpe"), ("long_flat", "lf_sharpe")]:
            sharpe = performance(portfolio_returns(stock_returns(g, strategy, 5)))["sharpe"]
            assert ci.loc[(model, k), "estimate"] == pytest.approx(sharpe, abs=1e-9)


def test_intervals_find_a_real_edge_and_not_noise():
    ci = bootstrap(fake_preds(edge=0.1), reps=1000, block=20, strongest="reversal")
    ci = ci.set_index(["model", "metric"])
    assert ci.loc[("model", "mcc"), "ci_low"] > 0
    assert ci.loc[("model", "mcc_minus_reversal"), "ci_low"] > 0
    assert ci.loc[("reversal", "mcc"), "ci_low"] < 0 < ci.loc[("reversal", "mcc"), "ci_high"]
    assert ci.loc[("always_long", "mcc"), ["ci_low", "ci_high"]].tolist() == [0.0, 0.0]
    mcc = ci.xs("mcc", level="metric")
    assert ((mcc["ci_low"] <= mcc["estimate"]) & (mcc["estimate"] <= mcc["ci_high"])).all()
    assert ("always_long", "ls_sharpe_minus_always_long") not in ci.index


def test_reproducible_and_sensitive_to_the_seed():
    preds = fake_preds(days=300)
    a = bootstrap(preds, reps=200, seed=1, strongest="reversal")
    b = bootstrap(preds, reps=200, seed=1, strongest="reversal")
    c = bootstrap(preds, reps=200, seed=2, strongest="reversal")
    pd.testing.assert_frame_equal(a, b)
    assert not a["ci_low"].equals(c["ci_low"])
