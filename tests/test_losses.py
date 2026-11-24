"""Tests for the daily loss series (losses.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import classification_metrics, portfolio_returns, stock_returns  # noqa: E402
from losses import classification_losses, economic_losses, write_losses  # noqa: E402

DAYS = pd.bdate_range("2005-01-03", periods=4)


def tiny():
    """Two stocks, four days, two models. Day 3: both stocks in the dead zone."""
    up = {"AAA": [1, 0, None, 1], "BBB": [1, 1, None, 0]}
    fwd = {"AAA": [0.01, -0.02, 0.0001, 0.03], "BBB": [0.02, 0.01, -0.0001, -0.01]}
    pred = {("long", "AAA"): [1, 1, 1, 1], ("long", "BBB"): [1, 1, 1, 1],
            ("mixed", "AAA"): [1, 0, 0, 0], ("mixed", "BBB"): [0, 1, 1, 0]}
    rows = []
    for (model, t), p in pred.items():
        rows.append(pd.DataFrame({"ticker": t, "model": model, "pred": p,
                                  "up": pd.array(up[t], dtype="Int8"), "fwd_return": fwd[t]},
                                 index=pd.Index(DAYS, name="date")))
    return pd.concat(rows)


def test_classification_losses_by_hand():
    c = classification_losses(tiny(), balanced=False)
    assert list(c.columns) == ["long", "mixed"] and list(c.index) == list(DAYS)
    assert c["long"].tolist()[:2] == [0.0, 0.5] and c["long"].iloc[3] == 0.5
    assert c["mixed"].tolist()[:2] == [0.5, 0.0] and c["mixed"].iloc[3] == 0.5
    assert c.iloc[2].isna().all()                       # all dead zone: no loss that day


def test_economic_losses_by_hand():
    e = economic_losses(tiny(), cost_bps=0)
    assert e["long"].tolist() == pytest.approx([-0.015, 0.005, 0.0, -0.01])
    # mixed, long/short: day 1 AAA +0.01, BBB short -0.02 -> -0.005, loss +0.005
    assert e["mixed"].iloc[0] == pytest.approx(0.005)
    with_cost = economic_losses(tiny(), cost_bps=10)
    assert with_cost["long"].iloc[0] == pytest.approx(-0.015 + 0.001)   # entering costs one side
    assert (with_cost["long"].iloc[1:] == e["long"].iloc[1:]).all()     # no trades after that


def random_preds(seed=0, days=600, stocks=5):
    rng = np.random.default_rng(seed)
    idx = pd.Index(pd.bdate_range("2005-01-03", periods=days), name="date")
    rows = []
    for t in range(stocks):
        fwd = rng.normal(0, 0.01, days)
        up = pd.array(np.where(np.abs(fwd) < 0.001, None, (fwd > 0).astype(int)), dtype="Int8")
        for model in ["a", "b"]:
            rows.append(pd.DataFrame({"ticker": f"S{t}", "model": model,
                                      "pred": rng.integers(0, 2, days), "up": up, "fwd_return": fwd},
                                     index=idx))
    return pd.concat(rows)


def test_losses_agree_with_the_headline_metrics():
    preds = random_preds()
    c, e = classification_losses(preds), economic_losses(preds)
    for model, g in preds.groupby("model"):
        lab = g[g["up"].notna()]
        acc = classification_metrics(lab["up"].astype(int), lab["pred"])["accuracy"]
        plain = classification_losses(preds, balanced=False)
        # equal weight per day vs per stock-day: close, not identical
        assert 1 - plain[model].mean() == pytest.approx(acc, abs=0.01)
        bal = classification_metrics(lab["up"].astype(int), lab["pred"])["balanced_accuracy"]
        assert 1 - c[model].mean() == pytest.approx(bal, abs=0.01)
        net = portfolio_returns(stock_returns(g, "long_short", 5))
        pd.testing.assert_series_equal(-e[model], net, check_names=False)


def test_write_losses(tmp_path):
    names = write_losses(random_preds(days=50), tmp_path)
    assert names == ["losses_classification.csv", "losses_misclassification.csv", "losses_economic.csv"]
    for name in names:
        t = pd.read_csv(tmp_path / name, index_col="date", parse_dates=True)
        assert list(t.columns) == ["a", "b"] and len(t) == 50


def test_balanced_loss_by_hand():
    """tiny(): 2005 labels are up, up, up, down, up, down -> pi = 4/6.
    Weights: up 1/(2 x 2/3) = 0.75, down 1/(2 x 1/3) = 1.5."""
    c = classification_losses(tiny())
    # long: day 2 misses AAA's down day (1.5) out of 2 -> 0.75; day 4 misses BBB's down day -> 0.75
    assert c["long"].tolist()[:2] == pytest.approx([0.0, 0.75]) and c["long"].iloc[3] == pytest.approx(0.75)
    # mixed: day 1 misses BBB's up day (0.75) out of 2 -> 0.375
    assert c["mixed"].iloc[0] == pytest.approx(0.375)
    assert c.iloc[2].isna().all()


def test_constant_predictions_score_one_half():
    """Always up and always down both average about 0.5, as their MCC is 0,
    however unbalanced the classes; the plain loss favours always up."""
    rng = np.random.default_rng(4)
    idx = pd.Index(pd.bdate_range("2005-01-03", "2008-12-31"), name="date")
    rows = []
    for t in range(22):
        up = pd.array((rng.random(len(idx)) < np.where(idx.year == 2008, 0.45, 0.56)).astype(int), dtype="Int8")
        for model, p in [("always_up", 1), ("always_down", 0)]:
            rows.append(pd.DataFrame({"ticker": f"S{t}", "model": model, "pred": p, "up": up,
                                      "fwd_return": 0.0}, index=idx))
    preds = pd.concat(rows)
    bal, plain = classification_losses(preds).mean(), classification_losses(preds, balanced=False).mean()
    assert bal["always_up"] == pytest.approx(0.5, abs=0.01) and bal["always_down"] == pytest.approx(0.5, abs=0.01)
    assert plain["always_up"] < 0.47 < 0.53 < plain["always_down"]


def test_a_year_with_one_direction_is_refused():
    preds = tiny()
    preds["up"] = pd.array([1] * len(preds), dtype="Int8")
    with pytest.raises(ValueError):
        classification_losses(preds)
