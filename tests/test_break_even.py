"""Tests for the break-even cost (break_even.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from break_even import break_even, break_even_table  # noqa: E402
from evaluate import performance, portfolio_returns, stock_returns  # noqa: E402

DAYS = pd.Index(pd.bdate_range("2005-01-03", periods=1000), name="date")


def preds_for(pred, fwd, model="m", ticker="AAA"):
    return pd.DataFrame({"ticker": ticker, "model": model, "pred": pred,
                         "up": pd.array((np.asarray(fwd) > 0).astype(int), dtype="Int8"),
                         "fwd_return": fwd}, index=DAYS)


def test_net_return_is_zero_at_the_break_even_cost():
    rng = np.random.default_rng(0)
    fwd = rng.normal(0, 0.01, len(DAYS))
    pred = np.where(rng.random(len(DAYS)) < 0.6, fwd > 0, fwd <= 0).astype(int)   # 60% right
    frames = [preds_for(pred, fwd, ticker=t) for t in ["AAA", "BBB"]]
    preds = pd.concat(frames)
    out = break_even(preds, "long_short")
    c = out["break_even_bps"]
    assert 0 < c < 1000
    at = performance(portfolio_returns(stock_returns(preds, "long_short", c)))["ann_return"]
    assert at == pytest.approx(0, abs=1e-6)
    above = performance(portfolio_returns(stock_returns(preds, "long_short", c + 1)))["ann_return"]
    assert above < 0


def test_never_profitable_and_no_limit():
    rng = np.random.default_rng(1)
    fwd = rng.normal(0, 0.01, len(DAYS))
    wrong = (fwd <= 0).astype(int)                                  # always wrong: loses before costs
    assert np.isnan(break_even(preds_for(wrong, fwd), "long_short")["break_even_bps"])
    up = np.abs(fwd) + 0.0001                                      # a market that only rises
    assert np.isinf(break_even(preds_for(np.ones(len(DAYS), int), up), "long_short")["break_even_bps"])


def test_more_trading_lowers_the_break_even():
    rng = np.random.default_rng(2)
    fwd = 0.002 + rng.normal(0, 0.01, len(DAYS))
    steady = np.ones(len(DAYS), int)
    steady[::50] = 0                                               # trades rarely
    flippy = np.ones(len(DAYS), int)
    flippy[::4] = 0                                                # trades often
    a = break_even(preds_for(steady, fwd), "long_short")
    b = break_even(preds_for(flippy, fwd), "long_short")
    assert b["turnover_per_year"] > a["turnover_per_year"]
    assert b["break_even_bps"] < a["break_even_bps"]


def test_table_has_every_model_and_strategy():
    rng = np.random.default_rng(3)
    fwd = rng.normal(0.0005, 0.01, len(DAYS))
    preds = pd.concat([preds_for(rng.integers(0, 2, len(DAYS)), fwd, model=m) for m in ["a", "b"]])
    t = break_even_table(preds)
    assert sorted(zip(t["model"], t["strategy"])) == [("a", "long_flat"), ("a", "long_short"),
                                                       ("b", "long_flat"), ("b", "long_short")]
