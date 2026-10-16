"""Tests for the backtest: positions, costs and turnover by hand, portfolio, performance figures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import (backtest, performance, portfolio_returns, positions,  # noqa: E402
                      stock_returns)


def one_stock(pred, fwd, ticker="AAA", start="2005-01-03"):
    dates = pd.bdate_range(start, periods=len(pred))
    return pd.DataFrame({"ticker": ticker, "model": "m", "test_year": dates.year, "pred": pred,
                         "up": pd.array([pd.NA] * len(pred), dtype="Int8"),
                         "fwd_return": fwd}, index=dates)


def test_config_values():
    from features import load_config
    b = load_config()["backtest"]
    assert b["cost_bps_per_side"] == 5 and b["cost_grid_bps"] == [0, 1, 2, 5, 10, 20]
    assert b["strategies"] == ["long_short", "long_flat"]


def test_positions():
    assert positions([1, 0, 1], "long_short").tolist() == [1, -1, 1]
    assert positions([1, 0, 1], "long_flat").tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        positions([1], "short_only")


def test_returns_costs_and_turnover_by_hand():
    #            up    down   down   up
    preds = one_stock([1, 0, 0, 1], [0.02, -0.01, 0.03, 0.01])
    ls = stock_returns(preds, "long_short", cost_bps=10)
    # positions +1, -1, -1, +1; turnover from cash: 1, 2, 0, 2
    assert ls["position"].tolist() == [1, -1, -1, 1]
    assert ls["turnover"].tolist() == [1, 2, 0, 2]
    assert ls["gross"].tolist() == pytest.approx([0.02, 0.01, -0.03, 0.01])
    assert ls["net"].tolist() == pytest.approx([0.02 - 0.001, 0.01 - 0.002, -0.03, 0.01 - 0.002])
    lf = stock_returns(preds, "long_flat", cost_bps=10)
    assert lf["turnover"].tolist() == [1, 1, 0, 1]
    assert lf["net"].tolist() == pytest.approx([0.019, -0.001, 0.0, 0.009])


def test_dead_zone_days_are_traded_and_last_day_dropped():
    preds = one_stock([1, 1, 1], [0.001, 0.002, np.nan])     # 'up' is all <NA> (dead zone)
    daily = stock_returns(preds, "long_flat", 0)
    assert len(daily) == 2 and daily["gross"].sum() == pytest.approx(0.003)


def test_turnover_never_mixes_stocks():
    a = one_stock([1, 1], [0.01, 0.01], "AAA")
    b = one_stock([0, 0], [0.01, 0.01], "BBB")
    daily = stock_returns(pd.concat([a, b]), "long_short", 0)
    # each stock pays only its own entry; no flip between AAA's +1 and BBB's -1
    assert daily["turnover"].tolist() == [1, 0, 1, 0]


def test_positions_carry_across_test_years():
    preds = one_stock([1] * 6, [0.001] * 6, start="2005-12-27")   # spans 2005 -> 2006
    assert set(preds["test_year"]) == {2005, 2006}
    assert stock_returns(preds, "long_flat", 5)["turnover"].sum() == 1   # no re-entry at new year


def test_equal_weight_portfolio():
    a = one_stock([1, 1], [0.02, 0.00], "AAA")
    b = one_stock([1, 1], [0.00, -0.02], "BBB")
    port = portfolio_returns(stock_returns(pd.concat([a, b]), "long_flat", 0))
    assert port.tolist() == pytest.approx([0.01, -0.01])


def test_performance_by_hand():
    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    p = performance(r, days_per_year=252)
    assert p["ann_return"] == pytest.approx((1.01 * 0.98 * 1.03) ** 63 - 1)
    assert p["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert p["max_drawdown"] == pytest.approx(0.98 - 1)       # from the 1.01 peak down to 1.01*0.98
    assert np.isnan(performance(pd.Series([0.001] * 10), 252)["sharpe"])   # undefined with no volatility


def test_costs_only_ever_hurt_and_scale_linearly():
    rng = np.random.default_rng(0)
    preds = one_stock(rng.integers(0, 2, 500), rng.normal(0.0005, 0.01, 500))
    res = backtest(preds, strategies=["long_short"], costs_bps=[0, 5, 10]).set_index("cost_bps")
    assert res.loc[0, "ann_return"] > res.loc[5, "ann_return"] > res.loc[10, "ann_return"]
    assert res.loc[10, "cost_per_year"] == pytest.approx(2 * res.loc[5, "cost_per_year"])
    assert res.loc[0, "cost_per_year"] == 0


def test_perfect_and_inverted_predictors():
    rng = np.random.default_rng(1)
    fwd = rng.normal(0, 0.01, 1000)
    perfect = backtest(one_stock((fwd > 0).astype(int), fwd), ["long_short"], [0]).iloc[0]
    wrong = backtest(one_stock((fwd < 0).astype(int), fwd), ["long_short"], [0]).iloc[0]
    assert perfect["sharpe"] > 10 and wrong["sharpe"] < -10


def test_always_long_long_flat_is_buy_and_hold():
    rng = np.random.default_rng(2)
    stocks = [one_stock([1] * 300, rng.normal(0.0004, 0.01, 300), t) for t in ["AAA", "BBB", "CCC"]]
    preds = pd.concat(stocks)
    daily = stock_returns(preds, "long_flat", 5)
    hold = preds.groupby(level=0)["fwd_return"].mean()
    # identical except the one-time entry cost on day one
    assert portfolio_returns(daily).iloc[1:].tolist() == pytest.approx(hold.iloc[1:].tolist())
    assert daily["turnover"].sum() == 3


def test_backtest_refuses_holdout_dates():
    with pytest.raises(PermissionError):
        backtest(one_stock([1, 1], [0.01, 0.01], start="2023-01-03"))
