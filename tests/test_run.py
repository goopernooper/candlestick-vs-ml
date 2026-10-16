"""End-to-end tests: the pooled table, the holdout cut, and a full baseline run."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import build_table  # noqa: E402
from run import baseline_checks, baseline_models, run_models, summary_table  # noqa: E402
from splits import walk_forward  # noqa: E402

CAL = pd.bdate_range("2001-10-01", "2025-12-31")
STOCKS = ["AAA", "BBB", "CCC", "DDD"]


def fake_loader(ticker):
    rng = np.random.default_rng(STOCKS.index(ticker))
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(CAL))))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, len(CAL)))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.004,
                         "Low": np.minimum(o, c) * 0.996, "Close": c}, index=CAL)


@pytest.fixture(scope="module")
def table():
    return build_table(STOCKS, feature_sets=["normalized_ohlc", "encoding_b_symbolic"], loader=fake_loader)


def test_table_layout(table):
    assert set(table["ticker"]) == set(STOCKS)
    assert table.index.min() == pd.Timestamp("2002-01-02")
    assert {"n_close", "b_symbol", "up", "fwd_return"} <= set(table.columns)
    assert (table.groupby("ticker").size() == table.groupby("ticker").size().iloc[0]).all()


def test_table_stops_before_holdout_minus_horizon(table):
    # last 2022 business day is 2022-12-30; its label would use 2023 prices, so it's dropped
    assert table.index.max() == pd.Timestamp("2022-12-29")
    assert table["fwd_return"].notna().all() or table.index.max() < pd.Timestamp("2023-01-01")


@pytest.fixture(scope="module")
def preds(table):
    return run_models(table, walk_forward(table.index), baseline_models(), ["n_close"])


def test_every_model_predicts_every_test_day(preds, table):
    test_days = table[table.index >= "2005-01-01"]
    for _, g in preds.groupby("model"):
        assert len(g) == len(test_days)
    assert set(preds["test_year"]) == set(range(2005, 2023))


def test_predictions_follow_the_contract(preds):
    assert {"ticker", "test_year", "model", "p_up", "pred", "up", "fwd_return"} <= set(preds.columns)
    assert preds["p_up"].between(0, 1).all()
    assert (preds["pred"] == (preds["p_up"] >= 0.5)).all()


def test_random_seed_differs_by_fold(preds):
    r = preds[(preds["model"] == "random") & (preds["ticker"] == "AAA")]
    a = r[r["test_year"] == 2005]["p_up"].to_numpy()[:50]
    b = r[r["test_year"] == 2006]["p_up"].to_numpy()[:50]
    assert not np.allclose(a, b)


def test_summary_and_baseline_checks(preds):
    s = summary_table(preds, cost_bps=5, stamp="abc1234", run_at="now")
    assert set(s["model"]) == {"majority", "always_long", "previous_day", "random"}
    assert (s["git_commit"] == "abc1234").all()
    assert {"ls_sharpe", "lf_sharpe", "mcc"} <= set(s.columns)
    checks = baseline_checks(s)
    assert len(checks) == 3 and all(passed for _, passed in checks)
