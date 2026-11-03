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


# ------------------------------------------------------- pattern miner in the runner
from features import symbol_window_columns  # noqa: E402
from run import experiments, run_experiments  # noqa: E402


def test_pattern_miner_is_registered():
    spec = experiments()["pattern_miner"]
    assert spec["feature_sets"] == ["encoding_b_symbol_window"]
    assert spec["features"] == symbol_window_columns()
    assert spec["make"](None).name == "pattern_miner"


@pytest.fixture(scope="module")
def miner_run():
    table = build_table(STOCKS, feature_sets=["normalized_ohlc", "encoding_b_symbol_window"],
                        loader=fake_loader)
    specs = {name: experiments()[name] for name in ["always_long", "pattern_miner"]}
    diag = []
    preds = run_experiments(table, walk_forward(table.index), specs, diagnostics=diag)
    return table, preds, pd.DataFrame(diag)


def test_pattern_miner_predicts_the_same_days_as_the_baselines(miner_run):
    table, preds, _ = miner_run
    n = preds.groupby("model").size()
    assert n["pattern_miner"] == n["always_long"] == (table.index >= "2005-01-01").sum()
    p = preds.loc[preds["model"] == "pattern_miner", "p_up"]
    assert p.between(0, 1).all() and p.nunique() > 1


def test_pattern_miner_diagnostics(miner_run):
    _, _, diag = miner_run
    assert set(diag["model"]) == {"pattern_miner"}               # baselines have no diagnostics
    assert sorted(diag["test_year"]) == list(range(2005, 2023))   # one row per fold
    used = diag[["used_len3", "used_len2", "used_len1", "used_len0"]]
    assert np.allclose(used.sum(axis=1), 1)
    assert (diag["kept_len1"] <= 20).all() and diag["base_rate"].between(0, 1).all()
    # more training data -> more long contexts qualify
    assert diag["kept_len3"].iloc[-1] >= diag["kept_len3"].iloc[0]


# ------------------------------------------------------------------ comparing runs
from compare_runs import compare  # noqa: E402


def test_compare_runs_statuses():
    cols = ["days", "accuracy", "balanced_accuracy", "mcc", "ls_ann_return", "ls_sharpe",
            "lf_ann_return", "lf_sharpe"]
    old = pd.DataFrame([[100, .5, .5, 0.0, .01, .1, .02, .2],
                        [100, .52, .51, .02, .03, .3, .04, .4],
                        [100, .5, .5, 0.0, 0, 0, 0, 0]],
                       index=["always_long", "lightgbm_b", "gone"], columns=cols)
    new = old.drop(index="gone").copy()
    new.loc["lightgbm_b", "mcc"] = 0.021
    new.loc["pattern_miner"] = [100, .51, .5, .01, .01, .1, .01, .1]
    status = compare(old, new).set_index("model")["status"]
    assert status.to_dict() == {"always_long": "same", "lightgbm_b": "CHANGED",
                                "gone": "dropped", "pattern_miner": "new"}
