"""Tests for the statistical tests (significance.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import load_config  # noqa: E402
from significance import analyse, distinct, dm_test, newey_west_lag, strongest_baseline  # noqa: E402

DAYS = pd.bdate_range("2005-01-03", periods=2000)


def test_dm_without_lags_is_the_t_statistic():
    rng = np.random.default_rng(0)
    d = rng.normal(0.3, 1, 500)
    out = dm_test(d, lags=0)
    se = d.std(ddof=0) / np.sqrt(len(d))
    assert out["dm_stat"] == pytest.approx(d.mean() / se) and out["p_value"] < 0.05
    assert newey_west_lag(4530) == 9 == dm_test(rng.normal(0, 1, 4530))["nw_lags"]


def test_dm_widens_for_autocorrelated_differences():
    rng = np.random.default_rng(1)
    d = pd.Series(rng.normal(0.02, 1, 3000)).rolling(10, min_periods=1).mean()
    assert abs(dm_test(d, lags=0)["dm_stat"]) > abs(dm_test(d, lags=30)["dm_stat"])


def test_dm_size_under_the_null():
    """Equal losses: about 5% of tests reject at 5%."""
    rng = np.random.default_rng(2)
    rejects = np.mean([dm_test(rng.normal(0, 1, 1000))["p_value"] < 0.05 for _ in range(400)])
    assert 0.02 < rejects < 0.09


def test_distinct_drops_identical_columns():
    t = pd.DataFrame({"always_long": [1.0, 2.0], "majority": [1.0, 2.0], "x": [1.0, 3.0]})
    kept, same = distinct(t)
    assert list(kept.columns) == ["always_long", "x"] and same == {"majority": "always_long"}


def fake_results(edge: float, seed: int = 3):
    """Daily losses for 4 baselines and 3 models. 'good' has lower losses than
    reversal by `edge` on both series; 'noise' and 'worse' do not."""
    rng = np.random.default_rng(seed)
    n = len(DAYS)
    base_c = rng.normal(0.49, 0.1, n)
    base_e = rng.normal(0, 0.01, n)
    cls = pd.DataFrame({
        "always_long": rng.normal(0.5, 0.1, n), "previous_day": rng.normal(0.51, 0.1, n),
        "random": rng.normal(0.5, 0.1, n), "reversal": base_c + rng.normal(0, 0.05, n),
        "good": base_c - edge + rng.normal(0, 0.05, n), "noise": base_c + rng.normal(0, 0.05, n),
        "worse": base_c + 0.03 + rng.normal(0, 0.05, n)}, index=DAYS)
    cls["majority"] = cls["always_long"]
    eco = pd.DataFrame({c: base_e + rng.normal(0, 0.004, n) for c in cls.columns}, index=DAYS)
    eco["majority"] = eco["always_long"]
    eco["good"] -= edge / 20                       # also a better trading return
    summary = pd.DataFrame({
        "mcc": {"always_long": 0, "majority": 0, "previous_day": -0.02, "random": 0.0, "reversal": 0.02,
                "good": 0.05, "noise": 0.021, "worse": 0.01},
        "ls_sharpe": {"always_long": 0.8, "majority": 0.8, "previous_day": -1, "random": -1, "reversal": -0.3,
                      "good": 1.5, "noise": 0.5, "worse": 0.2},
        "lf_sharpe": {"always_long": 0.8, "majority": 0.8, "previous_day": 0, "random": 0, "reversal": 0.4,
                      "good": 1.2, "noise": 0.7, "worse": 0.5}})
    return {"classification": cls, "economic": eco, "long_flat": eco.copy()}, summary


def test_strongest_baseline():
    _, summary = fake_results(0.0)
    assert strongest_baseline(summary, load_config()["significance"]["baselines"]) == "reversal"


def test_a_real_edge_passes_every_criterion():
    losses, summary = fake_results(edge=0.03)
    out = analyse(losses, summary, reps=300)
    v = out["summary"].set_index("model")
    assert out["strongest"] == "reversal"
    assert v.loc["good", ["beats_classification", "beats_economic", "beats_buy_and_hold",
                          "genuine_value"]].all()
    assert not v.loc[["noise", "worse"], "genuine_value"].any()
    m = out["mcs"].set_index(["loss", "model"])["in_mcs"]
    assert m[("classification", "good")] and not m[("classification", "reversal")]
    assert m[("classification", "majority")] == m[("classification", "always_long")]
    assert out["spa"].set_index("loss").loc["classification", "p_consistent"] < 0.05


def test_no_edge_no_verdict():
    losses, summary = fake_results(edge=0.0)
    out = analyse(losses, summary, reps=300)
    v = out["summary"]
    assert not v["beats_classification"].any() and not v["genuine_value"].any()
    m = out["mcs"].set_index(["loss", "model"])["in_mcs"]
    assert m[("classification", "reversal")]


def test_higher_mcc_alone_is_not_enough():
    """'noise' has a slightly higher MCC than reversal but no lower loss."""
    losses, summary = fake_results(edge=0.0)
    assert summary.loc["noise", "mcc"] > summary.loc["reversal", "mcc"]
    v = analyse(losses, summary, reps=200)["summary"].set_index("model")
    assert not v.loc["noise", "beats_classification"]


def test_dead_zone_days_are_skipped():
    losses, summary = fake_results(edge=0.03)
    losses["classification"].iloc[:5] = np.nan              # all-dead-zone days
    out = analyse(losses, summary, reps=200)
    dm = out["dm"]
    assert (dm.loc[dm["loss"] == "classification", "days"] == len(DAYS) - 5).all()
