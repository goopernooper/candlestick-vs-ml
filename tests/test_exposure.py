"""Tests for the market-exposure check (exposure.py): beta and alpha against buy-and-hold."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exposure import alpha_beta, exposure_table, market_returns, newey_west_lag  # noqa: E402

DAYS = pd.bdate_range("2005-01-03", periods=3000)


def test_alpha_beta_recovers_a_known_relationship():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0.0004, 0.01, len(DAYS)), index=DAYS)
    r = 0.0004 + 0.6 * m + rng.normal(0, 0.002, len(DAYS))     # alpha ~10% a year
    out = alpha_beta(r, m)
    assert out["beta"] == pytest.approx(0.6, abs=0.02)
    assert out["alpha_ann"] == pytest.approx(0.0004 * 252, abs=0.03)    # 3 standard errors
    assert out["alpha_t"] > 2
    assert out["nw_lags"] == newey_west_lag(len(DAYS)) == 8          # floor(4 x 30^(2/9))


def test_the_market_itself_has_beta_one_and_no_alpha():
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0.0004, 0.01, len(DAYS)), index=DAYS)
    out = alpha_beta(m, m)
    assert out["beta"] == pytest.approx(1.0) and abs(out["alpha_ann"]) < 1e-12
    assert np.isnan(out["alpha_t"]) and np.isnan(out["info_ratio"])     # an exact fit has no error


def test_dates_must_line_up():
    m = pd.Series(np.ones(10), index=DAYS[:10])
    with pytest.raises(ValueError):
        alpha_beta(np.ones(10), m)                       # no dates: nothing to match


def test_newey_west_widens_the_interval_for_autocorrelated_noise():
    rng = np.random.default_rng(2)
    m = pd.Series(rng.normal(0, 0.01, len(DAYS)), index=DAYS)
    e = pd.Series(rng.normal(0, 0.005, len(DAYS))).rolling(10, min_periods=1).mean().to_numpy()
    r = pd.Series(0.0001 + e, index=DAYS)
    assert abs(alpha_beta(r, m, lags=0)["alpha_t"]) > abs(alpha_beta(r, m, lags=20)["alpha_t"])


def fake_preds():
    """Two stocks, three models: always long, a market-neutral coin flip, and a
    model that knows tomorrow's direction (an upper bound, not a real model)."""
    rng = np.random.default_rng(3)
    frames = []
    for t in ["AAA", "BBB"]:
        fwd = rng.normal(0.0004, 0.015, len(DAYS))
        for model, pred in [("always_long", np.ones(len(DAYS), int)),
                            ("coin", rng.integers(0, 2, len(DAYS))),
                            ("oracle", (fwd > 0).astype(int))]:
            frames.append(pd.DataFrame({"ticker": t, "model": model, "pred": pred, "fwd_return": fwd},
                                       index=pd.Index(DAYS, name="date")))
    return pd.concat(frames)


def test_exposure_table():
    preds = fake_preds()
    m = market_returns(preds)
    assert len(m) == len(DAYS)
    t = exposure_table(preds, strategies=["long_short", "long_flat"], cost_bps=0)
    ls = t[t["strategy"] == "long_short"].set_index("model")
    assert ls.loc["always_long", "beta"] == pytest.approx(1.0) and ls.loc["always_long", "net_exposure"] == 1
    assert abs(ls.loc["always_long", "alpha_ann"]) < 1e-12
    assert abs(ls.loc["coin", "beta"]) < 0.1 and abs(ls.loc["coin", "net_exposure"]) < 0.05
    assert ls.loc["oracle", "alpha_t"] > 10
    lf = t[t["strategy"] == "long_flat"].set_index("model")
    assert lf.loc["coin", "beta"] == pytest.approx(0.5, abs=0.1)        # long half the time
