"""The look-ahead test for every feature set.

For a cut-off day T, rewrite the prices after T and check that no feature dated T
or earlier changes. Four kinds of rewrite are used:

  scaled     every later price multiplied by 1.7
  noise      every later price multiplied by its own random factor
  reversed   the later days replayed in reverse order
  truncated  the later days deleted entirely, as if computed live on day T

"Truncated" is the strictest: it catches anything that uses the whole series,
such as a mean or quantile over all dates, or a centred rolling window.

Every function in features.FEATURE_SETS is checked, on synthetic data and, if
the snapshot is present, on real stocks around 2008 and 2020. Labels look ahead
by design, so they get their own check: exactly `horizon` days, no more.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features  # noqa: E402
from features import FEATURE_SETS  # noqa: E402
from labels import make_labels  # noqa: E402

PRICES = ["Open", "High", "Low", "Close"]
REWRITES = ["scaled", "noise", "reversed", "truncated"]


# ----------------------------------------------------------------- helpers

def synthetic(n=500, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.exponential(0.006, n))
    low = np.minimum(open_, close) * (1 - rng.exponential(0.006, n))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=pd.bdate_range("2002-01-02", periods=n))


def rewrite_future(df, t, how):
    """Copy of df with every row after position t rewritten (or removed)."""
    if how == "truncated":
        return df.iloc[:t + 1].copy()
    out = df.copy()
    future = out.iloc[t + 1:][PRICES].to_numpy()
    if how == "scaled":
        future = future * 1.7
    elif how == "noise":
        future = future * np.random.default_rng(t).uniform(0.6, 1.6, size=future.shape)
    elif how == "reversed":
        future = future[::-1]
    # Keep each rewritten candle valid (High highest, Low lowest). Only the rewritten
    # rows are touched: real data has tiny rounding errors in past candles, and
    # "repairing" those would change the past and look like a leak.
    o, h, l, c = future.T
    future = np.column_stack([o, np.maximum.reduce([o, h, l, c]), np.minimum.reduce([o, h, l, c]), c])
    out.iloc[t + 1:, [out.columns.get_loc(col) for col in PRICES]] = future
    return out


def check_no_lookahead(func, df, t, how):
    """Raise AssertionError if func's output for days <= t depends on days after t."""
    original = func(df)
    changed = func(rewrite_future(df, t, how))
    pd.testing.assert_frame_equal(original.iloc[:t + 1], changed.iloc[:t + 1], check_exact=True)
    return original, changed


def cut_points(n):
    # just after ATR warms up, early, middle, and the second-to-last day
    return [16, 60, n // 2, n - 2]


# ------------------------------------------------- every feature, synthetic

@pytest.mark.parametrize("how", REWRITES)
@pytest.mark.parametrize("name", sorted(FEATURE_SETS))
def test_no_feature_looks_ahead(name, how):
    df = synthetic()
    for t in cut_points(len(df)):
        check_no_lookahead(FEATURE_SETS[name], df, t, how)


@pytest.mark.parametrize("name", sorted(FEATURE_SETS))
def test_rewriting_the_future_does_change_later_features(name):
    """Guards against a test that can't fail: later rows must actually respond."""
    df, t = synthetic(), 250
    original, changed = check_no_lookahead(FEATURE_SETS[name], df, t, "noise")
    assert not original.iloc[t + 1:].equals(changed.iloc[t + 1:])


# ---------------------------------------------------- every feature, real data

REAL_CASES = [("AAPL", "2008-09-12"), ("JPM", "2008-09-12"), ("BRK-B", "2008-10-09"),
              ("XOM", "2020-03-06"), ("NVDA", "2023-05-24"), ("SO", "2002-03-01")]


@pytest.mark.parametrize("ticker,day", REAL_CASES)
def test_no_feature_looks_ahead_on_real_stocks(ticker, day):
    import data
    if not data.raw_path(ticker).exists():
        pytest.skip(f"{ticker} not in data/raw (snapshot not present)")
    df = data.load_raw(ticker)[PRICES]
    t = int(df.index.get_indexer([pd.Timestamp(day)])[0])
    assert t >= 0, f"{day} is not a trading day for {ticker}"
    for name, func in FEATURE_SETS.items():
        for how in REWRITES:
            try:
                check_no_lookahead(func, df, t, how)
            except AssertionError as err:
                raise AssertionError(f"{name} looks ahead ({how}, {ticker} {day})") from err


# ------------------------------------------- the check itself catches leaks

LEAKY = {
    "tomorrow's close": lambda df: (df["Close"].shift(-1) / df["Close"]).to_frame("x"),
    "whole-period mean": lambda df: (df["Close"] / df["Close"].mean()).to_frame("x"),
    "centred window": lambda df: df["Close"].rolling(5, center=True).mean().to_frame("x"),
    "full-sample quantile bins": lambda df: pd.qcut(df["Close"], 5, labels=False).to_frame("x"),
}


@pytest.mark.parametrize("leak", sorted(LEAKY))
def test_check_catches_a_deliberate_leak(leak):
    df = synthetic()
    caught = False
    for how in REWRITES:
        for t in cut_points(len(df)):
            try:
                check_no_lookahead(LEAKY[leak], df, t, how)
            except AssertionError:
                caught = True
    assert caught, f"the leakage check missed a known leak: {leak}"


# ------------------------------------------- nothing can skip the test

def test_every_feature_function_is_registered():
    """A new encoding must be added to FEATURE_SETS, or this fails."""
    prefixes = ("encoding_", "normalized_", "lagged_")
    found = {name for name, obj in inspect.getmembers(features, inspect.isfunction)
             if name.startswith(prefixes) and obj.__module__ == features.__name__}
    missing = found - set(FEATURE_SETS)
    assert not missing, f"add these to FEATURE_SETS in features.py: {sorted(missing)}"


# ------------------------------------- labels: look ahead exactly `horizon` days

@pytest.mark.parametrize("how", REWRITES)
@pytest.mark.parametrize("horizon", [1, 5])
def test_labels_look_ahead_exactly_horizon_days(horizon, how):
    df = synthetic()
    func = lambda d: make_labels(d, horizon=horizon)  # noqa: E731
    for t in cut_points(len(df)):
        # Labels dated t - horizon or earlier only need prices up to day t.
        safe = t - horizon
        original, changed = func(df), func(rewrite_future(df, t, how))
        pd.testing.assert_frame_equal(original.iloc[:safe + 1], changed.iloc[:safe + 1], check_exact=True)
