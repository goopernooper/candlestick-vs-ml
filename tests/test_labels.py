"""Tests for the labels: values by hand, the dead zone, and exactly how far ahead they look."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from labels import LABEL_COLUMNS, make_labels  # noqa: E402


def random_ohlc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    idx = pd.bdate_range("2002-01-02", periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close},
                        index=idx)


def test_config_defaults():
    lab = make_labels(random_ohlc())
    assert list(lab.columns) == LABEL_COLUMNS
    explicit = make_labels(random_ohlc(), horizon=1, dead_zone_k=0.1, vol_window=20)
    pd.testing.assert_frame_equal(lab, explicit)


def test_forward_return_by_hand():
    df = random_ohlc()
    lab = make_labels(df)
    c = df["Close"]
    assert lab["fwd_return"].iloc[100] == pytest.approx(c.iloc[101] / c.iloc[100] - 1)
    assert np.isnan(lab["fwd_return"].iloc[-1])          # no next day yet
    assert pd.isna(lab["up"].iloc[-1])


def test_volatility_by_hand():
    df = random_ohlc()
    lab = make_labels(df)
    r = df["Close"].pct_change()
    assert lab["vol"].iloc[100] == pytest.approx(r.iloc[81:101].std())
    assert lab["vol"].iloc[:20].isna().all()             # needs 20 returns


def test_dead_zone_and_direction():
    lab = make_labels(random_ohlc()).dropna(subset=["fwd_return", "threshold"])
    small = lab["fwd_return"].abs() < lab["threshold"]
    assert (lab["dead_zone"] == small).all()
    assert lab.loc[small, "up"].isna().all()
    big = lab.loc[~small]
    assert (big["up"].astype(int) == (big["fwd_return"] > 0).astype(int)).all()


def test_handmade_dead_zone_day():
    df = random_ohlc()
    lab = make_labels(df)
    t = 200
    df.iloc[t + 1, df.columns.get_loc("Close")] = df["Close"].iloc[t] * (1 + 0.1 * lab["threshold"].iloc[t])
    lab2 = make_labels(df)
    assert bool(lab2["dead_zone"].iloc[t]) and pd.isna(lab2["up"].iloc[t])


def test_same_labels_at_any_price_level():
    df = random_ohlc()
    pd.testing.assert_frame_equal(make_labels(df).drop(columns="up"),
                                  make_labels(df * 50).drop(columns="up"))
    assert make_labels(df)["up"].equals(make_labels(df * 50)["up"])


@pytest.mark.parametrize("horizon", [1, 5])
def test_labels_look_exactly_horizon_days_ahead(horizon):
    """Change prices from day `cut` on. The label for day t uses Close[t + horizon], so
    labels up to day cut - horizon - 1 must be unchanged, and the label at day
    cut - horizon must change. This is why the splitter needs a purge gap of `horizon`."""
    df, cut = random_ohlc(), 250
    future = df.copy()
    future.iloc[cut:] *= 1.5
    before, after = make_labels(df, horizon=horizon), make_labels(future, horizon=horizon)
    last_safe = cut - horizon - 1
    pd.testing.assert_frame_equal(before.iloc[:last_safe + 1], after.iloc[:last_safe + 1])
    assert before["fwd_return"].iloc[last_safe + 1] != after["fwd_return"].iloc[last_safe + 1]


def test_volatility_uses_no_future_data():
    df, cut = random_ohlc(), 250
    future = df.copy()
    future.iloc[cut:] *= np.random.default_rng(1).uniform(0.5, 2.0, size=future.iloc[cut:].shape)
    pd.testing.assert_series_equal(make_labels(df)["vol"].iloc[:cut], make_labels(future)["vol"].iloc[:cut])
