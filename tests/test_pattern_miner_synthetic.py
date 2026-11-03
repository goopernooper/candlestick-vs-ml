"""Synthetic checks for the pattern miner: does it find a pattern that is really
there, and stay quiet when there is nothing to find?

Data are sized like the first walk-forward training window (22 stocks x ~3 years),
and the miner runs with its config.yaml settings (shrinkage m and min_count), so
these tests check the model exactly as it is used in the experiments.

1. Planted pattern: random symbols, except that a fixed 3-day sequence is
   planted about once every 100 days, and the day after it is up 70% of the
   time instead of the usual 52%. The miner should find that context, rank it
   first, and predict up whenever it appears in new data.
2. Pure noise: symbols and labels independent. Every estimate should stay close
   to the base rate, out-of-sample MCC should be about zero, and the miner should
   not fit the training data much better than new data. Without shrinkage and
   min_count (the raw miner) it overfits badly, which is why they exist.
3. End to end: the pattern is planted in candle prices (a big down day followed
   by a doji with a long lower wick), then goes through the real encoding,
   labels and modelling table. The miner should find it there too, and lose it
   when the labels are deliberately shifted by one day, which shows the test
   would catch a misalignment between features and labels.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import build_table  # noqa: E402
from evaluate import classification_metrics  # noqa: E402
from features import symbol_label, symbol_window_columns  # noqa: E402
from models.pattern import PatternMiner  # noqa: E402

PLANTED = (2, 2, 9)          # oldest first: "-- u l", "-- u l", "0 u L"
STOCKS, DAYS = 22, 750       # about the first training window (2002-2004)
BASE_UP, PLANTED_UP = 0.52, 0.70


def synthetic_symbols(seed, plant_rate=0.01, planted_up=PLANTED_UP):
    """Pooled symbol windows for STOCKS stocks of DAYS days each.

    Symbols are uniform over the 20 symbols. With probability plant_rate per
    day, the next three days are overwritten with PLANTED; the label on the
    third day (whether the following day is up) is drawn with planted_up,
    every other label with BASE_UP. Lags are built per stock, never across.
    Returns X (b_symbol windows), y, and a flag marking the planted days.
    """
    rng = np.random.default_rng(seed)
    frames, ys, flags = [], [], []
    for _ in range(STOCKS):
        sym = rng.integers(0, 20, DAYS)
        planted = np.zeros(DAYS, dtype=bool)
        t = 0
        while t < DAYS - 3:
            if plant_rate and rng.random() < plant_rate:
                sym[t:t + 3] = PLANTED
                planted[t + 2] = True
                t += 3
            else:
                t += 1
        y = (rng.random(DAYS) < np.where(planted, planted_up, BASE_UP)).astype(int)
        s = pd.Series(pd.array(sym, dtype="Int8"))
        frames.append(pd.DataFrame({"b_symbol": s, "b_symbol_lag1": s.shift(1),
                                    "b_symbol_lag2": s.shift(2)}))
        ys.append(y)
        flags.append(planted)
    return pd.concat(frames, ignore_index=True), np.concatenate(ys), np.concatenate(flags)


def all_estimates(miner):
    """Every kept context's estimate, across lengths: {context tuple: p_up}."""
    out = {}
    for table in miner.tables_.values():
        for key, p in table.items():
            out[key if isinstance(key, tuple) else (key,)] = p
    return out


# ---------------------------------------------------------------- planted pattern
@pytest.fixture(scope="module")
def planted():
    X, y, flag = synthetic_symbols(seed=0)
    return PatternMiner().fit(X, y), X, y, flag


def test_planted_pattern_is_frequent_enough_to_find(planted):
    _, _, y, flag = planted
    assert flag.sum() >= 100                                   # about 1% of 16,500 days
    assert abs(y[flag].mean() - PLANTED_UP) < 0.1              # the planted edge is really there


def test_miner_finds_the_planted_pattern(planted):
    miner, *_ = planted
    assert PLANTED in miner.tables_[3].index                   # kept: seen >= min_count times
    p = miner.tables_[3].loc[PLANTED]
    # shrinkage pulls the estimate part of the way back toward the base rate,
    # so it lands between the base rate and the true 70%
    assert miner.base_rate_ + 0.05 < p < PLANTED_UP
    # The only 3-day context common enough to keep, and near the top of all
    # ~400 kept contexts. (Not always first: with ~140 occurrences the observed
    # up rate itself varies from about 62% to 78% between samples, and a few
    # noise contexts reach 0.58 after shrinkage.)
    assert miner.tables_[3].idxmax() == PLANTED
    estimates = all_estimates(miner)
    assert len(estimates) > 300
    assert sorted(estimates, key=estimates.get, reverse=True).index(PLANTED) < 5


def test_planted_pattern_predicts_up_in_new_data(planted):
    miner, *_ = planted
    X_new, y_new, flag_new = synthetic_symbols(seed=100)       # a fresh "test year"
    e = miner.explain(X_new)
    assert (e.loc[flag_new, "context_length"] == 3).all()      # read from the full 3-day context
    assert (e.loc[flag_new, "p_up"] >= 0.5).all()              # always a prediction of up
    hit_rate = y_new[flag_new].mean()                          # accuracy on those days
    assert hit_rate > 0.6
    # Everywhere else the miner stays near the base rate. (Days that end with the
    # planted pattern's last two symbols by chance are left out: they legitimately
    # use the 2-day context, which carries part of the edge.)
    last_two = list(zip(X_new["b_symbol_lag1"].fillna(-1).astype(int), X_new["b_symbol"].astype(int)))
    other = ~flag_new & np.array([k != PLANTED[1:] for k in last_two])
    assert (e.loc[other, "p_up"] - miner.base_rate_).abs().max() < 0.1


# -------------------------------------------------------------------- pure noise
@pytest.fixture(scope="module")
def noise():
    X, y, _ = synthetic_symbols(seed=1, plant_rate=0)
    X_new, y_new, _ = synthetic_symbols(seed=101, plant_rate=0)
    return X, y, X_new, y_new


def test_noise_gives_estimates_near_the_base_rate(noise):
    X, y, X_new, y_new = noise
    miner = PatternMiner().fit(X, y)
    p = miner.predict_proba(X_new)
    assert np.abs(p - miner.base_rate_).max() < 0.1
    assert abs(classification_metrics(y_new, (p >= 0.5).astype(int))["mcc"]) < 0.025


def test_shrinkage_and_min_count_prevent_overfitting(noise):
    """Training accuracy minus accuracy on new data: how much the miner has
    memorised. On noise, any gap is pure overfitting."""
    X, y, X_new, y_new = noise

    def gap(miner):
        miner.fit(X, y)
        return (miner.predict(X) == y).mean() - (miner.predict(X_new) == y_new).mean()

    raw = PatternMiner(m=0, min_count=1)                        # raw rates, every context
    assert np.abs(raw.fit(X, y).predict_proba(X_new) - raw.base_rate_).max() > 0.4
    assert gap(raw) > 0.2                                       # memorises rare contexts
    assert gap(PatternMiner()) < 0.06                           # config settings: little to memorise


# -------------------------------------------------------------------- end to end
CAL = pd.bdate_range("2001-10-01", "2004-12-31")


def planted_candles(seed, plant_rate=0.03):
    """Synthetic daily prices with a planted two-day candle pattern.

    Ordinary days: close-to-close return ~ N(0, 1%), small random wicks.
    Planted: day 1 falls 3% with no wicks (symbol "-- u l"); day 2 opens and
    closes at the previous close with a 1.5% lower wick (symbol "0 u L"); the
    next day closes up 1% with probability PLANTED_UP, otherwise down 1%.
    Returns the prices and a flag on each planted day 2.
    """
    rng = np.random.default_rng(seed)
    n = len(CAL)
    plan, t = {}, 40                      # leave room for ATR to warm up
    while t < n - 3:
        if rng.random() < plant_rate:
            plan.update({t: "drop", t + 1: "doji", t + 2: "after"})
            t += 3
        else:
            t += 1
    o, h, low, c = (np.empty(n) for _ in range(4))
    prev = 100.0
    for t in range(n):
        kind = plan.get(t)
        if kind == "drop":
            o[t], c[t] = prev, prev * 0.97
            h[t], low[t] = o[t], c[t]
        elif kind == "doji":
            o[t] = c[t] = h[t] = prev
            low[t] = prev * 0.985
        else:
            r = (0.01 if rng.random() < PLANTED_UP else -0.01) if kind == "after" else rng.normal(0, 0.01)
            o[t], c[t] = prev * np.exp(rng.normal(0, 0.003)), prev * np.exp(r)
            h[t] = max(o[t], c[t]) * (1 + abs(rng.normal(0, 0.004)))
            low[t] = min(o[t], c[t]) * (1 - abs(rng.normal(0, 0.004)))
        prev = c[t]
    flag = pd.Series([plan.get(t) == "doji" for t in range(n)], index=CAL)
    return pd.DataFrame({"Open": o, "High": h, "Low": low, "Close": c}, index=CAL), flag


@pytest.fixture(scope="module")
def end_to_end():
    tickers = [f"S{i:02d}" for i in range(10)]
    data = {t: planted_candles(seed=i) for i, t in enumerate(tickers)}
    table = build_table(tickers, ["encoding_b_symbol_window"], lambda t: data[t][0])
    flags = pd.concat([data[t][1].rename("planted").to_frame().assign(ticker=t) for t in tickers])
    table = (table.reset_index()
             .merge(flags.rename_axis("date").reset_index(), on=["date", "ticker"], how="left"))
    return table


def fit_on(table, label="up"):
    train = table[table[label].notna()]
    return PatternMiner().fit(train[symbol_window_columns()], train[label].astype(int))


def test_planted_candles_get_the_expected_symbols(end_to_end):
    t = end_to_end
    pairs = t.loc[t["planted"], ["b_symbol_lag1", "b_symbol"]].astype(int)
    assert len(pairs) > 150
    assert (pairs["b_symbol_lag1"] == 0).all() and (pairs["b_symbol"] == 9).all()
    assert (symbol_label(0), symbol_label(9)) == ("-- u l", "0 u L")


def test_miner_finds_planted_candles_through_the_full_pipeline(end_to_end):
    miner = fit_on(end_to_end)
    two_day = miner.tables_[2]
    assert two_day.idxmax() == (0, 9)                           # top 2-day context
    assert two_day.max() > miner.base_rate_ + 0.08


def test_misaligned_labels_lose_the_pattern(end_to_end):
    """Negative control: give each day the next day's label (within each stock).
    The planted edge should vanish, so an off-by-one between features and
    labels anywhere in the pipeline would make the previous test fail."""
    t = end_to_end.sort_values(["ticker", "date"]).copy()
    t["up_next"] = t.groupby("ticker")["up"].shift(-1)
    miner = fit_on(t, label="up_next")
    p = miner.tables_[2].get((0, 9), miner.base_rate_)
    assert abs(p - miner.base_rate_) < 0.05
