"""Tests for the pattern miner. This part: counting contexts of consecutive symbols."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import build_table  # noqa: E402
from features import encoding_b_symbol_window, symbol_window_columns  # noqa: E402
from models.pattern import context_columns, context_table, count_contexts  # noqa: E402


def symbols_frame(seq, ups):
    """One stock's symbol sequence turned into the b_symbol window columns."""
    s = pd.Series(pd.array(seq, dtype="Int8"))
    X = pd.DataFrame({"b_symbol": s, "b_symbol_lag1": s.shift(1), "b_symbol_lag2": s.shift(2)})
    return X, np.asarray(ups)


def test_context_columns_are_oldest_first():
    assert context_columns(1) == ["b_symbol"]
    assert context_columns(3) == ["b_symbol_lag2", "b_symbol_lag1", "b_symbol"]
    assert symbol_window_columns() == ["b_symbol", "b_symbol_lag1", "b_symbol_lag2"]


def test_counts_by_hand():
    # day:      0  1  2  3  4  5
    # symbol:   4  8  4  8  4  8
    # next up:  1  0  1  1  0  1
    X, y = symbols_frame([4, 8, 4, 8, 4, 8], [1, 0, 1, 1, 0, 1])
    c = count_contexts(X, y, max_len=3)
    one, two, three = c[1], c[2], c[3]
    assert one.loc[4].tolist() == [3, 2] and one.loc[8].tolist() == [3, 2]       # [n, ups]
    # length 2 starts on day 1: (4,8) on days 1,3,5 -> ups 0,1,1 ; (8,4) on days 2,4 -> ups 1,0
    assert two.loc[(4, 8)].tolist() == [3, 2] and two.loc[(8, 4)].tolist() == [2, 1]
    # length 3 starts on day 2: (4,8,4) on days 2,4 -> ups 1,0 ; (8,4,8) on days 3,5 -> ups 1,1
    assert three.loc[(4, 8, 4)].tolist() == [2, 1] and three.loc[(8, 4, 8)].tolist() == [2, 2]


def test_order_matters():
    """(yesterday 4, today 8) and (yesterday 8, today 4) are different contexts."""
    X, y = symbols_frame([4, 8, 4, 8], [1, 1, 0, 0])
    two = count_contexts(X, y, max_len=2)[2]
    assert two.loc[(4, 8)].tolist() == [2, 1] and two.loc[(8, 4)].tolist() == [1, 0]


def test_totals_and_missing_symbols():
    rng = np.random.default_rng(0)
    X, y = symbols_frame(rng.integers(0, 20, 500), rng.integers(0, 2, 500))
    X.loc[100, "b_symbol_lag2"] = pd.NA            # missing only for length 3
    c = count_contexts(X, y)
    assert c[1]["n"].sum() == 500
    assert c[2]["n"].sum() == 499                  # the first day has no yesterday
    assert c[3]["n"].sum() == 497                  # first two days, and day 100
    assert (c[1]["ups"].sum(), c[2]["ups"].sum()) == (y.sum(), y[1:].sum())
    assert len(c[1]) <= 20 and len(c[2]) <= 400 and len(c[3]) <= 8000


def test_mismatched_lengths_are_an_error():
    X, y = symbols_frame([1, 2, 3], [1, 0, 1])
    with pytest.raises(ValueError):
        count_contexts(X, y[:2])


def test_context_table_is_readable():
    X, y = symbols_frame([4, 8, 4, 8, 4, 8], [1, 0, 1, 1, 0, 1])
    t = context_table(count_contexts(X, y))
    row = t[(t["length"] == 2) & (t["symbols"] == "4 8")].iloc[0]
    assert row["context"] == "- u l  >  0 u l" and row["n"] == 3 and row["up_rate"] == pytest.approx(2 / 3)


def test_symbol_window_never_mixes_stocks():
    cal = pd.bdate_range("2001-10-01", "2004-12-31")
    def loader(t):
        rng = np.random.default_rng({"AAA": 0, "BBB": 1}[t])
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(cal))))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, len(cal)))
        return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.004,
                             "Low": np.minimum(o, c) * 0.996, "Close": c}, index=cal)
    table = build_table(["AAA", "BBB"], ["encoding_b_symbol_window"], loader)
    for t in ["AAA", "BBB"]:
        own = encoding_b_symbol_window(loader(t)).loc[table.index.min():]
        mine = table[table["ticker"] == t][symbol_window_columns()]
        pd.testing.assert_frame_equal(mine, own.loc[mine.index], check_names=False)


# ------------------------------------------------------------------ shrinkage
from models.pattern import shrink  # noqa: E402


def counts_for(n, ups):
    return {1: pd.DataFrame({"n": n, "ups": ups}, index=pd.Index(range(len(n)), name="b_symbol"))}


def test_shrinkage_by_hand():
    c = shrink(counts_for([3, 300], [3, 180]), base_rate=0.5, m=100)[1]
    assert c.loc[0, "p_up"] == pytest.approx((3 + 50) / 103)        # 3 of 3 up -> only 0.515
    assert c.loc[1, "p_up"] == pytest.approx((180 + 50) / 400)      # 60% raw -> 0.575


def test_shrinkage_extremes():
    c = counts_for([10, 40], [9, 10])
    raw = shrink(c, 0.52, m=0)[1]["p_up"]
    assert raw.tolist() == pytest.approx([0.9, 0.25])                # m = 0: raw rate
    huge = shrink(c, 0.52, m=1e9)[1]["p_up"]
    assert huge.tolist() == pytest.approx([0.52, 0.52], abs=1e-6)    # huge m: base rate


def test_estimate_lies_between_raw_rate_and_base_rate():
    rng = np.random.default_rng(0)
    n = rng.integers(1, 500, 200)
    ups = rng.binomial(n, 0.5)
    p = shrink(counts_for(n, ups), base_rate=0.52, m=100)[1]["p_up"].to_numpy()
    raw = ups / n
    assert ((p >= np.minimum(raw, 0.52) - 1e-12) & (p <= np.maximum(raw, 0.52) + 1e-12)).all()


def test_rare_contexts_are_pulled_harder():
    """Same raw rate (80%), seen 5 times vs 5,000 times."""
    p = shrink(counts_for([5, 5000], [4, 4000]), base_rate=0.5, m=100)[1]["p_up"]
    assert abs(p[0] - 0.5) < 0.02 and abs(p[1] - 0.8) < 0.01


def test_shrinkage_leaves_counts_alone_and_uses_config():
    from features import load_config
    c = counts_for([10], [7])
    out = shrink(c, 0.5)
    assert "p_up" not in c[1] and "p_up" in out[1]
    m = load_config()["models"]["pattern_miner"]["shrinkage_m"]
    assert m == 100 and out[1]["p_up"].iloc[0] == pytest.approx((7 + m * 0.5) / (10 + m))


def test_shrinkage_rejects_bad_inputs():
    with pytest.raises(ValueError):
        shrink(counts_for([1], [1]), 0.5, m=-1)
    with pytest.raises(ValueError):
        shrink(counts_for([1], [1]), 1.5, m=10)


def test_shrinkage_on_counted_contexts():
    X, y = symbols_frame([4, 8, 4, 8, 4, 8], [1, 0, 1, 1, 0, 1])
    c = shrink(count_contexts(X, y), base_rate=y.mean(), m=4)
    assert c[2].loc[(4, 8), "p_up"] == pytest.approx((2 + 4 * 4 / 6) / (3 + 4))
    assert "p_up" in context_table(c).columns


# --------------------------------------------------------- prediction with backoff
from models.pattern import PatternMiner  # noqa: E402


def row(sym, lag1=None, lag2=None):
    return pd.DataFrame({"b_symbol": pd.array([sym], dtype="Int8"),
                         "b_symbol_lag1": pd.array([lag1], dtype="Int8"),
                         "b_symbol_lag2": pd.array([lag2], dtype="Int8")})


@pytest.fixture
def trained():
    """Training sequence repeating 1, 2, 3 for 300 days (100 of each 3-day context
    (1,2,3), (2,3,1), (3,1,2)), plus symbol 9 seen just 5 times. min_count 50, m 0
    so the estimates are raw rates and easy to check by hand."""
    seq = [1, 2, 3] * 100 + [9] * 5
    ups = ([1, 0, 1] * 100) + [1] * 5
    X, y = symbols_frame(seq, ups)
    return PatternMiner(max_context=3, m=0, min_count=50).fit(X, y), X, y


def test_uses_longest_qualifying_context(trained):
    miner, _, _ = trained
    e = miner.explain(row(3, 2, 1))                # context (1, 2, 3): seen ~100 times
    assert e["context_length"].iloc[0] == 3 and e["p_up"].iloc[0] == pytest.approx(1.0)


def test_backs_off_when_long_context_is_unseen(trained):
    miner, _, _ = trained
    e = miner.explain(row(3, 2, 7))                # (7, 2, 3) never seen; (2, 3) seen ~100 times
    assert e["context_length"].iloc[0] == 2 and e["p_up"].iloc[0] == pytest.approx(1.0)
    e = miner.explain(row(3, 7, 7))                # only symbol 3 itself is known
    assert e["context_length"].iloc[0] == 1


def test_backs_off_past_rare_contexts(trained):
    miner, _, _ = trained
    e = miner.explain(row(9, 9, 9))                # seen only 5 times, below min_count 50
    assert e["context_length"].iloc[0] == 0
    assert e["p_up"].iloc[0] == pytest.approx(miner.base_rate_)


def test_missing_lags_use_shorter_contexts(trained):
    miner, _, _ = trained
    e = miner.explain(row(3, 2, None))             # two days ago unknown
    assert e["context_length"].iloc[0] == 2
    e = miner.explain(row(3, None, None))
    assert e["context_length"].iloc[0] == 1


def test_unknown_symbol_gets_base_rate(trained):
    miner, _, _ = trained
    e = miner.explain(row(17, 17, 17))
    assert e["context_length"].iloc[0] == 0 and e["p_up"].iloc[0] == pytest.approx(miner.base_rate_)


def test_predictions_follow_the_shared_interface(trained):
    miner, X, _ = trained
    p = miner.predict_proba(X)
    assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all()
    assert set(miner.predict(X)) <= {0, 1}


def test_shrinkage_and_backoff_work_together():
    """With shrinkage on, a qualifying context's estimate matches shrink() exactly."""
    rng = np.random.default_rng(3)
    X, y = symbols_frame(rng.integers(0, 20, 3000), rng.integers(0, 2, 3000))
    miner = PatternMiner(m=100, min_count=10).fit(X, y)
    expected = shrink(count_contexts(X, y), y.mean(), 100)
    e = miner.explain(X)
    for i in [100, 500, 2000]:
        L = e["context_length"].iloc[i]
        key = tuple(int(X.iloc[i][c]) for c in context_columns(L))
        key = key if L > 1 else key[0]
        assert e["p_up"].iloc[i] == pytest.approx(expected[L].loc[key, "p_up"])


def test_config_defaults():
    from features import load_config
    cfg = load_config()["models"]["pattern_miner"]
    miner = PatternMiner()
    assert (miner.max_context, miner.m, miner.min_count) == (3, cfg["shrinkage_m"], cfg["min_count"])
    assert cfg["min_count"] == 30
