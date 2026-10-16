"""Tests for the walk-forward splitter: fold layout, the purge gap, and the locked holdout."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from labels import make_labels  # noqa: E402
from splits import check_not_holdout, walk_forward  # noqa: E402

CAL = pd.bdate_range("2001-10-01", "2025-12-31")   # stand-in trading calendar


def test_config_settings():
    from features import load_config
    s = load_config()["splits"]
    assert s["unlock_holdout"] is False, "the holdout must stay locked until week 9"
    assert (s["first_test_year"], s["last_test_year"]) == (2005, 2022)


def test_one_fold_per_development_year():
    folds = walk_forward(CAL)
    assert [f.test_year for f in folds] == list(range(2005, 2023))


def test_holdout_never_appears_while_locked():
    for f in walk_forward(CAL):
        for part in (f.train, f.purged, f.test):
            assert part.max() < pd.Timestamp("2023-01-01")


def test_unlocking_adds_the_holdout_years():
    folds = walk_forward(CAL, unlock_holdout=True)
    assert [f.test_year for f in folds][-3:] == [2023, 2024, 2025]


def test_check_not_holdout_blocks_holdout_dates():
    check_not_holdout(pd.DatetimeIndex(["2022-12-30"]))            # fine
    with pytest.raises(PermissionError):
        check_not_holdout(pd.DatetimeIndex(["2022-12-30", "2023-01-03"]))


def test_expanding_window_from_2002():
    folds = walk_forward(CAL)
    assert all(f.train[0] == pd.Timestamp("2002-01-02") for f in folds)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_each_test_set_is_one_whole_year():
    for f in walk_forward(CAL):
        assert set(f.test.year) == {f.test_year}
        assert len(f.test) == (CAL.year == f.test_year).sum()


def test_every_development_day_is_tested_exactly_once():
    tested = np.concatenate([f.test.values for f in walk_forward(CAL)])
    dev = CAL[(CAL >= "2005-01-01") & (CAL < "2023-01-01")]
    assert len(tested) == len(np.unique(tested)) == len(dev)


@pytest.mark.parametrize("horizon", [1, 5])
def test_purge_gap_is_exactly_horizon_days(horizon):
    for f in walk_forward(CAL, horizon=horizon):
        assert len(f.purged) == horizon
        gap = CAL.get_loc(f.test[0]) - CAL.get_loc(f.train[-1])
        assert gap == horizon + 1                       # horizon purged days in between
        assert f.train[-1] < f.purged[0] <= f.purged[-1] < f.test[0]
        assert not set(f.train) & set(f.test)


@pytest.mark.parametrize("horizon", [1, 5])
def test_training_labels_never_see_test_year_prices(horizon):
    """Rewrite every price from the first test day on: no training label may change.
    Without the purge gap, the last `horizon` training labels would change."""
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(CAL)))), index=CAL)
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close})
    for f in walk_forward(CAL, horizon=horizon)[::6]:
        future = df.copy()
        future.loc[f.test[0]:] *= 1.5
        before = make_labels(df, horizon=horizon).loc[f.train]
        after = make_labels(future, horizon=horizon).loc[f.train]
        pd.testing.assert_frame_equal(before, after)
        # and the purged days really would have leaked:
        assert not make_labels(df, horizon=horizon).loc[f.purged].equals(
            make_labels(future, horizon=horizon).loc[f.purged])


def test_select_works_on_stacked_stocks():
    """With all stocks in one table, every stock's row for a date lands in the same part."""
    f = walk_forward(CAL)[0]
    one = pd.DataFrame({"x": 1.0}, index=CAL)
    stacked = pd.concat([one.assign(ticker=t) for t in ["AAA", "BBB", "CCC"]])
    tr, te = f.select(stacked, "train"), f.select(stacked, "test")
    assert len(tr) == 3 * len(f.train) and len(te) == 3 * len(f.test)
    assert tr.index.max() < te.index.min()
