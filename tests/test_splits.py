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


# ---------------------------------------------------------------- tuning sets

def test_tuning_year_is_the_last_training_year():
    for f in walk_forward(CAL):
        t = f.tuning()
        assert set(t.test.year) == {f.test_year - 1}


@pytest.mark.parametrize("horizon", [1, 5])
def test_tuning_split_partitions_the_training_window(horizon):
    for f in walk_forward(CAL, horizon=horizon):
        t = f.tuning()
        # fit, inner purge and tuning set together are exactly the training window, in order
        assert t.train.append(t.purged).append(t.test).equals(f.train)
        assert len(t.purged) == horizon
        assert t.train[-1] < t.purged[0] and t.purged[-1] < t.test[0]
        # the tuning set ends before the outer purge and the test year
        assert t.test[-1] < f.purged[0] < f.test[0]


@pytest.mark.parametrize("horizon", [1, 5])
def test_fit_labels_never_see_tuning_year_prices(horizon):
    """The same leak check as the outer split, one level down."""
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(CAL)))), index=CAL)
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close})
    for f in walk_forward(CAL, horizon=horizon)[::6]:
        t = f.tuning()
        future = df.copy()
        future.loc[t.test[0]:] *= 1.5
        pd.testing.assert_frame_equal(make_labels(df, horizon=horizon).loc[t.train],
                                      make_labels(future, horizon=horizon).loc[t.train])
        assert not make_labels(df, horizon=horizon).loc[t.purged].equals(
            make_labels(future, horizon=horizon).loc[t.purged])


def test_tuning_never_touches_the_test_year_or_holdout():
    for f in walk_forward(CAL, unlock_holdout=True):
        t = f.tuning()
        for part in (t.train, t.purged, t.test):
            assert part.max() < f.test[0]


def test_first_fold_still_has_data_to_fit():
    first = walk_forward(CAL)[0].tuning()
    assert set(first.train.year) == {2002, 2003} and set(first.test.year) == {2004}


def test_too_many_tuning_years_is_an_error():
    with pytest.raises(ValueError):
        walk_forward(CAL)[0].tuning(years=3)    # 2002-2004 would leave nothing to fit
