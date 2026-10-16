"""Walk-forward splits with a purge gap, and the locked final holdout.

Usage
-----
    python src/splits.py      # print every fold

Each fold tests on one calendar year (2005, 2006, ..., 2022). Its training set
is every trading day from 2002-01-02 up to the start of that year, minus a
purge gap: the last `horizon` trading days before the test year are dropped,
because their labels are computed from prices inside the test year.

Splits are by calendar date, never by row number, so they stay correct when
all stocks are stacked into one table: every stock's rows for a given date
land in the same part.

2023-2025 is the final holdout. While config.yaml has unlock_holdout: false,
no fold tests on those years, and check_not_holdout() raises an error if any
code tries to use them.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_config  # noqa: E402


@dataclass(frozen=True)
class Fold:
    test_year: int
    train: pd.DatetimeIndex   # trading days the models learn from
    purged: pd.DatetimeIndex  # dropped: their labels reach into the test year
    test: pd.DatetimeIndex    # trading days the models are scored on

    def select(self, frame: pd.DataFrame, part: str) -> pd.DataFrame:
        """Rows of `frame` (indexed by date, dates may repeat across stocks)
        that belong to `part`: "train", "purged" or "test"."""
        return frame[frame.index.isin(getattr(self, part))]

    def describe(self) -> dict:
        return {"test_year": self.test_year,
                "train_from": self.train[0].date(), "train_to": self.train[-1].date(),
                "purged": ", ".join(str(d.date()) for d in self.purged),
                "test_from": self.test[0].date(), "test_to": self.test[-1].date(),
                "train_days": len(self.train), "test_days": len(self.test)}


def holdout_locked() -> bool:
    return not load_config()["splits"]["unlock_holdout"]


def holdout_start() -> pd.Timestamp:
    return pd.Timestamp(f"{load_config()['splits']['holdout_years'][0]}-01-01")


def check_not_holdout(dates) -> None:
    """Raise if any date is in the locked holdout. Call this anywhere results are computed."""
    if not holdout_locked():
        return
    dates = pd.DatetimeIndex(dates)
    if len(dates) and dates.max() >= holdout_start():
        raise PermissionError(
            f"Dates from {holdout_start().date()} onward are the locked holdout. "
            "They can only be used after ANALYSIS_PLAN.md is frozen (see config.yaml).")


def walk_forward(calendar, horizon: int | None = None, unlock_holdout: bool | None = None) -> list[Fold]:
    """Build every fold from a calendar of trading dates (for example SPY's)."""
    cfg = load_config()
    s = cfg["splits"]
    horizon = cfg["labels"]["horizon"] if horizon is None else horizon
    unlock = s["unlock_holdout"] if unlock_holdout is None else unlock_holdout
    if horizon < 1:
        raise ValueError("horizon must be at least 1 trading day")

    cal = pd.DatetimeIndex(calendar).unique().sort_values()
    cal = cal[cal >= pd.Timestamp(s["train_start"])]
    years = list(range(s["first_test_year"], s["last_test_year"] + 1))
    if unlock:
        first, last = s["holdout_years"]
        years += list(range(first, last + 1))

    folds = []
    for year in years:
        test = cal[cal.year == year]
        if len(test) == 0:
            raise ValueError(f"no trading days in {year} in the calendar")
        start = cal.get_loc(test[0])          # position of the first test day
        train = cal[:start - horizon]         # everything before, minus the purge gap
        purged = cal[start - horizon:start]   # labels of these days reach into the test year
        folds.append(Fold(year, train, purged, test))
    return folds


def trading_calendar() -> pd.DatetimeIndex:
    """SPY's trading days: the calendar every stock was validated against."""
    import data
    return data.load_raw("SPY").index


def main() -> int:
    folds = walk_forward(trading_calendar())
    cfg = load_config()
    print(f"{len(folds)} folds, horizon {cfg['labels']['horizon']} day(s), "
          f"holdout {'LOCKED' if holdout_locked() else 'UNLOCKED'} "
          f"({cfg['splits']['holdout_years'][0]}-{cfg['splits']['holdout_years'][1]}).\n")
    print(pd.DataFrame([f.describe() for f in folds]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
