"""Build the pooled modelling table: every selected stock's features and labels, stacked.

One row per stock per day, indexed by date (dates repeat across stocks), with a
`ticker` column, the requested feature columns, and the label columns from
labels.make_labels(). The table starts on the first study day (2002-01-02);
features are computed on the full history first, so the 2001 warm-up is used.

While the holdout is locked, the table stops before it, and also drops the last
`horizon` trading days before it, because their labels use holdout prices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_SETS, load_config  # noqa: E402
from labels import make_labels  # noqa: E402
from splits import holdout_locked, holdout_start  # noqa: E402


def selected_stocks() -> list[str]:
    import data
    u = pd.read_csv(data.UNIVERSE_CSV)
    return u.loc[u["role"] == "selected", "ticker"].tolist()


def build_table(stocks=None, feature_sets=("normalized_ohlc",), loader=None) -> pd.DataFrame:
    if loader is None:
        import data
        loader = data.load_raw
    stocks = selected_stocks() if stocks is None else list(stocks)
    cfg = load_config()
    start = pd.Timestamp(cfg["splits"]["train_start"])

    frames = []
    for t in stocks:
        df = loader(t)
        parts = [FEATURE_SETS[name](df) for name in feature_sets] + [make_labels(df)]
        frame = pd.concat(parts, axis=1).loc[start:]
        frames.append(frame.assign(ticker=t))
    table = pd.concat(frames)
    table.index.name = "date"

    if holdout_locked():
        horizon = cfg["labels"]["horizon"]
        dates = table.index.unique().sort_values()
        before = dates[dates < holdout_start()]
        # the last `horizon` days before the holdout have labels built from holdout prices
        last_usable = before[-horizon - 1] if len(before) > horizon else before[0]
        table = table[table.index <= last_usable]
    return table.sort_index(kind="stable")
