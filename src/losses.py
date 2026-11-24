"""Daily loss series: the input to the statistical tests (ANALYSIS_PLAN.md section 8).

Usage
-----
    python src/losses.py v4        # from results/v4/predictions.parquet

run.py also writes these for every new run.

The 22 stocks move together, so their daily results are not independent
observations. The tests therefore work on one number per model per day,
averaged across stocks first (DECISIONS.md D-005):

  classification loss   balanced misclassification: that day's labelled
                        stock-days the model got wrong, each weighted so that
                        up days and down days count equally (DECISIONS.md
                        D-027). A wrong call on an up day weighs 1 / (2 pi) and
                        on a down day 1 / (2 (1 - pi)), where pi is the share of
                        up days among the labelled stock-days of that test year.
                        The daily loss is the weighted errors divided by the
                        number of labelled stock-days. A constant prediction
                        (always up, or always down) averages about 0.5, as its
                        MCC is 0; it is the day-by-day form of 1 - balanced
                        accuracy. Using the test year's own share is not
                        look-ahead: it is part of scoring, as in balanced
                        accuracy, and no model sees it.
  misclassification     the plain share of that day's labelled stock-days the
                        model got wrong, for reference only: up days are about
                        52% of labelled days, so it rewards predicting "up"
                        more often, the same bias as accuracy.
  economic loss         minus the equal-weight long/short portfolio's net return
                        that day, at the headline cost (5 bps per side): a
                        loss of 0.01 means the strategy lost 1% that day.

Lower is better for both. A difference between two models' series, day by
day, is what the Diebold-Mariano tests compare, and the Model Confidence Set
and SPA test use the whole table.

Days on which every stock is in the dead zone have no labelled stock-day and
are left empty in the classification tables.

Writes results/<version>/losses_classification.csv (balanced; used by the
tests), losses_misclassification.csv and losses_economic.csv: one row per test
day, one column per model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import portfolio_returns, stock_returns  # noqa: E402
from features import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def class_weights(preds: pd.DataFrame) -> pd.Series:
    """Per labelled row: 1 / (2 pi) for an up day, 1 / (2 (1 - pi)) for a down
    day, pi being the share of up days in that row's test year. The year's
    labels are the same for every model, so the weights are too."""
    labelled = preds[preds["up"].notna()]
    up = labelled["up"].astype(int)
    one = labelled[labelled["model"] == labelled["model"].iloc[0]]
    share = one["up"].astype(int).groupby(one.index.year).mean()
    pi = pd.Series(labelled.index.year, index=labelled.index).map(share)
    if ((pi <= 0) | (pi >= 1)).any():
        raise ValueError("a test year has only up days or only down days")
    return pd.Series(np.where(up == 1, 1 / (2 * pi), 1 / (2 * (1 - pi))), index=labelled.index)


def classification_losses(preds: pd.DataFrame, balanced: bool = True) -> pd.DataFrame:
    """Date x model: balanced misclassification that day (see the module
    docstring), or with balanced=False the plain share misclassified."""
    labelled = preds[preds["up"].notna()]
    wrong = (labelled["pred"].astype(int) != labelled["up"].astype(int)).astype(float)
    if balanced:
        wrong = wrong * class_weights(preds).to_numpy()
    table = (wrong.groupby([labelled.index, labelled["model"]]).mean()
             .unstack("model").sort_index())
    days = pd.Index(sorted(preds.index.unique()), name="date")     # keep all-dead-zone days, empty
    table = table.reindex(days)
    table.index.name, table.columns.name = "date", None
    return table


def economic_losses(preds: pd.DataFrame, strategy: str = "long_short",
                    cost_bps: float | None = None) -> pd.DataFrame:
    """Date x model: minus the portfolio's net return that day."""
    cost_bps = load_config()["backtest"]["cost_bps_per_side"] if cost_bps is None else cost_bps
    cols = {model: -portfolio_returns(stock_returns(g, strategy, cost_bps))
            for model, g in preds.groupby("model", sort=True)}
    table = pd.DataFrame(cols).sort_index()
    table.index.name = "date"
    return table


def write_losses(preds: pd.DataFrame, folder: Path, cost_bps: float | None = None) -> list[str]:
    classification_losses(preds).to_csv(folder / "losses_classification.csv")
    classification_losses(preds, balanced=False).to_csv(folder / "losses_misclassification.csv")
    economic_losses(preds, cost_bps=cost_bps).to_csv(folder / "losses_economic.csv")
    return ["losses_classification.csv", "losses_misclassification.csv", "losses_economic.csv"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="results version, e.g. v4")
    args = parser.parse_args()
    folder = ROOT / "results" / args.version
    path = folder / "predictions.parquet"
    if not path.exists():
        print(f"{path} not found (predictions aren't committed; they exist on the computer that ran "
              f"`python src/run.py --version {args.version}`).")
        return 1
    preds = pd.read_parquet(path)
    names = write_losses(preds, folder)
    c, m, e = (pd.read_csv(folder / n, index_col=0) for n in names)
    summary = pd.DataFrame({"balanced_loss": c.mean(), "plain_misclassification": m.mean(),
                            "economic_loss_bps": e.mean() * 1e4})
    print(f"{len(c)} test days, {c.shape[1]} models. Mean daily losses (lower is better; the tests use "
          "balanced_loss and economic_loss_bps):\n")
    print(summary.sort_values("balanced_loss").to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote results/{args.version}/: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
