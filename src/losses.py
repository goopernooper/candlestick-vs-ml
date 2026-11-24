"""Daily loss series: the input to the statistical tests (ANALYSIS_PLAN.md section 8).

Usage
-----
    python src/losses.py v4        # from results/v4/predictions.parquet

run.py also writes these for every new run.

The 22 stocks move together, so their daily results are not independent
observations. The tests therefore work on one number per model per day,
averaged across stocks first (DECISIONS.md D-005):

  classification loss   the share of that day's labelled stock-days the model
                        got wrong (0 = all right, 1 = all wrong). Days on which
                        every stock is in the dead zone have no labelled
                        stock-day and are left empty.
  economic loss         minus the equal-weight long/short portfolio's net return
                        that day, at the headline cost (5 bps per side): a
                        loss of 0.01 means the strategy lost 1% that day.

Lower is better for both. A difference between two models' series, day by
day, is what the Diebold-Mariano tests compare, and the Model Confidence Set
and SPA test use the whole table.

Writes results/<version>/losses_classification.csv and losses_economic.csv:
one row per test day, one column per model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import portfolio_returns, stock_returns  # noqa: E402
from features import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def classification_losses(preds: pd.DataFrame) -> pd.DataFrame:
    """Date x model: the share of labelled stock-days misclassified that day."""
    labelled = preds[preds["up"].notna()]
    wrong = (labelled["pred"].astype(int) != labelled["up"].astype(int)).astype(float)
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
    economic_losses(preds, cost_bps=cost_bps).to_csv(folder / "losses_economic.csv")
    return ["losses_classification.csv", "losses_economic.csv"]


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
    c = pd.read_csv(folder / names[0], index_col=0)
    e = pd.read_csv(folder / names[1], index_col=0)
    summary = pd.DataFrame({"mean_classification_loss": c.mean(), "mean_economic_loss_bps": e.mean() * 1e4})
    print(f"{len(c)} test days, {c.shape[1]} models. Mean daily losses (lower is better):\n")
    print(summary.sort_values("mean_classification_loss").to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote results/{args.version}/{names[0]} and {names[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
