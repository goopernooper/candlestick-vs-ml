"""Bootstrap confidence intervals for MCC, accuracy and the Sharpe ratio.

Usage
-----
    python src/bootstrap_ci.py v4             # 10,000 replications
    python src/bootstrap_ci.py v4 --reps 1000

Implements ANALYSIS_PLAN.md section 8, item 5. The 22 stocks move together, so
the bootstrap resamples whole dates: every stock's result on a drawn date comes
along with it. Dates are drawn by the stationary bootstrap (Politis & Romano,
1994): blocks of consecutive test days, of random length with mean
block_size (20 trading days), joined end to start, so day-to-day dependence
within a block is kept. The same resampled dates are used for every model in a
replication, so differences between models are bootstrapped consistently.

For each model, 95% percentile intervals for:
  mcc, accuracy, balanced_accuracy   pooled over the resampled labelled stock-days
  ls_sharpe, lf_sharpe               long/short and long/flat net Sharpe ratio at
                                     the headline cost (5 bps per side)
  mcc_minus_<strongest baseline>     MCC difference from the baseline with the
                                     highest pooled MCC (reversal in practice)
  ls_sharpe_minus_always_long        long/short Sharpe difference from buy-and-hold

An interval for a difference that contains 0 means the gap is within what
resampling the test period produces. These intervals describe uncertainty;
the formal tests are in significance.py (section 9).

How it works: each date is reduced to per-model counts (true and false
positives and negatives) and per-model sums of the daily portfolio return and
its square. A replication is then a weighted sum of those per-date numbers,
the weights counting how often each date was drawn, which makes 10,000
replications take seconds.

Writes results/<version>/confidence_intervals.csv.
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
LEVEL = 0.95


def stationary_indices(n: int, reps: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """reps x n array of date positions from the stationary bootstrap."""
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    jump = rng.random((reps, n)) < 1.0 / block
    fresh = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def date_counts(preds: pd.DataFrame, dates: pd.Index, models) -> np.ndarray:
    """dates x models x 4 array of (tp, fp, tn, fn) on labelled stock-days."""
    lab = preds[preds["up"].notna()]
    y, p = lab["up"].astype(int).to_numpy(), lab["pred"].astype(int).to_numpy()
    cells = pd.DataFrame({"date": lab.index, "model": lab["model"].to_numpy(),
                          "tp": (p == 1) & (y == 1), "fp": (p == 1) & (y == 0),
                          "tn": (p == 0) & (y == 0), "fn": (p == 0) & (y == 1)})
    sums = cells.groupby(["date", "model"])[["tp", "fp", "tn", "fn"]].sum()
    out = np.zeros((len(dates), len(models), 4))
    for j, m in enumerate(models):
        if m in sums.index.get_level_values("model"):
            s = sums.xs(m, level="model").reindex(dates).fillna(0)
            out[:, j, :] = s.to_numpy()
    return out


def daily_returns(preds: pd.DataFrame, dates: pd.Index, models, strategy: str, cost_bps: float) -> np.ndarray:
    """dates x models net portfolio returns (NaN where a model has no return that day)."""
    cols = {m: portfolio_returns(stock_returns(preds[preds["model"] == m], strategy, cost_bps))
            for m in models}
    return pd.DataFrame(cols).reindex(dates).to_numpy(dtype=float)


def metrics_from_counts(c: np.ndarray) -> dict:
    """MCC, accuracy and balanced accuracy from (..., 4) counts; same
    conventions as evaluate.classification_metrics."""
    tp, fp, tn, fn = (c[..., k] for k in range(4))
    n = tp + fp + tn + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.where(denom > 0, denom, 1), 0.0)
        acc = (tp + tn) / n
        bal = np.nanmean(np.stack([tp / (tp + fn), tn / (tn + fp)]), axis=0)
    return {"mcc": mcc, "accuracy": acc, "balanced_accuracy": bal}


def sharpe_from_sums(s1, s2, n, days_per_year):
    """Sharpe ratio (sample sd, ddof 1) from the sum, sum of squares and count."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = (s2 - n * mean ** 2) / (n - 1)
        sd = np.sqrt(np.clip(var, 0, None))
        return np.where(sd > 1e-12, mean / np.where(sd > 1e-12, sd, 1) * np.sqrt(days_per_year), np.nan)


def bootstrap(preds: pd.DataFrame, reps: int | None = None, block: float | None = None,
              seed: int | None = None, chunk: int = 500, strongest: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    sig = cfg["significance"]
    reps = sig["reps"] if reps is None else reps
    block = sig["block_size"] if block is None else block
    seed = cfg["seed"] if seed is None else seed
    cost, days_per_year = cfg["backtest"]["cost_bps_per_side"], cfg["backtest"]["trading_days_per_year"]
    models = sorted(preds["model"].unique())
    dates = pd.Index(sorted(preds.index.unique()), name="date")
    n = len(dates)

    counts = date_counts(preds, dates, models)                       # T x M x 4
    ls = daily_returns(preds, dates, models, "long_short", cost)    # T x M
    lf = daily_returns(preds, dates, models, "long_flat", cost)
    have = ~np.isnan(ls)
    ls0, lf0 = np.nan_to_num(ls), np.nan_to_num(lf)

    point = {**{k: v for k, v in metrics_from_counts(counts.sum(axis=0)).items()},
             "ls_sharpe": sharpe_from_sums(ls0.sum(0), (ls0 ** 2).sum(0), have.sum(0), days_per_year),
             "lf_sharpe": sharpe_from_sums(lf0.sum(0), (lf0 ** 2).sum(0), have.sum(0), days_per_year)}

    flat_counts = counts.reshape(n, -1)
    draws = {k: [] for k in point}
    rng = np.random.default_rng(seed)
    for start in range(0, reps, chunk):
        r = min(chunk, reps - start)
        idx = stationary_indices(n, r, block, rng)
        w = np.zeros((r, n))
        np.add.at(w, (np.repeat(np.arange(r), n), idx.ravel()), 1.0)
        for k, v in metrics_from_counts((w @ flat_counts).reshape(r, len(models), 4)).items():
            draws[k].append(v)
        cnt = w @ have
        draws["ls_sharpe"].append(sharpe_from_sums(w @ ls0, w @ ls0 ** 2, cnt, days_per_year))
        draws["lf_sharpe"].append(sharpe_from_sums(w @ lf0, w @ lf0 ** 2, cnt, days_per_year))
    draws = {k: np.vstack(v) for k, v in draws.items()}               # reps x M

    lo, hi = (1 - LEVEL) / 2 * 100, (1 + LEVEL) / 2 * 100
    rows = []

    def add(model, metric, est, d):
        d = d[~np.isnan(d)]
        rows.append({"model": model, "metric": metric, "estimate": float(est),
                     "ci_low": float(np.percentile(d, lo)) if len(d) else np.nan,
                     "ci_high": float(np.percentile(d, hi)) if len(d) else np.nan,
                     "boot_sd": float(d.std(ddof=1)) if len(d) > 1 else np.nan})

    for j, m in enumerate(models):
        for k in point:
            add(m, k, point[k][j], draws[k][:, j])
    if strongest is None:
        base = [b for b in sig["baselines"] if b in models]
        strongest = base[int(np.argmax([point["mcc"][models.index(b)] for b in base]))] if base else None
    econ = sig["economic_benchmark"]
    for j, m in enumerate(models):
        if strongest and m != strongest:
            s = models.index(strongest)
            add(m, f"mcc_minus_{strongest}", point["mcc"][j] - point["mcc"][s],
                draws["mcc"][:, j] - draws["mcc"][:, s])
        if econ in models and m != econ:
            e = models.index(econ)
            add(m, f"ls_sharpe_minus_{econ}", point["ls_sharpe"][j] - point["ls_sharpe"][e],
                draws["ls_sharpe"][:, j] - draws["ls_sharpe"][:, e])
    out = pd.DataFrame(rows)
    return out.assign(reps=reps, block_size=block, level=LEVEL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="results version, e.g. v4")
    parser.add_argument("--reps", type=int, help="bootstrap replications (default: config.yaml)")
    args = parser.parse_args()
    folder = ROOT / "results" / args.version
    path = folder / "predictions.parquet"
    if not path.exists():
        print(f"{path} not found (predictions aren't committed; they exist on the computer that ran "
              f"`python src/run.py --version {args.version}`).")
        return 1
    preds = pd.read_parquet(path)
    reps = args.reps or load_config()["significance"]["reps"]
    print(f"Bootstrapping {preds['model'].nunique()} models over {preds.index.nunique():,} test days, "
          f"{reps:,} replications...", flush=True)
    ci = bootstrap(preds, reps=reps)
    ci.to_csv(folder / "confidence_intervals.csv", index=False)

    def show(metric, digits=4):
        t = ci[ci["metric"] == metric].sort_values("estimate", ascending=False)
        return "\n".join(f"  {r.model:<15} {r.estimate:+.{digits}f}  [{r.ci_low:+.{digits}f}, {r.ci_high:+.{digits}f}]"
                         for r in t.itertuples())
    print(f"\nMCC, 95% interval:\n{show('mcc')}")
    diff = next(m for m in ci["metric"].unique() if m.startswith("mcc_minus_"))
    print(f"\n{diff} (an interval containing 0: no clear difference):\n{show(diff)}")
    print(f"\nLong/short Sharpe at 5 bps, 95% interval:\n{show('ls_sharpe', 2)}")
    print(f"\nls_sharpe_minus_always_long:\n{show('ls_sharpe_minus_always_long', 2)}")
    print(f"\nWrote results/{args.version}/confidence_intervals.csv (also accuracy, balanced accuracy, "
          "long/flat Sharpe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
