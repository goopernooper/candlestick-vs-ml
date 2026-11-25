"""The confirmatory hypothesis of ANALYSIS_PLAN.md section 13.1 (D-029).

Usage
-----
    python src/confirmatory.py v4          # development years: exploratory, for reference
    python src/confirmatory.py holdout     # the confirmatory test, once the holdout run exists

Needs results/<version>/predictions.parquet and predictions_validation.parquet
(written by `python src/coverage.py <version>`).

What it tests, exactly as fixed in the plan
-------------------------------------------
For each of six models chosen from the development results, on its 10% most
confident stock-days (cutoff from each fold's validation year, as in
coverage.py):

    delta_m = MCC of the model - MCC of reversal on exactly the same stock-days

  H1 (primary, one test)   mean of the six delta_m > 0; one-sided, alpha 0.05
  H2 (secondary)           each delta_m > 0; one-sided, Holm-Bonferroni across
                           the six at a family-wise alpha of 0.05

p-values come from the stationary bootstrap of section 8 item 5 (whole dates,
mean block 20 trading days, 10,000 replications, the same resampled dates for
every model): p = the share of replications in which the statistic is <= 0.

Trading on confident days (descriptive). For each model, and for reversal on
the same stock-days, a long/short portfolio holds positions only on the
model's selected stock-days: each selected stock-day gets an equal weight
within its date, and a date with none is held in cash. Weights move from close
to close; the cost (5 bps per side) is charged on every change of weight, so
dates with fewer selected stocks are traded more heavily. Reports annual
return, Sharpe ratio and break-even cost. No test is attached.

On the development years (v4) these numbers are exploratory: the six models
were chosen because they looked good there, so their p-values are biased
toward zero. Only the holdout run is confirmatory.

Writes results/<version>/confirmatory_tests.csv and confirmatory_backtest.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_ci import metrics_from_counts, stationary_indices  # noqa: E402
from coverage import BENCHMARK, same_days, selections  # noqa: E402
from features import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["svm_ohlc", "rf_b", "lightgbm_b", "knn_b3", "mlp_ohlc", "lightgbm_ohlc"]
TARGET = 0.10
ALPHA = 0.05
DEVELOPMENT_MEAN_DELTA = 0.038       # the plan's development estimate of the H1 statistic


def holm(pvalues: dict, alpha: float = ALPHA) -> dict:
    """Holm-Bonferroni: {name: rejected?}, family-wise error rate alpha."""
    order = sorted(pvalues, key=pvalues.get)
    out, still = {}, True
    for i, name in enumerate(order):
        still = still and pvalues[name] <= alpha / (len(order) - i)
        out[name] = still
    return out


def counts_by_date(frame: pd.DataFrame, pos: pd.Series, n_dates: int) -> np.ndarray:
    """dates x 4 (tp, fp, tn, fn) on frame's labelled rows."""
    lab = frame[frame["up"].notna()]
    y, p = lab["up"].astype(int).to_numpy(), lab["pred"].astype(int).to_numpy()
    cells = np.stack([(p == 1) & (y == 1), (p == 1) & (y == 0), (p == 0) & (y == 0),
                      (p == 0) & (y == 1)], axis=1).astype(float)
    out = np.zeros((n_dates, 4))
    np.add.at(out, pos.reindex(lab.index).to_numpy(), cells)
    return out


def tests(val: pd.DataFrame, test: pd.DataFrame, models=MODELS, reps: int | None = None,
          seed: int | None = None, chunk: int = 250) -> dict:
    """H1 and H2 with bootstrap p-values and 95% intervals."""
    cfg = load_config()
    reps = cfg["significance"]["reps"] if reps is None else reps
    block = cfg["significance"]["block_size"]
    seed = cfg["seed"] if seed is None else seed
    missing = [m for m in [*models, BENCHMARK] if m not in set(test["model"])]
    if missing:
        raise ValueError(f"no predictions for {missing}")
    sel = selections(val[val["model"].isin(models)], test[test["model"].isin(models)], [TARGET])
    dates = pd.Index(sorted(test.index.unique()))
    pos = pd.Series(np.arange(len(dates)), index=dates)
    C = np.concatenate([np.concatenate([counts_by_date(sel[(m, TARGET)], pos, len(dates)),
                                        counts_by_date(same_days(sel[(m, TARGET)], test, BENCHMARK), pos,
                                                       len(dates))], axis=1) for m in models], axis=1)
    k = len(models)
    point = metrics_from_counts(C.sum(axis=0).reshape(k, 2, 4))["mcc"]
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, reps, chunk):
        r = min(chunk, reps - start)
        idx = stationary_indices(len(dates), r, block, rng)
        w = np.zeros((r, len(dates)))
        np.add.at(w, (np.repeat(np.arange(r), len(dates)), idx.ravel()), 1.0)
        draws.append(metrics_from_counts((w @ C).reshape(r, k, 2, 4))["mcc"])
    d = np.concatenate(draws)                                  # reps x models x 2
    delta, delta_draws = point[:, 0] - point[:, 1], d[:, :, 0] - d[:, :, 1]
    mean_draws = delta_draws.mean(axis=1)
    p_models = {m: float((delta_draws[:, j] <= 0).mean()) for j, m in enumerate(models)}
    rejected = holm(p_models)
    rows = [{"test": "H2", "model": m, "stock_days": len(sel[(m, TARGET)]), "model_mcc": point[j, 0],
             "reversal_mcc": point[j, 1], "delta": delta[j],
             "ci_low": np.percentile(delta_draws[:, j], 2.5), "ci_high": np.percentile(delta_draws[:, j], 97.5),
             "p_one_sided": p_models[m], "holm_rejected": rejected[m]} for j, m in enumerate(models)]
    h1 = {"test": "H1", "model": "mean of six", "delta": float(delta.mean()),
          "ci_low": np.percentile(mean_draws, 2.5), "ci_high": np.percentile(mean_draws, 97.5),
          "p_one_sided": float((mean_draws <= 0).mean()), "rejected": bool((mean_draws <= 0).mean() <= ALPHA)}
    return {"h1": h1, "table": pd.DataFrame([h1, *rows]), "selections": sel, "reps": reps,
            "dates": f"{dates.min().date()} to {dates.max().date()}"}


def confident_backtest(selected: pd.DataFrame, cost_bps: float, days_per_year: int, all_dates) -> dict:
    """Long/short on the selected stock-days only, equal weight within each date,
    cash on dates with none; cost on every change of weight."""
    s = selected.assign(sign=np.where(selected["pred"].astype(int) == 1, 1.0, -1.0))
    n = s.groupby(level=0)["sign"].transform("size")
    w = (s["sign"] / n).rename("w")
    weights = (pd.DataFrame({"date": s.index, "ticker": s["ticker"].to_numpy(), "w": w.to_numpy()})
               .pivot(index="date", columns="ticker", values="w").reindex(all_dates).fillna(0.0))
    rets = (pd.DataFrame({"date": s.index, "ticker": s["ticker"].to_numpy(), "r": s["fwd_return"].to_numpy()})
            .pivot(index="date", columns="ticker", values="r").reindex(all_dates)
            .reindex(columns=weights.columns).fillna(0.0))
    gross = (weights * rets).sum(axis=1).to_numpy(copy=True)
    turnover = weights.diff().abs().sum(axis=1).to_numpy(copy=True)
    turnover[0] = np.abs(weights.iloc[0]).sum()

    def log_growth(c):
        net = gross - turnover * c / 10_000
        return -np.inf if (net <= -1).any() else np.log1p(net).sum()

    net = gross - turnover * cost_bps / 10_000
    sd = net.std(ddof=1)
    n_days = len(net)
    if log_growth(0) <= 0:
        be = np.nan
    else:
        hi = 1.0
        while log_growth(hi) > 0 and hi < 10_000:
            hi *= 2
        be = float(brentq(log_growth, 0, hi)) if log_growth(hi) <= 0 else np.inf
    return {"days_invested": int((weights.abs().sum(axis=1) > 0).sum()), "days": n_days,
            "ann_return": float(np.expm1(np.log1p(net).sum() * days_per_year / n_days)),
            "sharpe": float(net.mean() / sd * np.sqrt(days_per_year)) if sd > 1e-12 else np.nan,
            "turnover_per_year": float(turnover.mean() * days_per_year), "break_even_bps": be}


def backtests(test: pd.DataFrame, sel: dict, models=MODELS) -> pd.DataFrame:
    cfg = load_config()["backtest"]
    all_dates = pd.Index(sorted(test.index.unique()))
    rows = []
    for m in models:
        s = sel[(m, TARGET)]
        for who, frame in [(m, s), (f"{BENCHMARK} on {m}'s days", same_days(s, test, BENCHMARK))]:
            rows.append({"model": m, "traded": who, **confident_backtest(
                frame, cfg["cost_bps_per_side"], cfg["trading_days_per_year"], all_dates)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="results version: v4 (development) or the holdout run's")
    parser.add_argument("--reps", type=int, help="bootstrap replications (default: config.yaml)")
    args = parser.parse_args()
    folder = ROOT / "results" / args.version
    paths = [folder / "predictions.parquet", folder / "predictions_validation.parquet"]
    for path in paths:
        if not path.exists():
            print(f"{path} not found: run `python src/run.py --version {args.version}` and then "
                  f"`python src/coverage.py {args.version}` first.")
            return 1
    test, val = (pd.read_parquet(p) for p in paths)
    h0, h1_year = load_config()["splits"]["holdout_years"]
    holdout = set(test["test_year"].unique()) & set(range(h0, h1_year + 1))
    if holdout:          # the holdout run also re-predicts the development years: test only 2023-2025
        test = test[test["test_year"].isin(holdout)]
        val = val[val["test_year"].isin(holdout)]
    years = sorted(test["test_year"].unique())
    kind = "CONFIRMATORY (holdout)" if holdout else "EXPLORATORY (development years; the models were chosen here)"
    out = tests(val, test, reps=args.reps)
    out["table"].to_csv(folder / "confirmatory_tests.csv", index=False)
    bt = backtests(test, out["selections"])
    bt.to_csv(folder / "confirmatory_backtest.csv", index=False)

    h1, t = out["h1"], out["table"][out["table"]["test"] == "H2"]
    print(f"Section 13.1, results {args.version}, test years {years[0]}-{years[-1]}: {kind}")
    print(f"{out['reps']:,} bootstrap replications over whole dates, mean block 20 days.\n")
    print(f"H1: mean over the six models of (model MCC - {BENCHMARK} MCC on the same stock-days), "
          "on each model's 10% most confident days")
    print(f"    {h1['delta']:+.4f}  95% interval [{h1['ci_low']:+.4f}, {h1['ci_high']:+.4f}]  "
          f"one-sided p = {h1['p_one_sided']:.4f}  ->  "
          f"{'REJECTED: mean difference > 0' if h1['rejected'] else 'not rejected'}"
          f"   (development estimate {DEVELOPMENT_MEAN_DELTA:+.3f})\n")
    print("H2: each model, Holm-Bonferroni across the six (family-wise alpha 0.05):\n")
    print(f"  {'model':<15}{'stock-days':>11}{'model MCC':>11}{'reversal':>10}{'difference':>12}"
          f"{'95% interval':>22}{'p':>9}   Holm")
    for r in t.itertuples():
        print(f"  {r.model:<15}{r.stock_days:>11,}{r.model_mcc:>+11.4f}{r.reversal_mcc:>+10.4f}{r.delta:>+12.4f}"
              f"   [{r.ci_low:+.4f}, {r.ci_high:+.4f}]{r.p_one_sided:>9.4f}   "
              f"{'rejected' if r.holm_rejected else '-'}")
    print("\nTrading only on the confident stock-days (long/short, equal weight within each date, "
          "cash otherwise, 5 bps per side; descriptive):\n")
    print(f"  {'traded':<34}{'days in market':>15}{'annual return':>15}{'Sharpe':>9}{'break-even':>12}")
    for r in bt.itertuples():
        be = "never" if np.isnan(r.break_even_bps) else f"{r.break_even_bps:.1f} bps"
        print(f"  {r.traded:<34}{r.days_invested:>15,}{r.ann_return:>15.1%}{r.sharpe:>9.2f}{be:>12}")
    if not holdout:
        print("\nThese are development years: the six models were chosen because they looked good here, so "
              "these p-values are biased toward zero and are not evidence. Only the holdout run is.")
    else:
        print("\nInterpretation fixed in ANALYSIS_PLAN.md section 13.1: if H1 is not rejected, the result is "
              "reported as inconclusive (the test's expected power is below 50%), not as evidence against.")
    print(f"\nWrote results/{args.version}/confirmatory_tests.csv and confirmatory_backtest.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
