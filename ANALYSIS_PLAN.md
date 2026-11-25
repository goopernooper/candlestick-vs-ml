# Analysis Plan

**Status: DRAFT** (week 4, 2026-10-29; decisions confirmed 2026-11-03; confirmatory holdout hypothesis added 2026-11-25, section 13.1). To be frozen and tagged `analysis-plan-frozen` in week 8, before the 2023–2025 holdout is unlocked.

This document fixes, before the final results are seen, exactly what will be run and how it will be judged. Once frozen, nothing in it changes. Any later deviation is listed in the Deviations section at the end with its date and reason, and is reported in the paper.

Decision numbers (D-xxx) refer to DECISIONS.md. Items added after seeing development results are marked **post hoc**.

---

## 1. Questions

**Primary.** Do candlestick pattern-based models (a variable-order pattern miner and k-nearest neighbours) predict next-day stock direction better than standard machine learning models (random forest, LightGBM, SVM, MLP) and naive baselines, after transaction costs?

**Headline analysis: the model × encoding grid** (D-012). Every machine learning model is trained on each encoding, so the effect of the encoding can be separated from the effect of the model class.

**Secondary.**
1. *Fair-features ablation:* does binning candles into symbols or named patterns add anything beyond the same information given as normalized prices?
2. *Cost sensitivity:* at what transaction cost does any edge disappear?
3. *Regimes:* does the ranking of models change between low-, medium- and high-volatility periods?
4. *Reversal:* is any edge more than the known short-term reversal effect? Added after results v1 showed the previous-day baseline with a negative MCC in 15 of 18 years. **Post hoc** (D-019): this question arose from development results and is reported as such.
5. *Replication:* do the eight reversal patterns of Caginalp & Laurent (1998) predict returns on our stocks in 2002–2022, with and without their trend condition, and how does their per-signal significance test compare with a test that allows for signals on the same day moving together? **Post hoc, descriptive** (D-026).

---

## 2. Data (fixed)

- 22 stocks: the two largest current S&P 500 members by market capitalization in each GICS sector with price data on 2002-01-02 (README, D-011). SPY is the market proxy; VIX is used only in robustness checks.
- Daily open, high, low and close from Yahoo Finance via yfinance 1.7.0, adjusted for splits and dividends, downloaded 2026-09-28. The exact files are fixed by the hashes in `data/manifest.json`; every run stops if they don't match.
- Study period: 2002-01-02 to 2025-12-31.
  - Development test years: 2005–2022.
  - Final holdout: 2023–2025.

## 3. Target

- Direction of the next day's close-to-close return (horizon 1 day).
- **Dead zone:** a day is unlabelled when |return| < 0.1 × the standard deviation of the previous 20 daily returns (`config.yaml`, `labels`). Dead-zone days are excluded from training and from classification metrics, but kept in the backtest.

## 4. Encodings

All computed once, shared by every model, and tested for look-ahead (`tests/test_leakage.py`).

| Encoding | Columns | Used by |
|---|---|---|
| A: TA-Lib patterns | 61 signed pattern indicators for day *t* | ML models |
| B continuous, 3-day window | body, upper wick, lower wick, gap in ATR units, for days *t*, *t*−1, *t*−2 (12 columns) | ML models; k-NN |
| B continuous, 5-day window | as above, for five days (20 columns) | k-NN |
| B symbolic | 20 symbols per day (5 body levels × long/short upper wick × long/short lower wick) | pattern miner |
| Normalized OHLC, 3-day window | open, high, low, close minus the previous close, in ATR units, for days *t*, *t*−1, *t*−2 (12 columns): the same information as the 3-day window of continuous B. Results v3 used day *t* only; **post hoc** change (D-024) | ML models (ablation) |
| Today's return only | `n_close` (1 column) | LightGBM ablation, **post hoc** (D-020) |

Bin edges for symbolic B are fixed in `config.yaml` (body ±0.1 and ±0.6 ATR, wicks 0.25 ATR) and are not changed after this plan is frozen.

## 5. Models (final list)

**Baselines** (in every table):
- `majority`, `always_long`, `previous_day`, `random` (seeded per fold)
- `reversal`: predicts the opposite of today's direction. **Post hoc** (D-019).

**Pattern-based models:**
- `pattern_miner`: a variable-order Markov model on symbolic B. Contexts of 1–3 symbols; P(up | context) shrunk toward the training base rate; backs off to shorter contexts below a minimum count (Begleiter et al., 2004).
- `knn_b3`, `knn_b5`: k-nearest neighbours, Euclidean distance, standardized within each fold, on 3- and 5-day windows of continuous B.

**Machine learning models**, each trained on encoding A, B continuous (3-day window), and normalized OHLC:
- random forest (`rf`)
- LightGBM (`lightgbm`)
- SVM with an RBF kernel (`svm`). Its confidence score is the logistic function of its decision function (above 0.5 exactly when it predicts up). Training is capped at **20,000 rows per fit**, a fixed-seed random sample from the whole training window, because full pooled windows are too slow (D-022). The cap was set on 2026-11-06, before any SVM results were produced.
- small multilayer perceptron (`mlp`)

**Optional** (only if ahead of schedule; listed so that adding them is not a deviation): DTW distance for k-NN; an LSTM; the 5-day forward target.

Models are named `model_encoding` (for example `lightgbm_a`, `svm_ohlc`). All models are reported; none are dropped for performing badly.

## 6. Validation

- Walk-forward: an expanding training window from 2002-01-02, with one test year at a time from 2005 to 2022 (18 folds). Splits are by calendar date, so all stocks' rows for a day land together.
- **Purge:** the last *h* trading days before each test year are removed from training (*h* = 1).
- **Tuning:** inside each training window only. Fit on all but the last year, score on that last year (with an *h*-day purge), then refit the chosen settings on the whole window.
  - Every model family, the pattern models included, gets the same budget: 20 Optuna trials (TPE sampler, seeded with the project seed plus the retune year).
  - Selection criterion: MCC on the tuning year.
  - The first trial is always the family's starting settings from `config.yaml` (models), so tuning never picks settings that did worse than them on the tuning year.
  - Search spaces are fixed in `config.yaml` (tuning.spaces) and reported in the paper:
    - LightGBM: number of trees, learning rate, leaves, minimum leaf size, column sampling, L2 penalty.
    - Random forest: minimum leaf size, columns tried per split.
    - SVM: C and the RBF width gamma.
    - MLP: layer sizes, L2 penalty, learning rate.
    - k-NN: k.
    - Pattern miner: shrinkage weight m and minimum count.
  - Not tuned: the baselines (nothing to tune), and the k-NN window length and the miner's maximum context length, which define separate models.
  - Re-tuning happens in the folds starting 2005, 2010, 2015 and 2020; folds in between reuse the most recent settings.
  - The 2025 retuning applies only to the holdout run.

## 7. Metrics

**Primary:** MCC, pooled over all labelled test days 2005–2022.

**Secondary classification:** balanced accuracy and accuracy, each pooled, per year, and per stock.

**Economic** (equal-weight portfolio, positions from day *t*'s close to day *t*+1's close):
- long/short (primary) and long/flat (compared against buy-and-hold)
- Sharpe ratio (no risk-free rate) at 5 bps per side
- annualized return, maximum drawdown, turnover
- **break-even cost:** the cost per side at which the compounded annual net return reaches zero, for long/short and long/flat, found exactly from the daily gross returns and turnover (`results/<version>/break_even.csv`, src/break_even.py)
- **market exposure:** each strategy's daily net return regressed on the equal-weight return of the 22 stocks, giving beta, annualized alpha with a Newey–West t-statistic, the information ratio and the average net exposure. **Post hoc** (D-025), descriptive.

**Cost grid:** 0, 1, 2, 5, 10, 20 bps per side. Headline: 5 bps (D-013).

## 8. Statistical tests

Daily losses are averaged across the 22 stocks before testing, because the stocks are not independent.

1. **Loss series:** for each model, two daily series:
   - **balanced misclassification** (classification loss): the day's wrong calls on labelled stock-days, with a wrong call on an up day weighted 1/(2π) and on a down day 1/(2(1 − π)), where π is the share of up days in that test year, divided by the number of labelled stock-days. Up and down days count equally, so a constant prediction scores about 0.5, consistent with its MCC of 0. This is the day-by-day form of 1 − balanced accuracy (D-027).
   - minus the portfolio's net long/short return at 5 bps (economic loss)

   Both are written by every run to `results/<version>/losses_classification.csv` and `losses_economic.csv` (src/losses.py), one row per test day and one column per model, together with the plain share misclassified (`losses_misclassification.csv`), which is reported for reference but not tested. A day on which every stock is in the dead zone has no classification loss.
2. **Diebold–Mariano tests** on the difference in daily loss between two models: a regression on a constant with Newey–West standard errors, lag = ⌊4(T/100)^(2/9)⌋. Two-sided, α = 0.05.
3. **Model Confidence Set** (`arch.bootstrap.MCS`, range statistic): all models, both loss series, 90% confidence, 10,000 stationary block-bootstrap replications with mean block length 20 trading days (the same block length as the confidence intervals in item 5). Models with identical loss series (majority and always-long) enter once.
4. **SPA test** (`arch.bootstrap.SPA`, studentized, consistent p-value): does any model beat the benchmark? Benchmarks: always-long for the economic losses (long/short, and long/flat for the buy-and-hold claim); the strongest baseline for the classification loss. Same bootstrap as the Model Confidence Set.

   Items 2–4 and the verdicts of section 9 are computed by `python src/significance.py <version>` (settings in `config.yaml`, significance) and written to `results/<version>/significance_*.csv`.
5. **Confidence intervals** for MCC, accuracy and Sharpe: stationary block bootstrap over dates (all stocks kept together on each date), mean block length 20 trading days, 10,000 replications. 95% percentile intervals, for each model's MCC, accuracy, balanced accuracy and long/short and long/flat Sharpe ratio at 5 bps, and for each model's MCC minus the strongest baseline's and long/short Sharpe minus always-long's (the same resampled dates for every model). Computed by `python src/bootstrap_ci.py <version>` and written to `results/<version>/confidence_intervals.csv`.

## 9. What counts as beating a baseline

A model **beats** a baseline on classification if:
- its pooled MCC is higher, **and**
- a Diebold–Mariano test on the daily classification-loss (balanced misclassification) difference is significant at the 5% level.

A model **beats** a baseline economically if:
- its long/short Sharpe ratio at 5 bps is higher, **and**
- a Diebold–Mariano test on the daily economic-loss difference is significant at the 5% level.

Beating buy-and-hold requires the same comparison of long/flat against `always_long` long/flat.

A model **beats the baselines** only if it beats the strongest one: the baseline with the highest pooled MCC (including `reversal`) for classification, and buy-and-hold for the economic claim.

A model shows **genuine predictive value** if it beats the baselines *and* the strongest baseline is excluded from the 90% Model Confidence Set while the model is in it.

Consistency across years (the number of the 18 folds in which the model beats the baseline) is reported for context but is not a requirement.

**If no model meets these criteria, that is the finding** (D-006), and the paper says so plainly.

## 10. Coverage analysis

- On each fold's tuning year, choose confidence cutoffs (|P(up) − 0.5|) that would trade 100%, 50%, 25% and 10% of days.
- Apply them unchanged to the test year.
- Report accuracy and MCC against the coverage actually achieved, for every model.
- The tuning year's predictions come from the model fitted on the training window minus that year, with the fold's settings: the same split as tuning, so the cutoffs use only out-of-sample predictions the model could have made at the time.
- A model that gives every validation day the same confidence (majority, always-long, previous-day, reversal) cannot rank days and acts on all of them at every target.
- **Against reversal on the same days (post hoc, D-028).** For each model and target, the reversal baseline is scored on exactly the stock-days the model acts on, and the difference in MCC gets a 95% stationary-bootstrap interval over dates (mean block 20 days, 10,000 replications). This separates a model whose confidence picks days on which reversal happens to work from one whose candles add information on those days.
- Computed by `python src/coverage.py <version>`, written to `results/<version>/coverage.csv`, `coverage.png`, `coverage_vs_reversal.csv` and `coverage_vs_reversal.png`.

## 11. Regimes

- **Definition** (D-014): SPY's trailing 21-day realized volatility on day *t*, assigned to low / medium / high by terciles. The tercile cutoffs on day *t* use only volatility values through day *t* − 1, on an expanding window starting 2002-01-02.
- **Analysis:** every model is re-evaluated within each regime, with the same metrics and tests.
- The high-volatility results are also broken down by episode (2008–09, 2011, 2020, 2022, and the remaining high-volatility days as "other"), because a few episodes may dominate that regime. For each episode: its share of the high-volatility days, every model's metrics on its days alone, and each model's MCC minus the benchmark's on the same days. And for each named episode: the high-volatility results with that episode left out. Written to `results/<version>/regime_episodes.csv`, `regime_episodes_leave_one_out.csv` and `regime_episodes.png`.
- **Robustness:** VIX terciles in place of SPY volatility.
- **Details fixed before running:**
  - Volatility is the standard deviation of SPY's daily close-to-close returns over the 21 trading days ending on day *t*, annualized.
  - A day is assigned a regime only once 252 earlier values exist (only 2002–2003 lack one, before the first test year).
  - Within each regime, the benchmarks are the same as in section 9 over all test days: the strongest baseline overall (reversal) for the classification loss, and always-long for the economic loss.
  - The long/short return within a regime is the mean daily net return on that regime's days (annualized arithmetically), with positions carried across regime boundaries.
  - Computed by `python src/regimes.py <version>` and written to `results/<version>/regime_*.csv`.

## 12. Robustness checks

1. Per-stock training instead of pooled.
2. Positions entered at the next day's open instead of day *t*'s close.
3. Dead-zone threshold of 0 and of twice the default.
4. VIX-based regimes.

Robustness results are reported whether or not they agree with the main results.

## 13. Holdout protocol

1. Freeze this plan: commit, then tag `analysis-plan-frozen`.
2. Set `unlock_holdout: true` in `config.yaml` and log it in DECISIONS.md.
3. Run the frozen pipeline once on 2023–2025 and commit the results, whatever they show.
4. No model, setting or feature is changed after seeing the holdout. If a bug is found afterwards, it is fixed and both the original and corrected holdout results are reported.

### 13.1 Confirmatory hypothesis: confident days beat reversal (D-029)

Development results (2005–2022) showed that, on their 10% most confident stock-days, six models had a higher MCC than the reversal baseline scored on the same stock-days, with 95% bootstrap intervals above zero (`results/v4/coverage_vs_reversal.csv`, D-028). That finding was exploratory: the comparison was chosen after the coverage curves were seen, and the six models were picked because their intervals cleared zero. The 2023–2025 holdout is used to test it on data that played no part in choosing it.

**Models** (fixed now, chosen from development results): `svm_ohlc`, `rf_b`, `lightgbm_b`, `knn_b3`, `mlp_ohlc`, `lightgbm_ohlc`.

**Statistic.** For model *m*:

> Δₘ = MCC of *m* on its selected holdout stock-days − MCC of `reversal` on exactly the same stock-days,

where the selected stock-days are those whose confidence |P(up) − 0.5| is at or above a cutoff that would have selected the top 10% of that fold's validation-year stock-days. The procedure is section 10's, unchanged:
- The holdout folds are 2023, 2024 and 2025. Each fold's validation year is the year before (2022, 2023, 2024).
- The validation-year predictions come from the model fitted on the fold's training window minus that year, with the fold's settings: the 2020 retune for the 2023 and 2024 folds, and the 2025 retune for the 2025 fold (section 6).
- MCC is pooled over the labelled stock-days selected in all three holdout years.

**Primary test (H1), one test.** The average of the six differences, Δ̄ = (1/6) Σₘ Δₘ, is greater than zero.
- One-sided test at α = 0.05.
- The p-value is the share of bootstrap replications in which Δ̄ ≤ 0. The bootstrap is the stationary bootstrap of section 8 item 5: whole dates, mean block length 20 trading days, 10,000 replications, the same resampled dates for all six models.
- Development estimate: Δ̄ = 0.038.

**Secondary tests (H2), per model.** For each of the six models, Δₘ > 0, one-sided, with the same bootstrap p-values, and the Holm–Bonferroni correction across the six at a family-wise α = 0.05.

**Interpretation, fixed now.**
- If H1 is rejected, the paper reports that the development finding was confirmed out of sample: on their most confident days, these candle-based models beat short-term reversal on the same days.
- If H1 is not rejected, the paper reports that it was not confirmed. The holdout is small: from the development intervals, each Δₘ will have a standard error of about 0.03 on three years, and the expected power of H1 is below 50% even if the development estimates are accurate. Development estimates of models selected for looking good are likely to be too high, so the true power is probably lower. A non-rejection is therefore reported as inconclusive, not as evidence against the effect, with the holdout estimates and their 95% intervals in full.
- H2 results are reported for all six models, whichever way they come out.

**Trading on confident days (secondary, descriptive).** For each of the six models and for `reversal`, a long/short portfolio holds positions only on that model's selected stock-days. Each selected stock-day gets an equal weight within its date; a date with no selected stock-day is held in cash. Positions run from close to close, costs are 5 bps per side, and reversal is traded on exactly the same stock-days as the model it is compared with. The report gives annual return, Sharpe ratio and break-even cost, on the development years (before the holdout is unlocked) and on the holdout. No significance test is attached, and no claim is based on it.

## 14. Deviations after freezing

| Date | What changed | Why | Effect on results |
|---|---|---|---|
| | | | |
