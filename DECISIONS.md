# Decisions

A running log of design decisions for this project: what was decided, when, by whom, and why.

**How to use this file**

- Add an entry whenever a check-in (or anything else) settles a question.
- Never delete an entry. If a decision changes, mark the old one **Superseded**, add a new entry, and link the two.
- Commit this file right after each Monday check-in, e.g. `git commit -m "decisions: 2026-09-28 check-in"`.
- Statuses: **Proposed** (waiting for a check-in), **Decided**, **Superseded**.

Professor Danisman gave full latitude on design, so entries marked "Decided by: Joshua" were settled while writing the project plan. Entries marked **Proposed** are defaults to confirm with him.

## Index

| ID | Decision | Status | Date |
|---|---|---|---|
| D-001 | Prediction target | Decided | 2026-09-23 |
| D-002 | Candlestick encodings | Decided | 2026-09-23 |
| D-003 | Models and baselines | Decided | 2026-09-23 |
| D-004 | Validation scheme | Decided | 2026-09-23 |
| D-005 | Metrics and significance tests | Decided | 2026-09-23 |
| D-006 | Commitment to report a null result | Decided | 2026-09-23 |
| D-007 | Data source | Decided | 2026-09-23 |
| D-008 | Venue and deadline | Proposed | |
| D-009 | Pooled vs. per-ticker training | Decided | 2026-11-03 |
| D-010 | Date range and holdout | Decided | 2026-11-03 |
| D-011 | Ticker universe | Decided | 2026-11-03 |
| D-012 | Headline result | Decided | 2026-11-03 |
| D-013 | Trading rules and costs | Decided | 2026-11-03 |
| D-014 | Regime definition | Decided | 2026-11-03 |
| D-015 | Preprint and authorship | Proposed | |
| D-016 | Raw data storage | Decided | 2026-10-07 |
| D-017 | Review of daily moves over 20% | Decided | 2026-10-07 |
| D-018 | Model settings for the first end-to-end run | Decided | 2026-10-29 |
| D-019 | Reversal baseline (post hoc) | Decided | 2026-11-03 |
| D-020 | Today's-return ablation (post hoc) | Decided | 2026-11-03 |
| D-021 | Pattern miner settings for results v2 | Decided | 2026-11-03 |
| D-022 | Random forest, SVM and MLP: first settings and the SVM training cap | Decided | 2026-11-06 |
| D-023 | Tuning protocol details | Decided | 2026-11-06 |

---

## Settled while writing the plan

### D-001: Prediction target
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:** Primary target is next-day close-to-close direction. Days whose absolute return is below 0.1 × the trailing 20-day standard deviation form a dead zone: they are dropped from training labels and classification metrics but kept in the backtest. Secondary target is n-day forward return (on the cut list).
- **Why:** Direction is the natural question for candlestick patterns. The dead zone keeps near-zero moves from being labeled arbitrarily, and scaling it by volatility keeps it comparable across stocks and years.
- **Alternatives considered:** Fixed-percentage dead zone (not comparable across volatility levels); regression on returns (harder to compare with pattern rules).

### D-002: Candlestick encodings
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:**
  - Encoding A: TA-Lib CDL pattern functions, one signed column per pattern.
  - Encoding B (continuous): signed body, upper wick, lower wick, and gap from prior close, each divided by ATR(14) lagged one day.
  - Encoding B (symbolic): body and wicks binned with fixed edges (body at ±0.1 and ±0.6 ATR, wicks at 0.25 ATR), 20 symbols.
  - Normalized OHLC for the ablation: each price minus the prior close, in ATR units.
  - All encodings use only data available at each day's close, are computed once, and are shared by every model.
- **Why:** TA-Lib gives standard, citable pattern definitions. Fixed bin edges avoid leaking future data into the encoding. Signed bodies keep candle color; the gap captures how consecutive candles relate.
- **Alternatives considered:** Hand-coded patterns (non-standard); quantile bin edges over the full sample (leaks future information).

### D-003: Models and baselines
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:**
  - Pattern side: a variable-order pattern miner (contexts of 1–3 symbols, shrinkage toward the base rate, backoff to shorter contexts) and k-NN on continuous encoding B.
  - ML side: random forest, LightGBM, RBF-kernel SVM, small MLP.
  - Baselines: majority class, always-long, yesterday's direction, random.
  - Every model family gets the same hyperparameter tuning budget.
- **Why:** Covers the subsequence and similarity methods and the tree, SVM, and neural models named in the project description. Equal tuning budgets keep the comparison about model class rather than tuning effort.
- **Alternatives considered:** CNN/LSTM (kept as optional, first on the cut list); DTW distance for k-NN (optional, second on the cut list).

### D-004: Validation scheme
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:** Expanding-window walk-forward, one test year at a time, with a purge gap equal to the label horizon. The last year of each training window is the tuning set. Splits are by calendar date and never shuffled.
- **Why:** Mirrors how a model would actually be used and prevents look-ahead leakage.
- **Alternatives considered:** Rolling window (possible robustness check); random k-fold (invalid for time series).

### D-005: Metrics and significance tests
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:**
  - Classification metrics: accuracy, balanced accuracy, MCC.
  - Economic metrics: annualized return and Sharpe ratio, net of costs.
  - Significance: Diebold-Mariano tests with Newey-West standard errors on each day's loss averaged across tickers, plus the Model Confidence Set over all models.
  - Uncertainty: stationary block-bootstrap intervals that resample dates, keeping all tickers together.
- **Why:** Tickers move together, so treating them as independent samples would overstate significance. The Model Confidence Set handles the many comparisons.
- **Alternatives considered:** Paired tests treating tickers as independent (overstates confidence).

### D-006: Commitment to report a null result
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:** If no method reliably beats the baselines after costs, the paper reports that as its finding. This is fixed before any results are seen.
- **Why:** It is the most likely honest outcome, it is publishable, and committing in advance removes the temptation to keep searching until something looks significant.

### D-007: Data source
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:** Daily OHLC from yfinance with `auto_adjust=True` set explicitly. The library version is pinned, raw data is cached as Parquet, and each snapshot gets a manifest (download date, version, row counts, file hashes).
- **Why:** Free and widely used. Pinning and snapshotting make results reproducible even if Yahoo's data changes.
- **Alternatives considered:** Stooq (fallback if yfinance becomes unreliable).

---

## To confirm at the September 28 check-in

Fill in **Decided** during the meeting, update the index, and commit.

### D-008: Venue and deadline
- **Status:** Proposed.
- **Proposed:** No default; this is the most important question for the meeting.
- **Why it matters:** The deadline decides whether week 12 ends in a submission or a polished draft, and how much of the cut list gets used.
- **Decided:**
- **Notes:** Still open as of 2026-11-03; depends on the department funding Prof. Danisman offered. If NCUR is the target, its abstract deadline is December 4, 2026.

### D-009: Pooled vs. per-ticker training
- **Status:** Decided, 2026-11-03. Decided by: Joshua (Prof. Danisman gave full latitude on design).
- **Proposed:** One model per fold trained on all tickers together; per-ticker models as a robustness check.
- **Why:** Per-ticker models get about 250 samples a year, which starves both the ML models and the pattern miner. ATR normalization makes pooling legitimate, and pooling directly tests the claim that candlestick patterns are universal.
- **Decided:** Adopted as proposed: one model per fold trained on all 22 stocks pooled; per-stock training is a robustness check. Implemented in src/run.py since week 3.
- **Notes:**

### D-010: Date range and holdout
- **Status:** Decided, 2026-11-03. Decided by: Joshua (Prof. Danisman gave full latitude on design).
- **Proposed:** Daily data from January 2002 through December 2025. Development test years 2005–2022. 2023–2025 stays locked as a final holdout until the analysis plan is frozen in week 8.
- **Why:** Starting in 2002 is after decimalization and puts 2008 in the test set. Ending in 2025 adds the April 2025 selloff as another high-volatility episode. The holdout gives one clean out-of-sample check.
- **Decided:** Adopted as proposed: 2002-2025, development test years 2005-2022, holdout 2023-2025 locked in code (config.yaml splits.unlock_holdout) until ANALYSIS_PLAN.md is frozen in week 8.
- **Notes:**

### D-011: Ticker universe
- **Status:** Decided, 2026-11-03. Decided by: Joshua (Prof. Danisman gave full latitude on design).
- **Proposed:** About 25 large caps, two or three per sector, with continuous data since 2002, plus SPY. The selection rule is written in the README, and survivorship bias is acknowledged as a limitation.
- **Question for Prof. Danisman:** Does he prefer a universe from prior work, for consistency?
- **Decided:** Adopted as proposed: 22 stocks by the written rule in the README (two per GICS sector), plus SPY. No universe from prior work was requested. Survivorship bias is acknowledged as a limitation.
- **Notes:**

### D-012: Headline result
- **Status:** Decided, 2026-11-03. Decided by: Joshua (Prof. Danisman gave full latitude on design).
- **Proposed:** Lead with the model × encoding grid; regime conditioning is a secondary analysis.
- **Why:** The grid separates the effect of the encoding from the effect of the model class, which prior work usually mixes together. The high-volatility regime rests on only a handful of episodes, so it is weaker as a headline.
- **Decided:** Adopted as proposed: the model x encoding grid is the headline; regime conditioning is secondary. Short-term reversal is added as a secondary question (D-019).
- **Notes:**

### D-013: Trading rules and costs
- **Status:** Decided, 2026-11-03. Decided by: Joshua (Prof. Danisman gave full latitude on design).
- **Proposed:**
  - Signal at day t's close, trade at that close; next-day open as a robustness check.
  - Long/short is primary; long/flat against buy-and-hold is also reported.
  - Headline tables charge 5 bps per side; the cost curve runs from 0 to 20 bps.
- **Decided:** Adopted as proposed: trade at day t's close (next-day open as a robustness check); long/short primary, long/flat vs buy-and-hold also reported; 5 bps per side headline, cost grid 0-20 bps. Implemented in config.yaml (backtest).
- **Notes:**

### D-014: Regime definition
- **Status:** Decided, 2026-11-03. Decided by: Joshua (Prof. Danisman gave full latitude on design).
- **Proposed:** Terciles of SPY's trailing 21-day realized volatility, with cutoffs computed from past data only. VIX as a robustness check.
- **Why:** Using past data only means a regime finding is something a trader could have acted on at the time.
- **Decided:** Adopted as proposed: terciles of SPY's trailing 21-day realized volatility, with cutoffs from data through day t-1 only (expanding window); VIX terciles as a robustness check.
- **Notes:**

### D-015: Preprint and authorship
- **Status:** Proposed.
- **Proposed:** Agree on the author list and order, and whether to post a preprint before or after submission. First-time arXiv authors may need an endorsement, so raise it early.
- **Decided:**
- **Notes:**

---

## Data

### D-016: Raw data storage
- **Status:** Decided, 2026-10-07. Decided by: Joshua.
- **Decision:** Raw price files (data/raw/) stay out of git. data/manifest.json, which records each file's row count, date range, and SHA-256 hash, is committed instead. The snapshot downloaded on 2026-09-28 is backed up as a zip in Google Drive.
- **Why:** Yahoo's terms don't allow redistributing its data, and files committed to git stay in the history even if deleted later. The manifest still proves exactly which data produced the results.
- **Consequences:** Run `python src/manifest.py --check` before experiments to confirm the data hasn't changed.

### D-017: Review of daily moves over 20%
- **Status:** Decided, 2026-10-07. Decided by: Joshua.
- **Decision:** Every close-to-close return or overnight gap larger than 20% was reviewed: 125 moves across 21 tickers. ^VIX was excluded because moves that large are normal for a volatility index. Explanations for every move are in data/big_moves.csv. The 39 moves in the 22 selected stocks were checked individually. The 86 moves in alternates were explained in groups by episode (for example Equinix and American Tower in 2001–03, bank stocks in 2008–09) and will be checked individually if an alternate is promoted.
- **Findings:** No missed splits or bad ticks were found. The GE spin-offs and the Linde and Prologis mergers produced no false jumps. One anomaly: BRK-B opened about 22% below the prior close on 2008-10-10. This disorderly opening trade also appears in Google Finance's data, and the day's close was confirmed against BRK-A's close. The row was kept as recorded.
- **Second-source check:** The 8 selected-stock rows not tied to specific news were confirmed on 2026-10-07: marketwatch.com shows the same daily percentage move on each date. The rows are AMZN 2001-11-14 and 2001-11-26; NVDA 2002-11-08, 2003-02-14 and 2003-05-09; PLD 2008-11-12 and 2008-12-11; WELL 2020-03-19.
- **Why:** A missed split or bad price would create a fake extreme candle that the models could learn from. Alternates don't enter the analysis unless promoted, so checking them in groups is enough for now.
- **Consequences:** No changes to data/raw/.

---

## Since results v1

### D-018: Model settings for the first end-to-end run
- **Status:** Decided, 2026-10-29. Decided by: Joshua.
- **Decision:** Models given continuous encoding B see a 3-day window (today and the two previous candles). k-NN uses 3- and 5-day windows with k = 100, standardized inside each fold. LightGBM uses fixed, conservative settings (config.yaml, models). All of these are replaced by settings tuned inside each training window in week 6, with the same tuning budget for every model family.
- **Why:** A single candle can't express multi-day shapes, while encoding A's patterns already look back up to five candles. Fixed settings let the whole pipeline run end to end before tuning.

### D-019: Reversal baseline (post hoc)
- **Status:** Decided, 2026-11-03. Decided by: Joshua (Prof. Danisman gave full latitude on design).
- **Decision:** Add a `reversal` baseline that predicts the opposite of today's direction, and treat "is any edge more than short-term reversal?" as a secondary question.
- **Why:** In results v1 the previous-day baseline had MCC -0.0245, negative in 15 of 18 years, so betting against today has MCC +0.0245, above every model so far (LightGBM 0.021-0.022, k-NN 0.014-0.019). LightGBM's predictions agree with reversal far more than chance (61.5% and 59.6% vs about 49% expected). A reviewer would ask whether the candlestick models are just rediscovering reversal; the paper should answer it.
- **Post hoc:** This baseline was chosen after seeing development results (2005-2022). It is reported as post hoc in the paper. The holdout (2023-2025) has not been seen.

### D-020: Today's-return ablation (post hoc)
- **Status:** Decided, 2026-11-03. Decided by: Joshua (Prof. Danisman gave full latitude on design).
- **Decision:** Add LightGBM trained only on today's return in ATR units (`n_close`), alongside the full encodings.
- **Why:** If the full candle encodings beat it, candle shape adds information beyond today's return; if not, the candlestick models' edge is reversal. Either answer is a clear finding.
- **Post hoc:** Same as D-019.

### D-021: Pattern miner settings for results v2
- **Status:** Decided, 2026-11-03. Decided by: Joshua.
- **Decision:** The pattern miner uses contexts of 1 to 3 daily symbols (symbolic encoding B), shrinks each context's up rate toward the training base rate with m = 100, and predicts from the longest context seen at least 30 times in training, backing off to shorter contexts and then to the base rate (config.yaml, models: pattern_miner). As with D-018, m and min_count are replaced by settings tuned inside each training window in week 6.
- **Why:** The first training window (2002–2004) has about 16,600 stock-days for 8,000 possible 3-day contexts, about two per context, so most raw up rates are noise (the exact counts are in results/pattern_context_counts.csv, from `python src/pattern_report.py`). On synthetic noise sized like that window, the miner without shrinkage or a minimum count scored about 76% on its training data and 50% on new data; with these settings the gap was about 4 points. On synthetic data with a planted 3-day pattern (up 70% of the time, about 1% of days), the miner with these settings kept the pattern in all 20 samples tried, ranked it first of about 400 contexts in 18 and never lower than third (tests/test_pattern_miner_synthetic.py).
- **Consequences:** results/<version>/diagnostics.csv records, for each fold, how many contexts of each length were kept and which context length each test prediction used.

### D-022: Random forest, SVM and MLP: first settings and the SVM training cap
- **Status:** Decided, 2026-11-06. Decided by: Joshua.
- **Decision:**
  - Random forest, RBF-kernel SVM and a small MLP (two hidden layers of 32 and 16 units) from scikit-learn 1.9.1, with the shared interface and per-fold seeds used by LightGBM. Starting settings are in config.yaml (models); like D-018 and D-021, they are replaced by settings tuned inside each training window.
  - The SVM and MLP standardize every column using the training days only; the tree models use the raw columns.
  - **SVM training cap:** each SVM fit uses at most 20,000 training days, a random sample drawn with the fold's seed from the whole training window. Windows smaller than that (the 2005 fold has about 16,600 days) are used in full.
  - The SVM's confidence score is the logistic function of its decision function. It is above 0.5 exactly when the SVM predicts up and orders days by distance from the decision boundary, but it is not a calibrated probability.
- **Why:** On data this noisy, nearly every training day becomes a support vector (about 97% in timing runs on synthetic prices sized like the real data), and SVM training time grows roughly with the square of the number of rows. Timed on 2 CPU cores: 20,000 rows take 10–30 seconds per fit depending on the encoding, while a full 100,000-row window would take an estimated 10 minutes per fit, too slow for 18 folds, three encodings and the tuning budget. Sampling from the whole window keeps the variety of market conditions the other models see. SVC's built-in probabilities (`probability=True`) would fit five extra models per fit and can disagree with the SVM's own predictions; no analysis needs calibrated probabilities, since classification uses the 0.5 cut and the coverage analysis only ranks days by confidence.
- **Consequences:** The paper reports the cap as a limitation of the SVM comparison: the SVM sees at most 20,000 of the up to about 100,000 training days. The number of rows and support vectors in each SVM fit is recorded in results/<version>/diagnostics.csv, as are the MLP's epochs and whether it converged.

### D-023: Tuning protocol details
- **Status:** Decided, 2026-11-06. Decided by: Joshua.
- **Decision:**
  - Implements D-003's equal tuning budget and ANALYSIS_PLAN.md section 6 in src/tune.py. Every tuned model gets 20 Optuna trials (TPE sampler, seed = project seed + retune year) in each of the retune years 2005, 2010, 2015 and 2020 (and 2025 for the holdout run).
  - Each trial is fitted on the training window minus its last year and scored by MCC on that year, with the one-day purge in between. The best settings are refitted on each fold's full training window.
  - Trial 0 is always the starting settings from config.yaml (D-018, D-021, D-022).
  - Search spaces are in config.yaml (tuning.spaces).
  - Each model is tuned separately on its own encoding: lightgbm_a and lightgbm_b can end up with different settings.
  - Results are cached in results/tuning/ with every trial recorded, and the settings each fold used are written to results/<version>/tuned_settings.csv.
  - `python src/run.py --untuned` still runs the starting settings, which reproduces results v1 and v2.
- **Why:** Starting from the hand-set settings makes tuning a guaranteed improvement on the tuning year rather than a gamble. Tuning each model–encoding pair separately is what a practitioner would do and keeps the grid comparison fair. Caching makes reruns cheap after changes that don't affect training, such as new metrics or the backtest.
- **Known limitation:** A tuning year has about 5,500 labelled stock-days, so its MCC has a standard error of about 0.013, as large as the differences between settings. Tuning mostly guards against badly chosen settings rather than finding a strong signal, and the paper says so.

---

## Check-in log

One line per meeting, pointing to the decisions it touched.

| Date | Summary | Decisions |
|---|---|---|
| 2026-09-28 | | D-008 to D-015 |
| 2026-11-03 | Joshua confirmed D-009 to D-014 and added D-019, D-020 (full design latitude from Prof. Danisman). D-008 and D-015 remain open. | D-009 to D-014, D-019, D-020 |

---

## Entry template

```markdown
### D-0XX: Short title
- **Status:** Proposed / Decided (YYYY-MM-DD) / Superseded by D-0YY
- **Decided by:**
- **Decision:**
- **Why:**
- **Alternatives considered:**
- **Consequences:** (what changes in code, config, or the paper)
```
