## 7. Statistical Analysis

Differences between models in this study are small, so the main statistical risk is mistaking chance for skill. Two features of the data make this risk larger than ordinary tests assume. First, the 22 stocks move together: on a day when the market falls, most of them fall, and a model's 22 predictions for that day are not 22 independent observations. Second, returns and prediction errors are correlated over time, especially in volatile periods. Every test below therefore works on one value per day, averaged across stocks, uses standard errors that allow for correlation over time, and resamples whole dates in blocks. All tests, their settings and the criteria for a positive finding were fixed in an analysis plan before the final holdout was used (Section 5.4); analyses added after development results had been seen are identified as such.

### 7.1 Daily loss series

Each model is represented by two daily loss series, lower being better.

*Classification loss.* On each day, the model's wrong calls on labelled stock-days are counted, with a wrong call on an up day weighted 1/(2π) and a wrong call on a down day 1/(2(1 − π)), where π is the share of up days in that test year; the weighted count is divided by the number of labelled stock-days that day. This is balanced misclassification, the day-by-day counterpart of one minus balanced accuracy. A plain error rate would reward a model for predicting up days more often, because up days are the majority (52% of labelled stock-days); with the weights, up and down days count equally, and any constant prediction scores about 0.5, consistent with its MCC of zero. The weights use the test year's share of up days, which is part of scoring, as in balanced accuracy, and is never seen by a model. Days on which every stock is in the dead zone have no classification loss.

*Economic loss.* Minus the day's net return of the equal-weighted long/short portfolio at 5 basis points per side (Section 6.3).

### 7.2 Comparing two models: Diebold–Mariano tests

To compare two models, we take the day-by-day difference of their losses and test whether its mean is zero (Diebold & Mariano, 1995). The test is a regression of the difference on a constant, with Newey–West standard errors (Newey & West, 1987) using a Bartlett kernel and lag ⌊4(*T*/100)^(2/9)⌋, which is 9 for the 4,530 days of the development period. The resulting statistic is compared with a standard normal distribution; tests are two-sided at the 5% level, and a negative statistic means the first model has the lower loss. Each model is compared with the strongest baseline on the classification loss and with always-long, which is buy-and-hold, on the economic loss.

### 7.3 Comparing all models: the Model Confidence Set and the SPA test

Comparing many models pair by pair invites false positives. Two tests address the comparison as a whole. The Model Confidence Set (Hansen, Lunde & Nason, 2011) is the set of models that contains the best model with a given confidence; models significantly worse than the best are eliminated one at a time. We compute it at 90% confidence, with the range statistic, for both loss series. The superior predictive ability (SPA) test (Hansen, 2005) asks a different question: can any model beat a given benchmark at all? We report its studentized, consistent *p*-value, with the strongest baseline as the benchmark for the classification loss and always-long for the economic loss. Both use 10,000 replications of the stationary bootstrap (Politis & Romano, 1994) with a mean block length of 20 trading days, as implemented in the `arch` package (Sheppard, [YEAR]). Always-long and the majority baseline make identical predictions in every test year and enter both procedures once.

### 7.4 Confidence intervals

For each model's MCC, accuracy, balanced accuracy and Sharpe ratios, and for the difference between each model and the strongest baseline in MCC and between each model and buy-and-hold in Sharpe ratio, we report 95% percentile intervals from the same stationary bootstrap. The bootstrap resamples dates, in blocks of random length with a mean of 20 trading days, and every stock's result on a drawn date comes with it, so the dependence across stocks and over time is kept. Every model is evaluated on the same resampled dates in each of the 10,000 replications, so that differences between models are bootstrapped consistently.

### 7.5 What counts as beating a baseline

The criteria below were fixed before the holdout was used. A model beats a baseline in classification if its pooled MCC is higher *and* its classification loss is significantly lower by a Diebold–Mariano test. It beats a baseline economically if its long/short Sharpe ratio at 5 basis points is higher *and* its economic loss is significantly lower. A model beats the baselines only if it beats the strongest of them: the baseline with the highest pooled MCC for classification, which in the development period is the reversal baseline, and buy-and-hold for trading. A model shows *genuine predictive value* if, in addition, the strongest baseline is excluded from the 90% Model Confidence Set while the model is in it. The number of years in which a model beats the baseline is reported for context but is not a requirement. If no model meets these criteria, that is the study's finding.

### 7.6 Market exposure and break-even cost

A long/short strategy that predicts up days more often than down days is net long, and in a rising market part of its return is market exposure that buy-and-hold earns for free. For each strategy, we regress its daily net return on the daily return of the equal-weighted portfolio of the 22 stocks. The slope (beta) measures market exposure. The intercept (alpha), annualized and tested with the same Newey–West standard errors as in Section 7.2, measures the return the strategy earns beyond its exposure.

The break-even cost of a strategy is the cost per side at which its compounded net return over the test period is zero. Each day's net return falls linearly with the cost, by the day's turnover times the cost, so the break-even cost is the single root of the total log growth, which we find numerically. The cost sweep of Section 6.3 shows the same information at six fixed costs.

### 7.7 Confident predictions

A model's edge may be concentrated on the days it is most confident about, where the pooled MCC would hide it. A model's confidence on a stock-day is |*P*(up) − 0.5|, the distance of its prediction from a coin flip. Choosing confidence cutoffs on the test year itself would bias the result, so for each fold we fit the model, with that fold's settings, on its training window minus the last year, and predict that year (the validation year, the same split used for tuning). From these predictions, we find the cutoffs that would select the top 50%, 25% and 10% of the validation year's stock-days. We then apply them unchanged to the test year and report accuracy and MCC against the share of test stock-days actually selected. Models that give every day the same confidence, such as the reversal baseline, cannot rank days and act on all of them.

A model's most confident days might simply be days on which short-term reversal works especially well. To test for this, we score the reversal baseline on exactly the stock-days each model selects, and give the difference in MCC a bootstrap interval as in Section 7.4. This comparison was added after the coverage results had been seen and is exploratory.

### 7.8 Volatility regimes

Each day is assigned to a low-, medium- or high-volatility regime by the realized volatility of SPY over the 21 trading days ending that day, compared with the terciles of all earlier values since January 2002. The cutoffs for a day use only data available before it, so the regime is something a trader could have known at the time. Every model is re-evaluated within each regime with the same metrics and tests, against the same benchmarks as over all days. A robustness version uses the VIX in place of realized volatility. Because a few crises may dominate the high-volatility regime, its results are broken down by episode (2008–09, 2011, 2020, 2022 and the remaining days), and recomputed with each episode left out.

### 7.9 The confirmatory test on the holdout

The exploratory comparison of Section 7.7 suggested that, on their most confident days, several models beat reversal. Before the 2023–2025 holdout was unlocked, this was registered as the study's single confirmatory hypothesis. Six models were named in advance: those whose development intervals were above zero. For each of these models *m*, Δₘ is its MCC on its 10% most confident holdout stock-days minus reversal's MCC on the same stock-days.

- **Primary hypothesis (H1):** the mean of the six Δₘ is greater than zero. This is one test, one-sided, at the 5% level, with a bootstrap *p*-value equal to the share of replications in which the mean is at most zero.
- **Secondary hypotheses (H2):** each Δₘ > 0, with the Holm–Bonferroni correction (Holm, 1979) for six tests at a family-wise level of 5%.

A single averaged statistic was chosen as the primary test because the holdout is short. Judging from the development intervals, each Δₘ would have a standard error of about 0.03 over three years, and the expected power of H1 was below 50% even if the development estimates were accurate. Estimates for models chosen because they looked good tend to be too high, so the true power was probably lower. The plan therefore stated in advance that a non-rejection would be reported as inconclusive rather than as evidence against the effect.

### References for this section

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23(4), 365–380.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453–497.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Sheppard, K. *arch: ARCH models and other tools for financial econometrics in Python* (version 8.0.0) [Software]. https://github.com/bashtage/arch
