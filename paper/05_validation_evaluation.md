## 5. Validation

Financial returns are ordered in time, and observations close together in time are related, so ordinary cross-validation with randomly shuffled folds would let a model learn from days that come after the days it is tested on. We instead use walk-forward validation, which reproduces the position of an investor who fits a model on the past and uses it on the following year.

### 5.1 Walk-forward design

The data are split into 18 folds, one for each test year from 2005 to 2022. In each fold the model is trained on every trading day from January 2, 2002 up to the start of the test year, and then predicts every day of that year; the training window therefore grows by one year with each fold, from roughly three years for the 2005 fold to twenty years for the 2022 fold. Each day from 2005 to 2022 is predicted exactly once, by a model that was trained only on earlier data. Training data from all 22 stocks are pooled into one training set, which the volatility scaling of Section 3.1 makes legitimate and which gives the models about 22 times as many examples as training each stock separately.

Splits are made by calendar date rather than by row. Because stocks tend to move together on the same day, splitting rows independently could place one stock's observation for a day in the training set and another stock's observation for the same day in the test set, and the model could learn the day's market move from the first. Splitting by date places every stock's observation for a given day in the same part.

### 5.2 Purging

The label for day *t* is computed from the closing price on day *t* + 1. The last training day before a test year therefore has a label that depends on the first day of the test year. Following López de Prado (2018), we purge from each training set the last *h* trading days before the test year, where *h* is the label horizon (one day for the next-day target, *n* days for an *n*-day target). We verified the purge directly: rewriting every price from the first test day onward leaves every training label unchanged, while the purged days' labels do change. López de Prado also recommends an embargo, an additional gap after each test set, because in k-fold cross-validation some training data follow the test fold. In walk-forward validation no training data follow the test year within a fold, so no embargo is needed.

### 5.3 Hyperparameter tuning

Hyperparameters are chosen inside each training window, never on the test year. Each training window is split again in the same way: the model is fitted on all but its last year, scored on that last year (the tuning set), with a purge of *h* days between the two. The best settings are then refitted on the whole training window before the test year is predicted. Every model family receives the same tuning budget (Section 4), so that differences between models reflect the model class rather than the effort spent tuning it.

### 5.4 Final holdout

The years 2023 to 2025 are reserved as a final holdout and are excluded from all development work. The exclusion is enforced in code: the evaluation functions refuse holdout dates while a configuration flag is off, and the modelling table ends on December 29, 2022, because the label of the last trading day of 2022 depends on a 2023 price. The analysis plan, covering the models, features, metrics, statistical tests and cost assumptions, is fixed and recorded in the repository before the holdout is used, and the holdout is evaluated once.

### 5.5 Reproducibility

The full experiment runs from a single command. Before running, it checks every raw data file against the SHA-256 hashes recorded in the data manifest (Section 2) and stops if any file has changed. All random elements use fixed seeds, and every results file records the git commit of the code that produced it, so each number in this paper can be traced to a specific version of the code.

## 6. Evaluation

Each model outputs, for every stock and test day, a probability that the next day's return is positive; it predicts an up day when that probability is at least 0.5. Predictions are evaluated in two ways: as classifications of direction, and as the signals of a trading strategy.

### 6.1 Classification metrics

We report accuracy, balanced accuracy and the Matthews correlation coefficient (MCC; Matthews, 1975). Because up days outnumber down days, accuracy rewards a model for simply predicting "up" more often. Balanced accuracy, the average of the hit rates on up days and on down days, removes this advantage: a constant prediction scores 0.5. MCC is the correlation between predicted and actual direction, ranging from −1 to +1, and is 0 for any constant prediction regardless of how unbalanced the classes are; it is our headline classification metric (Chicco & Jurman, 2020). Dead-zone days (Section 3.5) have no label and are excluded from these metrics. Metrics are computed over all test days pooled, and separately for each fold and each stock.

### 6.2 Baselines

Every results table includes four simple baselines. The *majority* baseline predicts whichever direction was more common in the training window. The *always-long* baseline always predicts an up day; its accuracy equals the share of up days in the test period, and its MCC is zero. The *previous-day* baseline predicts that each stock moves the same direction tomorrow as it did today. The *random* baseline draws its probability uniformly from [0, 1], with a fixed seed that differs by fold. A model that does not outperform these baselines, and in particular always-long, has not demonstrated predictive value.

### 6.3 Trading backtest

Predictions are converted to positions taken at the close of day *t* and held until the close of day *t* + 1. We evaluate two strategies: *long/short*, which holds +1 unit of a stock when the model predicts an up day and −1 unit otherwise, and *long/flat*, which holds +1 unit or stays in cash. Capital is divided equally among the 22 stocks and rebalanced daily. Positions carry continuously from one test year to the next, so changing folds does not create artificial trades.

Transaction costs are charged in basis points per side on every unit of position changed: entering or exiting a position costs one side, and switching from long to short costs two. Headline results assume 5 basis points per side, and we report results across costs from 0 to 20 basis points to show how any advantage changes as trading becomes more expensive. Dead-zone days are included in the backtest, since a strategy holding a position on such a day still bears its return and any trading cost.

For each strategy we report the compounded annual return, the annualized volatility, the Sharpe ratio (Sharpe, 1966), the maximum drawdown (the largest fall from a previous peak), and the average turnover. The Sharpe ratio is computed as the mean daily return divided by its standard deviation, scaled by √252, without subtracting a risk-free rate, which is appropriate for comparing strategies with one another. The always-long baseline under the long/flat strategy is equivalent to an equal-weighted, daily-rebalanced buy-and-hold portfolio of the 22 stocks, the natural benchmark for any trading strategy built from them.

Trading at a closing price that has just been observed is an optimistic assumption, since in practice the signal would be computed after the close. Section [X] repeats the backtest with positions entered at the next day's open. Statistical tests for whether differences between models exceed what chance would produce are described in Section 7.

### References for these sections

- Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21, 6.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Matthews, B. W. (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme. *Biochimica et Biophysica Acta – Protein Structure*, 405(2), 442–451.
- Sharpe, W. F. (1966). Mutual fund performance. *Journal of Business*, 39(1), 119–138.
