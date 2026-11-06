## 4. Models

Every model is trained in each walk-forward fold on the same pooled training data and outputs, for each stock and test day, a probability that the next day is an up day; an up day is predicted when this probability is at least 0.5 (Section 6). The models fall into two families. *Pattern-based* models predict by looking up what followed similar candles in the past: a variable-order pattern miner, which matches exact sequences of candle symbols, and k-nearest neighbours, which matches similar candle shapes. *Machine learning* models learn a general function of the encoded candles. The naive baselines are described with the evaluation, in Section 6.2.

### 4.1 Variable-order pattern miner

Candlestick analysis holds that particular short sequences of candles tend to be followed by particular moves. The pattern miner tests this claim directly, without deciding in advance which sequences matter. For every sequence of one to three consecutive daily symbols from the symbolic version of encoding B (Section 3.3), it estimates from the training data the probability that the next day is up, and it predicts each test day from the longest sequence for which that estimate is reliable. In the terminology of sequence prediction, it is a variable-order Markov model over the 20-symbol alphabet (Begleiter et al., 2004).

**Contexts.** The *context* of length *L* on day *t* is the sequence of one stock's symbols on days *t* − *L* + 1 through *t*, oldest first. With 20 symbols there are 20 possible contexts of length 1, 400 of length 2, and 8,000 of length 3. The maximum length of three matches the three-day window given to the other models (Section 4.2) and covers most of the multi-candle patterns in encoding A. Contexts are formed within each stock and never join the last day of one stock to the first day of another.

**Counting.** For every context *c* in the training data, the miner records *n*<sub>c</sub>, the number of labelled training days that end with *c*, and *u*<sub>c</sub>, the number of those followed by an up day. Counts are pooled across the 22 stocks, which the volatility scaling of Section 3.1 makes legitimate. A day contributes to the count of every context length for which its symbols exist.

**Shrinkage.** The raw up rate *u*<sub>c</sub> / *n*<sub>c</sub> is unreliable for contexts seen only a few times, and most long contexts are rare. The first training window (2002–2004) contains 15,166 labelled stock-days, fewer than two per possible three-day context. Of the 8,000 possible three-day contexts, 4,933 occur in it, with a median of two occurrences each, and even the most common occurs only 21 times. Two-day contexts are better supported, with a median of 28 occurrences. A context seen twice has a raw up rate of 0, 0.5 or 1, and choosing the most extreme of thousands of such rates would find "patterns" in pure noise. The miner therefore shrinks each estimate toward the base rate π, the share of up days in the training window:

> *p̂*<sub>c</sub> = (*u*<sub>c</sub> + *m* π) / (*n*<sub>c</sub> + *m*).

This is the *m*-estimate of a probability (Cestnik, 1990), equivalent to the posterior mean under a beta prior centered on π that carries the weight of *m* observations. The base rate counts as *m* additional past days. A context seen *m* times is placed halfway between π and its raw rate, one seen far fewer times stays close to π, and one seen far more often keeps nearly its raw rate. The estimate always lies between the raw rate and the base rate.

**Backoff.** To predict a test day, the miner uses the longest of its contexts (three, two or one day) that occurred at least *n*ₘᵢₙ times in the training data, and the base rate if none did. This mirrors backoff in statistical language models (Katz, 1987). Shrinkage and the minimum count play different roles. Shrinkage sets how far to trust a context's own up rate, while the minimum count decides which contexts are eligible at all. Without a minimum count, a three-day context seen a handful of times would always be chosen over a well-estimated two-day context, even though its shrunk estimate is almost entirely the base rate, and the information in the shorter context would be lost. We shrink toward the base rate rather than toward the next shorter context's estimate, a hierarchical alternative, so that every prediction can be traced to a single context with its count and up rate.

**Transparency.** Because each prediction is a lookup, the miner records which context length produced it. As the training window grows from three years to twenty, more long contexts pass the minimum count. With the development settings (*m* = 100, *n*ₘᵢₙ = 30), no three-day context qualifies in the first training window, so the 2005 predictions come from two-day (78%) and one-day (22%) contexts. By the 2022 fold, 844 three-day contexts qualify, and they produce 45% of that year's predictions. No test day fell back to the base rate in any year.

**Settings.** The maximum context length is fixed at three. The shrinkage weight *m* and the minimum count *n*ₘᵢₙ are tuned inside each training window (Section 4.5). Development runs before tuning used *m* = 100 and *n*ₘᵢₙ = 30.

### 4.2 k-nearest neighbours

The pattern miner requires exact matches between symbols, so two nearly identical candles on either side of a bin edge count as different. k-nearest neighbours (k-NN; Cover & Hart, 1967) instead compares candles by similarity on the continuous version of encoding B. Each stock-day is described by the body, upper wick, lower wick and gap of that day and the preceding days, in ATR units: 12 values for a three-day window and 20 for a five-day window. Both window lengths are reported as separate models.

Within each fold, every feature is standardized using the mean and standard deviation of the training days only, so that no single feature dominates the distances and the test year does not influence the scaling. For each test day, the model finds the *k* training days with the smallest Euclidean distance, drawn from any stock and any earlier date. The predicted probability of an up day is the share of those neighbours that were followed by an up day, a value in steps of 1/*k*; an exact tie at 0.5 is predicted as up. We use an exact (brute-force) search rather than an approximate one. Training windows reach about 100,000 stock-days in the last folds, so the search is the most expensive step of the experiment, but an exact search makes the results exactly reproducible.

Next-day direction is a noisy label, so a small *k* would let a few chance neighbours decide each prediction. With *k* = 100, the up share among the neighbours of a day with no real signal has a standard deviation of about 0.05. *k* is tuned inside each training window (Section 4.5), and development runs used *k* = 100.

### 4.3 Checks on synthetic data

Before applying the pattern miner to real prices, we checked on synthetic data, sized like the first training window (22 stocks, three years), that it finds a pattern when one exists and does not invent one when none does. The miner was run with the same settings as on real data.

*A planted pattern.* Symbols were drawn at random, except that a fixed three-day sequence (two large down days followed by a doji-like day with a long lower wick) was inserted on about 1% of days, and the day after it was up 70% of the time instead of 52%. In each of 20 independent samples, the miner kept the planted context and ranked its estimate first of about 400 retained contexts in 18 samples, and never lower than third. In a newly generated test year, it predicted an up day every time the pattern appeared, using the full three-day context, and was correct on 64–76% of those days. Because shrinkage pulls the estimate toward the base rate, the estimated probability (0.58–0.68) was below the true 70%.

This exercise also shows how little a genuine but rare pattern contributes to the headline metric. Pooled over all test days, the miner's MCC averaged 0.000 across the 20 samples (range −0.022 to 0.011), no different from its MCC on pure noise. A pattern that is right 70% of the time on 1% of days is invisible in the pooled MCC. The coverage analysis (Section [X]) is designed to detect edges of this kind, which are concentrated in a model's most confident predictions.

*Pure noise.* When symbols and labels were independent, every estimate stayed within 0.1 of the base rate and the out-of-sample MCC was indistinguishable from zero. The difference between accuracy on the training data and on new data measures how much a model has memorized. That gap was about 25 percentage points for a miner without shrinkage or a minimum count (76% against 50%), and about 4 points (at most 5) with our settings.

*The full pipeline.* Finally, a two-day pattern was planted in synthetic open, high, low and close prices rather than in symbols, and passed through the same encoding, labelling and table construction as the real data. The miner recovered it as the strongest two-day context. When the labels were deliberately shifted by one day, the pattern disappeared, which confirms that the check would detect a misalignment between features and labels anywhere in the pipeline. All of these checks are part of the project's automated test suite.

### 4.4 Machine learning models

[To be written: random forest, LightGBM, RBF-kernel SVM and a small multilayer perceptron, each trained on encoding A, continuous encoding B (three-day window) and normalized prices, plus the LightGBM ablation on today's return only (Section [X]).]

### 4.5 Hyperparameter tuning

Hyperparameters are tuned inside each training window, as described in Section 5.3: each candidate setting is fitted on all but the last year of the window and scored by MCC on that last year. Every model family, the pattern-based models included, receives the same budget of 20 trials with Optuna (Akiba et al., 2019), so that differences between families reflect the model class rather than the effort spent tuning it. Settings are re-tuned in the folds beginning 2005, 2010, 2015 and 2020, and the folds in between reuse the most recent settings. The search ranges are listed in Table [X]. [To be completed when tuning is implemented: search ranges for *m*, *n*ₘᵢₙ and *k*, and for the machine learning models.]

### References for this section

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631).
- Begleiter, R., El-Yaniv, R., & Yona, G. (2004). On prediction using variable order Markov models. *Journal of Artificial Intelligence Research*, 22, 385–421.
- Cestnik, B. (1990). Estimating probabilities: A crucial task in machine learning. In *Proceedings of the 9th European Conference on Artificial Intelligence* (pp. 147–149).
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.
- Katz, S. M. (1987). Estimation of probabilities from sparse data for the language model component of a speech recognizer. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 35(3), 400–401.
