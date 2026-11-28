## 5. Models

Every model is trained in each walk-forward fold on the same pooled training data and outputs, for each stock and test day, a probability that the next day is an up day; an up day is predicted when this probability is at least 0.5 (Section 7). The models fall into two families. *Pattern-based* models predict by looking up what followed similar candles in the past: a variable-order pattern miner, which matches exact sequences of candle symbols, and k-nearest neighbours, which matches similar candle shapes. *Machine learning* models learn a general function of the encoded candles. The naive baselines are described with the evaluation, in Section 7.2.

### 5.1 Variable-order pattern miner

Candlestick analysis holds that particular short sequences of candles tend to be followed by particular moves. The pattern miner tests this claim directly, without deciding in advance which sequences matter. For every sequence of one to three consecutive daily symbols from the symbolic version of encoding B (Section 4.3), it estimates from the training data the probability that the next day is up, and it predicts each test day from the longest sequence for which that estimate is reliable. In the terminology of sequence prediction, it is a variable-order Markov model over the 20-symbol alphabet (Begleiter et al., 2004).

**Contexts.** The *context* of length *L* on day *t* is the sequence of one stock's symbols on days *t* − *L* + 1 through *t*, oldest first. With 20 symbols there are 20 possible contexts of length 1, 400 of length 2, and 8,000 of length 3. The maximum length of three matches the three-day window given to the other models (Section 5.2) and covers most of the multi-candle patterns in encoding A. Contexts are formed within each stock and never join the last day of one stock to the first day of another.

**Counting.** For every context *c* in the training data, the miner records *n*(*c*), the number of labelled training days that end with *c*, and *u*(*c*), the number of those followed by an up day. Counts are pooled across the 22 stocks, which the volatility scaling of Section 4.1 makes legitimate. A day contributes to the count of every context length for which its symbols exist.

**Shrinkage.** The raw up rate *u*(*c*) / *n*(*c*) is unreliable for contexts seen only a few times, and most long contexts are rare. The first training window (2002–2004) contains 15,166 labelled stock-days, fewer than two per possible three-day context. Of the 8,000 possible three-day contexts, 4,933 occur in it, with a median of two occurrences each, and even the most common occurs only 21 times. Two-day contexts are better supported, with a median of 28 occurrences. A context seen twice has a raw up rate of 0, 0.5 or 1, and choosing the most extreme of thousands of such rates would find "patterns" in pure noise. The miner therefore shrinks each estimate toward the base rate π, the share of up days in the training window:

> *p̂*(*c*) = (*u*(*c*) + *m* π) / (*n*(*c*) + *m*).

This is the *m*-estimate of a probability (Cestnik, 1990), equivalent to the posterior mean under a beta prior centered on π that carries the weight of *m* observations. The base rate counts as *m* additional past days. A context seen *m* times is placed halfway between π and its raw rate, one seen far fewer times stays close to π, and one seen far more often keeps nearly its raw rate. The estimate always lies between the raw rate and the base rate.

**Backoff.** To predict a test day, the miner uses the longest of its contexts (three, two or one day) that occurred at least *n*ₘᵢₙ times in the training data, and the base rate if none did. This mirrors backoff in statistical language models (Katz, 1987). Shrinkage and the minimum count play different roles. Shrinkage sets how far to trust a context's own up rate, while the minimum count decides which contexts are eligible at all. Without a minimum count, a three-day context seen a handful of times would always be chosen over a well-estimated two-day context, even though its shrunk estimate is almost entirely the base rate, and the information in the shorter context would be lost. We shrink toward the base rate rather than toward the next shorter context's estimate, a hierarchical alternative, so that every prediction can be traced to a single context with its count and up rate.

**Transparency.** Because each prediction is a lookup, the miner records which context length produced it. As the training window grows from three years to twenty, more long contexts pass the minimum count. With the development settings (*m* = 100, *n*ₘᵢₙ = 30), no three-day context qualifies in the first training window, so the 2005 predictions come from two-day (78%) and one-day (22%) contexts. By the 2022 fold, 844 three-day contexts qualify, and they produce 45% of that year's predictions. No test day fell back to the base rate in any year.

**Settings.** The maximum context length is fixed at three. The shrinkage weight *m* and the minimum count *n*ₘᵢₙ are tuned inside each training window (Section 5.5). Development runs before tuning used *m* = 100 and *n*ₘᵢₙ = 30.

### 5.2 k-nearest neighbours

The pattern miner requires exact matches between symbols, so two nearly identical candles on either side of a bin edge count as different. k-nearest neighbours (k-NN; Cover & Hart, 1967) instead compares candles by similarity on the continuous version of encoding B. Each stock-day is described by the body, upper wick, lower wick and gap of that day and the preceding days, in ATR units: 12 values for a three-day window and 20 for a five-day window. Both window lengths are reported as separate models.

Within each fold, every feature is standardized using the mean and standard deviation of the training days only, so that no single feature dominates the distances and the test year does not influence the scaling. For each test day, the model finds the *k* training days with the smallest Euclidean distance, drawn from any stock and any earlier date. The predicted probability of an up day is the share of those neighbours that were followed by an up day, a value in steps of 1/*k*; an exact tie at 0.5 is predicted as up. We use an exact (brute-force) search rather than an approximate one. Training windows reach about 100,000 stock-days in the last folds, so the search is the most expensive step of the experiment, but an exact search makes the results exactly reproducible.

Next-day direction is a noisy label, so a small *k* would let a few chance neighbours decide each prediction. With *k* = 100, the up share among the neighbours of a day with no real signal has a standard deviation of about 0.05. *k* is tuned inside each training window (Section 5.5), and development runs used *k* = 100.

### 5.3 Checks on synthetic data

Before applying the pattern miner to real prices, we checked on synthetic data, sized like the first training window (22 stocks, three years), that it finds a pattern when one exists and does not invent one when none does. The miner was run with the same settings as on real data.

*A planted pattern.* Symbols were drawn at random, except that a fixed three-day sequence (two large down days followed by a doji-like day with a long lower wick) was inserted on about 1% of days, and the day after it was up 70% of the time instead of 52%. In each of 20 independent samples, the miner kept the planted context and ranked its estimate first of about 400 retained contexts in 18 samples, and never lower than third. In a newly generated test year, it predicted an up day every time the pattern appeared, using the full three-day context, and was correct on 64–76% of those days. Because shrinkage pulls the estimate toward the base rate, the estimated probability (0.58–0.68) was below the true 70%.

This exercise also shows how little a genuine but rare pattern contributes to the headline metric. Pooled over all test days, the miner's MCC averaged 0.000 across the 20 samples (range −0.022 to 0.011), no different from its MCC on pure noise. A pattern that is right 70% of the time on 1% of days is invisible in the pooled MCC. The coverage analysis (Sections 8.7 and 9.5) is designed to detect edges of this kind, which are concentrated in a model's most confident predictions.

*Pure noise.* When symbols and labels were independent, every estimate stayed within 0.1 of the base rate and the out-of-sample MCC was indistinguishable from zero. The difference between accuracy on the training data and on new data measures how much a model has memorized. That gap was about 25 percentage points for a miner without shrinkage or a minimum count (76% against 50%), and about 4 points (at most 5) with our settings.

*The full pipeline.* Finally, a two-day pattern was planted in synthetic open, high, low and close prices rather than in symbols, and passed through the same encoding, labelling and table construction as the real data. The miner recovered it as the strongest two-day context. When the labels were deliberately shifted by one day, the pattern disappeared, which confirms that the check would detect a misalignment between features and labels anywhere in the pipeline. All of these checks are part of the project's automated test suite.

### 5.4 Machine learning models

Four standard classifiers represent the machine learning side of the comparison: a random forest, gradient-boosted trees (LightGBM), a support vector machine with a radial basis function kernel, and a small multilayer perceptron. They span the main families of general-purpose classifiers: averaged trees, boosted trees, kernel methods and neural networks. The random forest, support vector machine and perceptron come from scikit-learn 1.9.1 (Pedregosa et al., 2011), and gradient boosting from LightGBM 4.7.0 (Ke et al., 2017). Like the pattern-based models, each outputs a probability of an up day, and an up day is predicted when that probability is at least 0.5.

**The model × encoding grid.** Each of the four models is trained separately on each of three inputs, giving twelve models named by model and encoding (for example `svm_b`):

- *Encoding A*: the 61 TA-Lib pattern indicators for day *t* (Section 4.2).
- *Encoding B*: body, upper wick, lower wick and gap, in ATR units, for days *t*, *t* − 1 and *t* − 2 (12 values; Section 4.3).
- *Normalized prices*: open, high, low and close relative to the previous close, in ATR units, for the same three days (12 values; Section 4.4).

Encoding B and normalized prices contain exactly the same information, each computable from the other, so comparing them isolates the effect of describing a candle by its shape. Comparing encoding A with either isolates the effect of reducing candles to named patterns. Comparing models within one encoding isolates the effect of the model class.

**Random forest.** A random forest (Breiman, 2001) averages many decision trees, each grown on a bootstrap sample of the training days and choosing each split from a random subset of the features. Its probability of an up day is the average, over trees, of the share of up days in the leaf a test day falls into. We use 300 trees. Next-day direction is a noisy label, so leaves are required to contain many training days; the minimum leaf size and the number of features tried at each split are tuned (Section 5.5).

**Gradient-boosted trees.** LightGBM (Ke et al., 2017) builds trees one after another, each correcting the errors of those before it. Before tuning, it uses 200 trees with 15 leaves each, a learning rate of 0.05, at least 500 training days per leaf, and random subsamples of 80% of the training days and 80% of the features for each tree. These settings are deliberately conservative, for the same reason as the random forest's large leaves.

**Support vector machine.** A support vector machine with a radial basis function kernel (Cortes & Vapnik, 1995) separates up days from down days with a flexible boundary, controlled by the penalty *C* and the kernel width γ. Features are standardized using the mean and standard deviation of the training days, so that the test year cannot affect the scaling. Two adaptations were needed.

First, training time grows roughly with the square of the number of training days, and on data this noisy nearly all training days become support vectors. A full training window of up to about 100,000 stock-days would take on the order of ten minutes per fit, which is impractical across 18 folds, three encodings and the tuning budget. Each support vector machine is therefore trained on a random sample of at most 20,000 training days, drawn with a fixed seed from the whole training window, so that the sample covers all the market conditions the other models see. Windows smaller than this (the 2005 fold has 15,166 labelled days) are used in full. The cap was fixed before any support vector machine results were produced. Because it means these models see fewer days than the others in later folds, it works against them.

Second, a support vector machine produces a signed distance from its decision boundary rather than a probability. We use the logistic function of this distance as the model's score. The score is above 0.5 exactly when the model predicts an up day, and it ranks days by their distance from the boundary, which is all that classification and the coverage analysis require. It is not a calibrated probability. We do not use scikit-learn's built-in calibration, which fits five additional models by internal cross-validation and can disagree with the model's own predictions.

**Multilayer perceptron.** The neural network has two hidden layers of 32 and 16 units with ReLU activations and is trained with the Adam optimizer (Kingma & Ba, 2015) in batches of 512 days, with an L2 penalty on the weights. Training stops when accuracy on a random 10% of the training days has not improved for ten passes through the data, or after 200 passes. Features are standardized on the training days, as for the support vector machine. We keep the network small because the signal is weak, and larger networks would mainly add capacity to fit noise. Deeper sequence models are left to future work.

**Today's-return ablation.** Results during development showed a short-term reversal effect: the previous-day baseline was below chance in most years (Section 7.2). To test whether the candle-based models do more than detect reversal, LightGBM is also trained on a single feature, day *t*'s close-to-close return in ATR units. If this one-feature model performed as well as the models given full candles, any apparent candlestick signal would reduce to reversal. This model was added after the development results were seen, and we report it as a post hoc analysis.

**Reproducibility.** Every model that uses randomness is seeded with the project seed plus the test year, so each fold is reproducible while folds differ from one another. LightGBM runs with a fixed number of threads, and the random forest averages its trees' predictions on a single thread, because otherwise the order in which results are added can change predictions in the last decimal places. Repeated runs give identical results.

### 5.5 Hyperparameter tuning

Hyperparameters are tuned inside each training window, as described in Section 6.3: each candidate setting is fitted on all but the last year of the window and scored by MCC on that last year, with the same one-day purge between the two. Every model family, the pattern-based models included, receives the same budget of 20 trials, chosen with Optuna's tree-structured Parzen estimator (Akiba et al., 2019; Bergstra et al., 2011), so that differences between families reflect the model class rather than the effort spent tuning it. The first trial is always the family's starting settings (Sections 5.1, 5.2 and 5.4), so tuning never chooses settings that did worse than these on the tuning year. The best settings are refitted on the whole training window before the test year is predicted.

Settings are re-tuned in the folds beginning 2005, 2010, 2015 and 2020, and the folds in between reuse the most recent settings. Each model is tuned separately on its own encoding, so that, for example, LightGBM on encoding A and on encoding B may use different settings. The search ranges are listed in Table 3. The baselines have nothing to tune, and the k-NN window length and the pattern miner's maximum context length are not tuned, because they define separate models.

**Table 3. Hyperparameter search ranges** (log scale unless stated)

| Model | Hyperparameter | Range |
|---|---|---|
| LightGBM | number of trees | 50–500 |
| | learning rate | 0.01–0.2 |
| | leaves per tree | 4–64 |
| | minimum training days per leaf | 50–2,000 |
| | share of features per tree | 0.5–1.0 (linear) |
| | L2 penalty | 0.001–10 |
| Random forest | minimum training days per leaf | 50–2,000 |
| | features tried per split | √*p*, 25%, 50% or 100% of *p* |
| Support vector machine | penalty *C* | 0.01–100 |
| | kernel width γ | 0.001–1 |
| Multilayer perceptron | hidden layers | (16), (32, 16) or (64, 32) |
| | L2 penalty | 10⁻⁵–0.1 |
| | initial learning rate | 10⁻⁴–0.01 |
| k-nearest neighbours | *k* | 10–1,000 |
| Pattern miner | shrinkage weight *m* | 1–1,000 |
| | minimum count *n*ₘᵢₙ | 5–500 |

*p* is the number of input features.

A tuning year has about 5,000 labelled stock-days, so MCC measured on it has a standard error of roughly 0.014, comparable to the differences between settings. Tuning therefore cannot reliably find the best settings; it mainly protects each model family from being judged on poorly chosen ones. The development results bear this out. Across the 64 tuning runs (16 models × 4 retune years), the chosen settings averaged an MCC of 0.039 on their tuning years, against 0.021 for the starting settings. Yet test-year performance changed far less, in both directions. LightGBM on encoding B rose from an MCC of 0.022 with its starting settings to 0.030 when tuned, while LightGBM on encoding A fell from 0.021 to 0.019 and k-NN on three days from 0.019 to 0.017. The chosen settings also varied widely from one retune year to the next. For k-NN on three days, *k* was 717, 953, 49 and 944. For the pattern miner, *m* ranged from 2 to 315. This pattern is what tuning on a weak, noisy signal would be expected to produce, and it is the reason all comparisons in this paper are made on test years that played no part in tuning.

### References for this section

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631).
- Begleiter, R., El-Yaniv, R., & Yona, G. (2004). On prediction using variable order Markov models. *Journal of Artificial Intelligence Research*, 22, 385–421.
- Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. In *Advances in Neural Information Processing Systems 24* (pp. 2546–2554).
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Cestnik, B. (1990). Estimating probabilities: A crucial task in machine learning. In *Proceedings of the 9th European Conference on Artificial Intelligence* (pp. 147–149).
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.
- Katz, S. M. (1987). Estimation of probabilities from sparse data for the language model component of a speech recognizer. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 35(3), 400–401.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30* (pp. 3146–3154).
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations*.
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
