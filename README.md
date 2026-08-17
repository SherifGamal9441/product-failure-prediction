# Product Failure Prediction — a low-signal, distribution-shift problem

Second competition of my machine learning course (ITI Intake 46), on the Kaggle
*Tabular Playground Series — August 2022* dataset.

**Task:** a manufacturer tests each unit of a product and records a set of physical
measurements. Given those measurements, predict the probability that a unit **fails**
quality control.

| | |
|---|---|
| **Type** | Binary classification, probabilistic |
| **Metric** | ROC-AUC |
| **Train / test** | 26,570 / 20,775 rows, 24 features |
| **Best local CV** | **0.5912** AUC |
| **Final model** | Rank-average of a RankGauss + swish MLP and an L2 logistic regression |

If the headline number looks broken — it isn't. The maximum achievable AUC on this
dataset is around **0.59**. The signal is genuinely that weak, and that fact shaped
every decision in the notebook.

## The data

- `product_code` — `A`–`E` in train, **`F`–`I` in test**. The test set contains product
  codes that appear nowhere in training.
- `loading` — a continuous load measurement, heavily right-skewed.
- `attribute_0`–`attribute_3` — product-level properties (constant within a product code).
- `measurement_0`–`measurement_17` — per-unit measurements, with missing values scattered
  through most of them.
- `failure` — the target, ~21% positive.

## Approach

### 1. Validation before modelling

`product_code` differing between train and test means a random split flatters you: the
model memorises product-specific quirks that will never transfer. Everything in this
notebook is scored with **`StratifiedGroupKFold` grouped on `product_code`**, so
validation always measures generalisation to an *unseen product*. That's the honest
proxy for the leaderboard, and it's what makes the 0.59 ceiling visible.

### 2. Linear models beat trees

With the group-aware CV in place:

| Model | Mean AUC | Std |
|---|---|---|
| **Logistic regression** | **0.5893** | 0.0072 |
| XGBoost | 0.5835 | 0.0054 |
| CatBoost | 0.5770 | 0.0069 |

Gradient boosting lost, consistently. When the true signal is a weak, nearly-linear
relationship, trees spend their capacity carving up noise; a strongly regularised linear
model can't. The Optuna search over logistic regression converged to `C ≈ 0.0074` —
i.e. "regularise almost everything away," which is exactly the right answer here.

### 3. Imputation didn't matter

| Strategy | Mean AUC | Std |
|---|---|---|
| Simple mean | 0.5831 | 0.0066 |
| KNN | 0.5832 | 0.0071 |
| Iterative (MICE) | 0.5823 | 0.0076 |

Three approaches, differences well inside one standard deviation. I kept `SimpleImputer`.

### 4. Feature selection and interactions

L1-penalised logistic regression zeroed out `measurement_1`, `_3`, `_6`, `_8`, `_11`,
`_12`, `_15`, `_16` outright. SHAP interaction values on an XGBoost probe identified
`measurement_13` as a hub feature, so I added explicit products
(`measurement_13 × loading`, `× measurement_10`, `× measurement_17`, …). The final
feature set is 10 main effects plus 3 interactions — 13 columns out of 24.

**The bug worth recording:** I log-transformed `loading` in train but not in test, so the
`measurement_13 × loading` interaction was computed on values ~27× larger on one side.
Predicted probabilities came out around 0.02 instead of the expected ~0.21. The fix was
to concatenate train and test, transform once, then split back — and to add a sanity
assertion on the mean predicted probability before writing any submission.

### 5. Neural networks

`QuantileTransformer(output_distribution='normal')` — **RankGauss** — as the scaler,
swish activations, BatchNorm, dropout, and `ReduceLROnPlateau`. Architecture sweep:

| Architecture | Mean AUC | Std |
|---|---|---|
| Funnel 64→32 | 0.5584 | 0.0093 |
| Bottleneck 32→8 | 0.5769 | 0.0056 |
| High dropout 128→64 | 0.5860 | 0.0070 |
| Balanced scale-up | 0.5903 | 0.0048 |
| **Heavy net** | **0.5912** | 0.0055 |
| Funnel taper | 0.5901 | 0.0072 |

RankGauss was worth more than any architecture change — it maps each feature to a
Gaussian by rank, which suits a linear-ish decision boundary and is robust to the skew in
`loading`. I also reimplemented a public winning solution's "fractal tree" network
(recursive branching to depth 5, leaves concatenated) to see whether the exotic topology
was doing the work. It scored in line with the plain heavy net — the preprocessing was
the winner's real edge, not the architecture.

### 6. Blending on ranks, not probabilities

AUC only cares about ordering, so the models are blended by converting each prediction
vector to ranks (`scipy.stats.rankdata / N`) and averaging. This sidesteps the problem
that a well-calibrated logistic regression and an uncalibrated neural net live on
different probability scales — averaging those directly lets the wider distribution
dominate. Final submission: 50/50 rank average of the RankGauss MLP and the tuned
logistic regression. I also tried LDA in place of logistic regression, which held up
about as well.

## What I learned

**Design the validation split before touching a model.** The train/test `product_code`
disjointness is the defining property of this dataset. Grouped CV was the single most
important decision in the notebook; without it every subsequent number would have been
fiction.

**Weak signal inverts the usual model ranking.** "Try gradient boosting first" is good
default advice, and it was wrong here. Low signal-to-noise plus a near-linear
relationship favours a heavily regularised linear model, and `C ≈ 0.007` says so
explicitly.

**Preprocessing must be identical on both sides — and asserted, not assumed.** The
`loading` log-transform bug was invisible in every metric except the mean predicted
probability. Now I check the prediction distribution before every submission; it is the
cheapest bug detector I have.

**Read the noise floor.** With a fold standard deviation of ~0.007, differences smaller
than about 0.014 AUC are not real. That killed several "improvements" — including the
whole imputation comparison — and saved a lot of time.

**Rank-average when the metric is rank-based.** Free robustness, no calibration required.

**Reimplementing a winning solution is a good diagnostic.** Copying the fractal
architecture and finding it *not* better told me the real edge was in the preprocessing.
That's more useful than a score.

## Repo layout

```
notebooks/
  01_experiments.ipynb   full notebook: EDA, grouped CV harness, model and imputation
                         comparisons, SHAP interaction analysis, Optuna searches, the
                         NN architecture sweep, and the rank-blend submissions
submissions/             representative submission files
```

## Reproducing

Download `train.csv`, `test.csv`, and `sample_submission.csv` from the
[competition page](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data)
into the repo root.

```
pip install -r requirements.txt
```

Data files are not committed (see `.gitignore`).
