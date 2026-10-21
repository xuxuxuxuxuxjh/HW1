# Supervised Learning Assignment Report

## Selected tasks and setup

This report uses the supplied `sarcos_regression` and `execution_classification` datasets. Both contain 5,000 labeled rows and have no missing input cells. The supplied five-fold assignments are used unchanged: each run trains on 4,000 rows and validates on the remaining 1,000. The seed is 2026. Identifiers and fold labels are excluded from model inputs.

All preprocessing is in the training pipeline and is fit only on the training rows of each fold. For both tasks, the baseline uses median imputation followed by `StandardScaler`. The imputer has no effect on these complete datasets, but keeps the pipeline robust to missing values. The regression learner is Ridge (`alpha=1.0`); the classifier is Logistic Regression (`C=1.0`, `max_iter=2000`). The preprocessing comparison changes only the scaling step: `unscaled` replaces `StandardScaler` with passthrough. Hyperparameters and folds are held fixed.

Bagging uses 100 bootstrap samples, each the size of the fold training set, with replacement. Each sample fits the same Ridge or Logistic Regression learner on the same imputed and scaled feature representation; predictions or class-1 probabilities are averaged. The seed is 2026 and `n_jobs=1`.

## Experiment settings

All methods use seed 2026 and the supplied five folds. Baseline: median imputation, StandardScaler, Ridge (`alpha=1`) for regression or Logistic Regression (`C=1`, `max_iter=2000`) for classification. `unscaled` changes only the scaler. Bagging uses 100 bootstrap fits of the baseline learner.

### SARCOS regression

**Data diagnosis**

Rows: 5,000  
Features: 21 (7 positions, 7 velocities, 7 accelerations)  
Missing values: 0  
Feature ranges: positions -0.923 to 2.249; velocities -4.734 to 4.683; accelerations -47.764 to 70.845  
Skew/unusual values: `position_6` skewness -1.67; negative and extreme operating values are valid and retained  
Target: `torque_1`, -90.39 to 112.27; mean 13.26; errors use the original scale

**Five-fold results** (mean ± sample SD)

| Method | Train RMSE | Validation RMSE | Validation MAE | Validation R² |
|---|---:|---:|---:|---:|
| Baseline | 5.4912 ± 0.0277 | 5.5305 ± 0.1083 | 3.9639 ± 0.0923 | 0.9265 ± 0.0051 |
| Unscaled | 5.4914 ± 0.0277 | 5.5303 ± 0.1083 | 3.9629 ± 0.0914 | 0.9265 ± 0.0052 |
| Bagging | 5.4913 ± 0.0277 | 5.5303 ± 0.1079 | 3.9638 ± 0.0939 | 0.9265 ± 0.0051 |

**Preprocessing comparison by fold** (delta = unscaled − baseline; lower RMSE is better)

| Fold | Baseline RMSE | Unscaled RMSE | Delta |
|---:|---:|---:|---:|
| 0 | 5.543601 | 5.545206 | +0.001605 |
| 1 | 5.624360 | 5.623460 | -0.000901 |
| 2 | 5.471403 | 5.473398 | +0.001994 |
| 3 | 5.377460 | 5.375349 | -0.002111 |
| 4 | 5.635466 | 5.633868 | -0.001598 |

Scaling has virtually no effect: the small differences change direction across folds. Bagging also has virtually no effect for this Ridge setting.

**Learning curve** (mean ± sample SD across folds; validation rows fixed)

| Training rows | Train RMSE | Validation RMSE |
|---:|---:|---:|
| 500 | 5.2722 ± 0.1729 | 5.7050 ± 0.1256 |
| 1,000 | 5.4015 ± 0.1865 | 5.5761 ± 0.0888 |
| 2,000 | 5.4342 ± 0.1099 | 5.5548 ± 0.1140 |
| 3,000 | 5.4981 ± 0.0108 | 5.5471 ± 0.1020 |
| 4,000 | 5.4912 ± 0.0277 | 5.5305 ± 0.1083 |

Validation RMSE improves as training size grows, with smaller gains after about 1,000–2,000 rows. The train-validation gap narrows from about 0.43 at 500 rows to 0.04 at 4,000, suggesting more data helps this fixed Ridge model. The conclusion is limited to `alpha=1` and these folds. The plot is saved as `runs/sarcos_regression/learning_curve_rmse.png`.

**Selected method / full-development training score**: `unscaled`; RMSE 5.4953, MAE 3.9382, R² 0.9278.

### Execution classification

**Data diagnosis**

Rows: 5,000  
Features: 20 (four levels of bid/ask prices and quantities, order size, side, latency, volatility)  
Missing values: 0  
Feature ranges: prices 24.99 to 200.09; quantities and order size 25 to 4,365  
Skewness: `order_size` 0.93; price features about 0.71  
Side: -1 = sell (2,488 rows); +1 = buy (2,512 rows); both values are valid  
Target: class 0 = 3,608 (72.16%); class 1 = 1,392 (27.84%); class imbalance makes probability metrics useful

**Five-fold results** (mean ± sample SD)

| Method | Train log loss | Validation log loss | Validation ROC-AUC | Validation accuracy | Validation F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.5262 ± 0.0024 | 0.5291 ± 0.0099 | 0.7265 ± 0.0146 | 0.7218 ± 0.0085 | 0.2689 ± 0.0248 |
| Unscaled | 0.5243 ± 0.0024 | 0.5273 ± 0.0099 | 0.7289 ± 0.0146 | 0.7240 ± 0.0078 | 0.2781 ± 0.0237 |
| Bagging | 0.5261 ± 0.0024 | 0.5289 ± 0.0096 | 0.7267 ± 0.0143 | 0.7218 ± 0.0090 | 0.2712 ± 0.0280 |

**Preprocessing comparison by fold** (delta = unscaled − baseline; lower log loss is better)

| Fold | Baseline log loss | Unscaled log loss | Delta |
|---:|---:|---:|---:|
| 0 | 0.535485 | 0.533529 | -0.001956 |
| 1 | 0.513054 | 0.510817 | -0.002237 |
| 2 | 0.535559 | 0.533700 | -0.001860 |
| 3 | 0.526015 | 0.525111 | -0.000904 |
| 4 | 0.535377 | 0.533586 | -0.001791 |

Unscaled inputs reduce log loss in all five folds, but the average gain is small (about 0.00175). Bagging changes mean validation log loss from 0.5291 to 0.5289, so it gives little benefit for this Logistic Regression setting.

**Selected method / full-development training score**: `unscaled`; log loss 0.5244, ROC-AUC 0.7334, accuracy 0.7238, F1 0.2796.

These full-development scores are training scores, not held-out test results. The test data have not been provided.

## Reproduction and limitations

Open `homework.ipynb`, run the cells from top to bottom for one dataset, then change `DATASET` in the setup cell and repeat for the other task. Keep `train.py`, `test.py`, and `datasets/` beside it. The experiment cell writes fold scores and summaries under `runs/`. The tested environment was Python 3.14.7, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, and Matplotlib 3.11.2; the notebook also installs compatible package versions. All experiments use seed 2026. No public labels or held-out answers were retrieved.

The SARCOS source is the [SARCOS dataset page](https://gaussianprocess.org/gpml/data/). Execution risk is a supplied synthetic simulation; its outcomes include random future market changes and should not be interpreted as real trading performance. The evaluation compares a small set of linear learners and fixed hyperparameters, so it does not establish the best possible model.

## AI use disclosure

OpenAI Codex assisted with implementing the baseline comparison and Bagging methods in `train.py`, organizing notebook experiments, running the supplied cross-validation and learning-curve analysis, and drafting this report. The student should review the code and results and be prepared to explain the choices and conclusions.
