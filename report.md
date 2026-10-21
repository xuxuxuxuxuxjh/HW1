# Supervised Learning Assignment Report

## Selected tasks and setup

This report uses the supplied `sarcos_regression` and `execution_classification` datasets. Both contain 5,000 labeled rows and have no missing input cells. The supplied five-fold assignments are used unchanged: each run trains on 4,000 rows and validates on the remaining 1,000. The seed is 2026. Identifiers and fold labels are excluded from model inputs.

All preprocessing is in the training pipeline and is fit only on the training rows of each fold. For both tasks, the baseline uses median imputation followed by `StandardScaler`. The imputer has no effect on these complete datasets, but keeps the pipeline robust to missing values. The regression learner is Ridge (`alpha=1.0`); the classifier is Logistic Regression (`C=1.0`, `max_iter=2000`). The preprocessing comparison changes only the scaling step: `unscaled` replaces `StandardScaler` with passthrough. Hyperparameters and folds are held fixed.

Bagging uses 100 bootstrap samples, each the size of the fold training set, with replacement. Each sample fits the same Ridge or Logistic Regression learner on the same imputed and scaled feature representation; predictions or class-1 probabilities are averaged. The seed is 2026 and `n_jobs=1`.

## Data diagnosis

### SARCOS robot torque regression

The 21 inputs are seven joint positions, seven velocities, and seven accelerations. There are 5,000 complete rows. Their scales differ substantially: across the seven joint position columns, observed values range from about -0.923 to 2.249; velocities range from -4.734 to 4.683; accelerations range from -47.764 to 70.845. Negative values are valid operating conditions. `position_6` is left-skewed (skewness -1.67); most feature skewness is smaller. The target `torque_1` ranges from -90.39 to 112.27, with mean 13.26 and standard deviation 20.45. The target is kept on its original scale for all reported errors.

### Execution slippage classification

The 20 inputs describe four levels of bid/ask prices and quantities, order size, direction, latency, and volatility. There are 5,000 complete rows. Prices range from about 24.99 to 200.09; quantities and order size are more widely spread, with values up to 4,365. `order_size` is right-skewed (skewness 0.93); the price columns have skewness around 0.71. `side` contains 2,512 buys (`1`) and 2,488 sells (`-1`); it is retained as a numeric binary direction, where the signs represent opposite order directions. The target has 3,608 negatives and 1,392 positives (72.16% / 27.84%). Because of this imbalance, log loss and ROC-AUC are more informative than accuracy alone. No logarithmic transform is applied to prices or sizes.

Per-feature ranges, quartiles, missing rates, and skewness are saved in `runs/sarcos_regression/feature_diagnosis.csv` and `runs/execution_classification/feature_diagnosis.csv`.

## Cross-validation results

Values below are mean and sample standard deviation over the five validation folds.

| Task / method | Main metric | Other metrics |
|---|---:|---|
| SARCOS baseline | RMSE 5.5305 ± 0.1083 | MAE 3.9639 ± 0.0923; R² 0.9265 ± 0.0051 |
| SARCOS unscaled | RMSE 5.5303 ± 0.1083 | MAE 3.9629 ± 0.0914; R² 0.9265 ± 0.0052 |
| SARCOS bagging | RMSE 5.5303 ± 0.1079 | MAE 3.9638 ± 0.0939; R² 0.9265 ± 0.0051 |
| Execution baseline | Log loss 0.5291 ± 0.0099 | ROC-AUC 0.7265 ± 0.0146; accuracy 0.7218 ± 0.0085; F1 0.2689 ± 0.0248 |
| Execution unscaled | Log loss 0.5273 ± 0.0099 | ROC-AUC 0.7289 ± 0.0146; accuracy 0.7240 ± 0.0078; F1 0.2781 ± 0.0237 |
| Execution bagging | Log loss 0.5289 ± 0.0096 | ROC-AUC 0.7267 ± 0.0143; accuracy 0.7218 ± 0.0090; F1 0.2712 ± 0.0280 |

The main-metric preprocessing comparison by fold is shown below. Delta is `unscaled - baseline`; negative values favor the unscaled condition for both RMSE and log loss.

| Fold | SARCOS baseline RMSE | SARCOS unscaled RMSE | Delta | Execution baseline log loss | Execution unscaled log loss | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.543601 | 5.545206 | +0.001605 | 0.535485 | 0.533529 | -0.001956 |
| 1 | 5.624360 | 5.623460 | -0.000901 | 0.513054 | 0.510817 | -0.002237 |
| 2 | 5.471403 | 5.473398 | +0.001994 | 0.535559 | 0.533700 | -0.001860 |
| 3 | 5.377460 | 5.375349 | -0.002111 | 0.526015 | 0.525111 | -0.000904 |
| 4 | 5.635466 | 5.633868 | -0.001598 | 0.535377 | 0.533586 | -0.001791 |

The SARCOS scaling result changes direction across folds and the mean RMSE difference is only -0.0002, which is negligible beside fold-to-fold variation. Scaling has little practical effect for this Ridge configuration. For execution classification, unscaled inputs reduce log loss on all five folds, but the mean reduction is small (about 0.00175). This suggests scaling is not needed for this particular fitted Logistic Regression configuration; the result does not establish that scaling is unhelpful for other models or tuning settings.

Bagging makes almost no difference. For SARCOS, mean validation RMSE changes from 5.5305 to 5.5303. For execution, mean validation log loss changes from 0.5291 to 0.5289. Both baseline learners are relatively stable linear models here, so averaging bootstrap fits reduces little variance. These runs do not show a meaningful Bagging gain.

The mean training and validation primary scores across folds were 5.4912 and 5.5305 RMSE for SARCOS baseline, and 0.5262 and 0.5291 log loss for execution baseline. The small gaps indicate similar performance on training and validation folds at these settings.

## SARCOS learning curve

Each validation fold remains fixed while only the training portion is subsampled. The table reports mean and sample standard deviation over folds.

| Training rows per fold | Training RMSE | Validation RMSE |
|---:|---:|---:|
| 500 | 5.2722 ± 0.1729 | 5.7050 ± 0.1256 |
| 1,000 | 5.4015 ± 0.1865 | 5.5761 ± 0.0888 |
| 2,000 | 5.4342 ± 0.1099 | 5.5548 ± 0.1140 |
| 3,000 | 5.4981 ± 0.0108 | 5.5471 ± 0.1020 |
| 4,000 | 5.4912 ± 0.0277 | 5.5305 ± 0.1083 |

Validation RMSE improves as more training data are used, with diminishing gains after roughly 1,000–2,000 rows. Training RMSE rises as larger samples include more varied cases, while the training-validation gap narrows from about 0.43 at 500 rows to about 0.04 at 4,000. This pattern suggests that more data helps the fixed Ridge model and reduces its sampling variability. The remaining small gap and the flattening validation curve suggest that additional data alone may yield modest gains; model form and irreducible noise may also limit performance. The conclusion is specific to Ridge with `alpha=1.0` and this sample.

The plotted curve is saved as `runs/sarcos_regression/learning_curve_rmse.png`.

## Final models and training-row scores

The final method for each task is selected by the lowest mean validation score on that task's primary metric. `unscaled` wins for both tasks, although the difference is tiny for SARCOS. Each selected model is refit on all 5,000 development rows.

| Dataset | Selected method | Full-development training metrics |
|---|---|---|
| SARCOS | unscaled | RMSE 5.4953; MAE 3.9382; R² 0.9278 |
| Execution | unscaled | Log loss 0.5244; ROC-AUC 0.7334; accuracy 0.7238; F1 0.2796 |

These are training-row metrics, not held-out test results. The separate test data were not present in the supplied folder, so final test performance remains to be measured when the instructor provides them.

## Reproduction and limitations

Open `homework.ipynb`, run the cells from top to bottom for one dataset, then change `DATASET` in the setup cell and repeat for the other task. Keep `train.py`, `test.py`, and `datasets/` beside it. The experiment cell writes fold scores and summaries under `runs/`. The tested environment was Python 3.14.7, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, and Matplotlib 3.11.2; the notebook also installs compatible package versions. All experiments use seed 2026. No public labels or held-out answers were retrieved.

The SARCOS source is the [SARCOS dataset page](https://gaussianprocess.org/gpml/data/). Execution risk is a supplied synthetic simulation; its outcomes include random future market changes and should not be interpreted as real trading performance. The evaluation compares a small set of linear learners and fixed hyperparameters, so it does not establish the best possible model.

## AI use disclosure

OpenAI Codex assisted with implementing the baseline comparison and Bagging methods in `train.py`, organizing notebook experiments, running the supplied cross-validation and learning-curve analysis, and drafting this report. The student should review the code and results and be prepared to explain the choices and conclusions.
