# SARCOS: Robot Joint Torque Regression

This task uses real observations from a robot arm with seven degrees of freedom to predict the torque at the first joint from joint positions, velocities, and accelerations. Each row represents one observed robot state and contains 21 input variables and one scalar target.

The data come from the [SARCOS data page for Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/data/). In the source files, the first 21 columns are inputs and the final seven columns are joint torques. This assignment retains only the first joint's torque as the target. The other joint torques are neither used as inputs nor provided to students.

## Files and Fields

`train.csv` contains 5,000 labeled samples. The column order is `sample_id`, the input variables listed below, and `torque_1`.

| Field | Role | Description |
|---|---|---|
| `sample_id` | Identifier | A unique identifier generated for this assignment; it has no predictive meaning and must not be used as an input |
| `position_1`, `position_2`, `position_3`, `position_4`, `position_5`, `position_6`, `position_7` | 7 inputs | Positions of joints 1–7 |
| `velocity_1`, `velocity_2`, `velocity_3`, `velocity_4`, `velocity_5`, `velocity_6`, `velocity_7` | 7 inputs | Velocities of joints 1–7 |
| `acceleration_1`, `acceleration_2`, `acceleration_3`, `acceleration_4`, `acceleration_5`, `acceleration_6`, `acceleration_7` | 7 inputs | Accelerations of joints 1–7 |
| `torque_1` | Regression target | Torque at the first joint, retaining the numerical values in the source file |

This package preserves the source values without converting units. The public download page does not fully document the numerical units or any scaling that may have been applied. The assignment therefore does not label these columns as rad, rad/s, or N·m without verification. Express prediction errors in the original target units.

`cv_folds.csv` contains `sample_id` and `fold`, where `fold` ranges from 0 to 4. It defines the validation splits only and must not be included in model inputs. `metadata.json` provides machine-readable lists of feature and target column names.

## Preprocessing Notes

- No missing values have been artificially introduced. Inspect the actual data before deciding whether missing-value handling is needed.
- Position, velocity, and acceleration have different meanings and numerical scales. When comparing scaling methods, consider their effects on regularization and numerical optimization.
- Negative and zero values can represent valid joint states or torques; they do not indicate missingness.
- Extreme values may correspond to real operating conditions. Justify any clipping, transformation, or removal of training observations. Do not remove validation observations to improve the reported metrics.

This task uses a subset of a public dataset. Random splitting does not guarantee that dependencies from the original data collection process are eliminated. Interpret the results as held-out generalization performance within this data source.
