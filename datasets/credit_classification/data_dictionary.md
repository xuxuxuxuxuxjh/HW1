# Credit Card Default Classification: Real Data

This binary classification task predicts whether a customer will default in the following month using their historical records. Each row represents one customer and contains 20 input variables. The model should output the probability that `default_next_month=1`.

The source is [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), contributed by I-Cheng Yeh. The records concern customers in Taiwan in 2005. UCI gives 2009 as the recommended citation year and provides the dataset under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

The original dataset has 23 predictors. This assignment removes `SEX`, `EDUCATION`, and `MARRIAGE`, retaining the 20 predictors listed below. The original `ID` is also omitted and replaced with a newly generated `sample_id` that has no predictive meaning. Values and labels are preserved from the source records; preparation is limited to sampling, field selection, and renaming the target.

## Files and Fields

`train.csv` contains 5,000 labeled samples. All monetary amounts are in New Taiwan dollars (NTD).

| Field | Role | Description and units |
|---|---|---|
| `sample_id` | Identifier | A unique identifier generated for this assignment; must not be used as an input |
| `LIMIT_BAL` | Input | Credit limit, NTD |
| `AGE` | Input | Age, years |
| `PAY_0` | Input | Repayment status in September 2005 |
| `PAY_2` | Input | Repayment status in August 2005 |
| `PAY_3` | Input | Repayment status in July 2005 |
| `PAY_4` | Input | Repayment status in June 2005 |
| `PAY_5` | Input | Repayment status in May 2005 |
| `PAY_6` | Input | Repayment status in April 2005 |
| `BILL_AMT1` | Input | Bill amount in September 2005, NTD |
| `BILL_AMT2` | Input | Bill amount in August 2005, NTD |
| `BILL_AMT3` | Input | Bill amount in July 2005, NTD |
| `BILL_AMT4` | Input | Bill amount in June 2005, NTD |
| `BILL_AMT5` | Input | Bill amount in May 2005, NTD |
| `BILL_AMT6` | Input | Bill amount in April 2005, NTD |
| `PAY_AMT1` | Input | Payment amount in September 2005, NTD |
| `PAY_AMT2` | Input | Payment amount in August 2005, NTD |
| `PAY_AMT3` | Input | Payment amount in July 2005, NTD |
| `PAY_AMT4` | Input | Payment amount in June 2005, NTD |
| `PAY_AMT5` | Input | Payment amount in May 2005, NTD |
| `PAY_AMT6` | Input | Payment amount in April 2005, NTD |
| `default_next_month` | Classification target | Default in the following month: 1 for default, 0 for no default |

Month numbering is not consistent across field names: September's repayment status is `PAY_0`, while its bill and payment amounts are `BILL_AMT1` and `PAY_AMT1`. The original naming is preserved; the absence of a `PAY_1` column does not mean that data are missing.

`cv_folds.csv` contains `sample_id` and `fold`, where `fold` ranges from 0 to 4 and must not be used as an input. `metadata.json` provides the feature and target column names.

## Repayment Status Codes

UCI's description defines `-1` as paying on time; `1`–`8` indicate payment delays of 1–8 months, respectively, and `9` indicates a delay of at least nine months. The actual data may also contain `-2` and `0`, but this passage on the UCI page does not fully define those two codes.

This package preserves the original codes. It does not convert `-2` or `0` to missing values or assign undocumented meanings to them. You may treat the status codes as distinct categories. If you treat them as numerical variables or combine categories, state your assumptions and compare the choices through validation. Integer codes do not by themselves imply equal numerical distances between all values.

## Preprocessing Notes

- No missing values have been artificially introduced. Negative status codes and zero monetary amounts do not automatically indicate missingness.
- Bill amounts can be negative. The source documentation is insufficient to determine the reason for every negative bill. Preserve them and document your handling assumptions rather than declaring them erroneous without evidence.
- Check for zeros and negative values before applying logarithms to monetary amounts. Standard `log` or `log1p` transformations are not suitable for every column.
- Category encoding, imputation, scaling, and any resampling must respect the separation between training and validation data. Preserve the original class distribution in validation data.
- These historical records do not represent all present-day customers. This assignment compares learning procedures and is not intended for real lending decisions.

Suggested citation: Yeh, I.-C. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
