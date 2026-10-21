# Building Cooling Load Regression: Synthetic Data

This task simulates the thermal conditions of four building zones and predicts their total cooling load from five inputs per zone. Cooling load is the rate at which heat must be removed to maintain indoor conditions; the target is measured in kW. Each row represents an independently generated building operating condition with 20 input variables.

These data were generated for this assignment. Both the inputs and the target are synthetic. The target comes from a simplified heat-balance relationship with random disturbances; it is neither measured energy use nor the output of an engineering simulation calibrated against building data.

## Files and Fields

`train.csv` contains 5,000 labeled samples. The column order is `sample_id`, the five inputs for each of zones 1–4, and `cooling_load_kw`.

| Field template, where `j` is 1, 2, 3, or 4 | Role | Description and units |
|---|---|---|
| `sample_id` | Identifier | A unique identifier generated for this assignment; must not be used as an input |
| `zone_j_area_m2` | 4 inputs | Effective heat-transfer area of the zone, m² |
| `zone_j_delta_temp_c` | 4 inputs | Outdoor minus indoor temperature, °C; the numerical temperature difference is the same when expressed in K |
| `zone_j_u_value_w_m2k` | 4 inputs | Effective heat-transfer coefficient, W/(m²·K) |
| `zone_j_solar_w_m2` | 4 inputs | Solar irradiance, W/m² |
| `zone_j_internal_heat_w` | 4 inputs | Rate of internal heat generation from occupants, equipment, and other sources, W |
| `cooling_load_kw` | Regression target | Total cooling load of the four zones, kW |

The input columns are:

```text
zone_1_area_m2, zone_1_delta_temp_c, zone_1_u_value_w_m2k, zone_1_solar_w_m2, zone_1_internal_heat_w
zone_2_area_m2, zone_2_delta_temp_c, zone_2_u_value_w_m2k, zone_2_solar_w_m2, zone_2_internal_heat_w
zone_3_area_m2, zone_3_delta_temp_c, zone_3_u_value_w_m2k, zone_3_solar_w_m2, zone_3_internal_heat_w
zone_4_area_m2, zone_4_delta_temp_c, zone_4_u_value_w_m2k, zone_4_solar_w_m2, zone_4_internal_heat_w
```

`cv_folds.csv` contains `sample_id` and `fold`, where `fold` ranges from 0 to 4 and must not be used as a model input. `metadata.json` provides the feature and target column names.

## Missing and Valid Values

The instructor first generates the target from the complete inputs, then independently sets each value in each temperature-difference column to missing with probability 5%. Missing values appear as empty cells in the CSV and are typically read as `NaN` by pandas. The observed missingness rate varies slightly because of random sampling.

The development and held-out test data use the same missingness mechanism. A missing value means only that the value was not observed; it does not mean that the true temperature difference is zero or that the zone contributes no cooling load. No missing values have been artificially introduced into the other input columns.

- Zero solar irradiance is valid and does not by itself indicate a sensor fault.
- This version simulates summer conditions. All observed outdoor-minus-indoor temperature differences are positive; missing values must not be interpreted as true zero temperature differences.
- The target is cooling load, not the electrical power consumption of the cooling equipment. These quantities are not interchangeable.
- Interactions among area, heat-transfer coefficient, temperature difference, and solar irradiance can have a clear physical basis. Explain the meaning and units of any derived features.

Fit imputation parameters and all other preprocessing parameters estimated from data using only the training portion of the current fold.
