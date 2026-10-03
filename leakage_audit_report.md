# Clean Re-Evaluation and Leakage Audit

## Conclusion

- **LEAKAGE FIXED:** the reconstructed model feature matrix passed the strict allowlist audit before any split was created; target, future wear, and forbidden future-state columns were not model inputs.
- **PERFORMANCE VERIFIED:** true; the saved final-holdout results were produced after fitting each model only on TRAIN and after the holdout had remained untouched by fitting, tuning, or selection.
- The legacy XGBoost R² was 0.999999974514 and is **LEGACY**, not an estimate of next-race performance. The old code shows group-disjoint five-fold `GroupKFold`, but no fixed chronological final holdout. It also computed a future-wear alias, while its explicit feature allowlist excluded that alias and current wear. Therefore the old source does **not** establish that direct target leakage caused the score; temporal look-ahead between folds and lack of an untouched forward holdout make that score insufficient as an unseen-future estimate.
- New validation R² by model: NeuralNetwork 0.998478, RandomForest 1.000000, Ridge 0.673110, XGBoost 0.999998.
- New unseen-race holdout R² by model: NeuralNetwork 0.998411, RandomForest 1.000000, Ridge 0.659516, XGBoost 0.999999.
- The old 0.9999 score is not proven to have been caused by direct target leakage. The source excluded direct target aliases, but its non-chronological folds did not test future-race transfer; interpret the corrected scores alongside the synthetic target structure below.

## Protocol

**Legacy protocol:** `models/common.py` rebuilt the next-lap increment; each legacy regressor evaluated out-of-fold predictions with five-fold `GroupKFold` grouped by `race_id`, then fitted a final model on all rows. This kept a race out of its own fold's training set, but folds were not ordered chronologically and there was no permanently untouched final race block. The legacy summary file was read-only and was not overwritten.

**Corrected protocol:** source rows were sorted by season, race date, round, and race ID. The earliest 60% of races formed TRAIN, the next 20% VALIDATION, and the latest 20% FINAL HOLDOUT. All splits are race-disjoint, driver-race-disjoint, and stint-disjoint. Hyperparameters were fixed before fitting; validation and holdout scores did not drive model selection. Every imputer, encoder, and scaler was inside a pipeline fitted with TRAIN rows only.

Exact race IDs:

- TRAIN (118): `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]`
- VALIDATION (40): `[119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158]`
- FINAL HOLDOUT (40): `[159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198]`

Race, race-driver, and tyre-stint overlap checks all passed. The target builder retains a target only when the following row is the next lap, tyre age advances by exactly one, and the compound is unchanged. Model-fit provenance records TRAIN race IDs and asserts they are disjoint from both validation and holdout IDs.

## Metrics

| model         | split      | r2       | mae      | rmse     | number_of_rows | number_of_races |
| ------------- | ---------- | -------- | -------- | -------- | -------------- | --------------- |
| NeuralNetwork | holdout    | 0.998411 | 0.007269 | 0.031000 | 46084          | 40              |
| NeuralNetwork | train      | 0.999485 | 0.006249 | 0.017800 | 137311         | 118             |
| NeuralNetwork | validation | 0.998478 | 0.007023 | 0.030539 | 46295          | 40              |
| RandomForest  | holdout    | 1.000000 | 0.000000 | 0.000000 | 46084          | 40              |
| RandomForest  | train      | 1.000000 | 0.000000 | 0.000000 | 137311         | 118             |
| RandomForest  | validation | 1.000000 | 0.000000 | 0.000000 | 46295          | 40              |
| Ridge         | holdout    | 0.659516 | 0.322547 | 0.453837 | 46084          | 40              |
| Ridge         | train      | 0.680193 | 0.317281 | 0.443608 | 137311         | 118             |
| Ridge         | validation | 0.673110 | 0.318171 | 0.447588 | 46295          | 40              |
| XGBoost       | holdout    | 0.999999 | 0.000178 | 0.000848 | 46084          | 40              |
| XGBoost       | train      | 1.000000 | 0.000114 | 0.000361 | 137311         | 118             |
| XGBoost       | validation | 0.999998 | 0.000212 | 0.001105 | 46295          | 40              |

## Target Distribution

| split      | number_of_rows | number_of_races | mean     | std      | min      | max      | median   | zero_fraction |
| ---------- | -------------- | --------------- | -------- | -------- | -------- | -------- | -------- | ------------- |
| TRAIN      | 137311         | 118             | 1.470042 | 0.784435 | 0.000000 | 2.500000 | 1.500000 | 0.131410      |
| VALIDATION | 46295          | 40              | 1.469964 | 0.782857 | 0.000000 | 2.500000 | 1.500000 | 0.132217      |
| HOLDOUT    | 46084          | 40              | 1.478225 | 0.777778 | 0.000000 | 2.500000 | 1.500000 | 0.129286      |

## Target Structure

The target has 4 distinct values. Across 207 observed `(tire_age_laps, tire_compound)` pairs, 207 map to exactly one target value. TRAIN covers 207 pairs, VALIDATION covers 207 pairs, and FINAL HOLDOUT covers 207 pairs; unseen holdout pairs relative to TRAIN: 0. This deterministic mapping can explain very high tree-model scores without race overlap. It verifies transfer across held-out race IDs in this generated dataset, not external or real-world tyre degradation performance.

## Sanity Checks

**Shuffled-target baseline:** Ridge was trained after randomly permuting the target within TRAIN and evaluated on VALIDATION. R² = -0.000955; this should be near zero or below and confirms the pipeline is not producing a high score without the target relationship.

**Feature ablation:** each row below is the same fixed Ridge baseline, fitted only on TRAIN. Holdout scores are diagnostics for the requested ablations and were not used to choose a model or feature set.

| variant                          | model | feature_count | holdout_r2 | holdout_mae | holdout_rmse | training_races | holdout_races | holdout_used_for_model_selection |
| -------------------------------- | ----- | ------------- | ---------- | ----------- | ------------ | -------------- | ------------- | -------------------------------- |
| A_strict_causal                  | Ridge | 101           | 0.659516   | 0.322547    | 0.453837     | 118            | 40            | False                            |
| B_no_current_lap_telemetry       | Ridge | 81            | 0.617518   | 0.340881    | 0.481013     | 118            | 40            | False                            |
| C_no_fuel_features               | Ridge | 97            | 0.646784   | 0.325260    | 0.462244     | 118            | 40            | False                            |
| D_no_lap_time_or_sector_features | Ridge | 74            | 0.622207   | 0.341795    | 0.478055     | 118            | 40            | False                            |

**Leave-one-unseen-race-out checks:** each listed holdout race was excluded from fitting and evaluated separately.

| race_id | rows | r2       | mae      | rmse     | training_races | race_excluded_from_fit |
| ------- | ---- | -------- | -------- | -------- | -------------- | ---------------------- |
| 159     | 1136 | 0.708312 | 0.284816 | 0.413790 | 118            | True                   |
| 168     | 1281 | 0.632842 | 0.428504 | 0.554036 | 118            | True                   |
| 178     | 900  | 0.673906 | 0.303785 | 0.441876 | 118            | True                   |
| 188     | 1380 | 0.551738 | 0.456339 | 0.651352 | 118            | True                   |
| 198     | 1279 | 0.681954 | 0.279488 | 0.391682 | 118            | True                   |

## Dominant Features

Feature importance was computed without holdout data. The table ranks original features within each model first, then averages those ranks, avoiding direct comparisons between coefficients, tree importances, and permutation scores.

| feature                   | rank  |
| ------------------------- | ----- |
| tire_compound             | 1.00  |
| fuel_load_kg              | 3.00  |
| tire_age_laps             | 4.25  |
| tire_age_squared          | 5.25  |
| lap                       | 5.50  |
| fuel_load_kg_lag1         | 11.25 |
| q1_time_sec               | 23.25 |
| circuit_base_lap_time_sec | 23.50 |
| race_progress             | 24.75 |
| fuel_load_kg_lag3         | 25.00 |
| fuel_load_kg_lag2         | 27.25 |
| lap_time_sec_lag2         | 27.50 |

## Reproducibility

The clean runner, fixed seed (42), exact race IDs, dataset SHA-256, package versions, and fit-provenance checks are stored with the metrics. The procedure is reproducible from `train.csv`; this run has not been independently repeated in a second full four-model execution.

## Artifacts

- `artifacts/strict_holdout_metrics.json`
- `artifacts/strict_holdout_comparison.csv`
- `artifacts/strict_holdout_predictions.csv`
- `artifacts/strict_holdout_feature_importance.csv`

Historical `artifacts/model_metrics.json` and `artifacts/model_comparison.csv` were left unchanged. Existing model binaries, OOF predictions, and cached outputs were ignored; none were loaded for this evaluation.
