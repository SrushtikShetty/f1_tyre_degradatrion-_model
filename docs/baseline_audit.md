# Baseline Scientific Audit

**Baseline date:** 2026-10-03  
**Branch and starting commit:** `main`, `7bd8368` (`origin/main`)  
**Starting worktree:** clean

## Scope and Reproduction

The baseline was collected before project edits. The existing test suite passed: **19 passed** in 21.19 seconds. It emitted a Starlette `httpx` deprecation warning and an XGBoost device-mismatch warning during inference. No baseline code or data was changed before these checks.

The training table has **233,640 rows and 71 columns**, covers **198 races** over seasons **2015-2023**, and contains telemetry-shaped fields. The repository contains no corresponding original generator/source evidence establishing these rows as measured real F1 telemetry. Treat the dataset as constructed/synthetic and its real-world provenance as unverified.

## Target Determinism

The raw `tire_wear_pct` column has zero mismatches against:

`min(100, rate[compound] * (tire_age_laps - 1))`, with rates HARD=1.0, MEDIUM=1.5, SOFT=2.5.

The existing target builder shifts wear by one row within race-driver groups and defines `next_lap_wear_increment` as next wear minus current wear. Its valid-row checks enforce consecutive laps, consecutive tyre age, and unchanged compound. The resulting target has four values (`0.0`, `1.0`, `1.5`, `2.5`) and 229,690 rows. Across 207 observed `(tire_compound, tire_age_laps)` groups, every group has one target value and zero within-group standard deviation.

On the existing chronological final holdout (46,084 rows, 40 later races), the compound-age lookup obtains **R² 1.000000, MAE 0, RMSE 0**. The saved strict-holdout Random Forest also obtains exact predictions; XGBoost obtains R² 0.9999988, MAE 0.0001783, RMSE 0.0008477. This is consistent with learning/reproducing the constructed rule, not evidence of real-world degradation prediction.

## Existing Metrics and UI

`artifacts/model_metrics.json` contains these unlabelled saved scores:

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Ridge | 0.6561385 | 0.3274067 | 0.4590252 |
| RandomForest | 0.9991421 | 0.0132818 | 0.0229282 |
| XGBoost | 0.99999997 | 0.0000551 | 0.0001250 |
| NeuralNetworkMLP | 0.9791359 | 0.0371482 | 0.1130690 |

This file contains no split/protocol metadata, so its scores cannot be verified as cross-validation, validation, or holdout results. The API currently serves this file from `GET /api/model-metrics`, and the dashboard presents the values without an explicit legacy label.

The separately saved chronological evaluation uses 118 training races, 40 validation races, and 40 final holdout races with no race, race-driver, or tyre-stint overlap. Those metrics are more clearly scoped, but still measure prediction of the deterministic target. A chronological split alone does not repair the target design.

## Baseline Finding

**Primary scientific defect:** compound and tyre age encode the next-lap target exactly through a synthetic wear schedule. The near-perfect lookup and tree-model scores are therefore not a defensible claim of learned real-world tyre-degradation dynamics. Existing feature allowlists and chronological split checks address other leakage risks, but do not remove this target determinism.

This checkpoint preserves the existing data and metrics as evidence. It does not overwrite, conceal, or reinterpret them as results from a corrected pipeline.
