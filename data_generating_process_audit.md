# Data-Generating-Process Audit

## Finding

**target determinism / synthetic-data limitation** (not data leakage). `train.csv` contains a constructed wear schedule: `tire_wear_pct = min(100, rate[compound] * (tire_age_laps - 1)); rates: HARD=1.0, MEDIUM=1.5, SOFT=2.5`. The generator source itself is absent from the repository and its history, so this is an exact empirical reconstruction rather than a claim about the unavailable original program. The equation has 0 mismatches across 233,640 raw rows.

The target builder then computes `next_lap_wear_increment[t] = tire_wear_pct[t+1] - tire_wear_pct[t]` after sorting within `race_id` and `driver_id`. A target row is kept only when the next lap is consecutive, tyre age increases by one, and compound is unchanged. In symbols, if `W(a,c)=min(100, r(c)*(a-1))`, the target is `W(a+1,c)-W(a,c)` for retained rows. This yields the fixed compound wear rates until the 100% cap, followed by a zero increment; MEDIUM has a final 1.0 increment at the cap boundary.

The checked-in training table is not accompanied by a real telemetry source, measurement protocol, or generator code. Although it contains telemetry-shaped columns, the exact deterministic age/compound wear schedule and artificial rating/tier fields are inconsistent with treating this as measured real-world F1 tyre degradation. The supplied target should be treated as synthetic/constructed, not as real telemetry.

## Target Summary

- `target_unique_values`: 4
- `target_distribution` (count / proportion): `{'0.0': {'count': 30123, 'proportion': 0.13114632765901868}, '1.0': {'count': 45966, 'proportion': 0.20012190343506464}, '1.5': {'count': 91941, 'proportion': 0.4002829901171144}, '2.5': {'count': 61660, 'proportion': 0.2684487787888023}}`
- `target_mean`: 1.471668336
- `target_std`: 0.782788909
- `target_min`: 0.000000
- `target_max`: 2.500000
- target rows: 229,690
- compound-age groups: 207; groups with `std == 0`: 207 (100.0%)

`target_distribution.csv` contains the requested overall summary, frequency distribution by target value, and the full 207-row `groupby(["tire_compound", "tire_age_laps"])` table with `count`, `nunique`, `mean`, sample `std`, `min`, and `max`.

## Same-Holdout Results

The race order, split fractions, model implementations, baseline fits, and XGBoost all-feature predictions are reused from the prior chronological clean evaluation after validating its dataset hash, exact split IDs, feature list, target alignment, and fit provenance. TRAIN has 118 races; FINAL HOLDOUT has 40 later races, IDs `[159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198]`. Every baseline and ablation is evaluated on the same 46,084 holdout rows. The five changed-feature XGBoost variants are freshly fitted on TRAIN only; no model sees a holdout race during fitting.

| model          | r2       | mae      | rmse     | number_of_rows | number_of_races |
| -------------- | -------- | -------- | -------- | -------------- | --------------- |
| Lookup         | 1.000000 | 0.000000 | 0.000000 | 46084          | 40              |
| Ridge          | 0.659516 | 0.322547 | 0.453837 | 46084          | 40              |
| Random Forest  | 1.000000 | 0.000000 | 0.000000 | 46084          | 40              |
| XGBoost        | 0.999999 | 0.000178 | 0.000848 | 46084          | 40              |
| Neural Network | 0.998411 | 0.007269 | 0.031000 | 46084          | 40              |

The lookup baseline uses the median TRAIN target for `(tire_compound, tire_age_laps)` and falls back to the TRAIN global median for an unseen key. It encounters 0 unseen holdout keys. Lookup R² is 1.000000000; XGBoost R² is 0.999998812. The absolute difference is 0.000001188; the lookup is close to XGBoost. This is evidence the tree is recovering the deterministic lookup rule, not a rich degradation process.

## XGBoost Ablations

All variants use fixed hyperparameters from the clean evaluation and the same TRAIN/HOLDOUT partition; no tuning was performed.

| variant                               | feature_count | r2       | mae      | rmse     |
| ------------------------------------- | ------------- | -------- | -------- | -------- |
| A_all_features                        | 101           | 0.999999 | 0.000178 | 0.000848 |
| B_remove_tire_age_laps                | 100           | 0.999994 | 0.000405 | 0.001966 |
| C_remove_tire_compound                | 100           | 0.958339 | 0.065980 | 0.158751 |
| D_remove_age_and_compound             | 99            | 0.958393 | 0.066160 | 0.158649 |
| E_remove_age_compound_and_derived_age | 97            | 0.959278 | 0.064293 | 0.156952 |
| F_physical_causal_telemetry_only      | 77            | 0.999995 | 0.000244 | 0.001656 |

Variant F is explicitly defined as available race-state telemetry and deterministic transformations of it: compound and tyre age, lap/position/pace/sectors, fuel/ERS/DRS, gaps, weather and temperatures, humidity, wind/grip where present, plus causal lag/rolling telemetry features. Static driver/team/circuit ratings and identifiers are excluded. Its exact feature list is recorded in `lookup_baseline_results.json`.

## Determinism and Interpretation

The target is exactly determined by `(tire_compound, tire_age_laps)` in all 207 observed groups: `nunique == 1` in 207 groups and `std == 0` in 207. All 207 training key pairs also occur in holdout, so the lookup has no unseen key problem. `lap` and `fuel_load_kg` are not needed to derive the target once compound and age are known. They can serve as correlated proxies in ablations but are not part of the exact rule.

- **LEAKAGE:** this audit does not change or re-evaluate the feature leakage system. The target derivation uses next-lap wear by definition; the existing causal feature guardrails remain separate.
- **TARGET DETERMINISM:** confirmed, by exact formula reconstruction and zero within-group target variance.
- **GENERALIZATION:** the deterministic schedule transfers to later race IDs because the same compound-age combinations recur. This is rule transfer within the generated table, not evidence of generalization to independent real races.
- **REALISM:** unsupported as a research claim about measured real-world F1 tyre degradation. The wear target is constructed, discretized by compound rates, capped at 100, and has no observed race-level noise.

## Provenance and Reproducibility

`train.csv` was added to repository history without a corresponding generator program; no source-to-telemetry provenance is present in the repository. The audit records the dataset SHA-256, versions, exact split race IDs, group table, fixed model configurations, and random seed in `lookup_baseline_results.json`. It does not modify `train.csv`, any leakage code, or historical metrics.
