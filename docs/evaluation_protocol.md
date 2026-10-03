# Evaluation Protocol

The authoritative development split is chronological and whole-race-disjoint:

- TRAIN contains the earliest 60% of races.
- VALIDATION contains the next 20% and is the only partition used for development-time model or feature selection.
- FINAL HOLDOUT contains the latest 20%, and is reserved for a single final evaluation after development choices are complete.

`src/f1_tyre/evaluation/splits.py` orders races using consistent `season`/`race_round` and, when present, event date metadata. If those fields are unavailable, numeric race IDs are accepted only as an explicit chronological key; arbitrary string IDs are not treated as chronology. The splitter rejects duplicate `(race_id, driver_id, lap)` rows and verifies no race, race-driver, or inferred/explicit tyre-stint overlaps across partitions.

Preprocessing must be fitted on TRAIN only. Grouped cross-validation is supplementary and cannot replace the later-race holdout claim. Do not inspect final-holdout scores for feature selection, hyperparameter tuning, calibration, or reporting decisions.

Global-mean, compound-only, and compound-age lookup baselines are fitted on TRAIN and scored on VALIDATION during development. The seed-20261003 dataset was used for target-design acceptance diagnostics, including a compound-age lookup check; it is not the untouched final evaluation dataset. The authoritative run is locked to generator seed `20261004`; its final holdout is reserved until all model and feature decisions are complete.