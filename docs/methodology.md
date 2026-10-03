# Methodology

This repository is a synthetic-style tyre-wear benchmark with unverified provenance, not a validated real-world F1 degradation model.

## Target

The current simulator creates an explicit `next_lap_degradation_pct` outcome from end-of-lap race state and structured latent tyre/surface variation. The canonical target builder maps that outcome to `next_lap_wear_increment` without shifting future rows. The earlier shift-derived wear-schedule label remains isolated in `build_legacy_target_frame` for historical audit reproduction only.

## Features

The feature pipeline sorts by race-driver-lap, uses prior-row lags and shifted rolling windows, and does not infer total race distance from the observed maximum lap. The feature registry records availability, transformations, and lineage; target, future-state, and partition-key columns are rejected from model inputs.

## Evaluation

The headline score is FINAL CHRONOLOGICAL HOLDOUT performance on the latest unseen races after all model and feature choices are complete. Preprocessing is fit on TRAIN only. VALIDATION is used for development decisions. Five-fold GroupKFold by race is supplementary out-of-fold analysis only; it is not a future-race estimate and cannot replace or tune against the final holdout.

## Model Usage

Inference loads saved model artifacts and does not retrain. The final metrics artifact will report validation and final chronological holdout results separately; legacy and grouped-CV metrics must remain explicitly labeled.
