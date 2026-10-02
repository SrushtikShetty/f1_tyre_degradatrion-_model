# F1 Tyre Degradation Prediction

## Overview

This project predicts the change in tyre wear over the next lap for Formula 1 race data. The modelling objective is to estimate the next-lap tyre-wear increment using a strict causal feature set that excludes future-state and race-end information.

## Problem

Tyre degradation is highly sensitive to track conditions, grip, tyre age, stint length, fuel load, pace trends, and driver/race context. The challenge is to estimate how much tyre wear will increase on the next lap without using information that would not have been available at prediction time.

## Objective

The model predicts the next-lap wear increment, which can be interpreted as the incremental degradation during the next lap. The absolute next-lap wear is then reconstructed as:

current_wear + predicted_increment

## Methodology

The project keeps the strongest existing causal logic intact:

- race-grouped causal ordering
- lap-to-lap lag features
- rolling historical telemetry windows
- tyre-age and race-progress features
- grouped validation to prevent race overlap
- strict exclusion of future-state columns and end-of-race information

The training target is built from the immediate next lap within each race/driver group. This preserves a real temporal target while avoiding leakage from future laps.

## Models

This project supports four regressors with the same feature pipeline:

- Ridge regression
- Random forest regression
- XGBoost regression
- Neural network / MLP regressor

## Repository Structure

- `src/f1_tyre/` — canonical data, feature, validation, metrics, hardware, and model-loader logic
- `models/` — model implementations and registry
- `training/` — explicit model training commands
- `inference/` — batch and interactive inference entry points
- `evaluation/` — metrics and leakage-audit tools
- `artifacts/` — trained artifacts and saved outputs
- `tests/` — unit tests for data, features, leakage, and inference compatibility
- `docs/` — methodology and leakage documentation

## Installation

```bash
python -m venv .venv
. .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
```

## Using Existing Trained Models

The project already includes trained model artifacts. Normal usage should load the saved model from `artifacts/models/` or the legacy artifact directory and perform inference without retraining.

This is an explicit design rule:

- inference does not train
- training is never triggered automatically
- missing artifacts raise a clear error instructing the user to run the training command

## Prediction

```bash
python inference/predict.py --model xgboost --input test.csv --output artifacts/predictions/predictions.csv
```

## Interactive Prediction

```bash
python inference/interactive_predict.py --model xgboost
```

## Explicit Training

```bash
python training/train_all.py
```

This command explicitly trains or retrains the models and saves artifacts.

## Model Comparison

Saved metrics are read from the artifact files and comparison outputs instead of retraining on each run. The project stores model metadata and metrics in the `artifacts/` tree and loads them during evaluation and reporting.

## GPU Support

XGBoost checks for CUDA availability and uses GPU when available. If CUDA is not available, it automatically falls back to CPU and keeps the training/inference behaviour reproducible.

## Evaluation Metrics

The project reports:

- R²
- MAE
- RMSE

These metrics are generated from comparable grouped validation predictions and saved with the model metadata.

## Leakage Prevention

The project explicitly prevents leakage by excluding future-state or race-end columns such as:

- future lap values
- race-end/finish information
- points, fastest-lap fields, and pit-stop outcomes
- next-lap state variables

The dedicated leakage audit checks the final feature matrix and raises a clear error if these patterns appear.

## Reproducibility

The project uses stable settings and grouped validation. Model settings and the feature pipeline are centralized so that all supported regressors use the same canonical data preparation and validation logic.

## Limitations

- The project is still a tabular regression approach; it does not yet include a full race-simulation model.
- The saved model artifacts are only as good as the underlying data and validation regime.
- Performance depends on the current dataset and the causal assumptions used for feature construction.

## Future Work

Potential extensions include:

- temporal sequence models
- strategy optimisation
- pit-stop prediction
- SHAP explainability
- uncertainty estimation
- driver/team adaptation

## License

This project is distributed for research and portfolio use. Add a project-specific license before public release if required.
