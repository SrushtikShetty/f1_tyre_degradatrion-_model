# Model Comparison

The project stores all trained models and their metrics in the `artifacts/` tree. The comparison is meant to read those saved evaluation artifacts rather than retrain during normal inference.

## Supported Models

- Ridge regression
- Random forest regression
- XGBoost regression
- Neural network / MLP regressor

## Comparison Workflow

Training is explicit and should be run via `python training/train_all.py`. After training, metrics are saved and may be loaded for reporting and inspection.

## Metrics Reported

- R²
- MAE
- RMSE
- model name
- device information when relevant
