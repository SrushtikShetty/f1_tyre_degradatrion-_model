# Multiple F1 Tyre Degradation Models

This adds four separate regression implementations while preserving the strict causal setup: the models predict **next-lap tyre wear** and exclude current tyre wear, future-state columns, race-end information and pit-stop outcomes from the feature set.

## Models
- Ridge linear regression
- Random Forest regression
- XGBoost regression
- Neural-network regression using scikit-learn MLPRegressor

## Run

```bash
pip install -r requirements_models.txt
python train_all_models.py
```

Outputs are written to `artifacts/` including one serialized model per algorithm and a model comparison table. Validation is race-grouped with GroupKFold, and results report R², MAE and RMSE.
