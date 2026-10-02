# Multiple F1 Tyre Degradation Models

The four regression models share one target: **next-lap tyre-wear increment** (`next_lap_wear_increment`), calculated as next-lap wear minus current-lap wear. Rows are retained only when the next row is the immediately following lap, tyre age advances by one, and the compound is unchanged. Absolute next-lap wear can be reconstructed as current wear plus the predicted increment.

The current-lap features represent information available after that lap is complete. Current tyre wear is used only to calculate the target and is excluded from model features. Future-state columns, race-end information, and pit-stop outcomes are also excluded.

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

Outputs are written to `artifacts/` including one serialized model per algorithm and a model comparison table. Validation is race-grouped with GroupKFold, and results report R², MAE and RMSE for the delta-wear target. Prediction loads trained artifacts only; it does not retrain.
