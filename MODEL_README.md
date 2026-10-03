# Models for the Synthetic Tyre-Wear Benchmark

The four regression models share one derived target: **next-lap tyre-wear increment** (`next_lap_wear_increment`), calculated as next-row wear minus current-row wear. The wear values in the checked-in table exactly follow a constructed compound-and-age schedule. The target is consequently deterministic from compound and age in this dataset; model scores are rule-recovery results, not evidence of learned real-world F1 degradation. Dataset provenance and limitations are documented in [docs/data_provenance.md](docs/data_provenance.md).

The baseline current-lap features are intended to represent information available at lap end. Current tyre wear is used only to calculate the target and is excluded from model features. The field provenance and feature-time availability remain separate validation questions; see the project audit before making scientific claims.

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
