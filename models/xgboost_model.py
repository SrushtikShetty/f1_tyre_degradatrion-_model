from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.f1_tyre.config import PROJECT_ROOT
from src.f1_tyre.hardware import detect_xgboost_device
from src.f1_tyre.model_loader import save_model_artifact

from .common import RANDOM_STATE, build_preprocessor, prepare_dataset


def train(data_dir='.', out_dir='artifacts'):
    X, y, groups, _, cols = prepare_dataset(data_dir)
    cv = GroupKFold(n_splits=5)
    device = detect_xgboost_device()
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        tree_method='hist',
        device=device,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline([('prep', build_preprocessor(X)), ('model', model)])
    preds = cross_val_predict(pipe, X, y, cv=cv, groups=groups, n_jobs=1)
    metrics = {
        'model': 'XGBoost',
        'r2': float(r2_score(y, preds)),
        'mae': float(mean_absolute_error(y, preds)),
        'rmse': float(np.sqrt(mean_squared_error(y, preds))),
        'device': device,
    }
    pipe.fit(X, y)
    artifact = {
        'model': pipe,
        'feature_columns': cols,
        'target': 'future_tire_wear_pct',
        'metrics': metrics,
        'metadata': {'model_name': 'xgboost', 'device': device},
    }
    save_model_artifact('xgboost', artifact, PROJECT_ROOT / out_dir / 'models')
    return metrics


if __name__ == '__main__':
    print(train())
