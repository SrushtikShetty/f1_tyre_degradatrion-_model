from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.f1_tyre.config import PROJECT_ROOT
from src.f1_tyre.model_loader import save_model_artifact

from .common import RANDOM_STATE, build_preprocessor, prepare_dataset


def train(data_dir='.', out_dir='artifacts'):
    X, y, groups, _, cols = prepare_dataset(data_dir)
    cv = GroupKFold(n_splits=5)
    pipe = Pipeline([
        ('prep', build_preprocessor(X)),
        ('model', RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])
    preds = cross_val_predict(pipe, X, y, cv=cv, groups=groups, n_jobs=1)
    metrics = {
        'model': 'RandomForest',
        'r2': float(r2_score(y, preds)),
        'mae': float(mean_absolute_error(y, preds)),
        'rmse': float(np.sqrt(mean_squared_error(y, preds))),
    }
    pipe.fit(X, y)
    artifact = {
        'model': pipe,
        'feature_columns': cols,
        'target': 'next_lap_wear_increment',
        'metrics': metrics,
        'metadata': {'model_name': 'random_forest'},
    }
    save_model_artifact('random_forest', artifact, PROJECT_ROOT / out_dir / 'models')
    return metrics


if __name__ == '__main__':
    print(train())
