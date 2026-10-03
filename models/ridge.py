from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.f1_tyre.config import PROJECT_ROOT
from src.f1_tyre.model_loader import save_model_artifact

from .common import RANDOM_STATE, build_preprocessor, label_supplementary_grouped_cv, prepare_dataset


def train(data_dir='.', out_dir='artifacts'):
    X, y, groups, _, cols = prepare_dataset(data_dir)
    cv = GroupKFold(n_splits=5)
    pipe = Pipeline([('prep', build_preprocessor(X)), ('model', Ridge(alpha=10.0))])
    preds = cross_val_predict(pipe, X, y, cv=cv, groups=groups, n_jobs=1)
    metrics = label_supplementary_grouped_cv({
        'model': 'Ridge',
        'r2': float(r2_score(y, preds)),
        'mae': float(mean_absolute_error(y, preds)),
        'rmse': float(np.sqrt(mean_squared_error(y, preds))),
    })
    pipe.fit(X, y)
    artifact = {'model': pipe, 'feature_columns': cols, 'target': 'next_lap_wear_increment', 'metrics': metrics, 'metadata': {'model_name': 'ridge'}}
    save_model_artifact('ridge', artifact, PROJECT_ROOT / out_dir / 'models')
    return metrics
