from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.f1_tyre.config import PROJECT_ROOT
from src.f1_tyre.model_loader import save_model_artifact

from .common import RANDOM_STATE, build_preprocessor, prepare_dataset


def train(data_dir='.', out_dir='artifacts'):
    X, y, groups, _, cols = prepare_dataset(data_dir)
    cv = GroupKFold(n_splits=5)
    pipe = Pipeline([
        ('prep', build_preprocessor(X)),
        ('scale', StandardScaler(with_mean=False)),
        ('model', MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.1,
            batch_size=512,
            random_state=RANDOM_STATE,
        )),
    ])
    preds = cross_val_predict(pipe, X, y, cv=cv, groups=groups, n_jobs=1)
    metrics = {
        'model': 'NeuralNetworkMLP',
        'r2': float(r2_score(y, preds)),
        'mae': float(mean_absolute_error(y, preds)),
        'rmse': float(np.sqrt(mean_squared_error(y, preds))),
    }
    pipe.fit(X, y)
    artifact = {
        'model': pipe,
        'feature_columns': cols,
        'target': 'future_tire_wear_pct',
        'metrics': metrics,
        'metadata': {'model_name': 'neural_network'},
    }
    save_model_artifact('neural_network', artifact, PROJECT_ROOT / out_dir / 'models')
    return metrics


if __name__ == '__main__':
    print(train())
