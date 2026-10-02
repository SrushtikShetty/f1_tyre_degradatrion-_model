from __future__ import annotations

from .linear_regression import train as train_ridge
from .neural_network import train as train_nn
from .random_forest import train as train_rf
from .xgboost_model import train as train_xgb


MODEL_FACTORIES = {
    "ridge": train_ridge,
    "random_forest": train_rf,
    "xgboost": train_xgb,
    "neural_network": train_nn,
}


def list_model_names() -> list[str]:
    return list(MODEL_FACTORIES.keys())


def get_model_factory(name: str):
    key = name.lower().replace("-", "_")
    if key not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_FACTORIES[key]
