from __future__ import annotations

import pandas as pd

from f1_tyre.model_loader import DEFAULT_MODEL_NAMES, list_available_models, load_model
from models.common import clean_data, create_current_features


def predict_with_model(frame: pd.DataFrame, model_name: str = "xgboost") -> pd.DataFrame:
    try:
        artifact = load_model(model_name)
    except ValueError:
        available = list_available_models()
        resolved = available[0] if available else "xgboost"
        artifact = load_model(resolved)
    if not isinstance(artifact, dict):
        raise TypeError("Model artifact must be a dictionary with a fitted estimator.")

    model = artifact.get("model")
    if model is None:
        raise KeyError("Model artifact is missing the trained estimator.")

    feature_columns = artifact.get("feature_columns", [])
    available = create_current_features(clean_data(frame))
    missing = [column for column in feature_columns if column not in available.columns]
    if missing:
        raise ValueError(f"Input frame is missing required model features: {missing}")
    matrix = available[feature_columns]
    predictions = model.predict(matrix)
    return pd.DataFrame({"prediction": predictions})
