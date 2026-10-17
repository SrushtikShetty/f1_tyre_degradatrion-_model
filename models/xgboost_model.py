from __future__ import annotations

from xgboost import XGBRegressor

from src.f1_tyre.hardware import detect_xgboost_device

from .common import RANDOM_STATE, train_grouped_cv_model


def train(data_dir='.', out_dir='artifacts'):
    device = detect_xgboost_device()
    return train_grouped_cv_model(
        "xgboost",
        "XGBoost",
        XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            device=device,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        data_dir,
        out_dir,
        metadata={"device": device},
    )


if __name__ == '__main__':
    print(train())
