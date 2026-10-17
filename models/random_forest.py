from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from .common import RANDOM_STATE, train_grouped_cv_model


def train(data_dir='.', out_dir='artifacts'):
    return train_grouped_cv_model(
        "random_forest",
        "RandomForest",
        RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        data_dir,
        out_dir,
    )


if __name__ == '__main__':
    print(train())
