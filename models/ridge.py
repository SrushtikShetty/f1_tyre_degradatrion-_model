from __future__ import annotations

from sklearn.linear_model import Ridge

from .common import train_grouped_cv_model


def train(data_dir=".", out_dir="artifacts"):
    return train_grouped_cv_model(
        "ridge",
        "Ridge",
        Ridge(alpha=10.0),
        data_dir,
        out_dir,
        cv_jobs=-1,
    )


if __name__ == "__main__":
    print(train())
