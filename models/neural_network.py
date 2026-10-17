from __future__ import annotations

from sklearn.neural_network import MLPRegressor

from .common import RANDOM_STATE, train_grouped_cv_model


def train(data_dir='.', out_dir='artifacts'):
    return train_grouped_cv_model(
        "neural_network",
        "NeuralNetworkMLP",
        MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.1,
            batch_size=512,
            random_state=RANDOM_STATE,
        ),
        data_dir,
        out_dir,
        scale=True,
    )


if __name__ == '__main__':
    print(train())
