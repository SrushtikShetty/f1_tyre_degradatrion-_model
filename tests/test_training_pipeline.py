import pandas as pd

import tempfile
from pathlib import Path

from data_generation.generate_dataset import generate_dataset, load_config
from training.train_corrected_models import prepare_training_data


def test_corrected_training_features_exclude_targets_ids_and_history():
    config = load_config()
    config.update({"races": 6, "drivers": 4, "laps_per_race": 9})
    dataset = generate_dataset(config, seed=73)
    # Exercise the same CSV boundary used by the training command.
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "generated.csv"
        dataset.to_csv(path, index=False)
        labeled, partitions, feature_frames, feature_columns, protocol = prepare_training_data(path)

    forbidden = {"race_id", "driver_id", "next_lap_wear_increment", "next_lap_degradation_pct", "tire_wear_pct"}
    assert not forbidden.intersection(feature_columns)
    assert not any("_lag" in name or "_roll" in name for name in feature_columns)
    assert set(feature_frames) == {"train", "validation", "holdout"}
    assert protocol["integrity"]["zero_race_overlap"] is True
    assert len(labeled) == sum(len(frame) for frame in partitions.values())
    assert all(len(features.columns) == len(feature_columns) for features in feature_frames.values())