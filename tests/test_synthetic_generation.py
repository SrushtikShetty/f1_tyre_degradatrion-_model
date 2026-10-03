from copy import deepcopy

import pandas as pd
import pytest

from data_generation.generate_dataset import (
    TARGET_COLUMN,
    attach_simulated_targets,
    generate_dataset,
    generate_race_states,
    load_config,
)
from f1_tyre.strict_feature_policy import ALLOWED_FEATURES


def _small_config():
    config = load_config()
    config.update({"races": 6, "drivers": 4, "laps_per_race": 9})
    return config


def test_same_seed_produces_identical_dataset():
    config = _small_config()

    first = generate_dataset(config, seed=17)
    second = generate_dataset(config, seed=17)

    pd.testing.assert_frame_equal(first, second)


def test_compound_age_groups_have_target_variation():
    generated = generate_dataset(_small_config(), seed=22).dropna(subset=[TARGET_COLUMN])
    grouped = generated.groupby(["tire_compound", "tire_age_laps"])[TARGET_COLUMN]

    assert (grouped.nunique() > 1).any()
    assert (grouped.std() > 0).any()


def test_target_and_simulator_latents_are_not_model_features():
    generated = generate_dataset(_small_config(), seed=31)
    model_features = [
        column
        for column in generated.columns
        if column in ALLOWED_FEATURES and column not in {"race_id", "driver_id", "tire_wear_pct"}
    ]

    assert TARGET_COLUMN not in model_features
    assert "tire_wear_pct" not in model_features
    assert not {"_tyre_sensitivity", "_surface_shock", "tyre_sensitivity", "surface_shock"} & set(generated.columns)


def test_changing_current_causal_state_changes_target_with_same_seed():
    config = _small_config()
    states = generate_race_states(config, seed=44)
    baseline = attach_simulated_targets(states, seed=44)
    changed_states = states.copy()
    changed_states.loc[0, "race_track_temp_c"] += 5.0
    changed = attach_simulated_targets(changed_states, seed=44)

    assert baseline.loc[0, TARGET_COLUMN] != changed.loc[0, TARGET_COLUMN]


@pytest.mark.parametrize("bad_races", [0, 4])
def test_config_rejects_too_few_races(bad_races):
    config = deepcopy(_small_config())
    config["races"] = bad_races

    with pytest.raises(ValueError, match="races"):
        generate_dataset(config, seed=1)