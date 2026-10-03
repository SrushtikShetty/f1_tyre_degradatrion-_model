import pytest

from f1_tyre.synthetic_degradation import calculate_next_lap_degradation_pct


BASE_STATE = {
    "compound": "MEDIUM",
    "current_tire_age_laps": 12,
    "track_temp_c": 39.0,
    "air_temp_c": 25.0,
    "circuit_load_index": 0.6,
    "pace_stress": 0.5,
    "fuel_fraction": 0.4,
    "driver_aggression": 0.55,
    "traffic_intensity": 0.3,
    "race_progress": 0.5,
    "tyre_sensitivity": 1.0,
    "surface_shock": 1.0,
}


def _calculate(**changes):
    return calculate_next_lap_degradation_pct(**(BASE_STATE | changes))


def test_compound_and_age_do_not_determine_increment_alone():
    baseline = _calculate()
    hot_track = _calculate(track_temp_c=52.0)
    high_load = _calculate(circuit_load_index=0.95, pace_stress=0.9)
    high_traffic = _calculate(traffic_intensity=0.95)

    assert len({baseline, hot_track, high_load, high_traffic}) == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_tire_age_laps", 24),
        ("track_temp_c", 48.0),
        ("air_temp_c", 34.0),
        ("circuit_load_index", 0.9),
        ("pace_stress", 0.9),
        ("fuel_fraction", 0.9),
        ("driver_aggression", 0.9),
        ("traffic_intensity", 0.9),
        ("race_progress", 0.9),
        ("tyre_sensitivity", 1.2),
        ("surface_shock", 1.15),
        ("compound", "SOFT"),
    ],
)
def test_each_causal_or_latent_factor_changes_the_target(field, value):
    assert _calculate(**{field: value}) != _calculate()


def test_increment_is_positive_and_deterministic_given_full_simulator_state():
    first = _calculate()
    second = _calculate()

    assert first > 0
    assert first == second


def test_out_of_range_stochastic_factor_is_rejected():
    with pytest.raises(ValueError, match="surface_shock"):
        _calculate(surface_shock=1.5)