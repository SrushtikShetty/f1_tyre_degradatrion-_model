from __future__ import annotations

import math


COMPOUND_BASE_DEGRADATION = {
    "HARD": 0.38,
    "MEDIUM": 0.55,
    "SOFT": 0.78,
}

COMPOUND_OPTIMAL_TRACK_TEMP_C = {
    "HARD": 36.0,
    "MEDIUM": 38.0,
    "SOFT": 40.0,
}


def _finite_value(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _unit_interval(value: float, name: str) -> float:
    numeric = _finite_value(value, name)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return numeric


def calculate_next_lap_degradation_pct(
    *,
    compound: str,
    current_tire_age_laps: float,
    track_temp_c: float,
    air_temp_c: float,
    circuit_load_index: float,
    pace_stress: float,
    fuel_fraction: float,
    driver_aggression: float,
    traffic_intensity: float,
    race_progress: float,
    tyre_sensitivity: float,
    surface_shock: float,
) -> float:
    """Simulate next-lap wear in percentage points from current race state.

    ``tyre_sensitivity`` models per-stint tyre construction variation and
    ``surface_shock`` models a bounded next-lap surface/load realization.
    The generator samples these latent factors; neither is a model feature.
    """
    compound_key = str(compound).strip().upper()
    if compound_key not in COMPOUND_BASE_DEGRADATION:
        raise ValueError(f"Unsupported tyre compound: {compound!r}.")

    age = _finite_value(current_tire_age_laps, "current_tire_age_laps")
    if age < 0:
        raise ValueError("current_tire_age_laps cannot be negative.")
    next_age = age + 1.0

    track_temp = _finite_value(track_temp_c, "track_temp_c")
    air_temp = _finite_value(air_temp_c, "air_temp_c")
    circuit_load = _unit_interval(circuit_load_index, "circuit_load_index")
    pace = _unit_interval(pace_stress, "pace_stress")
    fuel = _unit_interval(fuel_fraction, "fuel_fraction")
    aggression = _unit_interval(driver_aggression, "driver_aggression")
    traffic = _unit_interval(traffic_intensity, "traffic_intensity")
    progress = _unit_interval(race_progress, "race_progress")
    tyre_factor = _finite_value(tyre_sensitivity, "tyre_sensitivity")
    surface_factor = _finite_value(surface_shock, "surface_shock")
    if not 0.75 <= tyre_factor <= 1.25:
        raise ValueError("tyre_sensitivity must be between 0.75 and 1.25.")
    if not 0.8 <= surface_factor <= 1.2:
        raise ValueError("surface_shock must be between 0.8 and 1.2.")

    age_effect = 1.0 + 0.012 * next_age + 0.0001 * next_age**2
    optimum = COMPOUND_OPTIMAL_TRACK_TEMP_C[compound_key]
    thermal_effect = 1.0 + 0.012 * abs(track_temp - optimum) + 0.004 * abs(air_temp - 25.0)
    circuit_effect = 0.9 + 0.3 * circuit_load
    pace_effect = 0.9 + 0.3 * pace
    fuel_effect = 0.92 + 0.16 * fuel
    driving_effect = 0.85 + 0.3 * aggression
    traffic_effect = 0.9 + 0.2 * traffic
    race_phase_effect = 0.95 + 0.1 * progress

    return (
        COMPOUND_BASE_DEGRADATION[compound_key]
        * age_effect
        * thermal_effect
        * circuit_effect
        * pace_effect
        * fuel_effect
        * driving_effect
        * traffic_effect
        * race_phase_effect
        * tyre_factor
        * surface_factor
    )