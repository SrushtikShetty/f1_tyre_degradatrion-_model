from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from f1_tyre.synthetic_degradation import calculate_next_lap_degradation_pct


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "generated" / "synthetic_race_telemetry.csv"
TARGET_COLUMN = "next_lap_degradation_pct"
GROUP_COLUMNS = ["race_id", "driver_id"]
COMPOUND_CHOICES = ("HARD", "MEDIUM", "SOFT")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_config(config: Mapping[str, Any]) -> None:
    if int(config["races"]) < 6:
        raise ValueError("races must be at least 6 for chronological evaluation.")
    if int(config["drivers"]) < 2 or int(config["laps_per_race"]) < 5:
        raise ValueError("drivers must be at least 2 and laps_per_race at least 5.")
    if int(config["seed"]) < 0:
        raise ValueError("seed must be non-negative.")
    compounds = tuple(str(value).upper() for value in config["compounds"])
    probabilities = np.asarray(config["compound_probabilities"], dtype=float)
    if not compounds or any(value not in COMPOUND_CHOICES for value in compounds):
        raise ValueError(f"compounds must be selected from {COMPOUND_CHOICES}.")
    if len(set(compounds)) != len(compounds) or len(probabilities) != len(compounds):
        raise ValueError("compounds must be unique and match compound_probabilities.")
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("compound_probabilities must be non-negative and sum to 1.")
    rain_probability = float(config["rain_probability"])
    if not 0.0 <= rain_probability <= 1.0:
        raise ValueError("rain_probability must be between 0 and 1.")
    for section in ("track", "weather"):
        for label, limits in config[section].items():
            if float(limits["min"]) >= float(limits["max"]):
                raise ValueError(f"{section}.{label} must have min < max.")


def _sample_track_value(rng: np.random.Generator, config: Mapping[str, Any], name: str, integer: bool = False) -> float | int:
    limits = config["track"][name]
    if integer:
        return int(rng.integers(int(limits["min"]), int(limits["max"]) + 1))
    return float(rng.uniform(float(limits["min"]), float(limits["max"])))


def _sample_weather_value(rng: np.random.Generator, config: Mapping[str, Any], name: str) -> float:
    limits = config["weather"][name]
    return float(rng.uniform(float(limits["min"]), float(limits["max"])))


def generate_race_states(config: Mapping[str, Any], seed: int | None = None) -> pd.DataFrame:
    """Generate current/end-of-lap race state without target or future columns."""
    validate_config(config)
    actual_seed = int(config["seed"] if seed is None else seed)
    if actual_seed < 0:
        raise ValueError("seed must be non-negative.")
    rng = np.random.default_rng(actual_seed)
    driver_count = int(config["drivers"])
    lap_count = int(config["laps_per_race"])

    drivers = []
    for driver_id in range(1, driver_count + 1):
        drivers.append(
            {
                "driver_id": driver_id,
                "team_id": (driver_id - 1) // 2 + 1,
                "driver_skill_rating": float(rng.uniform(0.45, 0.95)),
                "driver_aggression_rating": float(rng.uniform(0.2, 0.9)),
                "driver_consistency_rating": float(rng.uniform(0.45, 0.95)),
                "team_car_speed_rating": float(rng.uniform(0.45, 0.95)),
                "team_car_downforce_rating": float(rng.uniform(0.4, 0.95)),
                "team_car_reliability_rating": float(rng.uniform(0.7, 0.99)),
                "team_pit_crew_rating": float(rng.uniform(0.5, 0.98)),
                "team_budget_tier": int(rng.integers(1, 4)),
            }
        )

    rows: list[dict[str, Any]] = []
    for race_index in range(int(config["races"])):
        race_id = race_index + 1
        length_km = float(_sample_track_value(rng, config, "length_km"))
        turns = int(_sample_track_value(rng, config, "turns", integer=True))
        drs_zones = int(_sample_track_value(rng, config, "drs_zones", integer=True))
        circuit_id = race_index % 24 + 1
        base_lap_time = 58.0 + 14.0 * length_km + 0.55 * turns
        air_temp_start = _sample_weather_value(rng, config, "air_temp_c")
        track_temp_start = _sample_weather_value(rng, config, "track_temp_c")
        humidity_start = _sample_weather_value(rng, config, "humidity_pct")
        wind_speed = _sample_weather_value(rng, config, "wind_speed_kph")
        wet_race = bool(rng.random() < float(config["rain_probability"]))
        weather = "WET" if wet_race else "DRY"
        race_order = rng.permutation(np.arange(1, driver_count + 1))
        grid_by_driver = {int(driver_id): rank for rank, driver_id in enumerate(race_order, start=1)}
        compound_probabilities = np.asarray(config["compound_probabilities"], dtype=float)

        for driver in drivers:
            driver_id = int(driver["driver_id"])
            compound = str(rng.choice(config["compounds"], p=compound_probabilities)).upper()
            grid_position = int(grid_by_driver[driver_id])
            skill = float(driver["driver_skill_rating"])
            aggression = float(driver["driver_aggression_rating"])

            for lap in range(1, lap_count + 1):
                progress = (lap - 1) / max(1, lap_count - 1)
                track_temp = float(
                    np.clip(
                        track_temp_start + 1.8 * math.sin(math.pi * progress) + rng.normal(0.0, 0.9),
                        config["weather"]["track_temp_c"]["min"],
                        config["weather"]["track_temp_c"]["max"],
                    )
                )
                air_temp = float(
                    np.clip(
                        air_temp_start + 0.7 * math.sin(math.pi * progress) + rng.normal(0.0, 0.35),
                        config["weather"]["air_temp_c"]["min"],
                        config["weather"]["air_temp_c"]["max"],
                    )
                )
                humidity = float(np.clip(humidity_start + rng.normal(0.0, 2.0), 0.0, 100.0))
                ahead_gap = 5.0 if grid_position == 1 else float(rng.uniform(0.15, 4.8))
                behind_gap = 5.0 if grid_position == driver_count else float(rng.uniform(0.15, 4.8))
                traffic = float(np.clip(math.exp(-min(ahead_gap, behind_gap) / 2.0), 0.0, 1.0))
                fuel_load = max(0.0, 105.0 * (1.0 - 0.98 * progress))
                wet_penalty = 0.055 if wet_race else 0.0
                pace_ratio = (
                    1.02
                    - 0.07 * skill
                    + 0.07 * (lap / lap_count)
                    + 0.035 * traffic
                    + wet_penalty
                    + rng.normal(0.0, 0.012)
                )
                lap_time = base_lap_time * pace_ratio
                sector_shares = rng.dirichlet((9.0, 10.0, 8.0))
                position = int(np.clip(grid_position + rng.integers(-1, 2), 1, driver_count))
                track_grip = float(np.clip(0.97 - (0.18 if wet_race else 0.0) + 0.025 * progress + rng.normal(0.0, 0.012), 0.5, 1.0))
                rows.append(
                    {
                        "race_id": race_id,
                        "season": int(config["season_start"]) + race_index // 24,
                        "race_round": race_index % 24 + 1,
                        "circuit_id": circuit_id,
                        "circuit_length_km": length_km,
                        "circuit_turns": turns,
                        "circuit_drs_zones": drs_zones,
                        "circuit_overtake_difficulty": float(np.clip(1.0 - turns / 40.0, 0.1, 0.9)),
                        "circuit_base_lap_time_sec": base_lap_time,
                        **driver,
                        "grid_position": grid_position,
                        "lap": lap,
                        "position": position,
                        "lap_time_sec": lap_time,
                        "s1_time_sec": lap_time * float(sector_shares[0]),
                        "s2_time_sec": lap_time * float(sector_shares[1]),
                        "s3_time_sec": lap_time * float(sector_shares[2]),
                        "tire_compound": compound,
                        "tire_age_laps": lap,
                        "fuel_load_kg": fuel_load,
                        "fuel_ratio": fuel_load / 105.0,
                        "ers_deploy_pct": float(rng.uniform(25.0, 100.0)),
                        "ers_harvest_pct": float(rng.uniform(10.0, 75.0)),
                        "drs_activated": int(ahead_gap < 1.0),
                        "gap_to_leader_sec": max(0.0, (position - 1) * float(rng.uniform(0.8, 2.4))),
                        "gap_ahead_sec": ahead_gap,
                        "gap_behind_sec": behind_gap,
                        "track_status": "WET" if wet_race else "GREEN",
                        "weather_current": weather,
                        "race_weather_start": weather,
                        "race_air_temp_c": air_temp,
                        "race_track_temp_c": track_temp,
                        "race_humidity_pct": humidity,
                        "wind_speed_kph": wind_speed,
                        "track_grip_level": track_grip,
                        "race_total_laps": lap_count,
                        "race_progress": lap / lap_count,
                        "_pace_stress": float(np.clip((pace_ratio - 0.9) / 0.35, 0.0, 1.0)),
                        "_circuit_load_index": float(np.clip((turns / length_km - 1.5) / 6.0, 0.0, 1.0)),
                        "_traffic_intensity": traffic,
                        "_tyre_sensitivity": 0.0,
                        "_surface_shock": 0.0,
                    }
                )

    return pd.DataFrame(rows).sort_values(GROUP_COLUMNS + ["lap"]).reset_index(drop=True)


def attach_simulated_targets(states: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add next-lap labels and accumulated current wear, excluding latent inputs."""
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    required = {
        "race_id", "driver_id", "lap", "tire_compound", "tire_age_laps",
        "race_track_temp_c", "race_air_temp_c", "_circuit_load_index",
    }
    missing = sorted(required - set(states.columns))
    if missing:
        raise ValueError(f"Missing target-generation state columns: {missing}")

    out = states.copy().sort_values(GROUP_COLUMNS + ["lap"]).reset_index(drop=True)
    rng = np.random.default_rng(seed ^ 0x5EED)
    target = np.full(len(out), np.nan, dtype=float)
    for _, indices in out.groupby(GROUP_COLUMNS, sort=False).groups.items():
        ordered_indices = list(indices)
        tyre_sensitivity = 0.75 + 0.5 * float(rng.beta(8.0, 8.0))
        for index in ordered_indices[:-1]:
            row = out.loc[index]
            surface_shock = 0.8 + 0.4 * float(rng.beta(5.0, 5.0))
            target[index] = calculate_next_lap_degradation_pct(
                compound=str(row["tire_compound"]),
                current_tire_age_laps=float(row["tire_age_laps"]),
                track_temp_c=float(row["race_track_temp_c"]),
                air_temp_c=float(row["race_air_temp_c"]),
                circuit_load_index=float(row["_circuit_load_index"]),
                pace_stress=float(row["_pace_stress"]),
                fuel_fraction=float(np.clip(row["fuel_load_kg"] / 105.0, 0.0, 1.0)),
                driver_aggression=float(row["driver_aggression_rating"]),
                traffic_intensity=float(row["_traffic_intensity"]),
                race_progress=float(row["race_progress"]),
                tyre_sensitivity=tyre_sensitivity,
                surface_shock=surface_shock,
            )

    out[TARGET_COLUMN] = target
    previous_increment = out.groupby(GROUP_COLUMNS, sort=False)[TARGET_COLUMN].shift(1).fillna(0.0)
    out["tire_wear_pct"] = previous_increment.groupby([out[column] for column in GROUP_COLUMNS], sort=False).cumsum()
    return out.drop(columns=[column for column in out.columns if column.startswith("_")])


def generate_dataset(config: Mapping[str, Any] | None = None, seed: int | None = None) -> pd.DataFrame:
    actual_config = load_config() if config is None else dict(config)
    validate_config(actual_config)
    actual_seed = int(actual_config["seed"] if seed is None else seed)
    states = generate_race_states(actual_config, actual_seed)
    return attach_simulated_targets(states, actual_seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible synthetic F1-style race telemetry.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--races", type=int)
    parser.add_argument("--drivers", type=int)
    parser.add_argument("--laps", type=int)
    parser.add_argument("--compounds", nargs="+")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    for argument, key in ((args.races, "races"), (args.drivers, "drivers"), (args.laps, "laps_per_race"), (args.seed, "seed")):
        if argument is not None:
            config[key] = argument
    if args.compounds:
        config["compounds"] = [compound.upper() for compound in args.compounds]
        config["compound_probabilities"] = [1.0 / len(args.compounds)] * len(args.compounds)

    dataset = generate_dataset(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Wrote {len(dataset):,} rows to {args.output}")
    print(f"Seed: {config['seed']}; races: {config['races']}; drivers: {config['drivers']}; laps: {config['laps_per_race']}")


if __name__ == "__main__":
    main()