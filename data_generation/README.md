# Synthetic Dataset Generator

Run from the repository root:

```powershell
python -m data_generation.generate_dataset
```

The versioned defaults are in `config.json`. The seed, race count, driver count, laps per race, and compound set can be overridden from the command line. Track and weather ranges, rain probability, and compound sampling probabilities are configured in that file.

```powershell
python -m data_generation.generate_dataset --seed 7 --races 30 --drivers 10 --laps 40 --compounds HARD MEDIUM SOFT --output data/generated/example.csv
```

The generator first creates end-of-lap observable race state, then creates `next_lap_degradation_pct` from that state plus bounded simulator-only latent factors. Internal underscore-prefixed columns such as `_pace_stress`, `_circuit_load_index`, `_traffic_intensity`, `_tyre_sensitivity`, and `_surface_shock` exist only inside target generation. They are dropped before export and must not be model features. The final lap of each race-driver sequence has no next-lap label. Current `tire_wear_pct` is accumulated from prior generated increments and is retained as target-construction context, not as an input feature.

Outputs default to `data/generated/synthetic_race_telemetry.csv`, which is ignored by Git. Re-run the same config and seed to recreate the same CSV. This is an explicit synthetic benchmark simulator; it is not a source for historical or measured F1 telemetry.
