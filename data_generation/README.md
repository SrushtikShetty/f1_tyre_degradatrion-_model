# Synthetic Dataset Generator

Run from the repository root:

```powershell
python -m data_generation.generate_dataset
```

The versioned defaults are in `config.json`. The seed, race count, driver count, laps per race, and compound set can be overridden from the command line. Track and weather ranges, rain probability, and compound sampling probabilities are configured in that file.

```powershell
python -m data_generation.generate_dataset --seed 7 --races 30 --drivers 10 --laps 40 --compounds HARD MEDIUM SOFT --output data/generated/example.csv
```

The generator first creates end-of-lap observable race state, then samples per-stint tyre sensitivity and per-next-lap surface/load variation from fixed beta distributions to create `next_lap_degradation_pct`. Those latent draws are excluded from the exported table. The final lap of each race-driver sequence has no next-lap label. Current `tire_wear_pct` is accumulated from prior generated increments and is retained as target-construction context, not as an input feature.

Outputs default to `data/generated/synthetic_race_telemetry.csv`, which is ignored by Git. Re-run the same config and seed to recreate the same CSV. This is an explicit synthetic benchmark simulator; it is not a source for historical or measured F1 telemetry.