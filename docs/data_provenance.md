# Dataset Provenance and Research Scope

## Determination

The project dataset is a **constructed, synthetic-style F1 benchmark with unverified source provenance**. Git history contains the CSV additions, but no original data acquisition record or source-to-table generator that establishes the rows as real race telemetry. A telemetry-shaped schema, plausible circuit labels, season values, or driver/team names do not establish authenticity. It is not possible to separate potentially observed inputs from synthetic or manually constructed fields using the available provenance.

The wear column has stronger evidence: all 233,640 rows exactly match the constructed schedule `min(100, rate[compound] * (tire_age_laps - 1))` for HARD=1.0, MEDIUM=1.5, and SOFT=2.5. The next-lap wear increment is then derived from that schedule by shifting the wear column within race-driver rows. These labels are not documented as measured tyre degradation.

## Field Provenance Register

| Field family | Examples | Classification supported by repository evidence | Limitations |
|---|---|---|---|
| Row identifiers and chronology | `race_id`, `season`, `race_round`, `lap` | Supplied table fields; source unknown | Values do not verify an official event, calendar, or row collection process. |
| Circuit metadata | `circuit_id`, country, length, turns, DRS zones | Supplied metadata-shaped fields; source unknown | No reference source or validation record is included. |
| Driver and team metadata | names, nationality, `driver_*_rating`, `team_*_rating`, budget tier | Supplied fields; source unknown; ratings are constructed-looking but their generator is unavailable | No source, measurement method, date, or calibration is recorded. |
| Qualifying and race state | grid, qualifying times, position, lap and sector times, gaps | Telemetry-shaped supplied fields; observed-vs-simulated origin cannot be verified | No sensor, timing feed, or race-data source is identified. |
| Tyre and pit fields | compound, tyre age, pit-stop fields, `tire_wear_pct` | Compound/age provenance unknown; wear is demonstrably constructed from compound and age | Wear follows the exact capped compound-rate schedule described above. |
| Weather and track state | air/track temperature, humidity, wind, grip, weather and track status | Supplied fields; source unknown | No measurement source, timestamp, interpolation, or forecast/observation distinction is documented. |
| Model target | `next_lap_wear_increment` | Derived target, not an observed source column | Built as next-row wear minus current-row wear for rows passing consecutive-lap, age, and compound checks. The target inherits the deterministic wear schedule. |
| Engineered model inputs | lags, rolling telemetry, ratios, sector sums, gaps and position transforms | Derived features | Availability and causality are evaluated separately in the feature audit; derived does not imply measured or physically validated. |
| Predictions | model output columns in prediction artifacts | Model predictions | Predictions are not observations or ground truth. |

## Git and File Evidence

`train.csv` appears in two original-history commits dated 2026-09-29, both titled `first commit`. Those commits add the CSV and legacy training/prediction files but do not contain an original data generator or provenance manifest. The current audit/re-evaluation scripts reconstruct and test the wear formula; they are not the source generator and must not be represented as such.

## Research Scope

The dataset is suitable for software, pipeline, and synthetic-rule experiments after its generation and target design become reproducible. It is **not currently suitable for claims about real-world Formula 1 tyre degradation, race strategy, or generalization to real races**. Metrics describe prediction of labels in this constructed table only.

Until field-level provenance is independently established, describe the inputs as **telemetry-shaped benchmark fields**, the wear labels as **constructed**, and the target as **derived**. Do not call the target measured tyre wear, describe predictive importance as physical causality, or imply that saved-model accuracy validates real racing performance.
