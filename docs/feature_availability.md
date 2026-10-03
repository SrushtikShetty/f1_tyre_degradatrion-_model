# Feature Availability and Causal Construction

The prediction boundary is the end of lap `t`. A model row may use the state observed by that time and prior rows from the same race-driver sequence. The synthetic target is the simulated degradation realized during lap `t+1`; it is a label, never an input.

## Feature Groups

| Class | Fields or transformations | Availability rule |
|---|---|---|
| Current/end-of-lap state | Lap and position, current lap/sector times, compound and age, fuel, ERS/DRS, current gaps, current weather/temperature/grip/status | Available only after the represented lap is complete. For a prediction at end of lap `t`, these are current-state observations, not lap `t+1` values. |
| Previous-lap state | `*_lag1`, `*_lag2`, `*_lag3` for pace, position, gaps, fuel, ERS, temperature, humidity, wind, and grip inputs | Computed with within-race-driver `shift(k)` after stable sorting by race, driver, and lap. |
| Rolling history | Lap and sector timing rolling means over 3/5 rows; rolling standard deviation over 5 rows | The source is shifted by one before rolling, so row `t` uses rows strictly before `t`. |
| Current-state derived | `race_progress`, `tire_age_ratio`, `tire_age_squared`, `total_nearest_gap`, `position_change`, `lap_time_per_km`, `sector_total_sec` | Derived only from current state and a supplied total race distance. Age ratio is omitted when that denominator is unavailable; completed-race maximum lap is never used as a substitute. |
| Static/pre-race context | Circuit characteristics, driver/team descriptors, grid and qualifying fields, scheduled race distance | These are only valid if actually known before or by the prediction moment. Their provenance is unverified in the checked-in legacy dataset. The synthetic generator creates them as simulator fields. |
| Future-derived or unavailable | Next-lap telemetry, future wear, finishing position, race-end weather, final points/status, pit outcomes from after lap `t` | Not valid model inputs; rejected by the feature policy and excluded from the generated current-state rows. |

`prepare_feature_matrix` sorts by race-driver-lap before creating histories. Causal regression tests perturb a later row and verify an earlier engineered row is unchanged. These availability rules establish temporal construction in the simulator; they do not establish that legacy CSV fields are genuine measurements.