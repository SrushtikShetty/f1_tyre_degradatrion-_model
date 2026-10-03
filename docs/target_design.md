# Synthetic Target Design

## Prediction Question

At the end of lap `t`, estimate the tyre-wear increment, in percentage points, that the synthetic simulator will realize during lap `t+1`. The predictor is restricted to state available by the end of lap `t`; future lap telemetry is not an input. The generated label is a simulated outcome, not a measurement of a real tyre.

## Response Process

The first implementation is `calculate_next_lap_degradation_pct` in `src/f1_tyre/synthetic_degradation.py`. It multiplies a compound-specific base rate by transparent response factors:

`increment = compound_base * age * temperature * circuit_load * pace * fuel * driving * traffic * race_phase * tyre_sensitivity * surface_shock`

Age increases stress smoothly. Thermal stress grows with distance from a compound-specific operating point and ambient temperature. Circuit load, pace, fuel fraction, driver aggression, traffic, and race progress contribute bounded effects. The coefficients are benchmark assumptions, not values calibrated against real F1 measurements.

`tyre_sensitivity` represents per-stint variation in tyre construction. `surface_shock` represents a bounded next-lap realization of local surface and load variability. The generator samples these latent factors from documented distributions; they are not given to the model. They are structured simulator state, not an arbitrary Gaussian perturbation added to an otherwise deterministic label.

Since observable current race state and latent factors vary, compound and age alone cannot determine the label. The compound-age lookup must be computed on the generated chronological holdout and published beside statistical models. The existing `train.csv`, labels, and saved metrics remain the legacy baseline until the generator and training path are integrated; this design document does not claim that they are already corrected.

## Diagnostic Requirements

For every generated dataset, record target unique-value count and distribution, overall variance, within-compound-age variance, compound-age lookup holdout metrics, and target associations with individual input fields. Reject the generated benchmark if compound-age groups remain constant or the lookup reaches the project guard threshold. Report results as synthetic-simulation performance only.