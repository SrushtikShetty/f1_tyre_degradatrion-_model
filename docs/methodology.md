# Methodology

This project follows a strict causal workflow for tyre-degradation prediction.

## Target Construction

The target is created from the next lap within each race/driver group. The current tyre wear is preserved for reconstruction of absolute next-lap wear, but the raw current-wear value is not allowed to leak directly into the predictive feature matrix used for degradation increment estimation.

## Feature Engineering

The canonical feature pipeline includes:

- grouped lag features across prior laps
- rolling means for telemetry windows
- tyre-age and tire-progress ratios
- pace and track-state features
- race-grouped temporal ordering

## Leakage Control

The project avoids future-state information by:

- building lagged values only from prior laps
- aggregating rolling features only from historical values
- excluding race-end, finish, points, and pit-stop outcome fields
- validating on race-grouped splits rather than random rows

## Validation

The project uses grouped cross-validation so that all laps from the same race/driver segment remain in the same fold. This reduces optimistic leakage from near-duplicate race context.

## Model Usage

All four models use the same pipeline so that differences in performance reflect model choice rather than inconsistent preprocessing.
