# Leakage Prevention

The project includes a dedicated leakage audit to stop future-state information from entering the model matrix.

## Forbidden Patterns

The audit flags columns whose names match known leakage patterns such as:

- future
- next_
- finish
- points
- fastest
- pit_stop
- pit_new
- target
- race_end
- status

## Protection Strategy

The leakage guard is implemented as a feature-matrix audit that raises a `ValueError` if forbidden columns are present. This protects the training pipeline and prevents accidental leakage from duplicate or historical target columns.

## Validation Rule

All models must train and predict using only the canonical causal feature set and grouped validation logic. No future-lap information should be included in the model input matrix.
