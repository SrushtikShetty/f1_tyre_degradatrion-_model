# Model Feature Importance and Explainability

## What the Current Artifact Supports

The root `feature_importance.csv` contains only `feature` and `importance`; it has no model, split, or calculation-method metadata. The dashboard must therefore label its method `legacy_unspecified`. It is not evidence of causal or physical importance.

The `/api/explain` endpoint does not compute SHAP or local attribution. It may show the supplied input value alongside a saved global score, but neither the sign of that input nor its magnitude is a local contribution to the prediction. The response explicitly identifies this as global predictive association only, and reports the importance method as `legacy_unspecified` for the current CSV.

## Corrected Pipeline Method

The reusable `grouped_permutation_importance` utility permutes related raw features together on VALIDATION and reports the change in MAE. Related fields should be grouped by a documented theme (for example tyre state, thermal conditions, pace, or traffic) to reduce misleading conclusions from correlated inputs. The existing ablation experiments are complementary model-level sensitivity checks.

No SHAP values are currently computed, so reports and UI must not claim SHAP or feature-level causal effects. These measurements describe model dependence within the synthetic simulator and do not establish physical causality or real-world F1 effects.