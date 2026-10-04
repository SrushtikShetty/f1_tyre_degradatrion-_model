# Current Repository State Audit

Date: 2026-10-04

## Scope

This audit records the repository state before additional implementation changes. It is based on:

- `git status`, `git log --oneline -20`, `git branch --show-current`, and `git diff`
- `python -m pytest -q`
- Inspection of authoritative evaluation artifacts, documentation, model artifacts, backend, frontend, training, source, data-generation, and test directories

## Git State

- Branch: `main`
- Remote tracking: up to date with `origin/main`
- Latest commit: `04eec84 models: retrain and save models using corrected evaluation protocol`
- Working tree at audit start contained one modified tracked bytecode artifact:
  - `models/__pycache__/common.cpython-314.pyc`
- No source-code diff was present at audit start.

## Test State

Baseline command:

```powershell
python -m pytest -q
```

Result:

- `81 passed`
- `1 warning`
- Warning source: FastAPI/Starlette `TestClient` dependency deprecation

## Already Complete

- The original deterministic target problem has been addressed by replacing the old target with a synthetic causal degradation simulator.
- The authoritative target is `next_lap_wear_increment`.
- The prediction moment is documented as end of lap `t`, using features known no later than end of lap `t`, to predict simulated wear realized during lap `t+1`.
- The authoritative synthetic benchmark dataset exists at `data/generated/authoritative_seed_20261004.csv`.
- The authoritative evaluation identifies the dataset as synthetic and does not claim measured real-world F1 accuracy.
- The chronological evaluation split is documented and present in `artifacts/evaluation/final_metrics.json`:
  - Train: races 1-36, 20,304 rows
  - Validation: races 37-48, 6,768 rows
  - Final holdout: races 49-60, 6,768 rows
- Integrity checks in the authoritative metrics report all pass:
  - zero race overlap
  - zero race-driver overlap
  - zero tyre-stint overlap
  - zero duplicate-row overlap
- Preprocessing fit scope is recorded as train-only.
- The final holdout role is documented as final evaluation only after model and feature selection are complete.
- Corrected model artifacts exist under `artifacts/models/corrected/`.
- Development artifacts for seed `20261004` exist under `artifacts/models/development_seed_20261004/`.
- The target-determinism guard is present in the authoritative metrics and passes:
  - Threshold: `0.98`
  - Final-holdout compound-age lookup R2: `0.7899768657310919`
- Explainability is documented in the authoritative artifacts as grouped permutation validation importance, not SHAP.

## Authoritative Artifacts

Current authoritative evaluation artifacts:

- `artifacts/evaluation/final_metrics.json`
- `artifacts/evaluation/final_report.json`
- `docs/evaluation_protocol.md`

Authoritative dataset:

- Path: `data/generated/authoritative_seed_20261004.csv`
- SHA-256: `c9121cdc3f5d00b7ba9e65c3af6bda27a038a9eb2ce6f241f0b94f6af483127a`
- Seed: `20261004`
- Rows: `33,840`
- Races: `60`

Authoritative corrected model artifact directory:

- `artifacts/models/corrected/`

Corrected model artifacts present:

- `artifacts/models/corrected/ridge.joblib`
- `artifacts/models/corrected/random_forest.joblib`
- `artifacts/models/corrected/xgboost.joblib`
- `artifacts/models/corrected/neural_network.joblib`

## Authoritative Metrics Snapshot

Validation:

| Model | R2 | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Ridge | 0.8878156070 | 0.0888327862 | 0.1199119697 |
| Random Forest | 0.9067123703 | 0.0818360886 | 0.1093473708 |
| XGBoost | 0.9240633708 | 0.0735441701 | 0.0986556525 |
| Neural Network | 0.7762125098 | 0.1201641669 | 0.1693613156 |

Final holdout:

| Model | R2 | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Ridge | 0.8314400958 | 0.1009574129 | 0.1300485995 |
| Random Forest | 0.8782007870 | 0.0834272877 | 0.1105479727 |
| XGBoost | 0.9119949374 | 0.0706428040 | 0.0939685163 |
| Neural Network | 0.7937591083 | 0.1138619444 | 0.1438520087 |

Baselines:

| Baseline | Split | R2 | MAE | RMSE |
| --- | --- | ---: | ---: | ---: |
| GlobalMean | validation | -0.0053534648 | 0.2845505530 | 0.3589679752 |
| CompoundOnly | validation | 0.5339660162 | 0.1816896283 | 0.2444021992 |
| CompoundAgeLookup | validation | 0.7603291382 | 0.1278339719 | 0.1752685261 |
| GlobalMean | final_holdout | -0.0001225790 | 0.2542632770 | 0.3167781059 |
| CompoundOnly | final_holdout | 0.5382431390 | 0.1676746275 | 0.2152460648 |
| CompoundAgeLookup | final_holdout | 0.7899768657 | 0.1119913960 | 0.1451650638 |

## Current UI Data Sources

Backend file inspected:

- `backend/app.py`

Current observed API sources:

- `/api/models` calls `list_available_models()` from `src/f1_tyre/model_loader.py`.
- `/api/model-metrics` reads `artifacts/model_metrics.json` through `MODEL_METRICS_FILE = ARTIFACTS_DIR / 'model_metrics.json'`.
- `/api/feature-importance` reads root-level `feature_importance.csv`.
- `/api/predict` loads saved artifacts through `load_model()`.
- `/api/explain` uses the same root-level feature-importance response and states that no SHAP values are computed.

Important inconsistency:

- The dashboard metrics endpoint does not currently use the authoritative file `artifacts/evaluation/final_metrics.json`.
- This means the UI can display stale or legacy metrics even though corrected authoritative metrics exist.

## Current Model Artifact Sources

Model-loading file inspected:

- `src/f1_tyre/model_loader.py`

Observed load order:

1. `artifacts/models/`
2. `artifacts/`
3. project root
4. fallback legacy file `f1_tyre_wear_model.joblib`

Important inconsistency:

- `MODEL_ARTIFACT_DIR` resolves to `artifacts/models`, not `artifacts/models/corrected`.
- Legacy top-level model artifacts exist beside corrected artifacts:
  - `artifacts/models/ridge.joblib`
  - `artifacts/models/random_forest.joblib`
  - `artifacts/models/xgboost.joblib`
  - `artifacts/models/neural_network.joblib`
- Because the loader checks `artifacts/models/<model>.joblib` before `artifacts/models/corrected/<model>.joblib`, inference can currently load the legacy or non-authoritative artifact path.
- The loader also has a project-root legacy fallback, which could hide missing corrected artifacts.

## Stale Or Legacy Artifacts

Potentially stale or legacy files that need later classification before deletion or retention:

- `artifacts/model_metrics.json`
- `feature_importance.csv`
- `artifacts/models/*.joblib` at the top of `artifacts/models/`
- `artifacts/models/development_seed_20261004/*.joblib`
- `data/generated/cli_probe.csv`
- `data/generated/synthetic_race_telemetry.csv`
- tracked `__pycache__` and `.pyc` files

These files were not deleted during this audit.

## Current Correctness Risks

Remaining scientific or technical issues identified before implementation changes:

1. UI metrics are not sourced from the authoritative final evaluation artifact.
2. Model loading can select non-corrected artifacts before corrected artifacts.
3. The backend exposes root-level legacy feature importance rather than the grouped permutation importance in `artifacts/evaluation/final_metrics.json`.
4. Corrected model artifact provenance and loader policy need hardening so inference cannot silently fall back to legacy models.
5. Repository hygiene needs attention because tracked bytecode is present and one tracked `.pyc` file is modified.
6. Stale generated data and legacy model/metric files need explicit documentation or cleanup so a fresh clone can distinguish authoritative artifacts from historical artifacts.
7. The final audit document does not yet exist and must be created after verification/fixes.

## Incomplete Phases Remaining

The following work remains after this Phase 0 audit:

- Finish leakage and feature-lineage verification.
- Verify simulator target lineage and latent-variable isolation.
- Reconcile authoritative metric provenance with code and docs.
- Make corrected model artifacts authoritative for inference.
- Analyze tyre-state baseline contribution using validation.
- Document simulator behavior and synthetic-target diagnostics.
- Make UI metrics use `artifacts/evaluation/final_metrics.json`.
- Remove or correct stale SHAP/explainability terminology in UI and API surfaces.
- Verify end-to-end FastAPI inference against corrected artifacts.
- Audit and clarify stale repository content.
- Run full end-to-end reproducibility checks.
- Create the final scientific audit document.

## Phase 0 Conclusion

The corrected synthetic benchmark and final evaluation artifacts are present and internally coherent enough to serve as the authoritative baseline for subsequent phases. The main unresolved integration risk is that serving and UI code still point at legacy artifact locations for metrics, feature importance, and model loading. No implementation files were modified during this audit.
