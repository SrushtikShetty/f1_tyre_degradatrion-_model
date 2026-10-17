# Repository-to-Paper Alignment Audit

**Audit date:** 2026-10-17  
**Inspected branch head before this audit:** \`main\` at \`38bdff2fc8c3adbfb4e420eeb7ff7a82ef17d33d\`  
**Working branch:** \`research/path-a-provenance-audit\`  
**Scope:** GitHub source tree, target builder/generator, evaluation artifacts, model loader, backend metrics/explanation routes, README and submitted literature. This is source inspection of the public repository. The repository could not be cloned in the audit runtime because outbound DNS to GitHub failed; current repository tests were therefore not executed locally in this runtime.

## Findings summary

The current default experiment is a race-grouped, chronological evaluation on a **synthetic simulator-generated dataset**. The current training code does not use FastF1 and the primary target is not measured tyre wear. The old \`train.csv\` / \`test.csv\` files remain in the repository, but those have unverified source provenance and a legacy deterministic wear schedule. They must not be conflated with the authoritative seed-20261004 run.

The prior manuscript draft's central *real-F1 aggression improves degradation prediction* hypothesis has **not** been tested by the current saved artifacts. It must be stated as a research question for a future FastF1 experiment, not a result. Current artifacts support only the synthetic benchmark comparison and the feature-group permutation analysis described below.

## Repository-to-paper claim matrix

| Manuscript or project claim | Status | Evidence and exact location | Paper treatment |
|---|---|---|---|
| The active default training data are authentic historical F1 telemetry | **Contradicted** | \`training/train_corrected_models.py::DEFAULT_DATASET\` points to \`data/generated/authoritative_seed_20261004.csv\`; \`data_generation/generate_dataset.py\`; \`src/f1_tyre/synthetic_degradation.py\`; \`artifacts/evaluation/final_metrics.json\` says \`dataset.type = synthetic\`. | Removed. The paper describes a synthetic benchmark and a proposed FastF1 experiment separately. |
| Current target is measured tyre wear | **Contradicted** | \`src/f1_tyre/data.py::build_target_frame\` copies generated \`next_lap_degradation_pct\` into \`next_lap_wear_increment\`; \`src/f1_tyre/synthetic_degradation.py::calculate_next_lap_degradation_pct\` defines a simulation formula. | Target described as a simulated one-lap wear increment. No real-wear claim. |
| Legacy \`train.csv\` wear is measured | **Unable to verify / contradicted by its exact schedule** | \`docs/data_provenance.md\` legacy forensic finding and \`docs/baseline_audit.md\`: all 233,640 \`tire_wear_pct\` rows match \`min(100, rate[compound] * (tire_age_laps - 1))\`; next-lap target is shifted from that value. | Legacy schedule discussed only as a separate historical benchmark. Legacy scores excluded from current model-results claims. |
| Current simulator target is a simple deterministic compound-age lookup | **Not supported by current generator** | \`src/f1_tyre/synthetic_degradation.py\` uses latent \`tyre_sensitivity\` and \`surface_shock\`; \`artifacts/evaluation/target_determinism_seed_20261004.json\` reports within-compound-age variability and lookup \(R^2=0.78998\) on final holdout. | Explicitly distinguish from legacy target determinism. |
| Driver aggression in the active target is measured | **Contradicted** | \`data_generation/generate_dataset.py::generate_race_states\` samples synthetic \`driver_aggression_rating\`; that field contributes to the generator target. No psychological or on-track validation source is provided. | Call it a synthetic simulator input. Use “driving intensity proxy” for the future public-telemetry experiment. |
| The active split is chronological and race-disjoint | **Verified in saved artifact** | \`src/f1_tyre/evaluation/splits.py::chronological_race_splits\`; \`docs/evaluation_protocol.md\`; \`artifacts/evaluation/final_metrics.json\`: train 36 races, validation 12, holdout 12; zero race, race-driver, stint and duplicate-row overlap; preprocessing fitted on train only. | Report as synthetic benchmark protocol, not as an authentic-data split. |
| The active authoritative benchmark has 33,840 rows and 60 races | **Verified in saved artifact** | \`artifacts/evaluation/final_metrics.json\`: 34,560 race-state rows, 33,840 modeling rows, 60 races, seed 20261004, dataset SHA-256 \`c9121cdc3f5d00b7ba9e65c3af6bda27a038a9eb2ce6f241f0b94f6af483127a\`. | Include in Data / Benchmark. |
| XGBoost is best among current saved models on the synthetic final holdout | **Verified in saved metrics, not locally rerun here** | \`artifacts/evaluation/final_metrics.json\`: XGBoost \(R^2=0.9119949\), MAE \(=0.0706428\), RMSE \(=0.0939685\); lower MAE and RMSE and higher R² than Ridge, Random Forest and MLP. | Report with explicit simulated-target scope and source artifact. |
| Current evaluation calculates SHAP | **Contradicted** | \`artifacts/evaluation/final_metrics.json::explainability.shap_computed = false\`; \`src/f1_tyre/evaluation/explainability.py::grouped_permutation_importance\` calculates validation grouped-permutation MAE increases; \`backend/app.py::explain_prediction\` states no local attribution is computed. | No SHAP figure or SHAP-result claim. Report grouped permutation importance only. |
| Current feature importance indicates aggression causes tyre wear | **Unsupported** | The active importance is grouped permutation on synthetic validation predictions; it measures predictive reliance under this model, not a causal intervention. | State association only. No causal or driver-personality claim. |
| Current UI serves the authoritative final-holdout metrics | **Outdated / bug** | \`backend/app.py::MODEL_METRICS_FILE\` is \`artifacts/model_metrics.json\`, while authoritative metrics are \`artifacts/evaluation/final_metrics.json\`. | Fixed on this branch to read the authoritative artifact. |
| Current UI serves current group importance | **Outdated / bug** | \`backend/app.py::FEATURE_IMPORTANCE_FILE\` reads root \`feature_importance.csv\`, a legacy per-feature file; saved current groups are in \`artifacts/evaluation/feature_importance_seed_20261004.json\`. | Fixed on this branch to read the authoritative group artifact. |
| Selected-model inference never silently changes to another model | **Not met before this branch** | \`backend/app.py::_safe_execute_prediction\` caught any \`ValueError\` from \`load_model\` and loaded a fallback model. | Fixed on this branch: selected-model load errors now propagate; no silent model substitution. |
| The current project runs FastF1 end to end | **Planned / not evidenced** | No authoritative FastF1 data-download or real-data training pipeline is in the active training entry point. A live FastF1 experiment was not run during this audit. | The FastF1 path is delivered as a separate research script/protocol; no performance figures from it. |
| Current tests pass on the repository head inspected | **Unable to verify in this runtime** | \`docs/current_state_audit.md\` records 81 passing tests as of 2026-10-04, but this audit could not clone the current repository due DNS failure. | Historical test count labelled by date; it is not presented as a fresh result. |
| Current saved synthetic models have a complete SHAP artifact | **Not implemented** | No SHAP values appear in authoritative metrics; active explanation uses grouped permutation importance. | A real-data SHAP analysis remains future work after a valid authentic-data fit. |

## Current authoritative benchmark results

Metrics are read from \`artifacts/evaluation/final_metrics.json\`. Units for MAE/RMSE are the simulator's target units (simulated wear-increment percentage points, per the target's intended definition); they are **not seconds and not measured tread depth**.

| Model | Final-holdout R² | Final-holdout MAE | Final-holdout RMSE |
|---|---:|---:|---:|
| Ridge | 0.83144 | 0.10096 | 0.13005 |
| Random Forest | 0.87820 | 0.08343 | 0.11055 |
| XGBoost | 0.91199 | 0.07064 | 0.09397 |
| Neural network | 0.79376 | 0.11386 | 0.14385 |
| Compound-age lookup baseline | 0.78998 | 0.11199 | 0.14517 |

The group-importance artifact is \`artifacts/evaluation/feature_importance_seed_20261004.json\`, method \`grouped_permutation_validation\`. It reports the largest validation MAE increase for tyre-state features (mean +0.23058), then fuel/energy (+0.10857), driver/team synthetic ratings (+0.02957), circuit load (+0.01555), pace/sectors (+0.01476), thermal/weather (+0.01314), traffic/position (+0.00773), and race context (+0.00416). These values refer to **synthetic validation predictions**, not real-F1 aggression features. The saved artifact's baseline validation MAE is 0.07354.

## FastF1 channel and construct-validity notes

The official FastF1 \`Telemetry\` source documents Speed (km/h), RPM, nGear, Throttle (0–100; 104 may represent unavailable/error data), Brake (boolean), DRS, and X/Y/Z position (in tenths of a metre). When car and position telemetry are merged, coordinates may be interpolated. Therefore:
- The proposed index contains observable/derived signal proxies only.
- Speed derivatives require timestamp-aware smoothing; they are sensitive to sample spacing and noise.
- Curvature-derived lateral acceleration is approximate and depends on interpolated coordinates, repeated positions, coordinate smoothing and differentiation.
- Steering angle, brake pressure, tyre load and direct tyre wear are not assumed available.
- The index is not a validated psychological aggression measure.
- Pace degradation is inferred from lap time; it is not a tyre-tread measurement.
- A retrospective relative-pace target that subtracts the field median at lap \(t+1\) uses future data to form the **label**, though not as a feature. It can support offline prediction research, but a live system would need a predeclared reference that is available at the decision time.

## Literature audit

| Source | Verification status | Use in the manuscript |
|---|---|---|
| Todd et al., “Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy,” ACM SAC 2025, DOI 10.1145/3672608.3707765 | Publisher/proceedings record verified; pages 1082–1089 confirmed by King's College publication record. | Closest XGBoost + F1 telemetry + XAI related work. Their target is tyre energy, not this proposed relative pace-change proxy. |
| Cappello and Hoegh, “A state-space approach to modeling tire degradation in formula 1 racing,” *Journal of Sports Analytics*, DOI 10.1177/22150218261446170 | SAGE publisher page verified; first online 27 April 2026. | Supports latent/probabilistic pace degradation as adjacent methodology; reports 19 2025 sessions for Lewis Hamilton, not multiple-driver generalisation. |
| Mistry and Smith, “AutoF1: An RNN-Based Approach to Simulating Strategic Decision-Making in Formula 1,” ITSC 2025 | White Rose accepted-version record verified; still describes item as “In Press.” | Cite as accepted/author-produced version; do not invent final page range or proceedings DOI. |
| De Bellis et al., “Estimating tyre wear for real-time applications: a cascaded Kalman filter approach,” *Vehicle System Dynamics*, DOI 10.1080/00423114.2026.2730699 | Publisher record and user-supplied article verified. | Adjacent real-vehicle tyre-wear observers; this is not F1 data. |
| Gitis and Gitis, “Neural Network Decision Support System for Formulating a Racing Team Strategy,” DOI 10.31891/csit-2025-2-17 | Journal article and DOI verified. | Related strategy decision-support work, not direct validation of this task's outcome. |
| Thomas et al., “Explainable Reinforcement Learning for Formula One Race Strategy,” DOI 10.1145/3672608.3707766 | DOI and source paper verified. | Explainability for race-strategy decision making, but task/algorithm differs. |
| Chen and Guestrin, XGBoost (KDD 2016), DOI 10.1145/2939672.2939785 | Standard bibliographic record; widely verified. | Algorithm background. |
| Lundberg and Lee, “A Unified Approach to Interpreting Model Predictions,” NeurIPS 2017 | Standard proceedings record; widely verified. | SHAP background; cited as methodology, not claimed as run in current project. |
| Feng et al., “Can vehicle longitudinal jerk be used to identify aggressive drivers? An examination using naturalistic driving data,” *Accident Analysis & Prevention* 104 (2017), DOI 10.1016/j.aap.2017.04.012 | Elsevier record verified. | Supports investigating jerk-based behavioural proxies in road-driving data; construct/vehicle context differs from F1. |
| García Tejada 2023 and Rondelli 2022 theses; Rao 2023 MSc project | User-supplied academic documents; not used as primary sources for material claims. | May be cited only as theses/project reports when necessary. |
| Gupta, “Auditing Evaluation Leakage in Formula 1 Race-Strategy Machine Learning” | Supplied manuscript only; public publication record not independently verified. | Not cited as published evidence. |
| “Residual Hybrid Learning for Data-Limited Formula 1 Lap Time Prediction Across Multiple Circuits” | Supplied manuscript / proceedings information supplied in the PDF; DOI not confirmed. | Use only with publication status caveat; not required for a central result. |

## Changes on this branch

1. README updated to identify the active synthetic pipeline, current saved metrics, and distinction from legacy data.
2. Data-provenance register rewritten to document current generator, generated rating, legacy target and independent FastF1 path.
3. Backend metrics route changed to the authoritative final-evaluation artifact; feature-importance route changed to saved grouped-permutation output.
4. Silent inference fallback to a different model removed.
5. Added research protocol/code/tests and a canonical IEEE paper describing the current evidence honestly.

## Unresolved limitations

- No authenticated FastF1 experiment was executed in the audit runtime. FastF1 was not installed and DNS access to external package/GitHub hosts failed.
- The current synthetic benchmark cannot establish physical tire wear prediction, real driver aggression, real-track external validity or real-race accuracy.
- The proposed relative-pace target is a proxy derived from times and should be validated against an independent target if available.
- Current benchmark explainability is grouped permutation importance; real-data SHAP and explanation stability are uncomputed.
- Repository-wide pytest must be run on a connected checkout before merging this branch.
