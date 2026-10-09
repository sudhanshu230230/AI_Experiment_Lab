# ADR-002: Model and preprocessing handoff to the serving API

- **Status:** Accepted
- **Date:** 2026-10-10
- **Phase:** 1 (design), 5 (consumption)

## Context
The model is trained in Colab and served from a separate FastAPI service. If serving
applies different preprocessing from training, predictions silently become wrong
(training-serving skew).

## Options considered
1. **Single sklearn `Pipeline` (preprocessor + estimator) as one artifact.** One object,
   one `predict_proba` call; skew is structurally impossible. Requires matching library
   versions at load time.
2. **Separate preprocessor and model files.** Flexible, but two artifacts can be
   mismatched or updated independently.
3. **Reimplement preprocessing in the API.** Duplicated logic; high skew risk.

## Decision
Train and save one sklearn `Pipeline`. XGBoost and LightGBM models are used through their
scikit-learn wrappers so they fit inside the same Pipeline. The Pipeline is logged to
MLflow (Phase 2) and loaded by the API from the registry or a file (Phase 5).

## Consequences
- The API container pins the same scikit-learn, XGBoost and Python versions as training;
  record versions with every run.
- Input to the API must match the raw feature schema, so schemas are defined from the
  training data's columns.
- Feature engineering that needs the target is not allowed inside preprocessing.

## Interview note
I save preprocessing and model as one Pipeline so the exact transformations used in training
run at inference. That removes training-serving skew by design.
