# ADR-010: Any tabular dataset, with the schema stored in the model

- **Status:** Accepted
- **Date:** 2026-10-19
- **Phase:** Post-v1.0 extension

## Context
v1.0 worked for one dataset. The 19 Telco columns were written by hand in `serving/schemas.py`, the form labels in `serving/ui.py`, the response said `churn_probability`, and training, drift and retraining assumed a binary target. Using the platform on another CSV (for example a churn file with different columns, a multi-class label or a price to predict) meant editing code in several places.

Constraints: keep the Telco demo and its 163 tests working unchanged; keep the Colab (train) and Render (serve) split; no new runtime dependency; the free serving tier has about 0.5 GB of memory, so training stays in Colab.

## Options considered
1. **Edit the Telco code for each new dataset.** Simple, but not reusable and easy to get wrong.
2. **A config file per dataset (columns, types, allowed values) that the serving code reads.** Reusable, but a second source of truth that can disagree with the trained model.
3. **Store the schema inside the model export and build the API and page from it.** One source of truth, produced at training time from the same data the model saw.
4. **Train inside the web app on uploaded files.** Most convenient, but unsuited to the free tier (memory, time, no persistent disk) and widens the attack surface.

## Decision
Option 3. `automl.train_dataset` infers the task (binary, multi-class or regression), cleans the columns and writes `metadata.json` with the task, target, classes and a per-column schema (type, nullable, range, allowed values, a synthetic typical example). The API reads only that metadata at start-up: it builds the pydantic request model dynamically (column names are aliases, so "Usage Frequency" works), chooses the response shape for the task, and renders the web form from the same description. An export without a schema (the Telco champion) falls back to the original hand-written models, so nothing about the live demo changes.

Task-specific choices, all in `automl/tasks.py`:

| | binary | multiclass | regression |
|---|---|---|---|
| models | logistic regression, random forest, XGBoost | same | ridge, random forest, XGBoost |
| selected by (CV, training split) | F1 | macro F1 | RMSE |
| retraining promotion metric (ADR-009) | ROC AUC | balanced accuracy | R^2 |
| API output | `prediction`, `probability`, `threshold` | `prediction`, `probabilities` | `prediction` |

Other rules: identifier columns are detected (nearly-unique non-float columns, or an id-like name) and left out with a note; numbers stored as text are kept numeric (blank = missing) so they are not mistaken for ids; dates and constant or empty columns are dropped with a note; no real data row is stored in the schema.

## Consequences
- Deploying another dataset means committing a different `models/champion/` (plus the library versions in the Dockerfile `ARG` lines); no code change.
- The API's 422 handler no longer echoes rejected values. This also fixed a latent bug: a JSON body containing `NaN` caused a 500, because FastAPI's default error response could not serialise it.
- The serving image must install the same scikit-learn, and the same XGBoost when the winner is XGBoost. `scripts/check_model_export.py` now checks both.
- Limits: tabular CSV only; categorical columns with more than 50 distinct values become free text (no dropdown, and unseen values are accepted); only three model families; Optuna tuning reuses the Telco search spaces; the monitoring dashboard keeps its Telco metric names (`ael_churn_probability` is only filled for binary models).
- Not solved: whether a model is good for a given business problem. The notebook and docs say how to spot target leakage, and that near-perfect scores on real data are a warning.

## Interview note
"The model carries its own schema, so the API, the form and the CSV template are generated from the model instead of hand-written. That made the platform dataset-agnostic without adding a second place for the column list to go stale, and the old Telco export still works through a fallback path."
