# ML Lifecycle

**Status:** v1.0. All 13 stages are implemented (Phases 1 to 8); the last column shows where.

| # | Stage | What happens | Output | Phase |
|---|---|---|---|---|
| 1 | Ingest | Load dataset, record source, hash and version | Raw dataframe, dataset metadata | 1 |
| 2 | Profile | Rows, dtypes, missing, duplicates, distributions, imbalance, correlations | Profile report | 1 |
| 3 | Validate | Target exists, types valid, missing and duplicate limits, schema, class balance | Accept / warn / reject | 1 |
| 4 | Split | Stratified train / test split with fixed seed | Train and test sets | 1 |
| 5 | Preprocess | Impute, scale numerics; impute, one-hot encode categoricals; fit on train only | Fitted preprocessor inside a Pipeline | 1 |
| 6 | Train baselines | Logistic Regression, Random Forest, XGBoost | Fitted pipelines, metrics | 1 |
| 7 | Track | Log params, metrics, artifacts, seed, dataset version, code version | MLflow runs | 2 |
| 8 | Tune | Optuna search with tracked trials | Best trial, search space | 3 |
| 9 | Select and register | Choose by task metric; register version; stage transitions | Registered model | 4 |
| 10 | Serve | API loads registered pipeline and predicts | Live endpoint | 5 |
| 11 | Monitor | Request, latency, error, input and prediction distributions | Metrics, dashboards | 6 |
| 12 | Detect drift | Compare production data with reference data | Drift report | 7 |
| 13 | Retrain and gate | Train candidate; compare with production; promote only if gates pass | New version or rejection | 8 |

## Principles per stage

- **Leakage:** stages 4 and 5 are ordered deliberately. Nothing target-aware or
  validation-aware is computed before the split.
- **Drift is a signal, not a verdict.** Drift triggers investigation and a retraining
  candidate. It never replaces the production model on its own (stage 13).
- **Reproducibility:** every run records seed, dataset version, code version, dependency
  versions and parameters.

## Retraining gate (as implemented, ADR-009)

A candidate is promoted only if ROC AUC on shifted data improves by at least 0.005 and ROC AUC on natural data drops by no more than 0.01. Both are measured on data neither model trained on. The latency gate in the original plan was not implemented: the model is a logistic regression and latency is not a differentiator here.
