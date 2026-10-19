# Testing Strategy

**Status:** Current (Phase 9). 188 tests; `ruff check .` and `pytest` run in CI on every push.

| Level | What is tested | File |
|---|---|---|
| Smoke | Package imports, version | `test_smoke.py` |
| Unit: data | Ingestion, profiling, each validation rule (pass and fail) | `test_data.py`, `test_validation.py` |
| Unit: preprocessing | No leakage (fit on train only), same output at train and inference, unseen categories, `TotalCharges` text | `test_preprocessing.py` |
| Unit: models, training, metrics | Metric correctness on known inputs, seed reproducibility, CV | `test_models_metrics.py`, `test_training.py` |
| Tracking | Runs log params, metrics and model; logged model reproduces predictions; XGBoost round-trip | `test_tracking.py` |
| Tuning | Search spaces, trial 0 equals the baseline, nested-run logging | `test_tuning.py` |
| Registry and export | Alias moves with a new version, rollback, lineage tag, export round-trip without MLflow, tampered metadata refused | `test_registry.py` |
| API | Valid and invalid payloads (422 for bad categories and numbers), `TotalCharges` handling, batch, health when the model fails to load, metrics, prediction log robustness | `test_serving.py` |
| Drift | PSI and KS properties, constant and all-missing columns, text-typed numbers, verdict rules, insufficient data, script end to end | `test_drift.py` |
| Retraining | Trigger rules, promotion gain and guard, F1 never promotes alone, candidate is an unfitted clone, export check | `test_retraining.py` |
| System | Docker image build and `smoke_test_api.py` against the running container | CI job `docker` |
| Post-deploy | Smoke test plus served-version check against the live service | `redeploy.yml` |

**Rules:** tests use small synthetic fixtures, never the full dataset (the dataset is not committed). Heavy compute is not tested in CI: tuning and registry tests use 3 trials and tiny data. Notebooks are not unit-tested; they were dry-run end to end on synthetic data before being run on real data.

**Not covered:** the Grafana dashboard rendering and the Prometheus alert firing (checked statically only: all PromQL expressions parse), the Evidently report, and the live Render service beyond the smoke test.

## Any-dataset support (ADR-010): `tests/test_automl.py` (25 tests)
- Schema inference: task from the target (binary, multi-class, regression), identifier detection (and that numbers-as-text are not mistaken for ids), booleans, dates, constants, missing values, no real rows in the schema, clear errors for unusable files.
- Training for each task on small synthetic data, scikit-learn's wine data (3 classes) and diabetes data (numeric target), including Optuna tuning and the selection direction (lower RMSE first).
- The API built from each exported model: valid and invalid requests (including a `NaN` body), blank and missing nullable fields, batch equals single, the prediction log, Prometheus labels, `/schema`, and the drift check script on logged traffic.
- The generated page: every column present, ids safe for odd column names, a hostile column name cannot close the script block.
- Retraining policy for multi-class and regression; the export check for XGBoost version mismatch.
- Not in CI: the browser run (headless Chromium against a live server for binary, multi-class and regression, desktop and phone width, both colour schemes) was done by hand when this was built.
