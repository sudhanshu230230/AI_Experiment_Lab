# Changelog

Dated log of project changes, newest first. Entries describe what exists, not what is planned.







## 2026-10-17: Phase 7 drift detection (code)

- `monitoring/drift.py`: per-feature KS/chi-square + PSI, dataset verdict (`no_drift`, `drift`, `insufficient_data`), prediction drift reported separately. `monitoring/prediction_log.py` reads the API's JSONL log.
- Fixed three bugs found by the tests: constant reference column gave PSI 0; an all-missing column crashed chi-square; text-typed `TotalCharges` was treated as categorical.
- Added `configs/monitoring.yaml`, `scripts/check_drift.py` (`--fail-on-drift`), notebook 06, ADR-008 and the Phase 7 report; wrote the drift section of `docs/monitoring-and-drift.md`.
- 34 drift tests.

## 2026-10-16: Deployment target moved to Render

- First deployment verified: https://churn-api-xop3.onrender.com passed all 6 smoke checks. The bare URL gave 404, so `/` now redirects to `/docs`.
- Hugging Face Docker Spaces now require a paid plan, so the primary target is Render (free). Added `render.yaml`; `smoke_test_api.py` gained `--wait SECONDS` for cold starts; `docs/deployment.md` rewritten with the Render steps (Hugging Face kept as a paid alternative); ADR-006 amended; Phase 5 report updated.

## 2026-10-16: Phase 6 monitoring (code)

- Added `serving/metrics.py` and `/metrics`: request count and latency by route template and status, prediction counts by class, probability histogram, model loaded/info gauges. One registry per app.
- Added `monitoring/` (Prometheus config, five alert rules, Grafana datasource and dashboard provisioning, 12-panel "Churn API" dashboard) and `docker-compose.yml`.
- Added `scripts/load_generator.py` (normal, `--drift` and `--bad-fraction` traffic).
- Prediction-log write errors no longer fail a request. `prometheus-client` added to the `serve` extra; `data/predictions/` git-ignored.
- Added 6 tests (`tests/test_serving.py`). Total: 118 tests.
- Rewrote `docs/monitoring-and-drift.md` (monitoring part); added ADR-007 and the Phase 6 report (in progress).

## 2026-10-16: Phase 5 serving API, Docker, deployment (code)

- Added `serving/app.py` and `serving/schemas.py`: FastAPI service (`/health`, `/model`, `/predict`, `/predict/batch`), strict validation, startup schema check, optional JSONL prediction log.
- Added `Dockerfile`, `.dockerignore`, `scripts/smoke_test_api.py`, `scripts/deploy_hf_space.py` (Hugging Face Space deploy, with `--dry-run`).
- CI: added a `docker` job that builds and smoke-tests the image once `models/champion/model.skops` exists; test job installs the `serve` extra.
- Added `tests/test_serving.py` (22 tests). Total: 112 tests.
- Fixed: non-numeric `TotalCharges` text caused a 500 instead of a 422 (found by a test).
- Filled in `docs/api-reference.md` and `docs/deployment.md`; added ADR-006 and the Phase 5 report (in progress).

## 2026-10-13: Phase 4 model registry (code)

- Added `registry/mlflow_registry.py`: `find_run`, `register_model` (alias + lineage tag), `alias_version`, `load_by_alias`, `run_summary`.
- Added `registry/export.py`: `export_pipeline` / `load_exported` (skops file + `metadata.json`, trusted types restricted to known packages); usable without MLflow.
- Added `tests/test_registry.py` (5 tests). Total: 90 tests.
- Added `notebooks/05_model_registry.ipynb`, ADR-005 (champion = tuned Logistic Regression), Phase 4 report (in progress), `models/README.md`.
- `pyproject.toml` serve extra now includes `skops`; `.gitignore` allows `models/champion/*.skops`.

## 2026-10-13: Phase 3 Optuna tuning

- Added `tuning/tuner.py`: `tune_model` (CV-F1 objective on the training split, TPE sampler, median pruner, baseline as trial 0, refit, one test score) and `suggest_params`.
- Added `tracking/tuning_logger.py`: `log_tuning_result` (parent run per model, nested run per trial).
- Added `configs/tuning.yaml` (search spaces, 40 trials per model).
- Added `tests/test_tuning.py` (6 tests). Total: 85 tests.
- Added `notebooks/04_optuna_tuning.ipynb`; ADR-004; Phase 3 report (in progress); EXP-003 skeleton.
- Ran notebook 04 in Colab: tuned test F1 Random Forest 0.6365 (from 0.5305), XGBoost 0.6317 (from 0.6119), Logistic Regression 0.6195 (from 0.6136). The three are statistically tied. EXP-003 written; Phase 3 report marked complete.
- Phase 2 closed earlier: EXP-002 reproduced EXP-001 to four decimals.

## 2026-10-12: Phase 2 MLflow tracking

- Added `tracking/mlflow_logger.py`: `setup_tracking`, `log_baseline_results`, `load_pipeline`, `compare_runs`, `backup_tracking_db`, `get_code_version`, `library_versions`, `trusted_types` (skops trusted types for the custom `NumericCoercer`).
- Added `tests/test_tracking.py` (12 tests). Total: 79 tests.
- Added `notebooks/03_tracked_baselines.ipynb` with a reproducibility check against EXP-001 (tolerance 5e-4).
- `pyproject.toml` tracking extra: `mlflow>=3.0`, `skops`, `optuna`. CI now installs `.[dev,ml,tracking]`.
- Ran notebook 03 in Colab (commit `c6d18bd`): 3 runs and 3 models logged; EXP-002 reproduces EXP-001 to four decimals. Phase 2 report and EXP-002 completed.
- ADR-001 accepted earlier after the storage spike.

## 2026-10-12: ADR-001 accepted (MLflow storage)

- Ran the storage spike in Colab: SQLite database and artifacts on Drive work (0.26 s per run, 0 failures, no lock errors with two concurrent writers, data intact after a runtime restart). ADR-001 status changed to Accepted with the results.
- Fixed `scripts/mlflow_storage_spike.py`: unique batch labels per run, so repeat runs on a non-empty store are counted exactly (the first run reported FAIL for that reason, not because of data problems).
- Colab setup cells in all notebooks now use `sqlite:///.../mlflow/mlflow.db` instead of the deprecated `file:` store.
- New machine verified: Python 3.14.8 on Windows, 67 tests pass.

## 2026-10-11: Setup guide for a new machine

- Rewrote `docs/setup-guide.md` as a complete guide for a fresh Windows machine (tools check, Python, Git identity, clone, venv, dependency groups, VS Code, Colab/Drive, workflow, troubleshooting, macOS/Linux notes).
- Added `.vscode/settings.json` (pytest, Ruff, venv activation) and `.vscode/extensions.json` (recommended extensions); `.gitignore` now keeps both.

## 2026-10-10: Phase 2 started (ADR-001 spike)

- Checked MLflow docs: file store is legacy and being deprecated; SQLite database is the recommended default.
- Added `scripts/mlflow_storage_spike.py` (write, concurrent, verify) and `notebooks/spike_mlflow_storage.ipynb`.
- Rewrote ADR-001 with documentation findings, five options, explicit pass criteria and a results table (results pending the Colab run).

## 2026-10-10: Phase 1 complete

- Ran notebooks 01 and 02 in Colab on the real Telco dataset (commit `97bc105`).
- Recorded results in `docs/experiments/EXP-001-baselines.md`: Logistic Regression and XGBoost tied on CV F1 (0.628 vs 0.612), Random Forest behind (0.540).
- Phase 1 report marked Complete.

## 2026-10-10: Phase 1 baselines

- Added `models/classification.py` (`build_classifier`: logistic regression, random forest, XGBoost with auto `scale_pos_weight`).
- Added `evaluation/metrics.py` (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion counts).
- Added `training/trainer.py` (`run_baselines`: preprocessing + model as one Pipeline, CV-based selection, one-time test scoring).
- Added tests: `test_models_metrics.py`, `test_training.py`. Total: 67 tests.
- Added `notebooks/02_baseline_models.ipynb` and `docs/experiments/EXP-001-baselines.md` (results pending the Colab run).

## 2026-10-10: Phase 1 preprocessing

- Added `preprocessing/pipeline.py`: `split_data` (stratified, seeded, removes id and target), `encode_target`, `NumericCoercer`, `infer_feature_columns`, `build_preprocessing_pipeline` (impute, scale, one-hot inside one sklearn Pipeline).
- Added `preprocessing:` and `dataset.positive_label` / `dataset.file` to `configs/default.yaml`.
- Added `tests/test_preprocessing.py` (16 tests, including leakage, unseen-category and serving-consistency tests). Total: 55 tests.
- Recorded the real dataset hash (`88be4b93...`) in the Phase 1 report.

## 2026-10-10: Phase 1 validation

- Added `data/validation.py`: eleven checks (row count, target exists and valid, supported dtypes, missing values including blank strings, duplicates, class balance, numeric-like text, identifier columns, constant columns, optional expected schema) returning PASS / WARN / FAIL, with `report()` and `raise_if_rejected()`.
- Thresholds come from `configs/default.yaml` (`validation:` section, including `min_rows`).
- Added `tests/test_validation.py` (21 tests). Total: 39 tests.
- Added validation cell to `notebooks/01_data_exploration.ipynb`.
- Recorded the real Telco profile in the Phase 1 report; ADR-003 data issues marked verified.

## 2026-10-10: Phase 1 started, ingestion and profiling

- Added `data/ingestion.py`: CSV loading (no cleaning), SHA-256 dataset version, `DatasetInfo`.
- Added `data/profiling.py`: `profile_dataframe` (shape, types, NaN vs blank, duplicates, outliers, categorical summaries, target imbalance, correlations, numeric-like text detection) and `format_report`.
- Added `utils/config.py`: YAML loading and data directory resolution.
- Added `tests/test_data.py` (18 tests on synthetic data) and `notebooks/01_data_exploration.ipynb`.
- Added `docs/phases/phase-01-data-and-baselines.md` (in progress).

## 2026-10-10: Phase 0 documentation and Colab bootstrap

- Added documentation set: charter, architecture, ML lifecycle, setup guide, testing strategy.
- Added ADR-001 (MLflow storage, proposed), ADR-002 (model handoff), ADR-003 (initial dataset and selection metric).
- Added ADR, phase-report and experiment-report templates.
- Added `notebooks/00_colab_setup.ipynb` (Drive mount, clone, install, tracking URI).
- Added `LICENSE` (MIT), `.gitignore`, `pyproject.toml` optional dependency groups per phase.
- Added `docs/phases/phase-00-skeleton.md`.
- Fixed Colab setup cell: add `src/` to `sys.path`, because a running kernel does not load the editable-install `.pth` file (found on first Colab run).

## Earlier (Phase 0 skeleton)

- Created package layout under `src/ai_experiment_lab/`, smoke test, and basic lint/test CI workflow.
