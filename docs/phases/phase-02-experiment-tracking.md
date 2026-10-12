# Phase 2 Report: Experiment Tracking with MLflow

- **Dates:** 2026-10-11 to in progress
- **Git tag:** v0.2-phase2 (create after the Colab run is confirmed)
- **Status:** In progress (code done and tested locally; Colab run of notebook 03 pending)

## Objective
Record every training run (parameters, metrics, dataset hash, code version, model) in MLflow so
any result can be compared, reproduced and loaded back later.

## Progress by step

| Step | Component | Status |
|---|---|---|
| 1 | Storage decision (ADR-001): SQLite + Drive artifacts | Done; spike passed, ADR Accepted |
| 2 | Tracking module (`tracking/mlflow_logger.py`) | Done; 12 unit tests |
| 3 | Notebook `03_tracked_baselines.ipynb` | Written; ran locally on synthetic data; Colab run pending |
| 4 | Reproducibility check against EXP-001 | Built into notebook 03; pending real run |
| 5 | Backup helper (`backup_tracking_db`) | Done; keeps the newest 3 copies |
| 6 | Viewing the runs (MLflow UI) | Open question, see below |

## How to run it
- Colab: run `notebooks/03_tracked_baselines.ipynb` after `00_colab_setup.ipynb` settings.
- Local tests: `pip install -e ".[dev,ml,tracking]"` then `pytest` (79 tests).

## What is logged per run
- Params: `model.*`, seed, `split.*`, `cv.folds`, `selection.primary_metric`, `data.*`, `pre.*`.
- Tags: dataset name, sha256, rows; `code.version` (git short hash, `-dirty` if uncommitted); stage; library versions; experiment id.
- Metrics: CV and test metrics, `cv_f1_std`, confusion counts.
- Artifacts: `metrics.json`, `confusion.json`, `dataset.json`, `config.json`, and the model (full Pipeline, with signature and input example).

## Decisions made
- ADR-001: SQLite database plus file artifacts on Google Drive; one training session writes at a time.
- MLflow 3 saves sklearn models in skops format; custom classes (`NumericCoercer`) must be listed as trusted types, computed by `trusted_types(pipeline)`.
- `tracking_uri: null` in config means use `MLFLOW_TRACKING_URI`, set by the Colab setup cell.
- The model logged is the same Pipeline that is evaluated (ADR-002), so serving loads exactly what was scored.

## Acceptance criterion
Notebook 03 metrics must match EXP-001 within 5e-4 (fixed seed). A mismatch means something
in the pipeline is non-deterministic and must be understood before Phase 3.

## Known limitations
- Two Colab sessions writing at the same time, and very large model artifacts, were not tested.
- The MLflow signature may warn about integer columns that could contain missing values; harmless here.
- Viewing: the notebook compares runs as a DataFrame. Using the MLflow UI locally needs the downloaded DB and matching artifact paths; undecided.

## Next
Confirm the Colab results, then Phase 3 (Optuna tuning logged as nested MLflow runs), then Phase 4 (model registry).
