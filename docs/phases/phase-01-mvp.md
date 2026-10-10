# Phase 1 Report: Minimum Viable ML Platform

- **Dates:** 2026-10-10 to (in progress)
- **Git tag:** v0.1-phase1 (to be created)
- **Status:** In progress

## Objective
Dataset to trained and compared models: ingestion, profiling, validation, preprocessing, 2-3 baselines, evaluation.

## Progress

| Step | Module | Status |
|---|---|---|
| Ingestion | `data/ingestion.py` | Implemented, unit tested |
| Profiling | `data/profiling.py` | Implemented, unit tested |
| Validation | `data/validation.py` | Not started |
| Preprocessing | `preprocessing/pipeline.py` | Not started |
| Baseline training | `training/trainer.py`, `models/` | Not started |
| Evaluation | `evaluation/metrics.py` | Not started |

## How to run it
Ingestion and profiling: `notebooks/01_data_exploration.ipynb` in Colab.

## Results
To be filled from real runs. Dataset hash: _pending_. Rows: _pending_. Class split: _pending_.

## Tests
`tests/test_data.py`: 21 tests covering hashing, blank and numeric-text normalization, missing and duplicate counts (IDs ignored), outliers, imbalance ratio, correlations, JSON serializability and input errors.

## Decisions made
ADR-003 (dataset and metric). Ingestion is deliberately not target-aware, so it is safe to run before the train/test split.

## Known limitations
- Numeric-text columns that are really categorical codes would be converted to numeric (documented in `normalize_frame`).
- Not yet run on the real Telco file.

## Next
Validation rules (`data/validation.py`).
