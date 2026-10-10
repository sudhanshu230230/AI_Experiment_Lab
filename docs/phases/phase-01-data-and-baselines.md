# Phase 1 Report: Data, Preprocessing and Baselines

- **Dates:** 2026-10-10 to (in progress)
- **Git tag:** v0.1-phase1 (not yet created)
- **Status:** In progress

## Objective
Take the Telco churn dataset from raw CSV to a comparison of 2-3 baseline models, with
leakage-safe preprocessing and tests for each step.

## Progress by step

| Step | Component | Status |
|---|---|---|
| 4a | Ingestion (`data/ingestion.py`) | Done, tested |
| 4b | Profiling (`data/profiling.py`) | Done, tested |
| 5 | Validation (`data/validation.py`) | Not started |
| 6 | Preprocessing (`preprocessing/pipeline.py`) | Not started |
| 7 | Baseline training and evaluation | Not started |

## How to run it
Notebook `notebooks/01_data_exploration.ipynb` in Colab (profiling).
Unit tests: `pytest` (18 tests covering ingestion, profiling and config).

## Results
Dataset version (sha256) and profile findings: to be filled in after the first Colab run.

| Item | Value |
|---|---|
| Dataset file | |
| sha256 | |
| Rows / columns | |
| Churn fraction | |
| Blank `TotalCharges` | |

## Decisions made
- Ingestion loads data as stored and does not clean it. Cleaning belongs to validation and
  preprocessing so that raw data stays inspectable.
- The profiler reports blank strings separately from NaN and flags numeric-looking text
  columns, since the Telco dataset is known to have this problem (to be confirmed on the
  real file).
- Dataset version is the SHA-256 of the file bytes.

## Known limitations
- Profiling has only been tested on synthetic frames so far, not the real dataset.
- No datetime column handling yet.

## Next
Step 5: validation rules driven by `configs/default.yaml`.
