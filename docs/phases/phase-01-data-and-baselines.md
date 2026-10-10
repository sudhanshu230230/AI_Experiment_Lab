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
| 5 | Validation (`data/validation.py`) | Done, tested on synthetic data |
| 6 | Preprocessing (`preprocessing/pipeline.py`) | Not started |
| 7 | Baseline training and evaluation | Not started |

## How to run it
Notebook `notebooks/01_data_exploration.ipynb` in Colab (profiling).
Unit tests: `pytest` (36 tests covering ingestion, profiling, validation and config).

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

- Validation outcome per check is pass / warn / fail. FAIL blocks training; WARN is carried
  forward to be handled in preprocessing (class imbalance, numeric-like text, identifiers).
- Class imbalance is a warning, not a failure, because it is handled later with stratified
  splits and class weights. A ratio of 2:1 or more triggers the warning; below 5% minority
  share the message is marked severe.

## Known limitations
- Profiling has only been tested on synthetic frames so far, not the real dataset.
- No datetime column handling yet; datetime columns are rejected by validation.
- Validation thresholds are defaults and have not been tuned on the real dataset.

## Next
Step 6: preprocessing pipeline (impute, scale, encode; fit on train only).
