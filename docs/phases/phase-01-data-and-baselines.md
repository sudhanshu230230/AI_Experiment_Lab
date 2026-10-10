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
| 4a | Ingestion (`data/ingestion.py`) | Done, unit tested |
| 4b | Profiling (`data/profiling.py`) | Done, unit tested; run on the real dataset |
| 5 | Validation (`data/validation.py`) | Done, unit tested on synthetic data; not yet run on the real file |
| 6 | Preprocessing (`preprocessing/pipeline.py`) | Done, unit tested; checked on synthetic data with the real 21-column schema |
| 7 | Baseline training and evaluation | Not started |

## How to run it
- Notebook `notebooks/01_data_exploration.ipynb` in Colab: load, profile, validate.
- Unit tests: `pytest` (55 tests covering ingestion, profiling, config, validation and preprocessing).

## Results
Profile of the real dataset, from the first Colab run (2026-10-10):

| Item | Value |
|---|---|
| Dataset file | `Telco-Customer-Churn.csv` |
| sha256 | `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a` |
| Rows / columns | 7,043 / 21 |
| Numeric / categorical features | 3 / 17 (the 17 include `customerID` and `TotalCharges`; target excluded) |
| Missing cells | 0.01% |
| Duplicate rows | 0 |
| Blank `TotalCharges` | 11 (stored as text; flagged as numeric-like) |
| Churn distribution | No 73.46%, Yes 26.54% (imbalance 2.77:1) |

Observation: `SeniorCitizen` is stored as 0/1 integers, so the profiler counts it as numeric,
but it is a binary flag. Harmless for scaling, but worth remembering when reading feature
importance.

## Decisions made
- Ingestion loads data as stored and does not clean it. Cleaning belongs to preprocessing so
  that raw data stays inspectable.
- The profiler reports blank strings separately from NaN and flags numeric-looking text
  columns. This was confirmed on the real file (`TotalCharges`).
- Dataset version is the SHA-256 of the file bytes.
- Validation returns PASS / WARN / FAIL per check. FAIL blocks training; WARN is carried
  forward to preprocessing (class imbalance, numeric-like text, identifiers).
- Class imbalance is a warning, not a failure, because it is handled later with stratified
  splits and class weights. A ratio above 1.5:1 warns; a minority share under 5% is marked
  severe.
- Validation thresholds come from `configs/default.yaml` (`validation:` section), not code.
- Datetime columns are rejected as unsupported in v1.
- The pipeline takes **raw feature columns** and contains every step that learns from data
  (coercion, imputation, scaling, encoding), so one object serves training and serving
  (ADR-002). The train/test split happens first; the pipeline is fitted on the training split.
- `TotalCharges` is converted with `NumericCoercer`: blanks become missing (then imputed with the
  training median); any other non-numeric value raises an error instead of silently becoming NaN.
- `customerID` and the target are removed in `split_data`, so the API will not need them.
- The target is encoded 1 = `Yes` (positive class from `dataset.positive_label`), 0 = `No`.
- One-hot encoding ignores categories unseen in training, so a new value at serving time
  produces all zeros for that feature instead of an error.
- Missing a required column at serving time raises an error; extra columns are ignored.

## Known limitations
- Validation has only been tested on synthetic frames; run it on the real file in Colab.
- Validation thresholds are defaults and have not been tuned on the real dataset.
- No datetime column handling.

## Next
Step 7: baseline models (Logistic Regression, Random Forest, XGBoost) on the preprocessed data, with
evaluation metrics, in `notebooks/02_baseline_models.ipynb`.
