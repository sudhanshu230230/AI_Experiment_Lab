# Phase 1 Report: Data, Preprocessing and Baselines

- **Dates:** 2026-10-10 to 2026-10-10
- **Git tag:** v0.1-phase1 (create after pushing)
- **Status:** Complete

## Objective
Take the Telco churn dataset from raw CSV to a comparison of 2-3 baseline models, with
leakage-safe preprocessing and tests for each step.

## Progress by step

| Step | Component | Status |
|---|---|---|
| 4a | Ingestion (`data/ingestion.py`) | Done, unit tested |
| 4b | Profiling (`data/profiling.py`) | Done, unit tested; run on the real dataset |
| 5 | Validation (`data/validation.py`) | Done; ran on the real file (accepted, with warnings) |
| 6 | Preprocessing (`preprocessing/pipeline.py`) | Done, unit tested; checked on synthetic data with the real 21-column schema |
| 7 | Baseline training and evaluation | Done; run in Colab (commit `97bc105`), results in `docs/experiments/EXP-001-baselines.md` |

## How to run it
- Notebook `notebooks/01_data_exploration.ipynb` in Colab: load, profile, validate.
- Notebook `notebooks/02_baseline_models.ipynb` in Colab: split, train three baselines, compare.
- Unit tests: `pytest` (67 tests covering ingestion, profiling, config, validation, preprocessing, models, metrics and training).

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

**Validation on the real file:** accepted with warnings for class imbalance (26.5% minority, 2.77:1)
and `TotalCharges` stored as text. The Colab run used an older `validation.py` that also flagged
`TotalCharges` as identifier-like; this was fixed (numeric-like text is no longer treated as an ID).

**Baseline results** (full detail in `docs/experiments/EXP-001-baselines.md`):

| Model | CV F1 | Test F1 | Test recall | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.628 | 0.614 | 0.783 | 0.841 | 0.633 |
| XGBoost | 0.612 | 0.612 | 0.687 | 0.822 | 0.627 |
| Random Forest | 0.540 | 0.531 | 0.465 | 0.822 | 0.609 |

Logistic Regression and XGBoost are statistically tied (CV difference is smaller than the fold
spread); Random Forest is clearly behind. Accuracy would have ranked the models wrongly.

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

- Each baseline is one Pipeline (preprocessing + model). Models are compared by mean 5-fold
  stratified cross-validated F1 on the training split. The test split is scored once per model
  and is never used to choose between models, so the test numbers stay an honest estimate.
- F1 is the primary metric (ADR-003); recall, precision, ROC-AUC and PR-AUC are reported too.
  Accuracy is reported but not used, since "always No" would score about 73.5%.
- Imbalance is handled with `class_weight=balanced` (logistic regression, random forest) and
  `scale_pos_weight` computed from the training labels (XGBoost). The decision threshold stays 0.5
  for now; threshold tuning is a later experiment.

## Known limitations
- Validation thresholds are defaults and have not been tuned on the real dataset.
- No datetime column handling.
- Models use mostly default hyperparameters on purpose; tuning is Phase 3.

## Next
Phase 2: MLflow experiment tracking, starting with the ADR-001 storage spike (is a Drive-backed MLflow store stable?).
