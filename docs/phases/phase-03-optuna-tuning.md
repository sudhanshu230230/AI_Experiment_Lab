# Phase 3 Report: Hyperparameter Tuning with Optuna

- **Dates:** 2026-10-13 to in progress
- **Git tag:** v0.3-phase3 (create after the Colab run is confirmed)
- **Status:** In progress (code and tests done; Colab run of notebook 04 pending)

## Objective
Find out whether tuning XGBoost and Random Forest (and Logistic Regression) beats the
baselines of EXP-001/002 by more than noise, with every trial tracked in MLflow.

## Progress by step

| Step | Component | Status |
|---|---|---|
| 1 | Method decision (ADR-004) | Done |
| 2 | Search spaces (`configs/tuning.yaml`) | Done |
| 3 | Tuner (`tuning/tuner.py`): CV objective, pruning, warm start, refit, one test score | Done; 6 unit tests |
| 4 | MLflow logging (`tracking/tuning_logger.py`): parent + nested trial runs | Done; tested |
| 5 | Notebook `04_optuna_tuning.ipynb` | Written; ran locally on synthetic data; Colab run pending |
| 6 | EXP-003 report | Skeleton; results pending |

## How to run it
- Colab: `notebooks/04_optuna_tuning.ipynb` (about 10 to 25 minutes; 40 trials per model).
- Tests: `pip install -e ".[dev,ml,tracking]"` then `pytest` (85 tests).

## Tests of note
- Trial 0 of Logistic Regression and of XGBoost has exactly the same CV F1 as `run_baselines`, so the objective and folds are consistent.
- Tuning is deterministic for a fixed seed.
- The nested trial runs are hidden from `compare_runs` unless `include_trials=True`.

## Decisions made
See ADR-004.

## Known limitations
- Study not persisted; an interrupted search must be rerun.
- Single train/test split; small gains are within noise (ADR-004).
- Threshold tuning and cost assumptions are still open (see EXP-001 next actions).

## Next
Confirm EXP-003 results, then Phase 4 (model registry: promote the chosen model by alias).
