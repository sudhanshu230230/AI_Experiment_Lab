# Experiment Report: Tracked Baselines (Reproduction of EXP-001)

- **ID:** EXP-002
- **Date:** 2026-10-12
- **Phase:** 2
- **Git commit:** `c6d18bd` (Colab "code.version", clean tree)
- **Dataset:** `Telco-Customer-Churn.csv`, 7,043 rows, sha256 `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`
- **Environment (Colab):** Python 3.13.16, scikit-learn 1.6.1, pandas 2.2.3, numpy 2.1.3, xgboost 3.4.1, mlflow 3.17.0

## Question
Do the same three baselines, now logged to MLflow, reproduce EXP-001 within 5e-4, and are the
logged models stored and retrievable?

## Setup
Identical to EXP-001 (split, preprocessing, models, seed 42). Only the logging is new.
Results were read directly from the downloaded MLflow database
(`mlflow-20261012-025651.db`, a backup made by `backup_tracking_db`).

## Results

| Model | CV F1 (± std) | Test F1 | EXP-001 CV F1 | EXP-001 Test F1 | Match (≤ 5e-4) |
|---|---|---|---|---|---|
| Logistic Regression | 0.6279 (± 0.0232) | 0.6136 | 0.6279 | 0.6136 | Yes |
| XGBoost | 0.6115 (± 0.0145) | 0.6119 | 0.6115 | 0.6119 | Yes |
| Random Forest | 0.5395 (± 0.0241) | 0.5305 | 0.5395 | 0.5305 | Yes |

Test confusion counts also match EXP-001 exactly (LR 747/288/81/293, XGB 826/209/117/257, RF 927/108/200/174).

Store contents: experiment `telco-churn` with 3 runs, all FINISHED; 3 logged models, all READY,
with artifacts under the Drive artifact folder. Each run carries dataset hash, code version,
library versions and `experiment.id = EXP-002` tags.

## Conclusion
1. **Reproduced.** All CV and test metrics match EXP-001 to four decimals, so the pipeline is deterministic given the seed, and logging does not change results.
2. The EXP-001 conclusions stand: LR and XGBoost are tied, Random Forest is behind.
3. Reproduction held across a different Python/library set than the original run was recorded under; the library versions are now stored per run, so any future drift is traceable.

## Next actions
- Phase 3: Optuna tuning logged as nested MLflow runs.
- Decide how to browse runs in the MLflow UI (see Phase 2 report).
- Note: the `Default` experiment (id 0) points at `/content/mlruns/0`, a local Colab path. It is empty and harmless.
