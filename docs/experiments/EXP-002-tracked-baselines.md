# Experiment Report: Tracked Baselines (Reproduction of EXP-001)

- **ID:** EXP-002
- **Date:** pending Colab run
- **Phase:** 2
- **Git commit:** pending
- **Dataset:** `Telco-Customer-Churn.csv`, sha256 `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`

## Question
Do the same three baselines, now logged to MLflow, reproduce EXP-001 within 5e-4, and can the
best logged model be reloaded from its run id and score raw rows?

## Setup
Identical to EXP-001 (split, preprocessing, models, seed 42). Only the logging is new.

## Results
Pending. Paste the notebook 03 comparison table, the reproducibility line (REPRODUCED EXP-001 or DIFFERS) and the backup path here.

| Model | CV F1 | Test F1 | EXP-001 CV F1 | EXP-001 Test F1 | Match |
|---|---|---|---|---|---|
| Logistic Regression | pending | pending | 0.6279 | 0.6136 | |
| XGBoost | pending | pending | 0.6115 | 0.6119 | |
| Random Forest | pending | pending | 0.5395 | 0.5305 | |

## Conclusion
Pending.
