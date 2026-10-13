# Experiment Report: Optuna Tuning of the Baselines

- **ID:** EXP-003
- **Date:** pending Colab run
- **Phase:** 3
- **Git commit:** pending
- **Dataset:** `Telco-Customer-Churn.csv`, sha256 `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`

## Question
Does tuning give Logistic Regression, XGBoost or Random Forest a gain on the held-out test
set that is larger than the noise (fold spread about 0.015 to 0.025 F1)? Does either tree
model overtake Logistic Regression?

## Setup
Method and rules in ADR-004. 40 trials per model, 5-fold CV F1 on the training split,
search spaces in `configs/tuning.yaml`. Baselines are the same as EXP-001/002.

## Results
Pending. Paste the notebook 04 outputs here.

| Model | Base CV F1 | Tuned CV F1 | Base test F1 | Tuned test F1 | Test gain |
|---|---|---|---|---|---|
| Logistic Regression | 0.6279 | pending | 0.6136 | pending | |
| XGBoost | 0.6115 | pending | 0.6119 | pending | |
| Random Forest | 0.5395 | pending | 0.5305 | pending | |

## Conclusion
Pending.
