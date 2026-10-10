# Experiment Report: Baseline Models on Telco Churn

- **ID:** EXP-001
- **Date:** (fill in after the Colab run)
- **Phase:** 1
- **Git commit:** (printed by the setup cell: "Code version")
- **Dataset:** `Telco-Customer-Churn.csv`, sha256 `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`

## Question
Which of three baseline models gives the best churn detection with default-ish settings,
and how much does each differ from a trivial predictor?

## Setup
- Split: 80/20 stratified, seed 42. Test set used once per model, never for selection.
- Preprocessing: `TotalCharges` coerced to numeric; numeric median-imputed and standardised;
  categorical most-frequent-imputed and one-hot encoded. Fitted on training data only.
- Models: Logistic Regression and Random Forest (`class_weight=balanced`), XGBoost
  (`scale_pos_weight` = negatives / positives in the training split). Settings in
  `configs/model_configs.yaml`.
- Selection score: mean 5-fold stratified cross-validated F1 on the training split.
- Class balance: 26.54% churn (2.77:1). A model that always predicts "No" scores
  about 73.5% accuracy and 0 F1, so accuracy is not used for selection.

## Results
Results pending: run `notebooks/02_baseline_models.ipynb` and paste the table here.

| Model | CV F1 | Test F1 | Test recall | Test precision | Test ROC-AUC | Test PR-AUC | Fit time (s) |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Conclusion
(To be written after the run: winner by CV F1, gap between CV and test scores, likely reasons.)

## Next actions
- Phase 2: log these runs in MLflow with parameters, metrics and artifacts.
- Phase 3: tune the best model family with Optuna.
