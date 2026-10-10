# Experiment Report: Baseline Models on Telco Churn

- **ID:** EXP-001
- **Date:** 2026-10-10
- **Phase:** 1
- **Git commit:** `97bc105` (Colab "Code version")
- **Dataset:** `Telco-Customer-Churn.csv`, 7,043 rows, sha256 `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`

## Question
Which of three baseline models gives the best churn detection with near-default settings,
and does the winner clearly beat the others?

## Setup
- Split: 80/20 stratified, seed 42. Train 5,634 rows, test 1,409 rows; churn rate 26.54% in both.
- Preprocessing: `TotalCharges` coerced to numeric; numeric median-imputed and standardised;
  categorical most-frequent-imputed and one-hot encoded. Fitted on training data only.
- Models: Logistic Regression and Random Forest (`class_weight=balanced`), XGBoost
  (`scale_pos_weight` = negatives / positives in the training split). Settings in
  `configs/model_configs.yaml`. Decision threshold 0.5.
- Selection score: mean 5-fold stratified cross-validated F1 on the training split.
  The test split was scored once per model and not used for selection.
- Reference point: always predicting "No" gives 73.46% accuracy and 0 recall.

## Results

| Model | CV F1 (± std) | Test F1 | Test recall | Test precision | Test ROC-AUC | Test PR-AUC | Test accuracy | Fit time (s) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | **0.628** (± 0.023) | **0.614** | **0.783** | 0.504 | **0.841** | **0.633** | 0.738 | 0.12 |
| XGBoost | 0.612 (± 0.015) | 0.612 | 0.687 | 0.552 | 0.822 | 0.627 | 0.769 | 1.98 |
| Random Forest | 0.540 (± 0.024) | 0.531 | 0.465 | 0.617 | 0.822 | 0.609 | **0.781** | 2.25 |

Confusion counts on the test set (374 actual churners, 1,035 non-churners):

| Model | True neg. | False pos. | False neg. | True pos. | Churners caught |
|---|---|---|---|---|---|
| Logistic Regression | 747 | 288 | 81 | 293 | 293 / 374 (78.3%) |
| XGBoost | 826 | 209 | 117 | 257 | 257 / 374 (68.7%) |
| Random Forest | 927 | 108 | 200 | 174 | 174 / 374 (46.5%) |

## Conclusion

1. **Logistic Regression and XGBoost are effectively tied; Random Forest is clearly behind.**
   Logistic Regression leads XGBoost on CV F1 by 0.016, which is smaller than one standard
   deviation of the fold scores (0.015 to 0.023), so the ranking between those two is not
   reliable on this evidence. Random Forest is about 0.09 below them, well outside that noise.
2. **Accuracy would have picked the wrong model.** Random Forest has the highest accuracy
   (78.1%) but the lowest recall: it misses more than half of the churners. Logistic Regression
   has accuracy of 73.8%, barely above the 73.5% "always No" baseline, yet it catches 78% of churners.
   This confirms the decision in ADR-003 not to select on accuracy.
3. **No sign of leakage or overfitting to the selection process.** CV and test F1 agree within
   about 0.01 for every model, which is within the fold-to-fold spread.
4. **The models trade recall against precision differently.** Logistic Regression flags the most
   customers (581) and has the lowest precision (50.4%); Random Forest flags the fewest (282) with
   the highest precision (61.7%). Which is better depends on business costs, which are not yet
   defined (see next actions).

Likely explanations (hypotheses, not yet tested):
- Churn on this dataset appears to depend mostly on a few roughly additive factors (contract type,
  tenure, charges), which suits a linear model, and 5,634 training rows is small for tree ensembles.
- Random Forest and XGBoost were run with mostly default settings, so they may be overfitting;
  Random Forest's low recall suggests its probabilities are conservative at the 0.5 threshold.

## Next actions
- Phase 2: log these runs in MLflow (parameters, metrics, artifacts, dataset hash).
- Phase 3: tune XGBoost and Random Forest with Optuna and check whether either overtakes Logistic Regression.
- Experiment: tune the decision threshold (e.g. maximise F1 or set a target recall) on validation data, since 0.5 is an arbitrary choice.
- Write down a cost assumption (cost of a missed churner vs a wasted retention offer) to justify the final metric and threshold.
- Consider repeated cross-validation to tighten the comparison between Logistic Regression and XGBoost.
