# Experiment Report: Optuna Tuning of the Baselines

- **ID:** EXP-003
- **Date:** 2026-10-13
- **Phase:** 3
- **Git commit:** not recorded in the pasted output (code from commit `2c568f9` or later; the run's `code.version` tag in MLflow has the exact value)
- **Dataset:** `Telco-Customer-Churn.csv`, 7,043 rows, sha256 `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`

## Question
Does tuning give Logistic Regression, XGBoost or Random Forest a gain on the held-out test
set that is larger than the noise (fold spread about 0.015 to 0.025 F1)? Does either tree
model overtake Logistic Regression?

## Setup
Method and rules in ADR-004: 40 trials per model, TPE sampler, median pruner, objective = mean
5-fold stratified CV F1 on the training split (same folds as the baselines), baseline as trial 0
where expressible, test split scored once after the search. Spaces in `configs/tuning.yaml`.

## Results

| Model | Base CV F1 | Tuned CV F1 (± std) | CV gain | Base test F1 | Tuned test F1 | Test gain |
|---|---|---|---|---|---|---|
| XGBoost | 0.6115 | **0.6384** (± 0.0137) | +0.0269 | 0.6119 | 0.6317 | +0.0198 |
| Random Forest | 0.5395 | 0.6375 (± 0.0160) | +0.0980 | 0.5305 | **0.6365** | +0.1060 |
| Logistic Regression | 0.6279 | 0.6307 (± 0.0198) | +0.0027 | 0.6136 | 0.6195 | +0.0058 |

Other tuned test metrics:

| Model | Recall | Precision | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| XGBoost | 0.7888 | 0.5268 | 0.8448 | 0.6564 |
| Random Forest | 0.7888 | 0.5335 | 0.8439 | 0.6509 |
| Logistic Regression | 0.7834 | 0.5122 | 0.8399 | 0.6317 |

Best parameters (search time: LR 32 s, RF 272 s, XGBoost 135 s):

| Model | Best parameters |
|---|---|
| Logistic Regression | `C=0.0370`, `solver=lbfgs` (baseline `C=1`) |
| Random Forest | `n_estimators=192`, `max_depth=11`, `min_samples_leaf=12`, `max_features=log2` (baseline: unlimited depth, leaf 1, sqrt) |
| XGBoost | `n_estimators=495`, `learning_rate=0.01066`, `max_depth=5`, `min_child_weight=1.906`, `subsample=0.684`, `colsample_bytree=0.938`, `reg_lambda=2.270` (baseline: 300 trees, lr 0.1, depth 6) |

Search shape (top 5 trials per model):
- Logistic Regression: CV F1 0.6299 to 0.6307 for `C` between 0.031 and 0.108. A flat plateau.
- Random Forest: CV F1 0.6345 to 0.6375; all top 5 use `max_features=log2`, depth 11 to 17, leaf size 9 to 12.
- XGBoost: CV F1 0.6363 to 0.6384; all top 5 use learning rate 0.011 to 0.017, depth 5 to 6, subsample 0.61 to 0.70, `min_child_weight` 1.5 to 2.8.

Not recorded in the pasted output: number of pruned trials, the stored-runs table and the backup path.

## Conclusion
1. **Random Forest gained a lot (+0.106 test F1); XGBoost a little (+0.020); Logistic Regression nothing (+0.006).** Only the Random Forest gain is clearly larger than the noise. The XGBoost gain is about twice the fold spread in CV but only the size of the test-set noise (about ±0.02), so it is likely but not proven.
2. **The three tuned models are tied.** Tuned CV F1 spans 0.631 to 0.638 and test F1 spans 0.620 to 0.637, and the order differs between CV (XGBoost, Random Forest, Logistic Regression) and test (Random Forest, XGBoost, Logistic Regression). This evidence does not support naming a winner. The trees lead slightly on ROC-AUC (0.844 vs 0.840) and PR-AUC (0.651 to 0.656 vs 0.632), but the gap is small.
3. **The EXP-001 hypothesis that default Random Forest was overfitting is supported.** The tuned forest is much more regularised (depth 11 instead of unlimited, leaves of 12 instead of 1, `log2` features) and its recall at threshold 0.5 rose from 0.465 to 0.789. XGBoost also moved to a slow, strongly regularised setting (learning rate 0.011, subsample 0.68, `reg_lambda` 2.3). Logistic Regression preferred stronger regularisation too (`C=0.037`), but it barely mattered.
4. **Little sign the search overfit the CV folds.** Tuned CV and test F1 are close (differences of 0.007, 0.001 and 0.011).
5. **All three models now sit at nearly the same operating point** (recall about 0.78 to 0.79, precision 0.51 to 0.53 at threshold 0.5), so the remaining lever is probably the decision threshold, not the model.

## Limitations
- One 80/20 split: test F1 has noise of roughly ±0.02, so differences of that size cannot be called real.
- Several best values sit near the edge of the search space (XGBoost learning rate 0.0107 vs lower bound 0.01, subsample near the lower bound 0.6, `colsample_bytree` near 1.0). The top-5 plateau is flat (0.6363 to 0.6384), so widening the space would likely add very little, but this was not tested.
- The Random Forest baseline (`max_depth=None`) could not be placed in the space, so its trial 0 was not exactly the baseline.
- Trials were logged after the search, so a disconnect mid-search would have lost the study.

## Next actions
- Phase 4: choose which model to register as the first champion. The evidence says tied, so the choice should rest on other criteria (simplicity, speed, explainability, serving cost) and be recorded in an ADR.
- Experiment: tune the decision threshold on training-fold predictions and compare against the 0.5 operating point.
- Experiment: repeated stratified CV (for example 5×3) on the three tuned configurations to see if any ordering survives.
- Write down the churn cost assumption (missed churner vs wasted retention offer), which decides the threshold and the final metric.
