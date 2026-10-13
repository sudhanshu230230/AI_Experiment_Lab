# ADR-004: Hyperparameter tuning method

- **Status:** Accepted
- **Date:** 2026-10-13
- **Phase:** 3

## Context
Phase 3 tunes the three baselines. The main risk is fooling ourselves: with enough trials,
the best CV score is partly luck, and any use of the test split during the search would make
the final estimate dishonest.

## Decision
- **Library:** Optuna, TPE sampler (seeded), Median pruner (5 startup trials, prune after 2 folds).
- **Objective:** mean 5-fold stratified CV F1 on the training split, with the same folds as the baselines (same seed), so baseline and tuned CV scores are directly comparable.
- **Test split:** scored once per model, after the search, on the refit best model. It never influences the search.
- **Warm start:** trial 0 is the baseline configuration where expressible (parameters not set in `model_configs.yaml` use the `default` in `tuning.yaml`). Random Forest `max_depth=None` is not expressible and is sampled.
- **Spaces in config:** `configs/tuning.yaml`, not code. 40 trials per model.
- **Logging:** one MLflow parent run per model (`stage=tuned`, best params, CV and test metrics, `trials.json`, model) with one nested run per trial (`stage=trial`). Trials are logged after the search ends, so an interrupted search logs nothing.
- **Fixed settings:** Logistic Regression is tuned too, so the comparison does not favour the models with more knobs.
- **Interpretation rule:** a tuned model is "better" only if the test gain is clearly larger than the fold spread (about 0.015 to 0.025 F1) and the CV gain agrees. Otherwise the models are treated as tied.

## Consequences
- Best CV score is optimistically biased; the reported evidence is test F1, with CV as supporting evidence.
- The study is in memory (not persisted); the MLflow trial runs are the record. Resuming an interrupted study is out of scope for now.
- With one 20% test split (1,409 rows, 374 churners), test F1 has its own noise (roughly ±0.02). Small gains cannot be distinguished from noise; repeated CV or a second holdout would be needed to claim them.
