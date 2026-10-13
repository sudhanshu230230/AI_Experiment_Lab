# ADR-005: Champion model and registry workflow

- **Status:** Accepted
- **Date:** 2026-10-13
- **Phase:** 4

## Context
EXP-003 left three tuned models statistically tied (test F1 0.6195 to 0.6365, CV F1 0.631 to 0.638,
with the ranking differing between CV and test). Performance cannot pick the first model to serve, so
other criteria decide. Serving also cannot read the Drive-hosted MLflow store.

## Decision
**Champion: tuned Logistic Regression** (`C=0.037`, `solver=lbfgs`, `class_weight=balanced`, threshold 0.5).
- Tied on F1 and within about 0.02 on ROC-AUC and PR-AUC.
- Simplest, fastest to fit (retraining in Phase 8 takes seconds) and to serve, smallest artifact, and its coefficients are directly explainable.
- Lowest risk for the monitoring and retraining phases: fewer moving parts.
- Revisit when threshold tuning or repeated CV (EXP-003 next actions) separates the models. A better model can replace it by registering a new version and moving the alias.

**Registry workflow (MLflow Model Registry, aliases not stages):**
1. The registered model is `telco-churn-classifier`. Versions are created from a tuned run's logged model.
2. The alias `champion` points at the serving version. Promotion = move the alias; rollback = move it back. Old versions are kept.
3. Each version is tagged with `source_run_id`, so version -> run -> dataset hash, code version and parameters is traceable.
4. The run is chosen by CV F1 (never test F1), as in ADR-004.
5. Before export, the model loaded **by alias** must reproduce the run's logged test metrics (checked in notebook 05).

**Export for serving:** the champion is exported to `models/champion/` (`model.skops` + `metadata.json`) and committed to git (about tens of KB).
- The serving image then needs only scikit-learn, skops and this package, not MLflow or Drive.
- skops, not pickle, so loading does not run arbitrary code. Only types listed in `metadata.json` are trusted, and the loader refuses any type outside `ai_experiment_lab`, `sklearn`, `numpy`, `pandas`, `xgboost`.
- Larger future models would move to a release asset or object storage; `.gitignore` allows only `models/champion/*.skops`.

## Consequences
- Serving a model that is not in the registry is not possible by design: the export is always produced from an alias.
- The committed export must be refreshed whenever the alias moves (Phase 8 automates this).
- Three models were never compared on a second holdout; the champion choice is a trade-off decision, not a performance claim.
