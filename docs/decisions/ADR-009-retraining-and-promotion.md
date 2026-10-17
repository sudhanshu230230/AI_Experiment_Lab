# ADR-009: Retraining trigger and promotion rule

- **Status:** Accepted
- **Date:** 2026-10-17
- **Phase:** 8

## Context
Phase 7 gives a drift verdict. We need a repeatable way to turn "the inputs moved" into "a better model is serving", without letting a retrain silently make things worse, within the agreed scope (Colab for ML, one free host, manual trigger).

## Decision
- **Trigger:** retrain on a `drift` verdict (or when forced). Never on `insufficient_data`, never on `no_drift`. Drift is a *prompt to look*, not proof the model is worse (Phase 7: F1 can rise while ROC AUC falls), so the trigger starts a candidate and the promotion rule decides.
- **Candidate:** an unfitted `sklearn.base.clone` of the champion pipeline (same preprocessing, same hyperparameters) refit on the original training data plus newly labelled rows. This is retraining, not a new search; a new search is a new experiment (Phase 3).
- **Promotion rule (all on data neither model trained on):** promote only if ROC AUC on *shifted* data improves by at least 0.005 **and** ROC AUC on *natural* data drops by no more than 0.01. ROC AUC, not F1, because F1 moves with the churn rate. The natural-data guard stops a model that learned the new population by forgetting the old one.
- **Always log, register only on promotion.** The candidate run (metrics, decision, model) is logged to MLflow either way, so a rejected retrain leaves a record. Registering creates a new version and moves the `champion` alias; the old version stays, so rollback is moving the alias back (ADR-005).
- **Deploy is a separate, manual step.** The notebook produces `champion.zip`; a human commits it as `models/champion/`; the `Redeploy` GitHub Actions workflow (manual trigger) runs lint and tests, checks the export (`scripts/check_model_export.py`: loadable, metadata complete, scikit-learn version equals the Dockerfile's), calls the Render deploy hook, then polls `/model` until the deployed `registry_version` equals the committed one and runs the smoke test with `--expect-version`.
- **Simulated new data.** There are no live labelled customers, so the held-out test set stands in: split 50/50 *before* resampling, each half shifted separately, so no row is on both sides. The original test set is therefore consumed by this step.

## Alternatives considered
- Automatic retrain-and-deploy on every drift verdict: too risky without a human check, and labels for the new data do not exist in this project.
- Promote on any improvement: noise-sized gains would churn versions. A margin plus a guard is cheap protection.
- Promote on F1: rejected by the Phase 7 result.
- Full hyperparameter search on every retrain: costs Colab time and answers a different question.

## Consequences
- The margins (0.005, 0.01) are judgement calls, not tuned; they are constants in `retraining/policy.py` and covered by tests.
- Evaluation sets are small (about 700 rows each), so ROC AUC differences of 0.005 are within sampling noise. With so little data the rule will often say "keep the champion", which is the safe failure.
- Retraining happens in Colab by hand; the GitHub workflow only redeploys. Fully automated retraining is out of scope and said so.
- The Render deploy hook URL is a secret; anyone holding it can trigger a deploy. It lives in GitHub Actions secrets only.
- Free Render sleeps, so the first poll may wait for a cold start; the workflow allows about 15 minutes.
