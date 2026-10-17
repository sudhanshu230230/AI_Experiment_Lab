# ADR-008: Drift detection design

- **Status:** Accepted (thresholds to be confirmed by the notebook 06 sensitivity table)
- **Date:** 2026-10-17
- **Phase:** 7

## Context
After deployment the model sees data that may differ from its training data. We need a repeatable check that says "the inputs have moved", whose verdict can later trigger retraining (Phase 8). Constraints: the development laptop runs Python 3.14, where only an old Evidently (0.4.19) installs; the hosted API must not depend on heavy libraries; production labels (did the customer actually churn?) arrive weeks late.

## Decision
- **Reference and current windows.** Reference = the training split (X_train, rebuilt with the same seed). Current = logged production inputs (`PREDICTION_LOG`), optionally filtered to one model version.
- **Per-feature test plus effect size.** Numeric features: two-sample KS test and PSI (bins from reference deciles, open-ended outer bins). Categorical: chi-square on category counts and PSI. Numbers stored as text (`TotalCharges`) are treated as numeric. Numeric columns with 10 or fewer distinct values are treated as categorical.
- **A feature is "drifted" only if PSI >= 0.2 AND p < 0.05.** With thousands of rows the test alone flags harmless differences; PSI alone ignores sample size. Requiring both reduces false alarms.
- **Dataset verdict:** `drift` when >= 30% of features drifted, `no_drift` otherwise, and `insufficient_data` when fewer than 100 current rows exist (small windows must never trigger retraining).
- **Prediction drift** (predicted probability, KS + PSI) is reported but does **not** change the verdict. It is a symptom, and the model's own output can move for harmless reasons.
- **Own module for the verdict, Evidently for the picture.** `ai_experiment_lab.monitoring.drift` depends only on numpy, pandas and scipy, so it runs on the laptop, in CI and in a retraining job, and is unit-tested (34 tests). Evidently 0.7 is run in Colab (notebook 06) to produce a readable HTML report; our verdict is cross-checked against it, not replaced by it.
- **Thresholds live in `configs/monitoring.yaml`**, not in code.
- **Drift is not accuracy loss.** The report says "inputs moved". Whether the model degraded needs labels; notebook 06 section 5 shows the two on simulated data.

## Alternatives considered
- Use Evidently for everything: blocked on the laptop's Python 3.14 and would add a heavy dependency to the retraining workflow.
- KS p-value alone: fires on tiny, harmless shifts at large n.
- Page on prediction drift: noisy; kept as supporting evidence.

## Consequences
- Thresholds (PSI 0.2, 30% share) are conventional defaults, not tuned to this data. Notebook 06 shows the verdict at increasing shift strengths so they can be checked rather than assumed.
- Features are tested one at a time, so a shift in feature *combinations* is invisible. Accepted for now.
- 20 features at a 30% share means 6 drifted features are needed; a real change that only touches a few columns will not trigger. That is a deliberate precision/recall choice, revisited if the sensitivity table shows it is too lax.
- Logged inputs exist only while the container's log file does (Render's free disk is ephemeral). For a real deployment the log would go to durable storage; for this project the drift check is demonstrated on local-stack and generated traffic.
