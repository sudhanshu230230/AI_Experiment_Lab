# Interview preparation

Answers are grounded in what this project actually did and measured. Where a number is quoted it is from the cited report. Where the honest answer is "this was a limitation", it says so, because that is a stronger interview answer than a claim you cannot defend.

## ML

**Why F1 (and not accuracy) as the selection metric?** Churn is 26.5% of customers, so always predicting "No" scores 73.5% accuracy with zero recall. In EXP-001 Random Forest had the best accuracy (78.1%) and the worst recall (46.5%): it missed more than half of the churners. Logistic Regression had 73.8% accuracy and caught 78.3%. F1 balances precision and recall on the positive class (ADR-003). What I would add with real business costs: pick the decision threshold from the cost of a missed churner versus a wasted retention offer. All three tuned models ended at nearly the same operating point (recall about 0.79, precision about 0.52 at threshold 0.5), so the threshold is the remaining lever (EXP-003).

**How did you prevent data leakage?** Split first, then fit imputers, scaler and encoder on the training split only. Preprocessing and model are one sklearn `Pipeline`, so cross-validation refits preprocessing inside each fold. Selection used CV on the training split; the test split was scored once. There is a test that the fit uses only training data. The honest caveat: Phase 8 used the test set as "new data", so no untouched test set remains.

**How did you handle class imbalance?** `class_weight=balanced` for logistic regression and random forest, `scale_pos_weight` for XGBoost, and an imbalance-aware metric (F1, with PR-AUC reported). I did not resample, because that changes the data distribution and complicates drift comparison.

**Why did one model outperform another? Which won?** Defaults: Logistic Regression beat the trees (CV F1 0.628 vs 0.540 for Random Forest), probably because churn depends on a few roughly additive factors and 5,634 rows is small for ensembles; default Random Forest overfit (unlimited depth). After 40 Optuna trials each, the three were tied (test F1 0.620 to 0.637, noise about +-0.02), so I chose the logistic regression for simplicity, speed and interpretability (ADR-005). I did not claim a winner the evidence did not support.

**How does cross-validation work, and why stratified?** Split the training data into 5 folds, train on 4 and score on 1, rotate, average. Stratified keeps the 26.5% churn rate in every fold. The fold standard deviation (0.015 to 0.023) is what tells you whether a 0.016 difference means anything (it did not).

**What would you do about the tuning gain being small?** Report it as small. Tuning moved Random Forest a lot (+0.106 test F1) because its defaults were bad, XGBoost a little, and logistic regression not at all. That is the expected pattern: tuning rescues badly configured models, it does not create signal.

## MLOps

**What problem does MLflow solve?** "Which run produced this model, with what data, code and parameters?" Each run stores parameters, metrics, artifacts, the dataset hash, the git commit and library versions. I rebuilt the champion from the registry and reproduced its logged test F1 and ROC AUC exactly (0.6195, 0.8399).

**What is a model registry and why version models?** A named, versioned catalogue of models with lineage. I use an alias (`champion`) instead of the older stages. Promotion and rollback are moving the alias; old versions are never deleted. Each version carries a `source_run_id` tag, so version -> run -> dataset hash -> code is recoverable.

**What is reproducibility, concretely?** Fixed seeds, the same split, dataset sha256, code version and library versions logged per run, and a test that a logged pipeline reproduces predictions.

**What happens when a production model degrades?** Labels arrive late for churn, so I watch for drift first (inputs and predictions), then confirm with performance once labels exist. A drift verdict starts a retrain candidate; a promotion gate decides whether it replaces the champion. If a bad model ships, move the alias back, re-export, redeploy.

**Why did you not retrain automatically on drift?** Drift says inputs moved, not that the model is worse. Phase 7 showed it: on a shifted sample F1 rose (0.619 to 0.681) because the churn rate rose, while ROC AUC fell (0.840 to 0.761). So the promotion rule uses ROC AUC, requires a gain on shifted data and no regression on natural data.

**What did your retraining run actually do?** It triggered on a drift verdict, refit a clone of the champion on old plus new data, and compared them on two held-out sets. The gain was +0.0042 ROC AUC against a required +0.005, so the champion was kept. I did not lower the margin to make the demo promote; with about 700 evaluation rows a 0.004 difference is noise. The system working means declining as often as promoting.

## Deployment

**Why FastAPI? How does the prediction API work?** Pydantic gives request validation for free (an unknown category or a non-numeric `TotalCharges` returns 422, not a 500; a test found a real 500 there). The model loads once at startup from `models/champion/`; `/predict` builds a one-row DataFrame and calls `predict_proba` on the single Pipeline. If the model fails to load, the service stays up and reports `degraded` on `/health`.

**Who can actually use it? / How did you make it usable for non-technical people?** The API alone is for developers, so the same service serves a plain page at `/app`: a form for one customer and a CSV upload for many, with a downloadable results file. The form is generated from the pydantic schema, so a new category added to the API appears in the dropdown and the page cannot drift out of step. I show a "risk score" with a warning that it is not a literal percentage, because `class_weight=balanced` makes the probabilities uncalibrated. Row-level API errors are mapped back to CSV rows. I tested it in headless Chromium (single, batch, bad CSV, mobile width). Limits: no login, so no real customer data; it is a demo model.

**Why Docker?** The image pins the scikit-learn version the model was trained with. A pickled or skops model loaded under a different scikit-learn version can fail or silently differ, so `check_model_export.py` compares the model's recorded version with the Dockerfile's before every redeploy.

**Why skops, not pickle?** Unpickling runs arbitrary code. skops refuses types it was not told to trust, and I additionally restrict trusted types to a prefix allow-list, so a tampered metadata file cannot make the loader import arbitrary classes (tested).

**Why Render and not Hugging Face Spaces?** I first told the plan to use Hugging Face because I believed Docker Spaces were free; the current docs require a paid plan. I corrected course and moved to Render's free tier (ADR-006). Trade-off: it sleeps after 15 minutes idle, so the first request takes about a minute.

## Monitoring

**What is data drift versus concept drift, and how is drift different from performance degradation?** Data drift: input distribution changed. Concept drift: the relationship between inputs and the label changed. Neither is visible from inputs alone for concept drift; performance degradation is measured with labels. Drift is an early, label-free warning; performance is the confirmation.

**How do you detect drift here, and why that rule?** Per feature: KS test (numeric) or chi-square (categorical) plus PSI. A feature counts as drifted only if PSI >= 0.1 and p < 0.05, and the dataset verdict is `drift` at 30% of features. Requiring both guards against "significant but trivial" at large n. The first threshold (PSI 0.2) was too lax: on a clearly shifted sample it flagged 3 of 19 features while Evidently flagged 13; I lowered it to 0.1, with the untouched test set staying at 0 (largest PSI 0.0125). The caveat I state: I tuned the threshold after seeing one family of shifts.

**What do you monitor operationally?** Request count and latency by route template and status, prediction counts by class, the predicted-probability histogram, model loaded and version. Labels are bounded (route templates, class names) so cardinality stays small.

## CI/CD

**What happens after a git push?** `ci.yml`: ruff, pytest on two Python versions, then build the Docker image, run it and smoke-test it. Deployment is deliberately a separate manual workflow that verifies the export, triggers Render and checks the served `registry_version` equals the committed one.

**Why automate tests?** A real incident: I handed over a zip with a new `app.py` but not the module it imported, and a broken commit was pushed. After that, one self-contained zip per phase, and CI that fails on import errors.

## System design

**How would this scale to 100 models?** Replace SQLite on Drive with a hosted MLflow server on Postgres plus object storage; one registered name per model with the same alias convention; a parametrised serving image or a model-router; per-model drift configs; promotion policy as shared code (it already is a pure function).

**Multiple users and concurrent experiments?** SQLite allows one writer (ADR-001), so concurrent users need the hosted tracking server. Runs are already independent and tagged with seed, dataset hash and code version.

**How would you roll back a bad model?** Move the `champion` alias to the previous version, re-export (or `git revert` the `models/champion/` commit), push, run Redeploy. The workflow's version check confirms the rollback actually serves.

**What would you do next?** A larger evaluation window or real labels before judging retraining; a second, independent shift scenario to validate drift thresholds; decision-threshold selection from business costs; durable storage for the prediction log; authentication.

## Mistakes worth telling (shows judgement)
- Said Hugging Face Spaces was free; checked, was wrong, corrected with sources.
- First drift thresholds were too lax; recalibrated, disclosed that this was fitted to one scenario.
- A tuning claim ("baseline is trial 0") was not true for XGBoost until defaults were enqueued; fixed with a test.
- A serving bug (500 on bad `TotalCharges`) was found by a test, not by a user.
