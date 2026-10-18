# Resume text

**Rule:** only claims the repository supports. Numbers below are from the phase and experiment reports.

## Entry (4 bullets)

**AI Experiment Lab: End-to-End ML Experimentation and MLOps Platform** | Python, scikit-learn, XGBoost, MLflow, Optuna, FastAPI, Docker, Prometheus, Grafana, GitHub Actions | [GitHub link]

- Built a leakage-safe churn-modelling pipeline (validation, profiling, single sklearn Pipeline) and compared three model families by 5-fold CV F1 rather than accuracy, which would have picked the worst churn detector (recall 0.47 vs 0.78); tuned each with 40 Optuna trials (TPE, pruning) tracked as nested MLflow runs.
- Registered the champion in an MLflow model registry (aliases, lineage tags, rollback) and served it through a FastAPI service with strict pydantic validation and a pickle-free skops export, plus a browser page (single-customer form and CSV batch scoring with downloadable results) for non-technical users; containerised with Docker and deployed to Render, with Prometheus metrics, a Grafana dashboard and alert rules.
- Implemented data and prediction drift detection (KS or chi-square plus PSI, with an insufficient-data guard), calibrated on simulated shifts, and found that F1 can rise while ROC AUC falls under drift; used that to gate model promotion on ROC AUC.
- Built a retraining flow that promotes a candidate only if it clearly beats the champion on shifted data without regressing on natural data, plus GitHub Actions CI (lint, 163 tests on two Python versions, Docker build and smoke test) and a manual redeploy workflow that verifies the served model version.

## One-line version
End-to-end ML platform for churn prediction: leakage-safe training, Optuna tuning, MLflow registry, FastAPI on Docker with a web page for end users (live), Prometheus/Grafana monitoring, drift detection and gated retraining, with 163 tests and CI.

## Not to claim
- Latency, throughput or uptime figures: none were measured.
- "Automated retraining": the trigger and deploy are manual; the policy is automated and tested.
- "Production": it is a free-tier demo with no login or authentication; do not describe the web page as handling real customer data. Traffic is synthetic.
- Accuracy improvements from tuning as proof of a better model: the tuned models are statistically tied.
- Hosted monitoring or Evidently in production: both run locally or in Colab.
