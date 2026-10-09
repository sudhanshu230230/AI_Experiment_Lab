# Project Charter

**Status:** Approved  |  **Date:** 2026-10-10  |  **Owner:** Project author

## 1. Purpose

AI Experiment Lab is a reusable mini ML platform. It takes a tabular dataset through
validation, preprocessing, multi-model experimentation, hyperparameter optimization,
experiment tracking, model registry, API deployment, monitoring, drift detection and
retraining. It is a portfolio project demonstrating the **ML lifecycle**, not a single model.

## 2. Goals

1. Make ML experiments reproducible and comparable.
2. Demonstrate each lifecycle stage with working, tested code.
3. Run all heavy compute on Google Colab, with no dependence on the local laptop.
4. Keep every major design decision documented and defensible.

## 3. Scope

In scope: tabular classification first (customer churn), regression later; MLflow;
Optuna; FastAPI; Docker; Prometheus and Grafana; Evidently; GitHub Actions.

## 4. Non-goals

- Not an AutoML clone, and not just a Streamlit dashboard or notebook.
- No deep learning, NLP, time series or LLM work until the tabular platform is stable.
- No dependence on one huge dataset or on paid cloud infrastructure.
- No Kubernetes, Airflow, DVC or feature store in the initial build.

## 5. Constraints

| Constraint | Consequence |
|---|---|
| Low-power local laptop (AMD A9-9425) | ML runs in Colab; Docker builds run in CI or Codespaces |
| Colab disk is ephemeral | Data, MLflow store and artifacts must live on Drive or a hosted service (ADR-001) |
| No fixed deadline | Build every phase fully; do not cut documentation or tests |

## 6. Engineering principles

Reproducibility, leakage prevention, configuration over hardcoding, separation of concerns,
testing of critical components, structured logging, and versioning of code, data,
experiments and models.

## 7. Success criteria (definition of done)

A reviewer can: understand the architecture from the README; run the project from the
documented instructions; run and compare experiments in MLflow; tune and register a model;
call the prediction API; view monitoring metrics; generate a drift report; understand how
retraining works; inspect tests and CI/CD configuration.

## 8. Roadmap

| Phase | Scope | Exit criteria |
|---|---|---|
| 0 | Skeleton, docs, Colab bootstrap | Repo installs; CI green; docs index complete |
| 1 | Ingestion, profiling, validation, preprocessing, 2-3 baselines | Dataset to model comparison runs in Colab; tests pass |
| 2 | MLflow tracking | Every run logs params, metrics, artifacts; reproducible by run ID |
| 3 | Optuna | Trials tracked; best trial and search space preserved |
| 4 | Model registry | Versioning, promotion and rollback demonstrated |
| 5 | FastAPI, Docker, deploy | `/predict`, `/health`, `/model-info` live remotely |
| 6 | Prometheus and Grafana | Request, latency and error metrics visible |
| 7 | Evidently | Drift report from reference vs production data |
| 8 | Retraining | Evaluation gate blocks or promotes a candidate |
| 9 | CI/CD | Push triggers test, build, deploy |

## 9. Definition of a completed phase

1. Code merged with tests passing.
2. Phase report written in `docs/phases/`.
3. Architecture, lifecycle and changelog docs updated; ADRs added for new decisions.
4. Git tag created (`v0.N-phaseN`).
