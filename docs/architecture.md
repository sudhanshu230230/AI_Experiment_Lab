# Architecture

**Status:** v1.0 (as implemented, Phases 0 to 9)

## 1. System

```text
 Google Colab (ML compute)                          Local / CI / cloud
 ---------------------------                        --------------------------------------------
 data/        ingest, profile, validate             serving/     FastAPI app, pydantic schemas, metrics
 preprocessing/ one sklearn Pipeline                Docker image (models/champion baked in) -> Render
 training/    baselines (CV F1)                     monitoring/  drift module, prediction-log reader
 tuning/      Optuna (TPE, median pruner)           prometheus + grafana (docker compose, local)
 tracking/    MLflow runs (SQLite on Drive)         GitHub Actions: ci.yml, redeploy.yml
 registry/    alias "champion", skops export        scripts/     smoke test, load generator,
 retraining/  trigger + promotion policy                         drift check, export check
          \\______ GitHub: code, configs, docs, models/champion/ ______/
```

Request path: `client -> /predict (pydantic validation) -> Pipeline.predict_proba -> JSON {probability, prediction, model_version}`; every request updates Prometheus metrics, and an optional JSONL log (`PREDICTION_LOG`) records inputs and scores for the drift check.

## 2. Components and code locations

| Component | Path | Phase |
|---|---|---|
| Ingestion, profiling, validation | `src/ai_experiment_lab/data/` | 1 |
| Preprocessing pipeline | `src/ai_experiment_lab/preprocessing/` | 1 |
| Model definitions, trainer | `src/ai_experiment_lab/models/`, `training/` | 1 |
| Metrics | `src/ai_experiment_lab/evaluation/` | 1 |
| MLflow tracking | `src/ai_experiment_lab/tracking/` | 2 |
| Optuna tuning | `src/ai_experiment_lab/tuning/` | 3 |
| Registry and serving export | `src/ai_experiment_lab/registry/` | 4 |
| API, schemas, Prometheus metrics | `src/ai_experiment_lab/serving/` | 5, 6 |
| Drift detection, prediction log | `src/ai_experiment_lab/monitoring/` | 7 |
| Retraining policy | `src/ai_experiment_lab/retraining/` | 8 |
| Dockerfile, compose, Render blueprint | repo root, `monitoring/` | 5, 6 |
| CI and redeploy workflows | `.github/workflows/` | 0, 9 |

The planned `api/` and `dashboard/` folders were not created: the API lives inside the package as `serving/` so Docker installs it with the library, and no dashboard was built (see README, "What this is not").

## 3. Where code is written and where it runs

| Concern | Written in | Runs in |
|---|---|---|
| Library code under `src/` | VS Code | Colab (training), local and CI (tests), Docker (serving) |
| Notebooks | Colab | Colab |
| MLflow store and artifacts | n/a | Colab, SQLite and files on Drive (ADR-001) |
| API, Docker, CI, docs | VS Code | Local, CI, Render |

## 4. Key design rules

1. **One artifact for inference.** Preprocessing and model are a single `Pipeline` (ADR-002).
2. **No leakage.** Split first; fit preprocessing on the training split only; select by CV, score the test split once.
3. **Config-driven.** Dataset, models, search spaces and drift thresholds are in `configs/*.yaml`.
4. **Persistent state lives off Colab** (ADR-001).
5. **Task-appropriate selection metric**, never accuracy alone (ADR-003); ROC AUC for promotion because F1 moves with the churn rate (ADR-009).
6. **No pickle in serving**: skops with a restricted trusted-type list (ADR-005).
7. **Drift is a signal, not a verdict**; promotion needs an evaluation gate (ADR-008, ADR-009).

## 5. Decision log

See [decisions/](decisions/): 001 MLflow storage, 002 model handoff, 003 dataset and selection metric, 004 tuning method, 005 champion and registry, 006 serving and deployment, 007 monitoring, 008 drift detection, 009 retraining and promotion.
