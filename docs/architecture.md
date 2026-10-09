# Architecture

**Status:** v0.1 (target architecture; components are marked with their implementation phase)

## 1. Target system

```text
                       Streamlit dashboard (Phase 5+)
                                  │
                                  ▼
                         FastAPI service (Phase 5)
                                  │
         ┌────────────────────────┼─────────────────────────┐
         ▼                        ▼                         ▼
  Dataset engine (P1)      Training engine (P1-3)      Model API (P5)
  ingest, profile,         sklearn, XGBoost,           loads registered
  validate, preprocess     Optuna                      model, predicts
         │                        │                         │
         │                        ▼                         ▼
         │                 MLflow tracking (P2)      Monitoring (P6-7)
         │                        │                  Prometheus, Grafana,
         │                        ▼                  Evidently
         │                 Model registry (P4)              │
         │                        │                         ▼
         └────────────────────────┴──────────────── Retraining (P8)
```

## 2. Components and code locations

| Component | Package path | Phase |
|---|---|---|
| Ingestion, profiling, validation | `src/ai_experiment_lab/data/` | 1 |
| Preprocessing pipeline | `src/ai_experiment_lab/preprocessing/` | 1 |
| Model definitions | `src/ai_experiment_lab/models/` | 1 |
| Trainer, tuning | `src/ai_experiment_lab/training/` | 1, 3 |
| Metrics, evaluation | `src/ai_experiment_lab/evaluation/` | 1 |
| Runtime and drift monitoring | `src/ai_experiment_lab/monitoring/` | 6, 7 |
| Shared utilities (config, logging, seeding) | `src/ai_experiment_lab/utils/` | 1 |
| REST API | `api/` | 5 |
| Dashboard | `dashboard/` | 5+ |

## 3. Where code is written and where it runs

GitHub is the bridge between the two environments.

```text
VS Code (write, test lightweight code) ──push──▶ GitHub ──clone──▶ Colab (run ML)
                                                   ▲                  │
                                                   └── configs, docs ─┘
```

| Concern | Written in | Runs in |
|---|---|---|
| Library code under `src/` | VS Code | Colab (training); local (unit tests) |
| Notebooks | Colab | Colab |
| MLflow runs and artifacts | n/a | Colab, stored per ADR-001 |
| API, dashboard, Docker, CI, docs | VS Code | Local (API), CI (Docker build), cloud (deploy) |

Notebooks are thin runners. They install the package and call modules in `src/`.

## 4. Key design rules

1. **One artifact for inference.** Preprocessing and model are saved as a single sklearn
   `Pipeline` so training and serving cannot diverge (ADR-002).
2. **No leakage.** Split first; fit all preprocessing on the training split only.
3. **Config-driven.** Dataset, models, thresholds and metrics come from `configs/`.
4. **Persistent state lives off Colab.** Data, MLflow store and artifacts are never kept
   only on Colab's local disk (ADR-001).
5. **Task-appropriate selection metric.** "Best" is never defined by accuracy alone (ADR-003).

## 5. Decision log

See [decisions/](decisions/). Current ADRs: 001 MLflow storage, 002 model handoff,
003 initial dataset and selection metric.
