# Phase 5 Report: Serving API, Docker and Deployment

- **Dates:** 2026-10-15 to in progress
- **Git tag:** v0.5-phase5 (create after the first deployment is verified)
- **Status:** In progress (API, image, deploy script, tests done; real champion export and first deployment pending)

## Objective
Serve the registry champion behind a validated HTTP API in a Docker image, and deploy it to a public free host.

## Progress by step

| Step | Component | Status |
|---|---|---|
| 1 | Design and deployment decisions (ADR-006) | Done |
| 2 | API (`serving/app.py`, `serving/schemas.py`): `/health`, `/model`, `/predict`, `/predict/batch`, validation, optional prediction log | Done; 22 tests |
| 3 | Dockerfile, `.dockerignore` | Written; not built here (no Docker daemon in the authoring environment) |
| 4 | Smoke test script (`scripts/smoke_test_api.py`) | Done; passes against a local server, including one started from a clean `.[serve]`-only virtualenv |
| 5 | CI job that builds the image and smoke-tests it | Written; runs once `models/champion/model.skops` is committed |
| 6 | Deploy script (`scripts/deploy_hf_space.py`) | Done; `--dry-run` staging verified, upload not run here |
| 7 | Real champion export committed (`models/champion/`) | Pending the Phase 4 Colab run |
| 8 | First deployment and public smoke test | Pending |

## How to run it
See `docs/api-reference.md` (local run) and `docs/deployment.md` (Docker and Hugging Face).

## Tests of note
- Schema rejects: missing field, unknown category, out-of-range value, wrong type, extra field, bad `TotalCharges` text, negative `TotalCharges`; accepts blank/null/number/text `TotalCharges`.
- Batch predictions equal single predictions; empty and oversized batches are rejected.
- No model gives `degraded` health and 503; a model whose input columns differ from the API schema is reported `degraded`.
- The prediction log records features, probability, model version and time.
- A clean virtualenv with only the serving dependencies (no MLflow, Optuna or XGBoost) serves requests, confirming the image will not need them.

## Bug found by testing
`TotalCharges` text such as `"n/a"` passed request validation and crashed inside the pipeline (HTTP 500). Fixed with a field validator (blank becomes null, other text is a 422).

## Decisions made
See ADR-006.

## Known limitations
- Docker image not yet built anywhere; first build is in CI or on the user's machine.
- No authentication or rate limiting (demo scope).
- Dependencies other than scikit-learn are not pinned.

## Next
Phase 6: Prometheus metrics in the API (request count, latency, prediction distribution) and a local Grafana dashboard via docker-compose.
