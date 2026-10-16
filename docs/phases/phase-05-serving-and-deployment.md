# Phase 5 Report: Serving API, Docker and Deployment

- **Dates:** 2026-10-15 to 2026-10-16
- **Git tag:** v0.5-phase5 (create after the first deployment is verified)
- **Status:** Complete (deployed on Render; CI Docker job not yet seen running)

## Objective
Serve the registry champion behind a validated HTTP API in a Docker image, and deploy it to a public free host (Render; see the ADR-006 amendment).

## Progress by step

| Step | Component | Status |
|---|---|---|
| 1 | Design and deployment decisions (ADR-006) | Done |
| 2 | API (`serving/app.py`, `serving/schemas.py`): `/health`, `/model`, `/predict`, `/predict/batch`, validation, optional prediction log | Done; 22 tests |
| 3 | Dockerfile, `.dockerignore` | Built successfully by Render on the first try; not built locally |
| 4 | Smoke test script (`scripts/smoke_test_api.py`) | Done; passes against a local server, including one started from a clean `.[serve]`-only virtualenv |
| 5 | CI job that builds the image and smoke-tests it | Written; runs once `models/champion/model.skops` is committed |
| 6 | Deploy configuration: `render.yaml` (primary, free) and `scripts/deploy_hf_space.py` (alternative, paid; `--dry-run` staging verified) | Done; neither deployed yet |
| 7 | Real champion export committed (`models/champion/`) | Done |
| 8 | First deployment on Render and public smoke test | Done: https://churn-api-xop3.onrender.com, all 6 smoke checks passed (example customer: probability 0.8748, prediction Yes, model version 1) |

## How to run it
See `docs/api-reference.md` (local run) and `docs/deployment.md` (Docker and Hugging Face).

## Tests of note
- Schema rejects: missing field, unknown category, out-of-range value, wrong type, extra field, bad `TotalCharges` text, negative `TotalCharges`; accepts blank/null/number/text `TotalCharges`.
- Batch predictions equal single predictions; empty and oversized batches are rejected.
- No model gives `degraded` health and 503; a model whose input columns differ from the API schema is reported `degraded`.
- The prediction log records features, probability, model version and time.
- A clean virtualenv with only the serving dependencies (no MLflow, Optuna or XGBoost) serves requests, confirming the image will not need them.

## Deployment notes
- Opening the bare URL returned `{"detail":"Not Found"}` because the API had no route at `/`. Added a redirect from `/` to `/docs` (test added).
- The served export reports `model_version` 1, while the registry alias was later moved to version 2 (the register cell was run twice on the same run). Both versions come from the same run, so predictions are the same; the next export refresh will carry version 2.

## Bug found by testing
`TotalCharges` text such as `"n/a"` passed request validation and crashed inside the pipeline (HTTP 500). Fixed with a field validator (blank becomes null, other text is a 422).

## Decisions made
See ADR-006.

## Change during the phase
The target moved from Hugging Face Spaces to Render on 2026-10-16: Docker Spaces now need a paid plan (ADR-006 amendment).

## Known limitations
- The CI `docker` job has not been observed running yet.
- Free Render instance sleeps after 15 minutes idle (about one minute to wake).
- No authentication or rate limiting (demo scope).
- Dependencies other than scikit-learn are not pinned.

## Next
Phase 6 (code done; see its report): Prometheus metrics in the API (request count, latency, prediction distribution) and a local Grafana dashboard via docker-compose.
