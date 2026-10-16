# ADR-006: Serving design and deployment target

- **Status:** Accepted
- **Date:** 2026-10-16
- **Phase:** 5

## Context
The champion (ADR-005) must be served over HTTP, containerised, and deployed somewhere free and public,
with a tight schedule and a low-powered development laptop.

## Decision
- **Framework:** FastAPI + pydantic. Request validation is the first line of defence: closed category lists, numeric ranges, `extra="forbid"`, and a validator for `TotalCharges` (blank means missing, other text is a 422). A test caught that junk text there would otherwise have crashed with a 500.
- **Raw rows in, probability out:** the API takes raw Telco columns and the exported Pipeline does all preprocessing (ADR-002), so training and serving cannot drift apart.
- **Fail loudly at startup, stay observable:** the model loads once at startup and its input columns must equal the API schema, otherwise the service reports `degraded` on `/health` and returns 503 on predictions (instead of crashing in a loop with no signal).
- **Prediction log:** optional JSONL log of every prediction (features, probability, model version) for Phase 7 drift analysis; off by default.
- **Image:** `python:3.12-slim`, non-root, scikit-learn pinned to the training version (a mismatch is logged as a warning at startup), model baked in so there are no runtime dependencies on Drive or MLflow.
- **Target (original decision, superseded, see Amendment): Hugging Face Spaces (Docker).** Free, no card, public URL, port 7860 convention. Deployed by `scripts/deploy_hf_space.py`, which stages only what the image needs and uploads it with a token from the environment, so the repository README stays normal and the same script can run from CI in Phase 9.
- **Verification:** `scripts/smoke_test_api.py` (standard library only) runs against any base URL: local, container, CI or the deployed Space.

## Alternatives considered
- Render free web service from GitHub: simpler git-based deploy, but free instances spin down aggressively and account requirements are less certain; HF Spaces kept as the primary target.
- Loading from MLflow at startup: couples serving to the Drive-hosted store (ADR-001) and adds a heavy dependency; rejected in favour of the committed export.
- Flask: fewer built-ins; FastAPI gives validation and OpenAPI docs for free.

## Consequences
- A new model means a new export committed to git and a redeploy (automated in Phase 8).
- The schema is specific to the Telco dataset; a different dataset needs a different `schemas.py`.
- The Docker build could not be run in the authoring environment (no Docker daemon). It was verified by installing `.[serve]` in a clean virtualenv and running the smoke test; the first real image build happens in CI or on the user's machine.

## Amendment 2026-10-16: deployment target changed to Render
- **Reason:** while preparing the first deployment, Hugging Face's documentation and community reports showed that Docker Spaces on free CPU now require a PRO subscription or billing credits. The "free, no card" assumption above was wrong (it was based on earlier knowledge and not re-checked).
- **New target: Render free web service**, built from the repository's Dockerfile via `render.yaml` (Docker runtime, plan `free`, health check `/health`, `PORT=10000`, auto-deploy off). Render's docs state a 15-minute idle spin-down (about one minute to wake) and 750 free instance hours a month, and do not list a card requirement for free services.
- **Kept:** `scripts/deploy_hf_space.py` stays as a documented paid alternative. The image, API and smoke test are host-independent; the smoke test gained `--wait SECONDS` for cold starts.
- **Not verified:** Render's docs do not explicitly state that Docker web services may use the free plan; the first deployment will confirm it. If Render refuses, the fallback is local-only demonstration with the compose stack.
- **Lesson recorded:** free-tier terms of third-party platforms change; check them at deployment time, not at design time.
