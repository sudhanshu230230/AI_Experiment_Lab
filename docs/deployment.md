# Deployment

**Status:** Image and deploy script implemented (Phase 5); first deployment pending (see the Phase 5 report). Decisions: ADR-006.

## What is deployed
A Docker image (`Dockerfile`) running the FastAPI service with the exported champion model baked in (`models/champion/`). It needs no database, Drive or MLflow at runtime.

- Base: `python:3.12-slim`; runs as a non-root user; listens on `$PORT` (default 7860, which Hugging Face Spaces expects).
- Installs `.[serve]` (FastAPI, uvicorn, pydantic, skops) plus `scikit-learn==${SKLEARN_VERSION}` (build arg, default 1.6.1, the version the model was trained with; it is stored in `models/champion/metadata.json` under `libraries`).
- A container `HEALTHCHECK` polls `/health`.
- If a future champion is an XGBoost model, add `xgboost` to the image (Dockerfile install line).

## Build and run locally (needs Docker Desktop)

```powershell
docker build -t ael-api .
docker run --rm -p 7860:7860 ael-api
# in another terminal
python scripts/smoke_test_api.py http://127.0.0.1:7860
```
If Docker Desktop is too heavy for the laptop, skip this: CI builds and smoke-tests the image on every push once `models/champion/model.skops` is committed (`.github/workflows/ci.yml`, job `docker`).

## Deploy to Hugging Face Spaces
1. Create a free account at huggingface.co and a write token (Settings, Access Tokens).
2. `pip install huggingface_hub`
3. Check what will be uploaded (no network):
   `python scripts/deploy_hf_space.py --space YOUR_USER/churn-api --dry-run`
4. Deploy:
   ```powershell
   $env:HF_TOKEN = "hf_..."          # this session only; never commit it
   python scripts/deploy_hf_space.py --space YOUR_USER/churn-api
   ```
5. Watch the build logs on the Space page. When it shows Running, the API is at `https://YOUR_USER-churn-api.hf.space` (`/docs`, `/health`).
6. Verify: `python scripts/smoke_test_api.py https://YOUR_USER-churn-api.hf.space`

The script stages only what the image needs (Dockerfile, `pyproject.toml`, `src/`, `models/champion/`, a Space README with the required front matter) so this repository's README is not touched.

## Redeploying a new model
Export the new champion (notebook 05), commit `models/champion/`, run the deploy script again. Phase 8 automates this step.

## Known limitations
- Dependencies in the image are not fully pinned (only scikit-learn is); a lockfile is a later hardening step.
- Free Spaces sleep after inactivity; the first request after a sleep is slow.
- No authentication on the API: acceptable for a demo with synthetic-risk data, not for production.
