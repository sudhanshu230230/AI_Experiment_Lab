# Deployment

**Status:** Deployed on Render (2026-10-16): https://churn-api-xop3.onrender.com (`/docs`, `/health`). Image, Render blueprint and Hugging Face script implemented (Phase 5). Decisions: ADR-006 (including the 2026-10-16 amendment).

## What is deployed
A Docker image (`Dockerfile`) running the FastAPI service with the exported champion model baked in (`models/champion/`). It needs no database, Drive or MLflow at runtime.

- Base: `python:3.12-slim`; runs as a non-root user; listens on `$PORT` (default 7860; Render sets 10000 through `render.yaml`).
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

## Deploy to Render (primary target, free)
**Live service:** https://churn-api-xop3.onrender.com. The bare URL redirects to `/docs`. The smoke test passed 6 of 6 checks against it on 2026-10-16 (first deployment). Steps to reproduce it:

Render builds the Dockerfile straight from the GitHub repository, so `models/champion/` must be committed and pushed first. `render.yaml` at the repo root describes the service (Docker runtime, free plan, health check `/health`, `PORT=10000`, manual deploys).

1. Sign up at render.com (sign in with GitHub is easiest). The free web service does not list a card requirement in Render's docs; if the dashboard asks for a payment method for a *free* service, stop and tell me.
2. Dashboard: **New** > **Blueprint**, connect the GitHub account when asked, choose the `AI_Experiment_Lab` repository, and **Apply**. Render reads `render.yaml` and starts the first build (a few minutes).
3. Open the service **Logs**. A healthy start ends with `Uvicorn running on http://0.0.0.0:10000` and the service shows **Live**.
4. Copy the service URL (`https://churn-api-xxxx.onrender.com`, shown at the top of the service page) and verify:
   ```powershell
   python scripts\smoke_test_api.py https://churn-api-xxxx.onrender.com --wait 120
   ```
   Expect `ALL CHECKS PASSED`. `--wait` covers the cold start. Also open `/docs` in a browser.
5. Redeploy later with **Manual Deploy** > **Deploy latest commit** (auto-deploy is off so docs-only pushes do not use free build minutes).

Free-instance behaviour (from Render's docs): the service spins down after 15 minutes without traffic and takes about a minute to wake on the next request; 750 free instance hours per month per workspace. Because of the spin-down, in-memory Prometheus counters also reset, and the prediction log (if enabled) is not kept between restarts.

If the build fails, open the build log; the common causes are `models/champion/` missing from the repo (the Dockerfile `COPY` fails) and a scikit-learn version that cannot be installed (check `libraries` in `models/champion/metadata.json` and pass `SKLEARN_VERSION` as a Docker build argument in the Render service settings).

## Alternative: Hugging Face Spaces (paid)
As of 2026-10-16, creating a Docker Space on free CPU requires a PRO subscription or billing credits (checked against Hugging Face's docs and forum reports; the price was not stated in the docs, a third-party source says about $9 a month). The deploy script still works if you subscribe:
```powershell
pip install huggingface_hub
python scripts\deploy_hf_space.py --space YOUR_USER/churn-api --dry-run
$env:HF_TOKEN = "hf_..."          # this session only; never commit it
python scripts\deploy_hf_space.py --space YOUR_USER/churn-api
python scripts\smoke_test_api.py https://YOUR_USER-churn-api.hf.space --wait 120
```
It stages only what the image needs (Dockerfile, `pyproject.toml`, `src/`, `models/champion/`, a Space README with the required front matter) and uploads it with the token from the environment.

## Redeploying a new model
Export the new champion (notebook 05), commit and push `models/champion/`, then trigger a Render manual deploy. Phase 8 and 9 automate this step.

## Known limitations
- Dependencies in the image are not fully pinned (only scikit-learn is); a lockfile is a later hardening step.
- Free hosts sleep after inactivity (Render: after 15 minutes); the first request after a sleep takes about a minute.
- No authentication on the API: acceptable for a demo with synthetic-risk data, not for production.
