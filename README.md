# AI Experiment Lab

An end-to-end ML experimentation and MLOps platform, built around one real problem (customer churn on the Telco dataset) so every lifecycle stage is exercised, tested and documented:

`dataset -> validation -> preprocessing -> experiments -> Optuna tuning -> MLflow tracking and registry -> FastAPI -> Docker -> deploy -> Prometheus/Grafana -> drift detection -> retraining -> CI/CD`

**Live demo (free tier, sleeps when idle, first request takes about a minute):** https://churn-api-xop3.onrender.com/ (web page for people) and `/docs` (API).

**For end users:** open the live demo, fill in a customer or upload a CSV of customers, and get a churn risk score and a downloadable results file. No install, no code. Locally: `docker run -p 7860:7860 ...` then open http://localhost:7860/.

> **Status:** Phases 0 to 9 complete. Every claim below is implemented and tested; limits are listed honestly in [What this is not](#what-this-is-not) and in [`docs/definition-of-done.md`](docs/definition-of-done.md).

## Architecture

```text
 Google Colab (all ML compute)                         Your machine / CI / cloud
 ------------------------------                        ---------------------------------------
 notebooks 01-07 (thin runners)                        FastAPI service  <-- models/champion/
   profile, validate, split                              /predict /health /model /metrics
   baselines -> MLflow runs (SQLite on Drive)            Docker image  --> Render (free)
   Optuna: 40 trials x 3 models, nested runs             Prometheus + Grafana (docker compose)
   registry: alias "champion", export (skops)            prediction log (JSONL) --> drift check
   drift report (own module + Evidently HTML)            GitHub Actions: CI, manual Redeploy
   retraining: candidate vs champion, promote?
   notebook 08 / train_any.py: any CSV -> model + schema
          \_____________ GitHub is the bridge: code, configs, docs, models/champion/ ____/
```

Key design rules (each has an ADR in [`docs/decisions/`](docs/decisions/)):
1. **Split first, fit preprocessing on the training split only.** Preprocessing and model are one sklearn `Pipeline`, so training and serving cannot diverge (ADR-002).
2. **Never select on the test set or on accuracy.** Models are chosen by 5-fold CV F1; the test split is scored once (ADR-003, ADR-004).
3. **Serve a skops file, not a pickle**, with a restricted list of trusted types (ADR-005).
4. **Drift is a prompt to look, not proof of failure.** Retraining promotes a candidate only if it is clearly better (ADR-008, ADR-009).

## Results on the Telco churn dataset (7,043 rows, 26.5% churn)

| Stage | Result |
|---|---|
| Baselines (EXP-001) | Logistic Regression CV F1 0.628, XGBoost 0.612, Random Forest 0.540. Accuracy would have picked the worst churn detector |
| Tuned, 40 Optuna trials each (EXP-003) | Test F1: Random Forest 0.637, XGBoost 0.632, Logistic Regression 0.620: a statistical tie (test noise about +-0.02) |
| Champion (ADR-005) | Tuned Logistic Regression (`C=0.037`): test F1 0.620, recall 0.78, ROC AUC 0.840. Chosen for simplicity and speed because the models tie |
| Drift (Phase 7) | Verdict flips to `drift` between shift strengths 3 and 10 of 19 features; untouched test data: 0 drifted. The first threshold (PSI 0.2) missed a clear shift and was lowered to 0.1 (ADR-008) |
| Drift vs performance | On a shifted sample F1 rose (0.619 to 0.681) while ROC AUC fell (0.840 to 0.761) |
| Retraining (Phase 8) | Candidate gained +0.004 ROC AUC on shifted data against a required +0.005: **rejected**, champion kept |

## Try it on your own data

Open `/lab` on the live site, upload any tabular CSV (not time series, images, audio or video), and it tells you which algorithm to use and with which settings. See [docs/model-advisor.md](docs/model-advisor.md).

## Run it

**Tests (local, any OS; takes 1.5 to 3 minutes):**
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,ml,tracking,serve]"
ruff check . && pytest
```

**API in Docker (needs Docker Desktop and `models/champion/`):**
```bash
docker compose up --build        # API :7860, Prometheus :9090, Grafana :3000
python scripts/smoke_test_api.py http://127.0.0.1:7860
python scripts/load_generator.py --rate 5 --duration 120          # add --drift for shifted customers
```

**Check drift on logged traffic:**
```bash
python scripts/check_drift.py --data data/raw/Telco-Customer-Churn.csv --log data/predictions/predictions.jsonl
```

**Your own dataset:** any tabular CSV with a binary, multi-class or numeric target: `python scripts/train_any.py --data my.csv --target <column> --out models/champion`, or Colab notebook 08. The API and web page then build themselves from the trained model, so there are no code changes. Guide: [`docs/any-dataset.md`](docs/any-dataset.md).

**ML work (Colab):** open a notebook from `notebooks/`, run the setup cell ([`docs/setup-guide.md`](docs/setup-guide.md)). Order: 01 explore, 02 baselines, 03 tracked baselines, 04 Optuna, 05 registry and export, 06 drift, 07 retraining, 08 your own dataset.

**Browse MLflow runs:** download the database backup from Drive (`mlflow/backups/`), then
```bash
pip install mlflow
mlflow ui --backend-store-uri sqlite:///path/to/mlflow-backup.db
```

**Call the API:**
```bash
curl -X POST https://churn-api-xop3.onrender.com/predict -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":5,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":85.5,"TotalCharges":"420.35"}'
```

## CI/CD

- [`ci.yml`](.github/workflows/ci.yml): on every push and pull request: ruff, pytest on two Python versions, then build the Docker image, run it and smoke-test it.
- [`redeploy.yml`](.github/workflows/redeploy.yml): manual. Verifies the committed model export, triggers a Render deploy, waits until the deployed `registry_version` equals the committed one, then smoke-tests. Steps: [`docs/retraining.md`](docs/retraining.md).

## Documentation

Start at [`docs/README.md`](docs/README.md). Most useful:
[architecture](docs/architecture.md) · [ML lifecycle](docs/ml-lifecycle.md) · [deployment](docs/deployment.md) · [monitoring and drift](docs/monitoring-and-drift.md) · [retraining](docs/retraining.md) · [API reference](docs/api-reference.md) · [definition of done](docs/definition-of-done.md) · [interview prep](docs/interview-prep.md) · [resume text](docs/resume.md) · [ADRs](docs/decisions/) · [phase reports](docs/phases/) · [experiment reports](docs/experiments/) · [changelog](docs/changelog.md)

## What this is not

- **Tabular data only** (binary, multi-class and regression targets; no images, free text or time series). Models: Logistic Regression (Ridge for regression), Random Forest, XGBoost. The deployed demo is still the Telco churn model; other datasets are trained with `scripts/train_any.py` or notebook 08 and replace it ([`docs/any-dataset.md`](docs/any-dataset.md)).
- The web page at `/app` only scores customers. There is no UI for running experiments; that workflow is notebooks plus the library in `src/`. No login, so do not put real customer data on the public demo.
- Prometheus and Grafana run locally (docker compose), not hosted. Evidently reports are produced in Colab; the in-repo drift module drives decisions.
- Retraining is triggered by hand and runs in Colab. New labelled data is simulated from the test set, which that step consumes.
- The hosted API has no authentication, and the free host sleeps and loses in-memory metrics and the prediction log on restart.
- Single 80/20 split per dataset: differences below about 0.02 F1 are noise. The cross-dataset support (ADR-010) is tested on synthetic data, the scikit-learn wine and diabetes data and one uploaded churn CSV, not on a wide range of real datasets.

## License

MIT. See [`LICENSE`](LICENSE).
