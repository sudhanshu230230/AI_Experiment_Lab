# API Reference

**Status:** Implemented (Phase 5). Code: `src/ai_experiment_lab/serving/`. Interactive docs: `/docs` (Swagger) and `/openapi.json` on the running service.

The service serves the exported champion model (`models/champion/`, ADR-005) and has no state besides the loaded model.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and whether the model loaded: `{"status": "ok" \| "degraded", "model_loaded": bool}` (always HTTP 200) |
| GET | `/model` | Registry name and version, source run, metrics, dataset hash, code and library versions, threshold, input columns. 503 if no model |
| POST | `/predict` | One customer in, one prediction out |
| POST | `/predict/batch` | `{"customers": [...]}` with 1 to 1000 customers; predictions come back in the same order |
| GET | `/schema` | The inputs the served model expects (names, types, allowed values) and the task; the web page and the smoke test read it |
| GET | `/app` | Web page for non-technical users: one-customer form and CSV batch upload (calls `/predict` and `/predict/batch`); `/` redirects here. Hidden from `/docs` |
| GET | `/metrics` | Prometheus metrics (see `docs/monitoring-and-drift.md`); not counted in its own metrics, hidden from `/docs` |

## Request: one customer (`POST /predict`)

All 19 feature columns of the Telco dataset, no `customerID`, no `Churn`:

```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": "420.35"
}
```

Validation (HTTP 422 on failure, with the offending field named):
- Every field is required except `TotalCharges`; unknown extra fields are rejected (catches typos).
- Categorical fields only accept the values seen in the dataset (see `/docs` for the lists).
- `tenure` 0 to 120 (integer), `MonthlyCharges` 0 to 1000.
- `TotalCharges` may be a number, numeric text, blank text or null. Blank/null means "no value yet" (new customers) and is imputed by the pipeline as in training. Other text or negative numbers are rejected.

## Response

```json
{"churn_probability": 0.7312, "prediction": "Yes", "threshold": 0.5, "model_version": 1}
```

`prediction` is "Yes" when `churn_probability >= threshold`. `model_version` is the registry version the served file was exported from.

## Other models (any dataset, ADR-010)

A model exported by `scripts/train_any.py` carries its own schema, so `/predict` and `/predict/batch` accept that model's columns instead of the Telco fields (`/docs` and `/schema` list them). Column names are used exactly as in the CSV, including spaces. Nullable columns may be omitted or null. The response depends on the task, and fields that do not apply are left out:

```json
{"prediction": "Yes", "probability": 0.73, "threshold": 0.5, "model_version": 1}
{"prediction": "class_b", "probabilities": {"class_a": 0.2, "class_b": 0.7, "class_c": 0.1}, "model_version": 1}
{"prediction": 155.69, "model_version": 1}
```
The Telco export keeps its original shape (`churn_probability`, no `/schema` fields beyond the form description).

## Errors

| Code | Meaning |
|---|---|
| 422 | Request failed validation. The body lists `loc` and `msg` for each problem and never echoes the submitted values (`NaN` and `Infinity` in a body are rejected as 422) |
| 503 | Model could not be loaded at startup (see container logs); `/health` reports `degraded` |

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `models/champion` | Folder with `model.skops` and `metadata.json` |
| `PREDICTION_LOG` | unset | If set, every prediction is appended to this JSONL file (timestamp, model version, `score`, `prediction`, features; Telco logs also keep `churn_probability`) for drift analysis in Phase 7 |
| `PORT` | `7860` | Port used by the Docker image |

## Try it

```powershell
# Terminal 1 (project root, venv active)
pip install -e ".[serve]"
uvicorn ai_experiment_lab.serving.app:app --port 7860
# Terminal 2
python scripts/smoke_test_api.py http://127.0.0.1:7860
```
