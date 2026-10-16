# Monitoring and Drift

**Status:** Monitoring implemented (Phase 6). Drift detection: not started (Phase 7).

## 1. Operational monitoring (Phase 6)

The API exposes Prometheus metrics at `GET /metrics`. A local stack (`docker-compose.yml`) runs the API, Prometheus and Grafana, with a provisioned dashboard and alert rules. Decisions: ADR-007.

### Run it (needs Docker Desktop and `models/champion/`)

```powershell
docker compose up --build
# in another terminal, make some traffic
python scripts/load_generator.py --rate 5 --duration 300 --bad-fraction 0.05
```
- Grafana: http://localhost:3000, dashboard **Churn API** (opens by default; anonymous viewer, `admin`/`admin` to edit)
- Prometheus: http://localhost:9090 (Status > Targets; Alerts)
- API: http://localhost:7860/docs

`docker compose down` stops everything. The prediction log is written to `data/predictions/predictions.jsonl` (git-ignored) for Phase 7.

### Metrics

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `ael_http_requests_total` | counter | method, path, status | Requests by route template and status code. Unknown URLs share `path="unmatched"` |
| `ael_http_request_duration_seconds` | histogram | method, path | Latency (buckets 5 ms to 5 s) |
| `ael_predictions_total` | counter | prediction | Predictions by class (Yes/No) |
| `ael_churn_probability` | histogram | none | Distribution of predicted probabilities (buckets of 0.1, plus 0.05) |
| `ael_model_loaded` | gauge | none | 1 if a model is loaded |
| `ael_model_info` | gauge | registered_name, version | Identity of the served model (value 1) |

`/metrics` itself is not counted. Metrics reset when the container restarts (counters restart from 0; PromQL `rate`/`increase` handle that).

### Dashboard panels
API status, model version, prediction rate, 5xx ratio, invalid (422) requests, p95 latency; requests per second by status; latency p50/p95/p99 of `/predict`; predictions per second by class; share of churn predictions; mean predicted probability; heatmap of predicted probabilities.

### Alert rules (`monitoring/alerts.yml`)

| Alert | Condition | For |
|---|---|---|
| ApiDown | Prometheus cannot scrape the API | 1 m |
| ModelNotLoaded | `ael_model_loaded == 0` | 1 m |
| HighServerErrorRate | more than 5% of requests are 5xx (5 m rate) | 5 m |
| HighLatencyP95 | p95 of `/predict` above 500 ms | 5 m |
| PredictionMixShift | more than 60% of predictions in 30 m are churn (training rate about 27%) | 10 m |

There is no Alertmanager: the rules show firing state in the Prometheus UI and document the thresholds. Wiring notifications is out of scope.

### What the operational metrics can and cannot tell you
They show the service is up, fast and answering, and give a coarse early signal that the *predictions* changed (share of churn, mean probability, probability heatmap). They do **not** show whether the *inputs* changed or whether the model is still *correct*; that needs the drift report (Phase 7) and, eventually, ground-truth churn labels.

### Demonstrating it
1. Normal traffic: `python scripts/load_generator.py --rate 5 --duration 300`. Share of churn predictions settles near the training rate.
2. Errors: add `--bad-fraction 0.2`; the 422 panel rises, 5xx stays at 0.
3. Input drift: `python scripts/load_generator.py --drift --rate 5 --duration 300`. The customer mix shifts to short-tenure, month-to-month, fiber, high-bill customers; the share of churn predictions and mean probability climb, and the heatmap moves toward 1. Phase 7 detects this from the logged inputs.

## 2. Data and model drift (Phase 7)

Not started. Plan: Evidently reports comparing the training data (reference) with the logged production inputs (`PREDICTION_LOG`), a drift-share threshold that triggers retraining (Phase 8), and an honest note about performance monitoring needing delayed labels.
