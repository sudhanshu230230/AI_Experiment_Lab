# Monitoring and Drift

**Status:** Monitoring implemented (Phase 6). Drift detection implemented (Phase 7); real-data results pending.

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

Decisions: ADR-008. Settings: `configs/monitoring.yaml`.

### What is compared
Reference = the training split. Current = logged production inputs (the API writes them when `PREDICTION_LOG` is set; `docker compose` does this). Each feature gets a KS test (numeric) or chi-square test (categorical) and a PSI. A feature has **drifted** only if PSI >= 0.1 **and** p < 0.05 (lowered from the usual 0.2 after testing on real data; ADR-008). The dataset verdict is:

| Verdict | Meaning |
|---|---|
| `no_drift` | fewer than 30% of features drifted |
| `drift` | 30% or more drifted: candidate trigger for retraining (Phase 8) |
| `insufficient_data` | fewer than 100 current rows; no decision is made |

The mean/shape of the predicted probability is also tested and reported, but does not affect the verdict.

### Run it on logged traffic
```powershell
docker compose up --build -d
python scripts/load_generator.py --drift --rate 10 --duration 120     # shifted customers
python scripts/check_drift.py --data data/raw/Telco-Customer-Churn.csv `
    --log data/predictions/predictions.jsonl --out reports/drift.json
```
Add `--fail-on-drift` to exit with code 1 on a `drift` verdict (used by the retraining workflow). `--model-version N` limits the check to one model version.

### Readable report in Colab
`notebooks/06_drift_detection.ipynb` runs the scenarios, a sensitivity table (shift strength vs verdict), a drift-versus-performance check and an Evidently HTML report saved to Drive (`reports/drift/`).

### What drift does and does not mean
- Drift says the **inputs** changed. It does not say the model got worse: a shifted population can leave ranking quality (ROC AUC) intact while changing the churn rate, precision and recall.
- Whether the model is still *right* needs true labels, which for churn arrive weeks later. So drift is the early warning and performance monitoring is the delayed confirmation.
- Features are tested one by one; a change in how features combine is not seen.
- Measured here: on a simulated shift toward month-to-month/fiber/short-tenure customers, F1 *rose* (0.619 to 0.681, because the churn rate went from 0.27 to 0.43) while ROC AUC *fell* (0.840 to 0.761). Watch ROC AUC, not F1 alone.
- Results on real data: see the Phase 7 report. In short, at PSI 0.2 even a very large shift flagged only 3 of 19 features, which is why the threshold is 0.1.

## Models trained on other datasets (ADR-010)
Drift detection works on any columns (numeric and categorical are told apart from the data). `python scripts/check_drift.py --data my.csv --model-dir models/champion --log <log>` takes the feature list from the model's schema. The prediction log records a `score` per row (probability of the positive class, top-class probability, or the predicted number). The Grafana dashboard keeps its Telco metric names; its probability panels only fill for binary models.

