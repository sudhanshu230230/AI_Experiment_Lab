# Phase 6 Report: Operational Monitoring (Prometheus and Grafana)

- **Dates:** 2026-10-16 to in progress
- **Git tag:** v0.6-phase6 (create after the local stack is verified)
- **Status:** In progress (code, config and tests done; first run of the compose stack pending)

## Objective
See whether the served model is up, fast, error-free and behaving consistently, with a reproducible local Prometheus and Grafana setup.

## Progress by step

| Step | Component | Status |
|---|---|---|
| 1 | Design decisions (ADR-007) | Done |
| 2 | Metrics in the API (`serving/metrics.py`, middleware, `/metrics`) | Done; 6 tests |
| 3 | Prometheus config and alert rules (`monitoring/`) | Done; YAML valid, all PromQL expressions parse |
| 4 | Grafana datasource, dashboard provisioning and the 12-panel dashboard | Done; JSON valid; not rendered here |
| 5 | `docker-compose.yml` (API, Prometheus, Grafana) | Written; not run here (no Docker daemon) |
| 6 | Traffic generator (`scripts/load_generator.py`, normal, `--drift`, `--bad-fraction`) | Done; verified against a live local server |
| 7 | Verify on the user's machine; add dashboard screenshot to the docs | Pending |

## How to run it
See `docs/monitoring-and-drift.md`.

## Tests of note
- Requests are counted by route template and status (200, 422, unmatched 404); `/metrics` is not self-counted.
- Each prediction in a batch increments the class counter and the probability histogram.
- Model-info and loaded gauges are correct with and without a model.
- Two app instances have independent registries.
- An unwritable prediction log does not turn a prediction into an error (found while adding the log to compose).

## Decisions made
See ADR-007.

## Known limitations
- Not verified end to end in a real Prometheus and Grafana (see ADR-007 consequences). If a panel is empty or a query errors, paste the panel's error and it will be fixed.
- Metrics reset on container restart; no Alertmanager; single worker only.
- Operational metrics cannot show input drift or accuracy (Phase 7 and later).

## Next
Phase 7: Evidently drift report from the logged inputs against the training data, with a threshold for the retraining trigger.
