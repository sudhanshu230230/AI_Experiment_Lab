# ADR-007: Monitoring design

- **Status:** Accepted
- **Date:** 2026-10-16
- **Phase:** 6

## Context
We want request, latency, error and prediction-behaviour visibility for the served model, runnable on a low-powered laptop, within the project schedule.

## Decision
- **Prometheus + Grafana locally** (docker-compose), not hosted. The hosted API exposes `/metrics`, but running Prometheus and Grafana online adds cost and accounts for little portfolio value over screenshots of the local stack.
- **Instrument in the app** with `prometheus_client`, one registry per app instance (tests create many apps; the global registry would raise duplicate-series errors).
- **Bounded labels:** the path label is the route template, never the raw URL; unknown URLs share `unmatched`. Predictions are labelled only by class. No per-customer or per-feature labels (cardinality, privacy).
- **Prediction-behaviour metrics, not just HTTP metrics:** class counts, a probability histogram and the model identity. These give a cheap early signal of drift in the outputs; input drift is tested properly in Phase 7.
- **Provisioned, version-controlled dashboard and alert rules** (`monitoring/`), so the setup is reproducible and reviewable. No Alertmanager (out of scope).
- **Single uvicorn worker**, so process-local metrics are correct. Multiple workers would need prometheus multiprocess mode.
- **Prediction log failures never fail a request** (a full disk or read-only mount is logged, the prediction is still returned).

## Consequences
- Metrics are lost when the container restarts (counters reset); acceptable for a demo.
- `/metrics` is public on the hosted Space. It exposes counts and latency only, no inputs.
- The dashboard JSON and compose stack could not be run in the authoring environment (no Docker daemon). They were checked statically: all YAML and JSON parse, and all 19 PromQL expressions (dashboard and alerts) parse with a PromQL parser; the metrics endpoint was checked against a live local server. First real run is on the user's machine.
