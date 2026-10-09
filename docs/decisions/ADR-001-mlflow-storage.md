# ADR-001: MLflow tracking and artifact storage

- **Status:** Proposed (validate with a spike at the start of Phase 2)
- **Date:** 2026-10-10
- **Phase:** 2

## Context
Training runs in Google Colab, whose local disk is wiped when the runtime ends. MLflow
runs, parameters, metrics and model artifacts must survive between sessions and be
reachable later from the API and dashboard. No paid infrastructure is available.

## Options considered
1. **Backend store and artifacts on Google Drive.** Free, simple, fully under our control.
   Risks: Drive is a network mount, so SQLite file locking may be unreliable, and
   concurrent sessions could corrupt the store. The MLflow file-based backend may also be
   deprecated in current versions; check the MLflow docs before choosing.
2. **Hosted MLflow (for example DagsHub).** Persistent, shareable, UI always available.
   Costs: third-party dependency, account setup, free-tier limits.
3. **Self-hosted MLflow server on a free cloud tier.** Closest to production. Costs: more
   setup, free tiers may sleep or reset storage.
4. **Local disk only.** Rejected: data is lost with every Colab session.

## Decision (proposed)
Start with option 1 (Drive-backed store) for Phase 2, keeping the tracking URI in config
so the backend can change without code changes. Run a short spike first to confirm the
store is stable on Drive. If it is not, switch to option 2.

## Consequences
- Tracking URI is configuration (`configs/default.yaml`), never hardcoded.
- Only one Colab session should write to the store at a time.
- A switch to a hosted or self-hosted server later is a config change plus a data migration.

## Interview note
Colab is ephemeral, so I put MLflow state on persistent storage and kept the URI in
config. That lets me move from Drive to a hosted server as the platform grows.
