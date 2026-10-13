# Phase 4 Report: Model Registry and Serving Export

- **Dates:** 2026-10-13 to in progress
- **Git tag:** v0.4-phase4 (create after the Colab run is confirmed)
- **Status:** In progress (code and tests done; Colab run of notebook 05 pending)

## Objective
Give the chosen model a versioned, traceable home (MLflow Model Registry) and produce a small
export that the API container can load without Drive or MLflow.

## Progress by step

| Step | Component | Status |
|---|---|---|
| 1 | Champion decision and registry workflow (ADR-005) | Done |
| 2 | Registry helpers (`registry/mlflow_registry.py`): find run, register, alias, load by alias, run summary | Done; tested |
| 3 | Serving export (`registry/export.py`): skops file + metadata, restricted trusted types | Done; tested |
| 4 | Notebook `05_model_registry.ipynb` | Written; ran locally on synthetic data; Colab run pending |
| 5 | Commit `models/champion/` to the repo | Pending the Colab download |

## How to run it
- Colab: `notebooks/05_model_registry.ipynb`. Run the register cell once (re-running makes a new version).
- Tests: `pytest` (90 tests).

## Tests of note
- Model loaded by alias gives identical predictions to the run's model.
- A new version moves the alias; the old version stays; rollback restores it.
- Export round-trip without MLflow gives identical predictions; metadata records input columns.
- A tampered `metadata.json` that lists an untrusted type is refused at load.

## Decisions made
See ADR-005 (champion = tuned Logistic Regression; aliases; export committed to git).

## Known limitations
- The registry lives in the same SQLite file on Drive as the runs (ADR-001), so it has the same single-writer rule.
- The exported model is a snapshot: it does not move when the alias moves until the export is redone.

## Next
Phase 5: FastAPI service that loads `models/champion/`, Docker image, deployment.
