# Phase 4 Report: Model Registry and Serving Export

- **Dates:** 2026-10-13 to 2026-10-17
- **Git tag:** v0.4-phase4
- **Status:** Complete

## Objective
Give the chosen model a versioned, traceable home (MLflow Model Registry) and produce a small
export that the API container can load without Drive or MLflow.

## Progress by step

| Step | Component | Status |
|---|---|---|
| 1 | Champion decision and registry workflow (ADR-005) | Done |
| 2 | Registry helpers (`registry/mlflow_registry.py`): find run, register, alias, load by alias, run summary | Done; tested |
| 3 | Serving export (`registry/export.py`): skops file + metadata, restricted trusted types | Done; tested |
| 4 | Notebook `05_model_registry.ipynb` | Done; run in Colab (see Results) |
| 5 | Commit `models/champion/` to the repo | Done (served by the deployed API, Phase 5) |

## How to run it
- Colab: `notebooks/05_model_registry.ipynb`. Run the register cell once (re-running makes a new version).
- Tests: `pytest` (152 tests at Phase 7).

## Results (Colab, 2026-10-17)
- Source run `5d24bb7c861d4b1bb581f694ce6e74e9` (tuned logistic regression, code `2c568f9`, dataset sha `88be4b93fbe0`): CV F1 0.6307, test F1 0.6195.
- Registered as `telco-churn-classifier`, alias `champion`, with lineage tag `source_run_id` pointing at that run.
- Loaded back by alias, the model reproduced the logged result exactly: test F1 0.6195 and ROC AUC 0.8399 (`REGISTRY MODEL REPRODUCES RUN`).
- The register cell was run three times, so the registry holds versions 1 to 3, all built from the same run; the alias now points at version 3. This is harmless (identical model), but it is the behaviour the notebook warns about: every re-run adds a version. The deployed API was exported when version 1 was the champion, so its `model_version` reads 1 although the alias has moved; the predictions are identical because the source run is the same. A future retrain (Phase 8) will refresh the export and the version together.
- MLflow warned that the run had no artifact at path `model` and registered from the logged-model entity instead (MLflow 3 stores models as logged-model entities); the registered version is correct, as the reproduction check shows.

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
