# Phase 0 Report: Skeleton, Documentation and Colab Bootstrap

- **Dates:** 2026-10-09 to 2026-10-10
- **Git tag:** v0.0-phase0 (to be created)
- **Status:** Complete

## Objective
Create a repository that installs cleanly, has CI, has a documentation system, and can be
run from Google Colab with a single setup cell.

## What was built
- Package layout under `src/ai_experiment_lab/` (empty modules, version constant).
- Smoke test and a CI workflow (ruff lint and pytest).
- Documentation set, ADR system, phase and experiment templates, changelog.
- `notebooks/00_colab_setup.ipynb` (Drive mount, clone, install, MLflow URI).
- Dependency groups per phase in `pyproject.toml`.

## How to run it
See `docs/setup-guide.md`.

## Results
No ML results yet. Phase 0 delivers structure only.

## Tests
`tests/test_smoke.py` verifies the package imports and exposes a version.

## Decisions made
ADR-001 (proposed), ADR-002, ADR-003.

## Known limitations
- The Colab notebook has not been run against a real GitHub repository yet; `REPO_URL`
  is a placeholder.
- ADR-001 depends on a Phase 2 spike.

## Next phase
Phase 1: ingestion, profiling, validation, preprocessing and baseline models on Telco churn.
