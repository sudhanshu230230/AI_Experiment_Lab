# Changelog

Dated log of project changes, newest first. Entries describe what exists, not what is planned.

## 2026-10-10: Phase 0 documentation and Colab bootstrap

- Added documentation set: charter, architecture, ML lifecycle, setup guide, testing strategy.
- Added ADR-001 (MLflow storage, proposed), ADR-002 (model handoff), ADR-003 (initial dataset and selection metric).
- Added ADR, phase-report and experiment-report templates.
- Added `notebooks/00_colab_setup.ipynb` (Drive mount, clone, install, tracking URI).
- Added `LICENSE` (MIT), `.gitignore`, `pyproject.toml` optional dependency groups per phase.
- Added `docs/phases/phase-00-skeleton.md`.

## Earlier (Phase 0 skeleton)

- Created package layout under `src/ai_experiment_lab/`, smoke test, and basic lint/test CI workflow.
