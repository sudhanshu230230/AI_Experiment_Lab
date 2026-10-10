# Changelog

Dated log of project changes, newest first. Entries describe what exists, not what is planned.

## 2026-10-10: Phase 1 started, ingestion and profiling

- Added `data/ingestion.py`: CSV loading, whitespace/blank/numeric-text normalization, SHA-256 dataset version, `DatasetMetadata`.
- Added `data/profiling.py`: `profile_dataset` (shape, types, missing, duplicates, outliers, categorical summaries, target imbalance, correlations) with a text `summary()`.
- Added `tests/test_data.py` (21 tests on synthetic data) and `notebooks/01_data_exploration.ipynb`.
- Added `docs/phases/phase-01-mvp.md` (in progress).

## 2026-10-10: Phase 1 started (ingestion and profiling)

- Added `data/ingestion.py` (CSV loading, SHA-256 dataset version, `DatasetInfo`).
- Added `data/profiling.py` (shape, types, missing vs blank, duplicates, outliers, correlations, target imbalance, numeric-like text detection, text report).
- Added `utils/config.py` (YAML loading, data directory resolution).
- Added `tests/test_data.py` (18 tests with config) and `notebooks/01_data_exploration.ipynb`.
- Added `docs/phases/phase-01-data-and-baselines.md` (in progress).

## 2026-10-10: Phase 0 documentation and Colab bootstrap

- Added documentation set: charter, architecture, ML lifecycle, setup guide, testing strategy.
- Added ADR-001 (MLflow storage, proposed), ADR-002 (model handoff), ADR-003 (initial dataset and selection metric).
- Added ADR, phase-report and experiment-report templates.
- Added `notebooks/00_colab_setup.ipynb` (Drive mount, clone, install, tracking URI).
- Added `LICENSE` (MIT), `.gitignore`, `pyproject.toml` optional dependency groups per phase.
- Added `docs/phases/phase-00-skeleton.md`.
- Fixed Colab setup cell: add `src/` to `sys.path`, because a running kernel does not load the editable-install `.pth` file (found on first Colab run).

## Earlier (Phase 0 skeleton)

- Created package layout under `src/ai_experiment_lab/`, smoke test, and basic lint/test CI workflow.
