# Phase 7 Report: Data and Prediction Drift

- **Dates:** 2026-10-17 to in progress
- **Git tag:** v0.7-phase7 (after notebook 06 results are recorded)
- **Status:** In progress (code, tests, config, notebook and docs done; notebook 06 not yet run on real data)

## Objective
Detect when production inputs have moved away from the training data, with a verdict the retraining flow (Phase 8) can act on, and explain honestly what drift does and does not mean.

## Progress by step

| Step | Component | Status |
|---|---|---|
| 1 | Design decisions (ADR-008) | Done |
| 2 | `monitoring/drift.py`: per-feature KS/chi-square + PSI, dataset verdict, prediction drift | Done |
| 3 | `monitoring/prediction_log.py`: reads the API's JSONL log, skips bad lines | Done |
| 4 | `configs/monitoring.yaml` thresholds | Done |
| 5 | `scripts/check_drift.py`: training split vs logged inputs, JSON report, `--fail-on-drift` | Done; tested end to end |
| 6 | Tests (`tests/test_drift.py`, 34) | Done |
| 7 | Notebook 06 (scenarios, sensitivity table, performance check, Evidently HTML) | Written; dry-run on synthetic data; real run pending |
| 8 | Record real results here and in `docs/monitoring-and-drift.md`; screenshot the Evidently report | Pending |

## Bugs found while testing
- A constant reference column made PSI 0 whatever the current values (both fell in one bin). Fixed.
- A current column that was entirely missing crashed the chi-square test. Fixed: nothing to compare gives p = 1, PSI = 0.
- `TotalCharges` is text in the raw data, so it would have been tested as a 6,000-category column. Fixed: numbers stored as text are compared as numbers.

## Results
To be filled from notebook 06: the no-shift verdict, the sensitivity table (strength vs verdict), the performance table (shifted vs test) and the Evidently cross-check.

## Limitations
See ADR-008 consequences: conventional thresholds, per-feature tests, labels arrive late, the free host's log is ephemeral.
