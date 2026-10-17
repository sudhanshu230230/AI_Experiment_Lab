# Phase 7 Report: Data and Prediction Drift

- **Dates:** 2026-10-17 to in progress
- **Git tag:** v0.7-phase7
- **Status:** Complete

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
| 7 | Notebook 06 (scenarios, sensitivity table, performance check, Evidently HTML) | Run in Colab on real data (twice: before and after the threshold change); Evidently report saved |
| 8 | Threshold amendment from the results (ADR-008) | Done; confirmed by the re-run |
| 9 | Performance table (drift vs accuracy) | Done |

## Bugs found while testing
- A constant reference column made PSI 0 whatever the current values (both fell in one bin). Fixed.
- A current column that was entirely missing crashed the chi-square test. Fixed: nothing to compare gives p = 1, PSI = 0.
- `TotalCharges` is text in the raw data, so it would have been tested as a 6,000-category column. Fixed: numbers stored as text are compared as numbers.

## Results (notebook 06, Colab, 2026-10-17)

Reference: 5,634 training rows, 19 features.

| Scenario | Rows | Drifted features (PSI 0.2) | Verdict | Prediction drift |
|---|---|---|---|---|
| Untouched test set | 1,409 | 0 / 19 (max PSI 0.0125) | no_drift | no (PSI 0.016) |
| Bootstrap, strength 0 | 1,000 | 0 / 19 | no_drift | no |
| Strength 1 | 1,000 | 0 / 19 | no_drift | no |
| Strength 3 and 10 | 1,000 | 2 / 19 | no_drift | yes |
| Strength 30 to 10,000 | 1,000 | 3 / 19 | no_drift | yes (PSI 1.24 at 10,000) |

Mean predicted probability rose from 0.42 (strength 0) to 0.61 (strength 100 and above).

**Finding: the first thresholds were too lax.** Even the most extreme sample (customers drawn almost only from month-to-month, fiber, short-tenure) flagged 3 of 19 features at PSI 0.2 (Contract 0.52, tenure 0.32, InternetService 0.21), below the 30% share needed for `drift`, while the model's own output had moved a lot (prediction PSI 1.24) and Evidently reported 13 of 19 columns drifted (68%; its dataset threshold is 50%). Seven further features sat at PSI 0.10 to 0.17. The PSI threshold was therefore lowered to 0.1 (ADR-008 amendment): the extreme sample then has 10 / 19 drifted (53%) while the untouched test set stays at 0 (its largest PSI is 0.0125). Caveat recorded in the ADR: this threshold was chosen after seeing this family of shifts.

Evidently HTML report: `reports/drift/evidently_shifted.html` on Drive (3.9 MB; not committed).

### After the change: re-run with PSI 0.1 (same data, same samples)

| Strength | Drifted at PSI 0.1 | Verdict | Drifted at PSI 0.2 | Verdict at 0.2 | Mean score |
|---|---|---|---|---|---|
| 0 (bootstrap control) | 0 / 19 | no_drift | 0 | no_drift | 0.417 |
| 1 | 2 / 19 | no_drift | 0 | no_drift | 0.528 |
| 3 | 4 / 19 (21%) | no_drift | 2 | no_drift | 0.567 |
| 10 | 9 / 19 (47%) | **drift** | 2 | no_drift | 0.603 |
| 30 to 10,000 | 10 / 19 (53%) | drift | 3 | no_drift | 0.612 to 0.614 |

The verdict flips between strength 3 and 10; the bootstrap control and the untouched test set stay at 0. At strength 10 the drifted features were tenure, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Contract, MonthlyCharges and TotalCharges; prediction drift PSI was 0.716. At PSI 0.2 the same sample is `no_drift` at every strength, which is the miss that motivated the change. The insufficient-data guard works: 30 rows gave `insufficient_data`.

Evidently on the strength-10 sample: 13 of 19 columns drifted (68%). Report: `reports/drift/evidently_shifted.html` on Drive (3.9 MB, not committed).

### Drift versus performance (strength-10 sample, labels known because this is a simulation)

| | F1 | ROC AUC | Precision | Recall | Churn rate |
|---|---|---|---|---|---|
| Test set | 0.619 | 0.840 | 0.512 | 0.783 | 0.265 |
| Shifted sample | 0.681 | 0.761 | 0.561 | 0.869 | 0.426 |

**The lesson: F1 went up while the model got worse.** The shifted population has a much higher churn rate (0.43 versus 0.27), which lifts precision, recall and F1 mechanically. ROC AUC, which does not depend on the churn rate, fell from 0.840 to 0.761: the model ranks customers less well on this population. A dashboard watching F1 would have said "better"; the drift verdict and ROC AUC both say "something changed and ranking quality dropped". Production monitoring of this model should track ROC AUC (or compare against the churn rate) rather than F1 alone, once labels arrive. This is one simulated shift; it shows the two can diverge, not how large the loss will be in general.

## Limitations
See ADR-008 consequences: conventional thresholds, per-feature tests, labels arrive late, the free host's log is ephemeral.
