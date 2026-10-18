# Definition of Done check

**Status:** Complete (Phase 9, 2026-10-17). The 16 criteria come from the project context. "Met" means a reviewer can do it from the repository and its documentation; gaps are stated, not hidden.

| # | Criterion | Status | Evidence / how | Gap |
|---|---|---|---|---|
| 1 | Open the GitHub repository | Met | https://github.com/sudhanshu230230/AI_Experiment_Lab | none |
| 2 | Understand the architecture from the README | Met | README diagram and design rules; `docs/architecture.md`; 9 ADRs | none |
| 3 | Run the project from documented instructions | Met | README "Run it", `docs/setup-guide.md` | The dataset is not committed (a project rule, see `data/README.md`); its source and version are in ADR-003 |
| 4 | Upload/use a dataset | Partly | Any CSV with the configured target works through `configs/default.yaml`; validation and profiling run first | No upload UI; one dataset tested; classification only |
| 5 | Run experiments | Met | Notebooks 02, 03; EXP-001, EXP-002 | none |
| 6 | View MLflow runs | Met | Runs stored in SQLite on Drive; `compare_runs` table in notebooks; README shows `mlflow ui` on a database backup | The MLflow UI is not hosted |
| 7 | Compare models | Met | EXP-001 (baselines), EXP-003 (tuned): tables of CV and test metrics | none |
| 8 | Tune a model | Met | Notebook 04, ADR-004; 40 Optuna trials per model, nested MLflow runs | Search spaces not widened beyond edges noted in EXP-003 |
| 9 | Register a model | Met | Notebook 05: `telco-churn-classifier`, alias `champion`, lineage tag, reproduction check | Registry versions 1 to 3 are the same model (notebook was re-run) |
| 10 | Call the prediction API | Met | Live: https://churn-api-xop3.onrender.com/ (web page) and `/docs`; smoke test 6 of 6 | Free host sleeps (about a minute to wake); no authentication |
| 11 | View monitoring metrics | Met locally | `/metrics`; `docker compose up` gives Prometheus and a 12-panel Grafana dashboard; 5 alert rules | Not hosted; dashboard and alerts were validated statically (PromQL parses) and run by the author locally, but no screenshots are committed |
| 12 | Generate/check a drift report | Met | `scripts/check_drift.py`; notebook 06 results in the Phase 7 report; Evidently HTML saved to Drive | The Evidently report is produced in Colab only (Python 3.14 locally cannot install a current Evidently) |
| 13 | Understand how retraining happens | Met | `docs/retraining.md`, ADR-009; notebook 07 run on real data (candidate rejected by the rule) | Manual trigger; new data simulated from the test set; promote path run only on synthetic data |
| 14 | Inspect automated tests | Met | 160 tests; `docs/testing-strategy.md` | Notebooks are not unit-tested |
| 15 | See CI/CD configuration | Met | `.github/workflows/ci.yml` (lint, tests on two Python versions, Docker build and smoke test) and `redeploy.yml` (manual) | Redeploy ran for real on 2026-10-18 (run #1, commit `63ac185`): verify 1m49s, deploy 41s, both green |
| 16 | Understand the whole ML lifecycle | Met | `docs/ml-lifecycle.md` maps 13 stages to code, ADRs and reports | none |

## Items in the original vision that were deliberately not built

| Item | Why |
|---|---|
| Streamlit dashboard | The lifecycle, not a UI, is the point (charter non-goal); the notebooks and MLflow UI cover inspection |
| LightGBM, SVM, regression models | One dataset and one task; three model families were enough to show a fair comparison |
| PostgreSQL | SQLite on Drive is enough for a single writer (ADR-001) |
| Hosted Prometheus, Grafana, MLflow server | Cost and no extra portfolio value over the local stack (ADR-007) |
| Automatic retrain-and-deploy | Needs real labelled data and a human check (ADR-009) |
| Latency gate for promotion | Logistic regression latency is not a differentiator |
| Real-time per-model Evidently monitoring in production | Out of scope; drift check runs on the logged inputs |

## Known weaknesses a reviewer could raise
1. The three tuned models tie, so the champion is chosen on simplicity, not on a measured win (EXP-003, ADR-005).
2. The retraining demo consumed the original test set; no untouched final test remains.
3. Drift thresholds were adjusted after seeing one family of simulated shifts (ADR-008 amendment).
4. One dataset, one split.
5. The first-deployment plan targeted Hugging Face, which turned out to require a paid plan; the move to Render is recorded in ADR-006.
