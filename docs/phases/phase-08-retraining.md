# Phase 8 Report: Retraining and Promotion

- **Dates:** 2026-10-17 to in progress
- **Git tag:** v0.8-phase8 (after notebook 07 is run and the redeploy workflow succeeds)
- **Status:** In progress (code, tests, notebook, workflow and docs done; notebook 07 and the workflow not yet run for real)

## Objective
Turn a drift verdict into a safer model in production: retrain, compare honestly with the champion, promote only if clearly better, and redeploy with a check that the right version is serving.

## Progress by step

| Step | Component | Status |
|---|---|---|
| 1 | Design decisions (ADR-009) | Done |
| 2 | `retraining/policy.py`: `should_retrain`, `decide_promotion`, `retrain_candidate`, `score`, `shifted_sample` | Done; 7 tests |
| 3 | Notebook 07 (simulate new data, candidate vs champion, log, register on promotion, export) | Written; dry-run end to end on synthetic data for both the reject and the promote path (MLflow registry, export, reload) |
| 4 | `scripts/check_model_export.py` (loadable, metadata, scikit-learn version vs Dockerfile) | Done; tested |
| 5 | `smoke_test_api.py --expect-version` | Done |
| 6 | `.github/workflows/redeploy.yml` (manual: verify, deploy hook, wait for version, smoke test) | Written; YAML valid; not run (needs the Render secret) |
| 7 | Run notebook 07 on real data; record the decision | Pending |
| 8 | Configure the GitHub secret and variable; run the workflow once | Pending |

## How to run it
`docs/retraining.md`.

## Results
To be filled from notebook 07 (verdict, comparison table, decision, registry version).

## Tests of note
- A drift verdict triggers; no-drift and insufficient data do not; force overrides; an unknown verdict raises.
- Promotion needs both the gain on shifted data and the guard on natural data; exactly-at-limit values pass (float tolerance); F1 alone never promotes.
- The candidate is an unfitted clone with the champion's hyperparameters, refit on old plus new data, and beats a stale champion on shifted data.
- The export check fails on a scikit-learn version mismatch, a missing folder and incomplete metadata.

## Limitations
See ADR-009 consequences: retraining is manual, new data is simulated and consumes the test set, small evaluation sets, margins are judgement calls.
