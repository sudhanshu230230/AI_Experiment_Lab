# Phase 8 Report: Retraining and Promotion

- **Dates:** 2026-10-17 to in progress
- **Git tag:** v0.8-phase8 (after notebook 07 is run and the redeploy workflow succeeds)
- **Status:** Retraining run on real data (candidate rejected, as the rule dictates); redeploy workflow run pending

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
| 7 | Run notebook 07 on real data; record the decision | Done: candidate rejected |
| 8 | Configure the GitHub secret and variable; run the Redeploy workflow once (with the current champion; it checks the version served equals the one committed) | Pending |

## How to run it
`docs/retraining.md`.

## Results
Notebook 07, Colab, 2026-10-17. Trigger: `drift` verdict. Champion: registry version 3. Original training rows 5,634 plus 704 new (shifted) rows. Evaluation: 705 shifted rows (churn rate 0.413) and 705 natural rows (churn rate 0.265), disjoint from the new training rows.

| Model | Eval data | ROC AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Champion | shifted | 0.7572 | 0.6658 | 0.5408 | 0.8660 | 0.6411 |
| Champion | natural | 0.8326 | 0.6253 | 0.5214 | 0.7807 | 0.7518 |
| Candidate | shifted | 0.7615 | 0.6720 | 0.5519 | 0.8591 | 0.6539 |
| Candidate | natural | 0.8323 | 0.6311 | 0.5399 | 0.7594 | 0.7645 |

**Decision: KEEP the current champion.** ROC AUC on shifted data gained +0.0042 against a required +0.0050; the natural-data guard was fine (-0.0002, allowed -0.0100). The candidate run is logged in MLflow (run `c5b3406413a14e52aa913e3c0f65267f`, tag `retrain.promoted=False`) and nothing was registered; the champion stays at version 3.

How to read it:
- The rule worked as designed: a gain of 0.004 on 705 rows is within sampling noise, so declining to promote is the right call. The candidate was marginally better on every shifted metric except recall, and not worse on natural data, so retraining on new data is not harmful, only not yet demonstrably worth a deploy.
- The margin was **not** lowered after seeing this result. Changing a threshold so that the one run we have passes would defeat its purpose. A real deployment would accumulate more labelled rows (a larger evaluation set shrinks the noise) before reconsidering.
- Drift was real (Phase 7) and ROC AUC on shifted data (0.757) is well below natural (0.833), so the model *is* weaker on the new population. Retraining on 704 rows did not close that gap, which suggests the loss comes from the shifted population being intrinsically harder to rank with these features, not just from stale training data. This is a hypothesis, not tested here.
- The promote path (register, alias move, export, reload) was exercised only on synthetic data in a dry run; it has not run on the real registry.

## Tests of note
- A drift verdict triggers; no-drift and insufficient data do not; force overrides; an unknown verdict raises.
- Promotion needs both the gain on shifted data and the guard on natural data; exactly-at-limit values pass (float tolerance); F1 alone never promotes.
- The candidate is an unfitted clone with the champion's hyperparameters, refit on old plus new data, and beats a stale champion on shifted data.
- The export check fails on a scikit-learn version mismatch, a missing folder and incomplete metadata.

## Limitations
See ADR-009 consequences: retraining is manual, new data is simulated and consumes the test set, small evaluation sets, margins are judgement calls.
