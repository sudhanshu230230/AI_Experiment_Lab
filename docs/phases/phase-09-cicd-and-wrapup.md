# Phase 9 Report: CI/CD and Project Wrap-up

- **Dates:** 2026-10-17
- **Git tag:** v1.0
- **Status:** Complete (repository side). The redeploy workflow still needs its first real run (needs the Render secret)

## Objective
Finish the automation (CI plus a safe redeploy path) and make the project reviewable: accurate README, a Definition of Done audit, resume text limited to what exists, and interview preparation.

## Progress by step

| Step | Component | Status |
|---|---|---|
| 1 | CI: ruff, pytest on Python 3.11 and 3.13, committed-export check, Docker build and smoke test; least-privilege permissions; cancel superseded runs | Done |
| 2 | Redeploy workflow (manual): verify, Render hook, wait for the committed version, smoke test with `--expect-version` | Written in Phase 8; first run pending |
| 3 | README rewritten (it still described Phase 0): architecture, results, run instructions, limits | Done |
| 4 | Architecture, ML lifecycle, testing strategy and docs index brought up to date | Done |
| 5 | `docs/definition-of-done.md`: 16 criteria with evidence and gaps, items deliberately not built, known weaknesses | Done |
| 6 | `docs/resume.md`: entry plus an explicit "not to claim" list | Done |
| 7 | `docs/interview-prep.md`: questions grounded in measured results, plus mistakes worth telling | Done |

## Findings from the Definition of Done audit
- 14 of 16 criteria are fully met, 2 partly or with a stated gap (dataset upload, local-only monitoring). Details and the reasons are in the audit.
- The README had been left at its Phase 0 text through all later phases. The rule "a phase is not done until its docs are committed" had not covered the README; it is now part of the wrap-up checklist.

## What to do after merging
1. Confirm CI is green on both Python versions and the Docker job.
2. In GitHub, add the secret `RENDER_DEPLOY_HOOK_URL` and the variable `RENDER_SERVICE_URL`; run **Redeploy** once.
3. Optional evidence for reviewers: a Grafana dashboard screenshot and the MLflow UI comparison page in `docs/img/`.
4. Tag `v1.0`.

## Limitations
See the audit. Highest-impact: retraining is manual and simulated; one dataset and split; monitoring is local.
