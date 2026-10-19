# Documentation index

**Rule: a phase is not done until its documentation is written and committed with the code.**

| Document | Purpose | Status |
|---|---|---|
| [00-project-charter.md](00-project-charter.md) | Goals, scope, non-goals, success criteria, roadmap | Complete |
| [architecture.md](architecture.md) | System components, data flow, Colab / VS Code split | Complete (v1.0) |
| [ml-lifecycle.md](ml-lifecycle.md) | Dataset to retraining, stage by stage | Complete (v1.0) |
| [setup-guide.md](setup-guide.md) | VS Code, GitHub and Colab setup | Complete |
| [deployment.md](deployment.md) | Docker, cloud deployment, environment variables | Implemented (Phase 5) |
| [monitoring-and-drift.md](monitoring-and-drift.md) | Metrics, dashboards, drift policy | Implemented (Phases 6-7) |
| [any-dataset.md](any-dataset.md) | Train on your own CSV (binary, multi-class, regression) and deploy it | Implemented (ADR-010) |
| [retraining.md](retraining.md) | Retrain, promote, redeploy, rollback | Implemented (Phase 8) |
| [definition-of-done.md](definition-of-done.md) | The 16 portfolio-ready criteria with evidence and gaps | Complete (Phase 9) |
| [interview-prep.md](interview-prep.md) | Questions and project-grounded answers | Complete (Phase 9) |
| [resume.md](resume.md) | Resume entry limited to what is implemented | Complete (Phase 9) |
| [api-reference.md](api-reference.md) | Endpoints, schemas, example requests | Implemented (Phase 5) |
| [testing-strategy.md](testing-strategy.md) | What is tested and how | Current (Phase 9) |
| [changelog.md](changelog.md) | Dated log of changes | Active |

## Folders

| Folder | Contents |
|---|---|
| [decisions/](decisions/) | Architecture Decision Records. One file per decision; never deleted, only superseded. |
| [phases/](phases/) | One report per completed phase, written from `_template.md`. |
| [experiments/](experiments/) | One report per experiment round, written from `_template.md`. |

## Conventions

- Documents state their **Status** at the top and are updated in the same commit as the change they describe.
- Features are described as implemented only when they are implemented and tested.
- Decisions that cost real trade-offs get an ADR. Number them sequentially: `ADR-NNN-short-title.md`.
- Dates use ISO format (`YYYY-MM-DD`).
