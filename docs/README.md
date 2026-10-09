# Documentation index

**Rule: a phase is not done until its documentation is written and committed with the code.**

| Document | Purpose | Status |
|---|---|---|
| [00-project-charter.md](00-project-charter.md) | Goals, scope, non-goals, success criteria, roadmap | Complete |
| [architecture.md](architecture.md) | System components, data flow, Colab / VS Code split | Complete (v0.1) |
| [ml-lifecycle.md](ml-lifecycle.md) | Dataset to retraining, stage by stage | Complete (v0.1) |
| [setup-guide.md](setup-guide.md) | VS Code, GitHub and Colab setup | Complete |
| [deployment.md](deployment.md) | Docker, cloud deployment, environment variables | Not started (Phase 5) |
| [monitoring-and-drift.md](monitoring-and-drift.md) | Metrics, dashboards, drift policy | Not started (Phases 6-7) |
| [api-reference.md](api-reference.md) | Endpoints, schemas, example requests | Not started (Phase 5) |
| [testing-strategy.md](testing-strategy.md) | What is tested and how | Draft |
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
