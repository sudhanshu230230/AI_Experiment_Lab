# AI Experiment Lab

An end-to-end ML experimentation and MLOps platform: dataset validation, preprocessing,
multi-model experiments, hyperparameter optimization, experiment tracking, model registry,
API deployment, monitoring, drift detection and retraining.

> **Status:** Phase 0 complete (repo skeleton and documentation). See
> [`docs/changelog.md`](docs/changelog.md) for progress. Features are only listed as done
> once they are implemented and tested.

## How the work is split

| Where | What |
|---|---|
| **VS Code (local)** | All source code, configs, tests, API, Docker, CI, docs |
| **GitHub** | Single source of truth; bridge between VS Code and Colab |
| **Google Colab** | All ML execution: profiling, training, Optuna, drift reports |

Colab notebooks are thin runners. They clone this repo, `pip install -e .`, and call the
modules in `src/`. No real logic lives in notebooks.

## Quickstart (Colab)

Open any notebook in `notebooks/`, then run the setup cell described in
[`docs/setup-guide.md`](docs/setup-guide.md).

## Quickstart (local, VS Code)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Documentation

Start at [`docs/README.md`](docs/README.md). Key documents:

- [Project charter](docs/00-project-charter.md)
- [Architecture](docs/architecture.md)
- [ML lifecycle](docs/ml-lifecycle.md)
- [Setup guide](docs/setup-guide.md)
- [Architecture decisions (ADRs)](docs/decisions/)
- [Phase reports](docs/phases/)

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo skeleton, docs, Colab bootstrap | Done |
| 1 | Ingestion, profiling, validation, preprocessing, baselines | Done |
| 2 | MLflow experiment tracking | Planned |
| 3 | Optuna hyperparameter optimization | Planned |
| 4 | Model registry | Planned |
| 5 | FastAPI, Docker, deployment | Planned |
| 6 | Prometheus and Grafana monitoring | Planned |
| 7 | Evidently drift detection | Planned |
| 8 | Retraining with evaluation gates | Planned |
| 9 | CI/CD with GitHub Actions | Planned (basic lint/test workflow exists) |

## License

MIT. See [`LICENSE`](LICENSE).
