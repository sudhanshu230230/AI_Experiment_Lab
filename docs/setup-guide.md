# Setup Guide

**Status:** Complete for Phase 0. Extended as phases add tooling.

## 1. One-time setup

1. Create a GitHub repository named `ai-experiment-lab` and push this project to it.
2. Create a Drive folder `MyDrive/ai-experiment-lab/` with subfolders `data/` and
   `mlflow/`. Colab will use these for datasets and tracking storage (ADR-001).
3. Place the dataset in `MyDrive/ai-experiment-lab/data/` (see ADR-003 for the source).

## 2. VS Code (local)

```bash
git clone https://github.com/<your-username>/ai-experiment-lab.git
cd ai-experiment-lab
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
ruff check .
```

Local machines run lightweight work only: unit tests, linting, API code, documentation.

## 3. Colab

Open `notebooks/00_colab_setup.ipynb` from GitHub (File > Open notebook > GitHub), or copy
its first cell to the top of any notebook. The cell:

1. Mounts Google Drive.
2. Clones (or pulls) the repo into `/content/ai-experiment-lab`.
3. Runs `pip install -e ".[ml,tracking]"`.
4. Points MLflow at the Drive-backed store.
5. Prints the commit hash, so each session records which code version it ran.

Edit `REPO_URL` in the cell before first use. For a private repo, use a GitHub token stored in
Colab Secrets (key icon in the sidebar). Never paste a token into a notebook cell.

## 4. Daily workflow

1. Write and test code in VS Code; commit and push.
2. In Colab, re-run the setup cell (it pulls the latest commit) and run the experiment.
3. Record results in `docs/experiments/` and update `docs/changelog.md`.
4. At the end of a phase, write the phase report and tag the release.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: ai_experiment_lab` in Colab | Running kernel did not pick up the editable install (fixed in setup cell: it adds `src` to `sys.path`); or runtime restarted | Re-run the setup cell |
| Code changes not visible in Colab | Old commit cloned | Push from VS Code, re-run setup cell (it runs `git pull`) |
| MLflow runs missing after a session | Store was on Colab local disk | Confirm the tracking URI points to Drive |
