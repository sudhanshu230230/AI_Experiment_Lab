# Setup Guide

**Status:** Complete. Written for a fresh Windows machine that already has Git and VS Code;
macOS and Linux differences are in section 9. Update this document whenever tooling changes.

## 1. Overview

| Where | Role |
|---|---|
| **VS Code (this machine)** | Write code, tests, docs; run unit tests and lint |
| **GitHub** | Single source of truth; the bridge between this machine and Colab |
| **Google Colab** | Runs all ML work. Needs nothing from this machine except what is pushed to GitHub |
| **Google Drive** | Holds the dataset, results and (Phase 2) the MLflow store |

Datasets and model files are never in Git. Moving to a new machine therefore means: install
tools, clone the repo, create the Python environment. Nothing else is copied.

## 2. Before you start: do not lose work from the old machine

On the **old** machine, from the project folder:

```powershell
git status
git log origin/main..HEAD --oneline
```

`git status` must say `nothing to commit, working tree clean`, and the second command must
print nothing. If either shows something, commit and `git push` first. Anything not pushed
exists only on the old machine.

## 3. Verify the tools

Open PowerShell and run:

```powershell
git --version
code --version
py --list
```

| Tool | Expected | If missing |
|---|---|---|
| Git | `git version 2.x` | Reinstall Git for Windows from git-scm.com, keep "Git Credential Manager" ticked |
| VS Code | a version number and a commit hash | In VS Code: `Ctrl+Shift+P`, "Shell Command: Install 'code' command in PATH" (or reinstall and keep "Add to PATH" ticked) |
| Python | a 3.11 or 3.12 entry in `py --list` | Section 4 |

Close and reopen PowerShell after installing anything, so the new PATH is picked up.

## 4. Install Python

Use **Python 3.12**. The project requires 3.10 or newer, and the newest release (3.14 at the time of
writing) also works for the current phases: a dry-run check on 2026-10-11 showed pandas, scikit-learn,
XGBoost, LightGBM, MLflow, Optuna and FastAPI all install on 3.14 for Windows. The exception is
**Evidently** (Phase 7, drift reports): on 3.14, pip resolves only the old 0.4.19 instead of the
current 0.7.23. 3.12 avoids that problem. The virtual environment is disposable, so if you start on
another version you can delete `.venv` and recreate it later.

```powershell
winget install --id Python.Python.3.12 -e
```

If `winget` is unavailable, download the installer from python.org and **tick "Add python.exe
to PATH"** on the first screen. Reopen PowerShell, then check:

```powershell
py -3.12 --version
```

## 5. Configure Git

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global --list
```

Commits show this email publicly on GitHub. To keep your real address private, use the
`ID+username@users.noreply.github.com` address shown in GitHub under Settings > Emails, and
tick "Keep my email addresses private".

## 6. Get the code

```powershell
cd $HOME\Documents
git clone https://github.com/<your-username>/ai-experiment-lab.git
cd ai-experiment-lab
code .
```

On the first clone or push, Git Credential Manager opens a browser window to sign in to GitHub.
Approve it once; Windows stores the credential. Do not use your GitHub password in the terminal
(GitHub no longer accepts it); the browser sign-in or a personal access token is required.

Confirm you have the latest work:

```powershell
git log --oneline -5
git status
```

## 7. Create the Python environment

From the project folder, in the VS Code terminal (`` Ctrl+` ``):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The prompt now starts with `(.venv)`. If `py -3.12` says "No suitable Python runtime found", Python 3.12 is
not installed: go back to section 4, then close and reopen the terminal. If activation is blocked
("running scripts is disabled"):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**Before installing, confirm the environment is really in use** (otherwise pip installs into your global
Python):

```powershell
python --version
(Get-Command python).Source
```

The first must show 3.12.x and the second must end in `\.venv\Scripts\python.exe`. Then install:

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev,ml]"
```

Dependency groups (defined in `pyproject.toml`); add groups as phases need them:

| Group | Install when | Contents |
|---|---|---|
| `dev` | Always | pytest, ruff |
| `ml` | Phase 1 onwards | xgboost, lightgbm, matplotlib |
| `tracking` | Phase 2 onwards (running MLflow code locally) | mlflow, optuna |
| `serve` | Phase 5 | fastapi, uvicorn, pydantic |
| `monitoring` | Phases 6-7 | prometheus-client, evidently |

Example, later: `pip install -e ".[dev,ml,tracking,serve]"`. Colab installs its own copy; it
does not use this environment.

## 8. Configure VS Code

1. When VS Code offers to install the **recommended extensions**, accept (Python, Pylance,
   Jupyter, Ruff). Or run:
   ```powershell
   code --install-extension ms-python.python
   code --install-extension ms-python.vscode-pylance
   code --install-extension ms-toolsai.jupyter
   code --install-extension charliermarsh.ruff
   ```
2. Select the interpreter: `Ctrl+Shift+P`, "Python: Select Interpreter", choose the one inside
   `.venv`. The bottom-right status bar should show it.
3. The repository's `.vscode/settings.json` already enables pytest in the Testing panel, activates
   the venv in new terminals, and uses Ruff for lint and fixes on save.
4. Notebooks are written to run in Colab. If you open one locally, pick the `.venv` kernel for
   reading or editing only; do not run them locally (they mount Google Drive).

### Verify the setup

```powershell
pytest
ruff check .
```

Expected: all tests pass (67 at the end of Phase 1) and `All checks passed!`. The same commands
run in CI on every push.

## 9. macOS and Linux differences

| Step | Windows | macOS / Linux |
|---|---|---|
| Create environment | `py -3.12 -m venv .venv` | `python3 -m venv .venv` |
| Activate | `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| Install Python | `winget` or python.org | `brew install python@3.12` or the distro package manager |
| Folder variable | `$HOME\Documents` | `~/Documents` |

All other commands are identical.

## 10. Colab and Drive

Nothing about Colab depends on this machine. It needs:

1. The **same Google account** that owns the Drive folder `MyDrive/ai-experiment-lab/` (with
   `data/`, `results/`).
2. The repository on GitHub, up to date with your last push.
3. If the repo is private: a GitHub token saved in Colab Secrets (key icon) named `GITHUB_TOKEN`.
   Secrets belong to the Google account, so they carry over; if you create a new token, replace
   the old value.

Open any notebook from GitHub (File > Open notebook > GitHub) and run the first cell. It pulls the
latest code, installs the package and prints the commit it is running.

## 11. Daily workflow

1. `git pull` (always first, in case Colab-side or other-machine changes were pushed).
2. Edit in VS Code; run `pytest` and `ruff check .`.
3. `git add .`, `git commit -m "message"`, `git push`.
4. In Colab, re-run the setup cell (it runs `git pull`) and run the experiment.
5. Record results in `docs/experiments/` and `docs/changelog.md`; at the end of a phase, write the
   phase report and tag the release (`git tag v0.N-phaseN`, `git push origin v0.N-phaseN`).

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `git` or `code` not recognised | PATH not refreshed | Close and reopen PowerShell; reinstall with "Add to PATH" |
| `py` not recognised | Python launcher not installed | Reinstall Python from python.org with the launcher ticked |
| `No suitable Python runtime found` for `py -3.12` | Only another version (for example 3.14) is installed | `winget install --id Python.Python.3.12 -e`, reopen the terminal, check `py -0` |
| Activate script "module could not be loaded" | The `.venv` folder was not created (venv step failed) | Fix the Python error above, rerun `py -3.12 -m venv .venv`, use `.\.venv\Scripts\Activate.ps1` |
| `pip` output mentions `Python314\\Lib\\site-packages` | Installing into the global Python, venv not active | `deactivate` if needed; recreate and activate the venv; check `(Get-Command python).Source` |
| Activation blocked | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `pip install` fails building a package, or installs an old library version | Python version too new for a dependency wheel | Delete `.venv`, recreate it with `py -3.12` |
| `pytest` not found | venv not active | Run the activate command; the prompt must show `(.venv)` |
| `ModuleNotFoundError: ai_experiment_lab` locally | Package not installed | `pip install -e ".[dev,ml]"` with the venv active |
| Push asks for a password or fails | Password auth is not supported | Sign in through the browser prompt, or use a personal access token |
| `Permission denied` / 403 on push | Signed in to the wrong GitHub account | Windows Credential Manager > remove the `git:https://github.com` entry, push again |
| VS Code shows wrong interpreter | Not selected | `Ctrl+Shift+P`, "Python: Select Interpreter", pick `.venv` |
| Colab shows old code | Latest work not pushed | `git push`, then re-run the Colab setup cell |
| Colab: `ModuleNotFoundError` | Kernel did not pick up the install | Re-run the setup cell (it adds `src/` to `sys.path`) |

## 13. Not needed on this machine

- **Docker:** built and tested in CI or Codespaces (Phase 5); a laptop is not required to run it.
- **The dataset:** it lives on Drive. Local tests use small synthetic data.
- **GPU or heavy compute:** all training runs in Colab.
