# Retraining and redeploying

**Status:** Implemented (Phase 8). Decisions: ADR-009. Real-data results: see the Phase 8 report.

## The loop

```
drift verdict (Phase 7)  ->  notebook 07: candidate vs champion  ->  promote? ->  champion.zip
        ->  commit models/champion/  ->  "Redeploy" workflow  ->  Render serves the new version
```

## 1. Decide and retrain (Colab)
1. Run notebook 06 so `reports/drift/drift_shifted.json` on Drive holds the latest verdict.
2. Run `notebooks/07_retraining.ipynb`. It stops if the verdict is not `drift` (set `FORCE = True` to override). It builds the candidate, compares it with the champion on shifted and natural data, prints the decision, logs the run, and **only if promoted** registers a new version, moves the `champion` alias and downloads `champion.zip`.
3. Run each cell once. A promoted run adds a registry version every time it executes.

## 2. Put the new model in the repo (Windows)
```powershell
cd D:\Projects\AI_Experiment_Lab
Remove-Item models\champion\* -Force
Expand-Archive -Force $HOME\Downloads\champion.zip models\champion
python scripts\check_model_export.py          # must print OK, with the new version number
git add models/champion
git commit -m "Promote champion version N after drift retrain"
git push
```
`check_model_export.py` fails if the metadata is incomplete, or if the model was trained with a different scikit-learn than the Dockerfile installs (then change `ARG SKLEARN_VERSION` in the Dockerfile to match). It loads the model only when the installed scikit-learn equals the training version; otherwise it prints `SKIP load check`, because a model saved by one scikit-learn version often cannot be loaded by another. The real load test is the Docker smoke test, which runs the pinned version.

## 3. Redeploy (GitHub Actions, manual)
One-time setup in the GitHub repository, **Settings > Secrets and variables > Actions**:
- Secret `RENDER_DEPLOY_HOOK_URL`: in Render, the service's **Settings > Deploy Hook**. Treat it like a password; never paste it in chat or commit it.
- Variable `RENDER_SERVICE_URL`: `https://churn-api-xop3.onrender.com`

Then **Actions > Redeploy > Run workflow**. It runs lint and tests, checks the export, triggers the Render deploy, waits until `/model` reports the committed `registry_version`, and runs the smoke test with `--expect-version`. A red run leaves the old version serving.

## Rollback
Move the alias back (`champion` -> previous version in the MLflow registry), re-export the previous version (notebook 05's export cell, or `git revert` the `models/champion/` commit), push, and run Redeploy. The previous version is never deleted.

## Promotion rule in one line
Promote only if ROC AUC on shifted data improves by at least 0.005 and ROC AUC on natural data does not drop by more than 0.01.

## Limits
Retraining is manual and runs in Colab. New labelled data is simulated from the test set (which this consumes). Evaluation sets are small, so small differences are noise and "keep the champion" is a common, safe outcome.
