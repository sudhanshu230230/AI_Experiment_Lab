# Using the platform on your own dataset

**Status:** Implemented (ADR-010). Code: `src/ai_experiment_lab/automl/`, `scripts/train_any.py`, `notebooks/08_any_dataset.ipynb`.

Any CSV with a header row and one column to predict. The platform infers what kind of problem it is, trains and compares models, and exports a folder that the API and the web page serve with **no code changes**.

## What it supports

| Target column | Task | Output |
|---|---|---|
| two values (Yes/No, 0/1, ...) | binary classification | label, probability of the positive class, threshold |
| 3 to 50 classes (text, or integer codes with up to 20 values) | multi-class classification | label and a probability for every class |
| a number with many distinct values | regression | a number |

Override the guess with `--task` (or `TASK` in the notebook). Not supported: images, free text, time-series forecasting, multi-label targets.

## Two ways to train

**Colab (recommended for real data):** open `notebooks/08_any_dataset.ipynb`, put the CSV in `MyDrive/ai-experiment-lab/data/`, set `DATA_FILE` and `TARGET`, Run all. It shows what it will do with your file before training, then trains, logs to MLflow, exports, and prints what to set in the Dockerfile.

**Command line (small files, or a local run):**
```powershell
python scripts/train_any.py --data my.csv --target Churn --out models/champion
python scripts/train_any.py --data houses.csv --target price --tune-trials 20
python scripts/train_any.py --data my.csv --target label --id-columns user_id --models logistic_regression,xgboost
```
`--mlflow --tracking-uri sqlite:///mlflow.db` also logs and registers the run. Run `python scripts/train_any.py --help` for all options.

## What it does to your columns

| Column | Treatment |
|---|---|
| nearly unique per row (ids), or named like an id | left out (the output says so; override with `--id-columns`, `none` disables detection) |
| numbers stored as text (blank = missing) | kept as numbers, blanks imputed |
| true/false | text categories |
| other text | categories, one-hot encoded; up to 50 values become a dropdown, more become free text |
| dates | dropped, with a note (add a numeric column such as "days since signup" yourself) |
| constant or empty | dropped, with a note |
| missing values | imputed inside the pipeline (median, or most common) |
| rows with a missing target | removed, with a note |

The same rules as the Telco work apply: split first, all preprocessing inside one Pipeline fitted on the training split, model chosen by cross-validation on the training split, the test split scored once.

## Deploying the result

1. Copy `model.skops` and `metadata.json` into `models/champion/` (the notebook and CLI write them there).
2. Set the library versions the trainer prints as the defaults of `ARG SKLEARN_VERSION=` (and `ARG XGBOOST_VERSION=` if XGBoost won) at the top of the `Dockerfile`. A mismatch loads silently in some cases and fails in others. `python scripts/check_model_export.py` checks this and fails the CI if they differ.
3. `ruff check . && pytest -q`, commit `models/champion` and the Dockerfile, push, then Render **Manual Deploy** or the **Redeploy** workflow.
4. `python scripts/smoke_test_api.py <url> --wait 120`. The smoke test reads `/schema`, so it works for any model.

The page at `/app` and the API at `/docs` now show your columns. `GET /schema` returns the machine-readable description.

## Monitoring and retraining for any model

- Prometheus metrics work as before. The probability histogram is filled only for binary models; the predicted-class counter is bounded (at most 50 classes, or one `value` series for regression).
- Drift: `python scripts/check_drift.py --data my.csv --model-dir models/champion --log data/predictions/predictions.jsonl` (reference = your training CSV cleaned the same way).
- Retraining decisions use `ai_experiment_lab.retraining` with `score(..., task=...)` and `promotion_metric(task)`: ROC AUC (binary), balanced accuracy (multi-class), R^2 (regression). The notebook 07 flow is written for the Telco data; for other data call these functions the same way.

## Honest limits

- **Near-perfect scores are a warning, not a result.** On real data, F1 or R^2 above about 0.99 usually means a column leaks the answer (something recorded after the outcome) or the data is synthetic. The platform cannot see leakage; remove such columns yourself.
- Probabilities are not calibrated (binary models use `class_weight=balanced`), so use the score to rank rows, not to quote odds.
- No confidence interval for regression.
- Categorical values never seen in training are rejected when the column has a dropdown (up to 50 values); free-text columns accept anything and the encoder ignores unknown values.
- A new dataset replaces the served model. Hosting two models at once needs a second service.
- The free host has no login: do not put real customer data on a public demo.
