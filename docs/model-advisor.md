# Model advisor

Open `/lab` on the live site (or locally at http://127.0.0.1:8000/lab).

1. Choose a CSV (up to 15 MB, comma, semicolon or tab separated, with a header row).
2. Pick the column to predict and tick columns to ignore (IDs, names). Likely IDs are ticked for you.
3. Press **Find the best algorithm**. It takes about 20 to 90 seconds.
4. Read the result: the recommended algorithm and why, the settings to use, how well it does on rows it never saw compared with just guessing, every candidate tried, which columns matter, warnings, and code to reproduce it. **Download the trained model** gives a zip you can serve with this project.

Works for a yes/no column, several classes, or a number. It does not handle time series, images, audio or video.

## Limits (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `LAB_ENABLED` | 1 | 0 switches the advisor off |
| `LAB_MAX_UPLOAD_MB` | 15 | largest file |
| `LAB_MAX_COLUMNS` | 100 | most columns |
| `LAB_MAX_ROWS` | 20000 | rows used for the search (random sample) |
| `LAB_BUDGET_SECONDS` | 60 | time spent tuning |
| `LAB_MAX_TRIALS` | 10 | Optuna trials per algorithm |
| `LAB_HARD_SECONDS` | 420 | job is stopped after this |
| `LAB_TTL_SECONDS` | 1800 | uploads and results are forgotten after this |

Uploads live in memory only. One run at a time. For big data or a long search use Colab (`notebooks/08_any_dataset.ipynb`) or `scripts/train_any.py`.

## API

`POST /lab/analyze` (raw CSV body) returns the columns and a suggested target. `POST /lab/run` with `{upload_id, target, id_columns}` returns a job id. `GET /lab/jobs/{id}` returns progress, then the result. `GET /lab/jobs/{id}/model` returns the zip.

Measured on a 64,374-row, 12-column file (laptop-class CPU, including the demo model in memory): 3,000 rows took 41 s and 322 MB peak; 25,000 rows 49 s and 376 MB; all 64,374 rows 76 s and 408 MB. The free Render server has about 0.5 GB and a much slower CPU, so the default is 20,000 rows; raise `LAB_MAX_ROWS` on a bigger plan.
