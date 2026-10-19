"""Train a model on any tabular CSV and export it for the API (ADR-010).

Usage (from the repo root):
    python scripts/train_any.py --data my.csv --target Churn --out models/champion
    python scripts/train_any.py --data houses.csv --target price --task regression --tune-trials 20
    python scripts/train_any.py --data my.csv --target label --id-columns user_id,row_no
    python scripts/train_any.py ... --mlflow --tracking-uri sqlite:///mlflow.db    # also register it

The task (binary, multiclass or regression) is inferred from the target unless --task is given.
The output folder holds model.skops and metadata.json, including the schema, and is exactly what the
Docker image serves: commit it as models/champion and redeploy. The serving image must install the
same scikit-learn version the model was trained with (the script prints the Docker build argument).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ai_experiment_lab.automl import DatasetError, export_result, train_dataset
from ai_experiment_lab.automl.schema import slugify
from ai_experiment_lab.data.ingestion import file_sha256


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV file with a header row")
    ap.add_argument("--target", required=True, help="column to predict")
    ap.add_argument("--task", choices=("binary", "multiclass", "regression"), default=None)
    ap.add_argument("--id-columns", default=None,
                    help="comma-separated columns to leave out; 'none' disables automatic detection")
    ap.add_argument("--positive-label", default=None, help="binary: the class to detect")
    ap.add_argument("--models", default=None, help="comma-separated subset, e.g. logistic_regression,xgboost")
    ap.add_argument("--tune-trials", type=int, default=0, help="Optuna trials per model (0 = no tuning)")
    ap.add_argument("--tune-seconds", type=float, default=120)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=None, help="train on a random sample of this many rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/champion")
    ap.add_argument("--name", default=None, help="model name (default: the CSV file name)")
    ap.add_argument("--title", default=None, help="title shown on the web page")
    ap.add_argument("--version", type=int, default=1, help="model version recorded in the export")
    ap.add_argument("--report", default=None, help="write the leaderboard and notes as JSON here")
    ap.add_argument("--mlflow", action="store_true", help="log the run and register the model in MLflow")
    ap.add_argument("--tracking-uri", default=None)
    args = ap.parse_args(argv)

    path = Path(args.data)
    name = args.name or path.stem
    df = pd.read_csv(path)
    ids = None
    if args.id_columns is not None:
        ids = [] if args.id_columns.strip().lower() == "none" else [c.strip() for c in args.id_columns.split(",") if c.strip()]
    try:
        result = train_dataset(
            df, args.target, task=args.task, id_columns=ids, positive_label=args.positive_label,
            models=tuple(args.models.split(",")) if args.models else None,
            cv_folds=args.cv_folds, seed=args.seed, tune_trials=args.tune_trials,
            tune_seconds=args.tune_seconds, max_rows=args.max_rows, progress=print)
    except (DatasetError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    sha = file_sha256(path)
    version = args.version
    if args.mlflow:
        from ai_experiment_lab.automl.tracking import log_and_register

        info = log_and_register(result, name, sha, args.tracking_uri)
        version = info["version"] or version
        print(f"MLflow run {info['run_id']}; registered '{info['name']}' version {info['version']}")
    out = export_result(result, args.out, name, version=version, dataset_sha256=sha, title=args.title)
    print("\nLeaderboard (best first):")
    cols = [c for c in result.table.columns if c in ("model", f"cv_{result.selection_metric}",
                                                      "cv_selection_std", "fit_seconds")
            or c.startswith("test_")]
    print(result.table[cols].round(4).to_string(index=False))
    print(f"\nTask: {result.schema.task}; target: {result.schema.target}; best: {result.best}; "
          f"rows: {result.n_rows:,}; features: {len(result.schema.features)}")
    if result.schema.excluded:
        print("Left out as identifiers:", result.schema.excluded)
    for note in result.notes:
        print("Note:", note)
    meta = json.loads((out / "metadata.json").read_text())
    libs = meta["libraries"]
    print(f"\nExported to {out}  (model {slugify(name)} v{version}).")
    args_hint = f"SKLEARN_VERSION={libs.get('scikit-learn')}"
    if any(t.startswith("xgboost.") for t in meta.get("skops_trusted_types", [])):
        args_hint += f" and XGBOOST_VERSION={libs.get('xgboost')}"
    print(f"Docker: set {args_hint} (edit the ARG lines in the Dockerfile; Render builds from those).")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({
            "task": result.schema.task, "target": result.schema.target, "best": result.best,
            "leaderboard": json.loads(result.table.to_json(orient="records")),
            "test_metrics": result.test_metrics, "notes": result.notes,
            "schema": result.schema.to_dict()}, indent=2))
        print("report written to", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
