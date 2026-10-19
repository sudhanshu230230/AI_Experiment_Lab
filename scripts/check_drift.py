"""Compare logged production inputs with the training data and print a drift verdict.

Usage (from the repo root, with the dataset CSV available):
    python scripts/check_drift.py --data data/raw/Telco-Customer-Churn.csv \
        --log data/predictions/predictions.jsonl --out reports/drift.json
    python scripts/check_drift.py ... --fail-on-drift     # exit code 1 when the verdict is "drift"

Reference = the training split (rebuilt with the same seed as training). Current = the logged
inputs, optionally only those of one model version. The verdict and per-feature results are
written as JSON; the retraining workflow (Phase 8) reads that file.

Any dataset (ADR-010): pass --model-dir (a folder exported by scripts/train_any.py) and --data. The
feature columns come from the model's own schema, and the reference is the training CSV cleaned the
same way. Prediction drift uses the logged ``score`` column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_experiment_lab.data import load_csv
from ai_experiment_lab.monitoring import DRIFT, detect_drift, read_prediction_log
from ai_experiment_lab.preprocessing import split_data
from ai_experiment_lab.utils.config import load_config


def generic_reference(df, model_dir):
    """The training CSV reduced to the model's input columns, cleaned as at training time."""
    from ai_experiment_lab.automl.schema import Schema, prepare_features

    meta = json.loads((Path(model_dir) / "metadata.json").read_text())
    if "schema" not in meta:
        raise SystemExit("this model has no schema; run without --model-dir for the Telco setup")
    schema = Schema.from_dict(meta["schema"])
    X, _, _ = prepare_features(df, schema.target, schema.excluded)
    missing = [c for c in schema.names if c not in X.columns]
    if missing:
        raise SystemExit(f"the data is missing the model's columns: {missing}")
    return X[schema.names], schema.names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the training CSV")
    ap.add_argument("--model-dir", default=None,
                    help="exported model folder with a schema (any-dataset mode); default is the Telco setup")
    ap.add_argument("--log", required=True, help="JSONL prediction log written by the API")
    ap.add_argument("--out", default=None, help="write the full report as JSON here")
    ap.add_argument("--model-version", type=int, default=None, help="only logged rows of this version")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--monitoring-config", default="configs/monitoring.yaml")
    ap.add_argument("--fail-on-drift", action="store_true", help="exit 1 if the verdict is drift")
    args = ap.parse_args(argv)

    thresholds = load_config(args.monitoring_config)["drift"]
    df, _ = load_csv(Path(args.data))

    if args.model_dir:
        reference, features = generic_reference(df, args.model_dir)
    else:
        cfg = load_config(args.config)
        d, sp = cfg["dataset"], cfg["split"]
        split = split_data(df, d["target"], d["positive_label"], d["id_columns"],
                           sp["test_size"], cfg["project"]["random_seed"], sp["stratify"])
        reference, features = split.X_train, list(split.X_train.columns)

    logged = read_prediction_log(args.log, model_version=args.model_version)
    if logged.empty:
        print("The prediction log has no usable rows; nothing to compare.")
        return 0
    current = logged[[c for c in features if c in logged.columns]]
    report = detect_drift(reference, current, **thresholds)

    print(report.summary())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        print("report written to", out)
    return 1 if (args.fail_on_drift and report.verdict == DRIFT) else 0


if __name__ == "__main__":
    sys.exit(main())
