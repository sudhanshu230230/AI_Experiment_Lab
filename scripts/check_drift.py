"""Compare logged production inputs with the training data and print a drift verdict.

Usage (from the repo root, with the dataset CSV available):
    python scripts/check_drift.py --data data/raw/Telco-Customer-Churn.csv \
        --log data/predictions/predictions.jsonl --out reports/drift.json
    python scripts/check_drift.py ... --fail-on-drift     # exit code 1 when the verdict is "drift"

Reference = the training split (rebuilt with the same seed as training). Current = the logged
inputs, optionally only those of one model version. The verdict and per-feature results are
written as JSON; the retraining workflow (Phase 8) reads that file.
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Telco churn CSV (the training data)")
    ap.add_argument("--log", required=True, help="JSONL prediction log written by the API")
    ap.add_argument("--out", default=None, help="write the full report as JSON here")
    ap.add_argument("--model-version", type=int, default=None, help="only logged rows of this version")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--monitoring-config", default="configs/monitoring.yaml")
    ap.add_argument("--fail-on-drift", action="store_true", help="exit 1 if the verdict is drift")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    d, sp = cfg["dataset"], cfg["split"]
    thresholds = load_config(args.monitoring_config)["drift"]

    df, _ = load_csv(Path(args.data))
    split = split_data(df, d["target"], d["positive_label"], d["id_columns"],
                       sp["test_size"], cfg["project"]["random_seed"], sp["stratify"])

    logged = read_prediction_log(args.log, model_version=args.model_version)
    if logged.empty:
        print("The prediction log has no usable rows; nothing to compare.")
        return 0
    features = list(split.X_train.columns)
    current = logged[[c for c in features if c in logged.columns]]
    report = detect_drift(split.X_train, current, **thresholds)

    print(report.summary())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        print("report written to", out)
    return 1 if (args.fail_on_drift and report.verdict == DRIFT) else 0


if __name__ == "__main__":
    sys.exit(main())
