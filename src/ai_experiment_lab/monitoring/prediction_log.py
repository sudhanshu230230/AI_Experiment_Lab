"""Read the API's JSONL prediction log (``PREDICTION_LOG``) into a DataFrame."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_prediction_log(path: str | Path, model_version: int | None = None) -> pd.DataFrame:
    """One row per logged prediction: the feature columns plus ``ts``, ``model_version`` and
    ``churn_probability``. Blank lines and lines that are not valid JSON (for example a write
    cut short by a restart) are skipped. ``model_version`` filters to one model version.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            rows.append(
                {**rec["features"], "ts": rec["ts"], "model_version": rec.get("model_version"),
                 "churn_probability": rec["churn_probability"]}
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if model_version is not None:
        df = df[df["model_version"] == model_version].reset_index(drop=True)
    return df
