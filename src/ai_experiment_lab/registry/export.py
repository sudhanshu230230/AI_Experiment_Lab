"""Export a fitted Pipeline for serving, without MLflow.

The serving image cannot reach Drive or the MLflow store, so the champion is exported to a
small folder that is committed to the repository (ADR-005):

    model.skops     the fitted preprocessing + model Pipeline (skops format, not pickle)
    metadata.json   what it is: registry name/version, source run, metrics, dataset hash,
                    code and library versions, input columns, and the skops trusted types

Loading needs only scikit-learn, skops and this package, not MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skops.io import dump, dumps, get_untrusted_types, load

MODEL_FILE = "model.skops"
METADATA_FILE = "metadata.json"

# A trusted type outside these packages in metadata.json is refused at load time, so a
# tampered metadata file cannot make the loader execute arbitrary classes.
ALLOWED_TYPE_PREFIXES = ("ai_experiment_lab.", "sklearn.", "numpy.", "pandas.", "xgboost.")


def export_pipeline(pipeline, out_dir: str | Path, metadata: dict[str, Any]) -> Path:
    """Write ``model.skops`` and ``metadata.json`` into ``out_dir``. Returns the folder."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trusted = sorted(get_untrusted_types(data=dumps(pipeline)))
    bad = [t for t in trusted if not t.startswith(ALLOWED_TYPE_PREFIXES)]
    if bad:
        raise ValueError(f"Pipeline contains types outside the allowed packages: {bad}")
    dump(pipeline, out / MODEL_FILE)
    meta = {
        **metadata,
        "input_columns": list(getattr(pipeline, "feature_names_in_", [])),
        "skops_trusted_types": trusted,
    }
    (out / METADATA_FILE).write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return out


def load_exported(model_dir: str | Path):
    """Load an exported Pipeline and its metadata: ``(pipeline, metadata)``."""
    d = Path(model_dir)
    meta = json.loads((d / METADATA_FILE).read_text())
    trusted = meta.get("skops_trusted_types", [])
    bad = [t for t in trusted if not t.startswith(ALLOWED_TYPE_PREFIXES)]
    if bad:
        raise ValueError(f"Refusing to trust types outside the allowed packages: {bad}")
    return load(d / MODEL_FILE, trusted=trusted), meta
