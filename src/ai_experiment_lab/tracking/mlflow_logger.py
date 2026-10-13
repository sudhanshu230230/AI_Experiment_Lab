"""Experiment tracking with MLflow (ADR-001: SQLite database and artifacts on Drive).

Every training run is logged with what is needed to reproduce and compare it: parameters,
metrics, the dataset version (SHA-256), the code version (git commit), library versions, and
the fitted preprocessing + model Pipeline as one artifact (ADR-002).
"""

from __future__ import annotations

import os
import platform
import sqlite3
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from ai_experiment_lab.data.ingestion import DatasetInfo
from ai_experiment_lab.preprocessing import SplitData
from ai_experiment_lab.training.trainer import BaselineResults

TRACKED_LIBRARIES = ("scikit-learn", "pandas", "numpy", "xgboost", "mlflow")


def get_code_version() -> str:
    """Short git commit of the code, with '-dirty' if there are uncommitted changes."""
    here = str(Path(__file__).resolve().parent)
    try:
        head = subprocess.run(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if head.returncode != 0:
            return "unknown"
        status = subprocess.run(
            ["git", "-C", here, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        dirty = "-dirty" if status.stdout.strip() else ""
        return head.stdout.strip() + dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def library_versions() -> dict[str, str]:
    """Versions of the libraries that affect results (missing ones are skipped)."""
    out = {"python": platform.python_version()}
    for lib in TRACKED_LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            continue
    return out


def _sqlite_path(uri: str) -> Path | None:
    """Database file for a ``sqlite:///...`` URI, or None for other kinds of store."""
    prefix = "sqlite:///"
    return Path(uri[len(prefix):]) if uri.startswith(prefix) else None


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise ValueError(
            "No MLflow tracking URI. Pass one or set MLFLOW_TRACKING_URI "
            "(the Colab setup cell sets it)."
        )
    return uri


def setup_tracking(experiment_name: str, tracking_uri: str | None = None) -> str:
    """Point MLflow at the store and select (creating if needed) the experiment.

    For a SQLite store, artifacts are kept in an ``artifacts`` folder next to the database
    (on Drive in Colab), so they survive with it. Returns the experiment id.
    """
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        db = _sqlite_path(uri)
        location = None
        if db is not None:
            artifacts = (db.parent / "artifacts").resolve()
            artifacts.mkdir(parents=True, exist_ok=True)
            location = artifacts.as_uri()
        exp_id = client.create_experiment(experiment_name, artifact_location=location)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def trusted_types(pipeline) -> list[str]:
    """Types inside a pipeline that the skops model format must be told to trust.

    MLflow saves scikit-learn models with skops, which refuses to load any type that is not
    explicitly allowed (for example our own ``NumericCoercer``). We compute the list from the
    pipeline we just built, so the allowed types are recorded, visible, in the model's MLmodel file.
    """
    from skops.io import dumps, get_untrusted_types

    return sorted(get_untrusted_types(data=dumps(pipeline)))


def _scalar_params(params: dict[str, Any]) -> dict[str, Any]:
    """Estimator parameters that MLflow can store as text (skips None and objects)."""
    return {
        k: v for k, v in params.items()
        if v is not None and isinstance(v, (bool, int, float, str))
    }


def log_baseline_results(
    results: BaselineResults,
    split: SplitData,
    dataset_info: DatasetInfo,
    cfg: dict[str, Any],
    cv_folds: int,
    extra_tags: dict[str, str] | None = None,
    log_models: bool = True,
) -> dict[str, str]:
    """Log one MLflow run per model in ``results``. Returns {model name: run id}.

    ``setup_tracking`` must be called first.
    """
    seed = cfg["project"]["random_seed"]
    pre = cfg.get("preprocessing", {})
    primary = results.primary_metric
    run_ids: dict[str, str] = {}

    for _, row in results.table.iterrows():
        name = row["model"]
        pipe = results.pipelines[name]
        with mlflow.start_run(run_name=name) as run:
            # ---- parameters: everything needed to repeat the run
            mlflow.log_param("model.name", name)
            mlflow.log_params({f"model.{k}": v for k, v in _scalar_params(
                pipe.named_steps["model"].get_params()).items()})
            mlflow.log_params({
                "seed": seed,
                "split.test_size": cfg["split"]["test_size"],
                "split.stratify": cfg["split"]["stratify"],
                "cv.folds": cv_folds,
                "selection.primary_metric": primary,
                "data.positive_label": cfg["dataset"]["positive_label"],
                "data.n_train": len(split.X_train),
                "data.n_test": len(split.X_test),
                "pre.coerce_numeric": ",".join(pre.get("coerce_numeric", [])),
                "pre.numeric_impute": pre.get("numeric_impute_strategy", "median"),
                "pre.categorical_impute": pre.get("categorical_impute_strategy", "most_frequent"),
            })

            # ---- tags: identity of data and code
            tags = {
                "dataset.name": dataset_info.name,
                "dataset.sha256": dataset_info.sha256,
                "dataset.rows": str(dataset_info.n_rows),
                "code.version": get_code_version(),
                "stage": "baseline",
            }
            tags.update({f"lib.{k}": v for k, v in library_versions().items()})
            tags.update(extra_tags or {})
            mlflow.set_tags(tags)

            # ---- metrics
            metrics = {}
            for col, val in row.items():
                if col == "model":
                    continue
                key = f"cv_{primary}_std" if col == "cv_primary_std" else col
                metrics[key] = float(val)
            for k, v in results.confusion[name].items():
                metrics[f"test_{k}"] = float(v)
            mlflow.log_metrics(metrics)

            # ---- artifacts
            mlflow.log_dict(metrics, "metrics.json")
            mlflow.log_dict(results.confusion[name], "confusion.json")
            mlflow.log_dict(dataset_info.to_dict(), "dataset.json")
            mlflow.log_dict(cfg, "config.json")
            if log_models:
                sample = split.X_train.head(5)
                signature = infer_signature(sample, pipe.predict(sample))
                mlflow.sklearn.log_model(
                    pipe,
                    name="model",
                    signature=signature,
                    input_example=sample,
                    skops_trusted_types=trusted_types(pipe),
                )
            run_ids[name] = run.info.run_id
    return run_ids


def load_pipeline(run_id: str, tracking_uri: str | None = None):
    """Load the fitted preprocessing + model Pipeline logged in a run."""
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def compare_runs(
    experiment_name: str,
    order_by: str = "cv_f1",
    tracking_uri: str | None = None,
    stage: str | None = None,
    include_trials: bool = False,
) -> pd.DataFrame:
    """Runs of an experiment as a clean table, best first (by a metric name such as 'cv_f1').

    ``stage`` keeps only runs with that ``stage`` tag (``baseline`` or ``tuned``). Nested
    per-trial runs are hidden unless ``include_trials`` is True.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    df = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{order_by} DESC"],
    )
    if df.empty:
        return df
    if not include_trials and "tags.stage" in df.columns:
        df = df[df["tags.stage"] != "trial"]
    if stage is not None:
        df = df[df.get("tags.stage") == stage]
    if df.empty:
        return df
    out = pd.DataFrame({"run_id": df["run_id"], "model": df["params.model.name"]})
    if "tags.stage" in df.columns:
        out["stage"] = df["tags.stage"]
    for col in df.columns:
        if col.startswith("metrics.") and not col.endswith(("_tn", "_fp", "_fn", "_tp")):
            out[col.removeprefix("metrics.")] = df[col]
    out["code_version"] = df.get("tags.code.version")
    out["dataset_sha256"] = df.get("tags.dataset.sha256", pd.Series(dtype=str)).str[:12]
    return out.reset_index(drop=True)


def backup_tracking_db(tracking_uri: str | None = None, keep: int = 3) -> Path | None:
    """Snapshot the SQLite database into ``backups/`` next to it and keep the newest ``keep``.

    Uses SQLite's backup API, so it is safe while the database is in use. Returns the backup
    path, or None if the store is not SQLite. Call at the end of a session (ADR-001).
    """
    db = _sqlite_path(resolve_tracking_uri(tracking_uri))
    if db is None or not db.exists():
        return None
    folder = db.parent / "backups"
    folder.mkdir(exist_ok=True)
    target = folder / f"{db.stem}-{time.strftime('%Y%m%d-%H%M%S')}{db.suffix}"
    src, dst = sqlite3.connect(db), sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    for old in sorted(folder.glob(f"{db.stem}-*{db.suffix}"))[:-keep]:
        old.unlink()
    return target
