"""Log an Optuna tuning result to MLflow: one parent run per model, one nested run per trial."""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from ai_experiment_lab.data.ingestion import DatasetInfo
from ai_experiment_lab.preprocessing import SplitData
from ai_experiment_lab.tracking.mlflow_logger import (
    _scalar_params,
    get_code_version,
    library_versions,
    trusted_types,
)
from ai_experiment_lab.tuning import TuningResult


def log_tuning_result(
    result: TuningResult,
    split: SplitData,
    dataset_info: DatasetInfo,
    cfg: dict[str, Any],
    cv_folds: int,
    n_trials_requested: int,
    extra_tags: dict[str, str] | None = None,
    log_model: bool = True,
) -> str:
    """Log ``result`` and return the parent run id. ``setup_tracking`` must be called first.

    Trials are logged after the search finishes, so an interrupted search logs nothing.
    """
    seed = cfg["project"]["random_seed"]
    with mlflow.start_run(run_name=f"tune:{result.model}") as parent:
        mlflow.log_params(
            {
                "model.name": result.model,
                "seed": seed,
                "cv.folds": cv_folds,
                "selection.primary_metric": "f1",
                "tuning.n_trials_requested": n_trials_requested,
                **{f"best.{k}": v for k, v in _scalar_params(result.full_params).items()},
            }
        )
        tags = {
            "dataset.name": dataset_info.name,
            "dataset.sha256": dataset_info.sha256,
            "code.version": get_code_version(),
            "stage": "tuned",
            "tuning.sampler": "TPE",
            "tuning.pruner": "Median",
        }
        tags.update({f"lib.{k}": v for k, v in library_versions().items()})
        tags.update(extra_tags or {})
        mlflow.set_tags(tags)

        folds = result.cv_fold_scores
        metrics = {
            "cv_f1": result.best_cv_f1,
            "cv_f1_std": float(np.std(folds)) if folds else float("nan"),
            "tuning.n_complete": float(result.n_complete),
            "tuning.n_pruned": float(result.n_pruned),
            "tuning.search_seconds": result.seconds,
            **{f"test_{k}": v for k, v in result.test_metrics.items()},
            **{f"test_{k}": float(v) for k, v in result.test_confusion.items()},
        }
        mlflow.log_metrics(metrics)
        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_dict(result.full_params, "best_params.json")
        mlflow.log_table(result.trials, "trials.json")

        for _, row in result.trials.iterrows():
            with mlflow.start_run(run_name=f"trial-{int(row['number']):03d}", nested=True):
                mlflow.set_tags({"stage": "trial", "trial.state": row["state"]})
                mlflow.log_params(
                    {k[2:]: v for k, v in row.items() if k.startswith("p_") and pd.notna(v)}
                )
                if row["state"] == "COMPLETE":
                    mlflow.log_metric("cv_f1", float(row["cv_f1"]))
                mlflow.log_metric("seconds", float(row["seconds"]))

        if log_model:
            sample = split.X_train.head(5)
            mlflow.sklearn.log_model(
                result.pipeline,
                name="model",
                signature=infer_signature(sample, result.pipeline.predict(sample)),
                input_example=sample,
                skops_trusted_types=trusted_types(result.pipeline),
            )
        return parent.info.run_id
