"""Optional MLflow logging and registry step for ``automl`` results (needs the ``tracking`` extra).

One run per training call: parameters, cross-validation and test metrics, the leaderboard, and the
fitted pipeline. The model is then registered and the ``champion`` alias moved to it, exactly as
for the Telco model (ADR-005), so the registry version in the exported metadata is real.
"""

from __future__ import annotations

from typing import Any

from ai_experiment_lab.automl.schema import slugify
from ai_experiment_lab.automl.train import TrainResult


def log_and_register(
    result: TrainResult,
    dataset_name: str,
    dataset_sha256: str | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    register: bool = True,
) -> dict[str, Any]:
    """Log the run, register the model and set the alias. Returns ``{run_id, version, name}``."""
    import mlflow
    import mlflow.sklearn

    from ai_experiment_lab.registry.mlflow_registry import register_model
    from ai_experiment_lab.tracking.mlflow_logger import (
        get_code_version,
        library_versions,
        resolve_tracking_uri,
        setup_tracking,
        trusted_types,
    )

    name = slugify(dataset_name)
    uri = resolve_tracking_uri(tracking_uri)
    experiment_id = setup_tracking(experiment_name or name, uri)
    s = result.schema
    with mlflow.start_run(run_name=f"{name}-{result.best}") as run:
        mlflow.set_tags({
            "stage": "automl", "dataset.name": dataset_name,
            "dataset.sha256": dataset_sha256 or "", "code.version": get_code_version(),
            "task": s.task, "target": s.target, "experiment.id": str(experiment_id),
            **{f"lib.{k}": v for k, v in library_versions().items()},
        })
        mlflow.log_params({"model.name": result.best, "task": s.task, "target": s.target,
                           "n_rows": result.n_rows, "n_features": len(s.features),
                           "selection_metric": result.selection_metric,
                           **{f"param.{k}": v for k, v in result.params.get(result.best, {}).items()}})
        best = result.table.iloc[0]
        mlflow.log_metrics({k: float(v) for k, v in best.items()
                            if isinstance(v, (int, float)) and k.startswith(("cv_", "test_"))})
        mlflow.log_table(result.table, "leaderboard.json")
        mlflow.log_dict(s.to_dict(), "schema.json")
        mlflow.sklearn.log_model(result.pipeline, name="model",
                                 skops_trusted_types=trusted_types(result.pipeline))
        run_id = run.info.run_id
    version = None
    if register:
        version = register_model(run_id, name, tracking_uri=uri,
                                 tags={"task": s.task, "dataset": dataset_name})
    return {"run_id": run_id, "version": version, "name": name, "tracking_uri": uri}
