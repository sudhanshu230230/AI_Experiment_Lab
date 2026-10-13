"""MLflow Model Registry helpers: register a run's model, give it an alias, load by alias.

Aliases (``champion``) are used instead of the deprecated registry stages. Serving and
monitoring refer to ``models:/<name>@champion``, so promoting a new version is one call and
rolling back is pointing the alias at the previous version.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ai_experiment_lab.tracking.mlflow_logger import resolve_tracking_uri

CHAMPION = "champion"


def find_run(
    experiment_name: str,
    model_name: str,
    stage: str = "tuned",
    order_by: str = "cv_f1",
    tracking_uri: str | None = None,
) -> str:
    """Run id of the best run of ``model_name`` with the given ``stage`` tag (best by CV metric).

    Selection uses a CV metric, never the test metric, so choosing a run does not leak the test set.
    """
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.stage = '{stage}' and params.`model.name` = '{model_name}'",
        order_by=[f"metrics.{order_by} DESC"],
        max_results=1,
    )
    if runs.empty:
        raise LookupError(f"No '{stage}' run for model '{model_name}' in '{experiment_name}'")
    return str(runs.loc[0, "run_id"])


def register_model(
    run_id: str,
    registered_name: str,
    alias: str | None = CHAMPION,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> int:
    """Register ``runs:/<run_id>/model`` as a new version and optionally point ``alias`` at it.

    Returns the new version number. The version is tagged with its source run id so the lineage
    registry version -> run -> dataset hash / code version / parameters is always recoverable.
    """
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    mv = mlflow.register_model(f"runs:/{run_id}/model", registered_name)
    version = int(mv.version)
    client.set_model_version_tag(registered_name, str(version), "source_run_id", run_id)
    for k, v in (tags or {}).items():
        client.set_model_version_tag(registered_name, str(version), k, v)
    if alias:
        client.set_registered_model_alias(registered_name, alias, str(version))
    return version


def alias_version(registered_name: str, alias: str = CHAMPION, tracking_uri: str | None = None):
    """The ModelVersion an alias points to, or None if the alias is not set."""
    client = MlflowClient(tracking_uri=resolve_tracking_uri(tracking_uri))
    try:
        return client.get_model_version_by_alias(registered_name, alias)
    except MlflowException:
        return None


def load_by_alias(registered_name: str, alias: str = CHAMPION, tracking_uri: str | None = None):
    """Load the fitted Pipeline behind ``models:/<name>@<alias>``."""
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    return mlflow.sklearn.load_model(f"models:/{registered_name}@{alias}")


def run_summary(run_id: str, tracking_uri: str | None = None) -> dict[str, Any]:
    """Metrics and tags of a run, in the shape stored in the exported ``metadata.json``."""
    run = MlflowClient(tracking_uri=resolve_tracking_uri(tracking_uri)).get_run(run_id)
    tags = run.data.tags
    return {
        "source_run_id": run_id,
        "model_name": run.data.params.get("model.name"),
        "metrics": {k: v for k, v in run.data.metrics.items() if not k.startswith("tuning.")},
        "dataset_sha256": tags.get("dataset.sha256"),
        "dataset_name": tags.get("dataset.name"),
        "code_version": tags.get("code.version"),
        "stage": tags.get("stage"),
        "experiment_id": tags.get("experiment.id"),
        "libraries": {k[4:]: v for k, v in tags.items() if k.startswith("lib.")},
    }
