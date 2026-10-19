"""Train, describe and serve any tabular dataset (binary, multi-class or regression), ADR-010."""

from ai_experiment_lab.automl.advisor import Advice, advise, column_profile
from ai_experiment_lab.automl.schema import (
    DatasetError,
    Feature,
    Schema,
    build_schema,
    infer_task,
    suggest_id_columns,
)
from ai_experiment_lab.automl.tasks import (
    MODELS,
    PROMOTION,
    SELECTION,
    evaluate,
    promotion_metric,
)
from ai_experiment_lab.automl.train import TrainResult, build_metadata, export_result, train_dataset

__all__ = [
    "MODELS",
    "PROMOTION",
    "SELECTION",
    "Advice",
    "DatasetError",
    "Feature",
    "Schema",
    "TrainResult",
    "advise",
    "build_metadata",
    "build_schema",
    "column_profile",
    "evaluate",
    "export_result",
    "infer_task",
    "promotion_metric",
    "suggest_id_columns",
    "train_dataset",
]
