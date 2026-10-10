"""Baseline training: preprocessing + model as one Pipeline, compared fairly.

Model selection uses stratified cross-validation on the TRAINING split only. The held-out
test split is evaluated once per model and reported as the final, unbiased estimate; it is
never used to choose between models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from ai_experiment_lab.evaluation import classification_metrics, confusion_counts
from ai_experiment_lab.models import build_classifier
from ai_experiment_lab.preprocessing import SplitData, build_preprocessing_pipeline

CV_SCORING = {
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


@dataclass
class BaselineResults:
    """Output of ``run_baselines``."""

    table: pd.DataFrame            # one row per model, sorted by the selection score
    pipelines: dict[str, Pipeline]  # fitted preprocessing + model, keyed by model name
    confusion: dict[str, dict[str, int]]
    primary_metric: str


def make_model_pipeline(
    name: str,
    split: SplitData,
    model_params: dict[str, Any] | None,
    preprocessing_cfg: dict[str, Any],
    seed: int,
) -> Pipeline:
    """Unfitted Pipeline: preprocessing followed by the classifier (one artifact, ADR-002)."""
    pre = build_preprocessing_pipeline(
        split.X_train,
        coerce_numeric=preprocessing_cfg.get("coerce_numeric"),
        numeric_impute_strategy=preprocessing_cfg.get("numeric_impute_strategy", "median"),
        categorical_impute_strategy=preprocessing_cfg.get(
            "categorical_impute_strategy", "most_frequent"
        ),
    )
    clf = build_classifier(name, model_params, seed=seed, y_train=split.y_train)
    return Pipeline([("preprocess", pre), ("model", clf)])


def run_baselines(
    split: SplitData,
    models_cfg: dict[str, dict[str, Any]],
    preprocessing_cfg: dict[str, Any],
    seed: int = 42,
    cv_folds: int = 5,
    primary_metric: str = "f1",
) -> BaselineResults:
    """Train every configured model and compare them.

    ``models_cfg`` maps model name to ``{"params": {...}}`` (the ``classification`` section
    of ``model_configs.yaml``).
    """
    if primary_metric not in CV_SCORING:
        raise ValueError(f"primary_metric must be one of {sorted(CV_SCORING)}")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    rows, pipelines, confusions = [], {}, {}
    for name, spec in models_cfg.items():
        params = (spec or {}).get("params", {})
        pipe = make_model_pipeline(name, split, params, preprocessing_cfg, seed)

        # Selection score: cross-validation on training data (pipeline is refit per fold,
        # so imputers, scalers and encoders never see the validation fold).
        cv_res = cross_validate(pipe, split.X_train, split.y_train, cv=cv, scoring=CV_SCORING)

        start = time.perf_counter()
        pipe.fit(split.X_train, split.y_train)
        fit_seconds = time.perf_counter() - start

        pred = pipe.predict(split.X_test)
        proba = pipe.predict_proba(split.X_test)[:, 1]
        test = classification_metrics(split.y_test, pred, proba)

        row: dict[str, Any] = {"model": name}
        for m in CV_SCORING:
            scores = cv_res[f"test_{m}"]
            row[f"cv_{m}"] = float(scores.mean())
        row["cv_primary_std"] = float(cv_res[f"test_{primary_metric}"].std())
        row.update({f"test_{k}": v for k, v in test.items()})
        row["fit_seconds"] = fit_seconds
        rows.append(row)
        pipelines[name] = pipe
        confusions[name] = confusion_counts(split.y_test, pred)

    table = (
        pd.DataFrame(rows)
        .sort_values(f"cv_{primary_metric}", ascending=False)
        .reset_index(drop=True)
    )
    return BaselineResults(table, pipelines, confusions, primary_metric)
