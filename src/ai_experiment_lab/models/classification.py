"""Candidate classification models, built from configuration."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SUPPORTED_MODELS = ("logistic_regression", "random_forest", "xgboost")


def build_classifier(
    name: str,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    y_train: pd.Series | None = None,
) -> BaseEstimator:
    """Create an unfitted classifier.

    Imbalance handling: logistic regression and random forest use ``class_weight`` from the
    config; XGBoost gets ``scale_pos_weight`` computed from the training labels when
    ``y_train`` is given and the config does not set it.
    """
    params = dict(params or {})
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "xgboost is not installed. Install the 'ml' extra: pip install -e \".[ml]\""
            ) from exc
        if "scale_pos_weight" not in params and y_train is not None:
            pos = int((y_train == 1).sum())
            neg = int((y_train == 0).sum())
            if pos:
                params["scale_pos_weight"] = neg / pos
        params.setdefault("eval_metric", "logloss")
        return XGBClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown model '{name}'. Supported: {', '.join(SUPPORTED_MODELS)}")
