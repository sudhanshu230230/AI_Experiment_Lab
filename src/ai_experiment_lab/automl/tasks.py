"""Everything that depends on the task: models, metrics, cross-validation, the selection metric."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from ai_experiment_lab.evaluation import classification_metrics

MODELS = {
    "binary": ("logistic_regression", "random_forest", "xgboost"),
    "multiclass": ("logistic_regression", "random_forest", "xgboost"),
    "regression": ("ridge", "random_forest", "xgboost"),
}

# Selection metric per task: (name, higher_is_better). Chosen on cross-validation of the
# training split only; the test split is scored once.
SELECTION = {
    "binary": ("f1", True),          # ADR-003: accuracy hides a missed minority class
    "multiclass": ("f1_macro", True),
    "regression": ("rmse", False),
}
# Retraining promotion metric per task (ADR-009): a metric that does not move just because the
# mix of classes moved. Always higher-is-better.
PROMOTION = {"binary": "roc_auc", "multiclass": "balanced_accuracy", "regression": "r2"}

CV_SCORING = {
    "binary": {"f1": "f1", "recall": "recall", "precision": "precision", "roc_auc": "roc_auc",
               "pr_auc": "average_precision"},
    "multiclass": {"f1_macro": "f1_macro", "accuracy": "accuracy",
                   "balanced_accuracy": "balanced_accuracy"},
    "regression": {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error",
                   "r2": "r2"},
}

DEFAULT_PARAMS: dict[str, dict[str, dict[str, Any]]] = {
    "binary": {
        "logistic_regression": {"max_iter": 1000, "class_weight": "balanced"},
        "random_forest": {"n_estimators": 200, "class_weight": "balanced", "n_jobs": -1},
        "xgboost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6},
    },
    "multiclass": {
        "logistic_regression": {"max_iter": 2000, "class_weight": "balanced"},
        "random_forest": {"n_estimators": 200, "class_weight": "balanced", "n_jobs": -1},
        "xgboost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6},
    },
    "regression": {
        "ridge": {"alpha": 1.0},
        "random_forest": {"n_estimators": 200, "n_jobs": -1},
        "xgboost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6},
    },
}

RIDGE_SPACE = {"alpha": {"type": "float", "low": 0.001, "high": 1000.0, "log": True}}


def build_estimator(task: str, name: str, params: dict[str, Any] | None, seed: int,
                    y_train=None) -> BaseEstimator:
    """An unfitted model for the task."""
    params = dict(params or {})
    if name not in MODELS[task]:
        raise ValueError(f"model '{name}' is not available for {task}; choose from {MODELS[task]}")
    if task == "regression":
        if name == "ridge":
            return Ridge(random_state=seed, **params)
        if name == "random_forest":
            return RandomForestRegressor(random_state=seed, **params)
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, **params)
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    from xgboost import XGBClassifier
    if task == "binary" and "scale_pos_weight" not in params and y_train is not None:
        pos = int((np.asarray(y_train) == 1).sum())
        neg = int((np.asarray(y_train) == 0).sum())
        if pos:
            params["scale_pos_weight"] = neg / pos
    if task == "binary":
        params.setdefault("eval_metric", "logloss")
    return XGBClassifier(random_state=seed, **params)


def cv_splitter(task: str, y, n_splits: int, seed: int):
    """Stratified folds for classification (never a fold without a class), plain folds otherwise."""
    if task == "regression":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def max_folds(task: str, y, wanted: int) -> int:
    if task == "regression":
        return max(2, wanted)
    smallest = int(np.bincount(np.asarray(y)).min())
    return max(2, min(wanted, smallest))


def evaluate(task: str, model, X, y) -> dict[str, float]:
    """Metrics of a fitted pipeline on (X, y)."""
    y = np.asarray(y)
    pred = model.predict(X)
    if task == "regression":
        mse = mean_squared_error(y, pred)
        return {"rmse": float(np.sqrt(mse)), "mae": float(mean_absolute_error(y, pred)),
                "r2": float(r2_score(y, pred)),
                "median_ae": float(median_absolute_error(y, pred))}
    proba = model.predict_proba(X)
    if task == "binary":
        return classification_metrics(y, pred, proba[:, 1])
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
    }
    labels = np.arange(proba.shape[1])
    try:
        out["log_loss"] = float(log_loss(y, proba, labels=labels))
        if len(np.unique(y)) == proba.shape[1]:
            out["roc_auc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    except ValueError:
        pass
    return out


def promotion_metric(task: str) -> str:
    return PROMOTION[task]
