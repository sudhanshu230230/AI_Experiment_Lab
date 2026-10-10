"""Evaluation metrics for binary classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Metrics for the positive class (label 1).

    ``y_proba`` is the predicted probability of class 1. Accuracy is reported for
    completeness but should not be used to pick a model on imbalanced data; PR-AUC is
    the most informative single number when the positive class is rare.
    """
    y_true = np.asarray(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Confusion matrix as named counts (positive class = 1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
