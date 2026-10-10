"""Tests for model construction and metrics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ai_experiment_lab.evaluation import classification_metrics, confusion_counts
from ai_experiment_lab.models import build_classifier


def test_build_known_models():
    assert isinstance(build_classifier("logistic_regression", {"max_iter": 50}), LogisticRegression)
    rf = build_classifier("random_forest", {"n_estimators": 7}, seed=3)
    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_estimators == 7 and rf.random_state == 3


def test_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_classifier("svm")


def test_xgboost_gets_scale_pos_weight():
    pytest.importorskip("xgboost")
    y = pd.Series([0] * 75 + [1] * 25)
    clf = build_classifier("xgboost", {"n_estimators": 5}, y_train=y)
    assert clf.get_params()["scale_pos_weight"] == pytest.approx(3.0)
    own = build_classifier("xgboost", {"scale_pos_weight": 9}, y_train=y)
    assert own.get_params()["scale_pos_weight"] == 9


def test_metrics_known_values():
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.4, 0.6, 0.2, 0.1, 0.3])
    m = classification_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert 0.0 <= m["pr_auc"] <= 1.0 and m["roc_auc"] == pytest.approx(0.875)


def test_metrics_handle_no_positive_predictions():
    m = classification_metrics([1, 0, 1, 0], [0, 0, 0, 0], [0.2, 0.1, 0.4, 0.3])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_confusion_counts():
    c = confusion_counts([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert c == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
