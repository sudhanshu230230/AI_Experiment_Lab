"""Integration test: raw frame to compared baseline models."""

import numpy as np
import pandas as pd
import pytest

from ai_experiment_lab.preprocessing import split_data
from ai_experiment_lab.training import run_baselines

PRE = {"coerce_numeric": ["TotalCharges"]}
MODELS = {
    "logistic_regression": {"params": {"max_iter": 500, "class_weight": "balanced"}},
    "random_forest": {"params": {"n_estimators": 40, "class_weight": "balanced"}},
}


@pytest.fixture(scope="module")
def split():
    rng = np.random.default_rng(0)
    n = 600
    tenure = rng.integers(0, 72, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    # Learnable signal: short tenure on a monthly contract churns far more often.
    logit = -1.0 + 2.0 * (contract == "Month-to-month") - 0.04 * tenure
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(n)],
            "tenure": tenure,
            "MonthlyCharges": rng.uniform(20, 110, n).round(2),
            "Contract": contract,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    df["TotalCharges"] = (df.tenure * df.MonthlyCharges).round(2).astype(str)
    df.loc[df.tenure == 0, "TotalCharges"] = " "
    return split_data(df, "Churn", "Yes", id_columns=["customerID"])


@pytest.fixture(scope="module")
def results(split):
    return run_baselines(split, MODELS, PRE, seed=1, cv_folds=3, primary_metric="f1")


def test_table_shape_and_columns(results):
    t = results.table
    assert set(t["model"]) == set(MODELS)
    for col in ("cv_f1", "cv_pr_auc", "test_f1", "test_roc_auc", "test_pr_auc", "fit_seconds"):
        assert col in t.columns
    metric_cols = [c for c in t.columns if c.startswith(("cv_f1", "cv_roc", "test_"))]
    assert ((t[metric_cols] >= 0) & (t[metric_cols] <= 1)).all().all()


def test_sorted_by_primary_metric(results):
    assert results.table["cv_f1"].is_monotonic_decreasing


def test_models_learn_something(results):
    assert (results.table["test_roc_auc"] > 0.65).all()  # well above chance (0.5)


def test_pipelines_are_single_artifacts_that_predict_raw_rows(results, split):
    for pipe in results.pipelines.values():
        assert list(pipe.named_steps) == ["preprocess", "model"]
        proba = pipe.predict_proba(split.X_test.iloc[:3])  # raw rows, including blank TotalCharges
        assert proba.shape == (3, 2) and np.isfinite(proba).all()


def test_confusion_sums_to_test_size(results, split):
    for c in results.confusion.values():
        assert sum(c.values()) == len(split.y_test)


def test_bad_primary_metric(split):
    with pytest.raises(ValueError):
        run_baselines(split, MODELS, PRE, primary_metric="accuracy")
