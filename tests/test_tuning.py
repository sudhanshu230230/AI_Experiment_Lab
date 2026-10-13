"""Tests for Optuna tuning and its MLflow logging (small synthetic data, few trials)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna")
pytest.importorskip("mlflow")

import mlflow
import optuna

from ai_experiment_lab.data import DatasetInfo
from ai_experiment_lab.preprocessing import split_data
from ai_experiment_lab.tracking import compare_runs, log_tuning_result, setup_tracking
from ai_experiment_lab.training import run_baselines
from ai_experiment_lab.tuning import suggest_params, tune_model

PRE = {"coerce_numeric": ["TotalCharges"]}
FIXED = {"max_iter": 300, "class_weight": "balanced"}
SPACE = {
    "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
    "solver": {"type": "categorical", "choices": ["lbfgs", "liblinear"]},
}


@pytest.fixture(scope="module")
def split():
    rng = np.random.default_rng(0)
    n = 300
    tenure = rng.integers(0, 72, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
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
    return split_data(df, "Churn", "Yes", id_columns=["customerID"], seed=1)


def test_suggest_params_handles_all_types_and_rejects_unknown():
    study = optuna.create_study()
    space = {
        "a": {"type": "float", "low": 0.1, "high": 1.0, "log": True},
        "b": {"type": "int", "low": 1, "high": 5},
        "c": {"type": "categorical", "choices": ["x", "y"]},
    }
    p = study.ask()
    out = suggest_params(p, space)
    assert 0.1 <= out["a"] <= 1.0 and 1 <= out["b"] <= 5 and out["c"] in {"x", "y"}
    with pytest.raises(ValueError):
        suggest_params(study.ask(), {"z": {"type": "weird"}})


def test_trial_zero_is_baseline_and_matches_baseline_cv(split):
    """Trial 0 uses the baseline params and the same folds, so its CV F1 equals run_baselines."""
    fixed = {"max_iter": 300, "class_weight": "balanced", "C": 1.0, "solver": "lbfgs"}
    base = run_baselines(
        split, {"logistic_regression": {"params": fixed}}, PRE, seed=1, cv_folds=3
    )
    res = tune_model("logistic_regression", split, SPACE, fixed, PRE, seed=1, cv_folds=3,
                     n_trials=4)
    t0 = res.trials.loc[res.trials["number"] == 0].iloc[0]
    assert t0["p_C"] == 1.0 and t0["p_solver"] == "lbfgs"
    assert t0["cv_f1"] == pytest.approx(base.table.loc[0, "cv_f1"], abs=1e-9)
    assert res.best_cv_f1 >= base.table.loc[0, "cv_f1"] - 1e-12


def test_tune_model_result_is_consistent(split):
    res = tune_model("logistic_regression", split, SPACE, FIXED, PRE, seed=1, cv_folds=3,
                     n_trials=5)
    assert len(res.trials) == 5 == res.n_complete + res.n_pruned
    assert set(res.best_params) == set(SPACE)
    assert res.full_params["max_iter"] == 300 and res.full_params["C"] == res.best_params["C"]
    assert len(res.cv_fold_scores) == 3
    assert res.best_cv_f1 == pytest.approx(np.mean(res.cv_fold_scores))
    assert {"f1", "recall", "roc_auc"} <= set(res.test_metrics)
    assert sum(res.test_confusion.values()) == len(split.y_test)
    # The refit pipeline takes raw rows.
    assert res.pipeline.predict_proba(split.X_test.head(3)).shape == (3, 2)


def test_tuning_is_deterministic(split):
    kw = {"seed": 1, "cv_folds": 3, "n_trials": 4}
    a = tune_model("logistic_regression", split, SPACE, FIXED, PRE, **kw)
    b = tune_model("logistic_regression", split, SPACE, FIXED, PRE, **kw)
    assert a.best_params == b.best_params and a.best_cv_f1 == b.best_cv_f1


def test_log_tuning_result_writes_parent_and_nested_trials(split, tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    setup_tracking("tune-test", tracking_uri=uri)
    res = tune_model("logistic_regression", split, SPACE, FIXED, PRE, seed=1, cv_folds=3,
                     n_trials=3)
    info = DatasetInfo("synthetic", "x.csv", "abc123", 300, 6)
    cfg = {"project": {"random_seed": 1}}
    run_id = log_tuning_result(res, split, info, cfg, cv_folds=3, n_trials_requested=3)

    run = mlflow.get_run(run_id)
    assert run.data.tags["stage"] == "tuned"
    assert run.data.metrics["cv_f1"] == pytest.approx(res.best_cv_f1)
    assert "test_f1" in run.data.metrics
    children = mlflow.search_runs(
        experiment_names=["tune-test"], filter_string=f"tags.mlflow.parentRunId = '{run_id}'"
    )
    assert len(children) == 3

    table = compare_runs("tune-test", tracking_uri=uri)
    assert len(table) == 1 and table.loc[0, "stage"] == "tuned"  # trials hidden by default
    assert len(compare_runs("tune-test", tracking_uri=uri, include_trials=True)) == 4


def test_xgboost_trial_zero_reproduces_baseline_via_space_defaults(split):
    """Parameters missing from the config fall back to the space's default for trial 0."""
    pytest.importorskip("xgboost")
    fixed = {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3}
    space = {
        "n_estimators": {"type": "int", "low": 10, "high": 40},
        "max_depth": {"type": "int", "low": 2, "high": 5},
        "min_child_weight": {"type": "float", "low": 1.0, "high": 20.0, "log": True,
                             "default": 1.0},
        "reg_lambda": {"type": "float", "low": 0.01, "high": 20.0, "log": True,
                       "default": 1.0},
    }
    base = run_baselines(split, {"xgboost": {"params": fixed}}, PRE, seed=1, cv_folds=3)
    res = tune_model("xgboost", split, space, fixed, PRE, seed=1, cv_folds=3, n_trials=3)
    t0 = res.trials.loc[res.trials["number"] == 0].iloc[0]
    assert t0["cv_f1"] == pytest.approx(base.table.loc[0, "cv_f1"], abs=1e-9)
