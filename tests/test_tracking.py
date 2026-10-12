"""Tests for MLflow tracking, using a throwaway SQLite store in a temp folder."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")

import mlflow
import mlflow.artifacts
from mlflow import MlflowClient

from ai_experiment_lab.data import DatasetInfo
from ai_experiment_lab.preprocessing import split_data
from ai_experiment_lab.tracking import (
    backup_tracking_db,
    compare_runs,
    get_code_version,
    library_versions,
    load_pipeline,
    log_baseline_results,
    setup_tracking,
)
from ai_experiment_lab.training import run_baselines

CFG = {
    "project": {"random_seed": 1},
    "split": {"test_size": 0.2, "stratify": True},
    "dataset": {"positive_label": "Yes"},
    "preprocessing": {"coerce_numeric": ["TotalCharges"]},
}
MODELS = {
    "logistic_regression": {"params": {"max_iter": 300, "class_weight": "balanced"}},
    "random_forest": {"params": {"n_estimators": 15, "class_weight": "balanced"}},
}
EXPERIMENT = "unit-test"


@pytest.fixture(scope="module")
def data():
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


@pytest.fixture(scope="module")
def tracked(tmp_path_factory, data):
    root = tmp_path_factory.mktemp("mlflow_store")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    exp_id = setup_tracking(EXPERIMENT, uri)
    results = run_baselines(data, MODELS, CFG["preprocessing"], seed=1, cv_folds=3)
    info = DatasetInfo("synthetic", "x.csv", "a" * 64, 300, 6)
    run_ids = log_baseline_results(results, data, info, CFG, cv_folds=3)
    return {"uri": uri, "root": root, "exp_id": exp_id, "results": results, "run_ids": run_ids,
            "info": info}


def test_setup_is_idempotent_and_artifacts_next_to_db(tracked):
    again = setup_tracking(EXPERIMENT, tracked["uri"])
    assert again == tracked["exp_id"]
    exp = MlflowClient(tracking_uri=tracked["uri"]).get_experiment(tracked["exp_id"])
    assert "artifacts" in exp.artifact_location
    assert str(tracked["root"].name) in exp.artifact_location


def test_missing_uri_raises(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(ValueError, match="tracking URI"):
        setup_tracking("x")


def test_one_run_per_model(tracked):
    assert set(tracked["run_ids"]) == set(MODELS)
    runs = MlflowClient(tracking_uri=tracked["uri"]).search_runs([tracked["exp_id"]])
    assert len(runs) == 2


def test_params_tags_metrics_recorded(tracked, data):
    client = MlflowClient(tracking_uri=tracked["uri"])
    run = client.get_run(tracked["run_ids"]["random_forest"])
    assert run.data.params["model.name"] == "random_forest"
    assert run.data.params["model.n_estimators"] == "15"
    assert run.data.params["seed"] == "1"
    assert run.data.params["data.n_train"] == str(len(data.X_train))
    assert run.data.tags["dataset.sha256"] == "a" * 64
    assert "code.version" in run.data.tags and "lib.scikit-learn" in run.data.tags
    row = tracked["results"].table.set_index("model").loc["random_forest"]
    assert run.data.metrics["test_f1"] == pytest.approx(row["test_f1"])
    assert run.data.metrics["cv_f1"] == pytest.approx(row["cv_f1"])
    assert run.data.metrics["cv_f1_std"] == pytest.approx(row["cv_primary_std"])
    assert "test_tp" in run.data.metrics


def test_artifacts_and_model_logged(tracked):
    client = MlflowClient(tracking_uri=tracked["uri"])
    names = {a.path for a in client.list_artifacts(tracked["run_ids"]["logistic_regression"])}
    assert {"metrics.json", "confusion.json", "dataset.json", "config.json"} <= names
    # MLflow 3 stores models as separate "logged model" entities linked to the run.
    models = client.search_logged_models(experiment_ids=[tracked["exp_id"]])
    assert {m.source_run_id for m in models} == set(tracked["run_ids"].values())
    mlmodel = mlflow.artifacts.download_artifacts(artifact_uri=f"{models[0].artifact_location}/MLmodel")
    with open(mlmodel, encoding="utf-8") as f:
        text = f.read()
    assert "NumericCoercer" in text  # allowed types are recorded in the model metadata


def test_logged_pipeline_reproduces_predictions(tracked, data):
    for name, run_id in tracked["run_ids"].items():
        loaded = load_pipeline(run_id, tracked["uri"])
        original = tracked["results"].pipelines[name]
        np.testing.assert_allclose(
            loaded.predict_proba(data.X_test), original.predict_proba(data.X_test)
        )


def test_compare_runs_sorted(tracked):
    table = compare_runs(EXPERIMENT, order_by="cv_f1", tracking_uri=tracked["uri"])
    assert len(table) == 2
    assert table["cv_f1"].is_monotonic_decreasing
    assert {"run_id", "model", "test_f1", "code_version", "dataset_sha256"} <= set(table.columns)
    assert not any(c.endswith(("_tn", "_fp")) for c in table.columns)


def test_training_is_deterministic(data):
    a = run_baselines(data, MODELS, CFG["preprocessing"], seed=1, cv_folds=3).table
    b = run_baselines(data, MODELS, CFG["preprocessing"], seed=1, cv_folds=3).table
    cols = [c for c in a.columns if c.startswith(("cv_", "test_"))]
    pd.testing.assert_frame_equal(a[["model", *cols]], b[["model", *cols]])


def test_backup_keeps_newest(tracked, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracked["uri"])
    import time
    paths = []
    for _ in range(3):
        paths.append(backup_tracking_db(keep=2))
        time.sleep(1.1)  # file names carry a one-second timestamp
    kept = sorted((tracked["root"] / "backups").glob("*.db"))
    assert len(kept) == 2 and paths[-1] in kept and paths[0] not in kept
    mlflow.set_tracking_uri(tracked["uri"])  # backup must leave the live store usable
    assert len(MlflowClient(tracking_uri=tracked["uri"]).search_runs([tracked["exp_id"]])) == 2


def test_backup_non_sqlite_returns_none():
    assert backup_tracking_db("http://localhost:5000") is None


def test_code_version_and_libraries():
    assert isinstance(get_code_version(), str)
    libs = library_versions()
    assert "python" in libs and "scikit-learn" in libs


def test_trusted_types_recorded_and_xgboost_roundtrip(tmp_path, data):
    pytest.importorskip("xgboost")
    uri = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    setup_tracking("xgb-test", uri)
    models = {"xgboost": {"params": {"n_estimators": 10, "max_depth": 3}}}
    results = run_baselines(data, models, CFG["preprocessing"], seed=1, cv_folds=3)
    info = DatasetInfo("synthetic", "x.csv", "b" * 64, 300, 6)
    run_id = log_baseline_results(results, data, info, CFG, cv_folds=3)["xgboost"]
    loaded = load_pipeline(run_id, uri)
    np.testing.assert_allclose(
        loaded.predict_proba(data.X_test), results.pipelines["xgboost"].predict_proba(data.X_test)
    )
