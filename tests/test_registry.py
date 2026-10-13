"""Tests for the model registry (alias, lineage) and the MLflow-free serving export."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")
pytest.importorskip("skops")

from ai_experiment_lab.data import DatasetInfo
from ai_experiment_lab.preprocessing import split_data
from ai_experiment_lab.registry import export_pipeline, load_exported
from ai_experiment_lab.registry.export import METADATA_FILE
from ai_experiment_lab.registry.mlflow_registry import (
    alias_version,
    find_run,
    load_by_alias,
    register_model,
    run_summary,
)
from ai_experiment_lab.tracking import log_tuning_result, setup_tracking
from ai_experiment_lab.tuning import tune_model

PRE = {"coerce_numeric": ["TotalCharges"]}
SPACE = {"C": {"type": "float", "low": 0.01, "high": 10.0, "log": True, "default": 1.0}}
FIXED = {"max_iter": 300, "class_weight": "balanced"}
EXPERIMENT = "registry-test"


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


@pytest.fixture(scope="module")
def store(split, tmp_path_factory):
    """A tracking store holding one tuned logistic-regression run."""
    uri = f"sqlite:///{(tmp_path_factory.mktemp('mlf') / 'mlflow.db').as_posix()}"
    setup_tracking(EXPERIMENT, tracking_uri=uri)
    res = tune_model("logistic_regression", split, SPACE, FIXED, PRE, seed=1, cv_folds=3,
                     n_trials=3)
    info = DatasetInfo("synthetic", "x.csv", "abc123", 300, 6)
    run_id = log_tuning_result(res, split, info, {"project": {"random_seed": 1}},
                               cv_folds=3, n_trials_requested=3)
    return uri, run_id, res


def test_find_run_picks_tuned_run_and_rejects_unknown_model(store):
    uri, run_id, _ = store
    assert find_run(EXPERIMENT, "logistic_regression", tracking_uri=uri) == run_id
    with pytest.raises(LookupError):
        find_run(EXPERIMENT, "xgboost", tracking_uri=uri)


def test_register_alias_and_load_reproduce_predictions(store, split):
    uri, run_id, res = store
    assert alias_version("churn-model", tracking_uri=uri) is None
    v = register_model(run_id, "churn-model", tags={"reason": "unit test"}, tracking_uri=uri)
    assert v == 1
    mv = alias_version("churn-model", tracking_uri=uri)
    assert int(mv.version) == 1 and mv.tags["source_run_id"] == run_id
    loaded = load_by_alias("churn-model", tracking_uri=uri)
    X = split.X_test
    np.testing.assert_allclose(loaded.predict_proba(X), res.pipeline.predict_proba(X))


def test_new_version_moves_alias_and_old_version_stays(store, split):
    uri, run_id, _ = store
    v2 = register_model(run_id, "churn-model", tracking_uri=uri)
    assert v2 == 2 and int(alias_version("churn-model", tracking_uri=uri).version) == 2
    # Rollback is just pointing the alias back.
    from mlflow import MlflowClient

    MlflowClient(tracking_uri=uri).set_registered_model_alias("churn-model", "champion", "1")
    assert int(alias_version("churn-model", tracking_uri=uri).version) == 1


def test_export_roundtrip_without_mlflow_and_metadata(store, split, tmp_path):
    uri, run_id, res = store
    meta = run_summary(run_id, tracking_uri=uri)
    assert meta["dataset_sha256"] == "abc123" and meta["stage"] == "tuned"
    export_pipeline(res.pipeline, tmp_path / "champ", {**meta, "registered_name": "churn-model"})
    pipe, loaded_meta = load_exported(tmp_path / "champ")
    X = split.X_test
    np.testing.assert_allclose(pipe.predict_proba(X), res.pipeline.predict_proba(X))
    assert loaded_meta["registered_name"] == "churn-model"
    assert loaded_meta["input_columns"] == list(split.X_train.columns)
    assert any(t.endswith("NumericCoercer") for t in loaded_meta["skops_trusted_types"])


def test_load_refuses_untrusted_types_in_tampered_metadata(store, tmp_path):
    _, _, res = store
    export_pipeline(res.pipeline, tmp_path / "m", {})
    meta_path = tmp_path / "m" / METADATA_FILE
    meta = json.loads(meta_path.read_text())
    meta["skops_trusted_types"].append("os.system")
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="allowed packages"):
        load_exported(tmp_path / "m")
