"""Any-dataset support (ADR-010): schema inference, training for all three tasks, and the API
and web page that build themselves from the exported model."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("skops")
from fastapi.testclient import TestClient
from sklearn.datasets import load_diabetes, load_wine

from ai_experiment_lab.automl import (
    DatasetError,
    build_schema,
    export_result,
    infer_task,
    promotion_metric,
    suggest_id_columns,
    train_dataset,
)
from ai_experiment_lab.automl.schema import Schema
from ai_experiment_lab.monitoring import detect_drift, read_prediction_log
from ai_experiment_lab.serving.app import create_app


def messy_binary(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Column names with spaces, an id, a bool, numbers-as-text with blanks, a constant, dates, NaNs."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    plan = rng.choice(["Basic", "Standard", "Premium"], n)
    calls = rng.integers(0, 10, n)
    logit = -2 + 0.04 * (age - 40) + 0.35 * calls + (plan == "Basic")
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    spend = np.round(rng.uniform(10, 500, n), 2).astype(str)
    spend[::17] = " "
    return pd.DataFrame({
        "Customer ID": [f"U{i:05d}" for i in range(n)],
        "Age": age,
        "Support Calls": calls,
        "Plan Type": plan,
        "Is Active": rng.random(n) < 0.6,
        "Total Spend": spend,
        "Region": np.where(rng.random(n) < 0.1, None, rng.choice(["N", "S", "E"], n)),
        "Constant": 1,
        "Joined": pd.date_range("2020-01-01", periods=n, freq="D"),
        "Churn": np.where(y, "Yes", "No"),
    })


def wine_df() -> pd.DataFrame:
    d = load_wine(as_frame=True)
    df = d.frame.copy()
    df["target"] = df["target"].map({0: "class_a", 1: "class_b", 2: "class_c"})
    return df


def diabetes_df() -> pd.DataFrame:
    return load_diabetes(as_frame=True).frame


# ---- task and schema inference -----------------------------------------------------------------

@pytest.mark.parametrize("values,expected", [
    (["a", "b"] * 30, "binary"),
    ([0, 1] * 30, "binary"),
    (["a", "b", "c"] * 20, "multiclass"),
    ([1, 2, 3, 4] * 20, "multiclass"),
    (list(np.linspace(0, 1, 80)), "regression"),
    (list(range(200)), "regression"),
])
def test_infer_task(values, expected):
    assert infer_task(pd.Series(values)) == expected


def test_infer_task_rejects_constant_and_huge_text_target():
    with pytest.raises(DatasetError):
        infer_task(pd.Series(["a"] * 10))
    with pytest.raises(DatasetError):
        infer_task(pd.Series([f"label{i}" for i in range(120)]))


def test_schema_cleans_messy_columns_and_records_why():
    schema, X, y, coerce, notes = build_schema(messy_binary(), "Churn")
    assert schema.task == "binary" and schema.positive_label == "Yes" and schema.classes == ["No", "Yes"]
    assert "Customer ID" not in X.columns and "Customer ID" in schema.excluded
    assert "Constant" not in X.columns and "Joined" not in X.columns
    kinds = {f.name: f for f in schema.features}
    assert kinds["Total Spend"].numeric_text and kinds["Total Spend"].nullable
    assert "Total Spend" in coerce
    assert kinds["Is Active"].kind == "categorical" and kinds["Is Active"].allowed == ["False", "True"]
    assert kinds["Region"].nullable and kinds["Age"].kind == "numeric" and kinds["Age"].integer
    assert set(y.unique()) == {0, 1}
    assert any("Constant" in n for n in notes) and any("Joined" in n for n in notes)


def test_schema_holds_no_real_rows():
    schema, *_ = build_schema(messy_binary(), "Churn")
    example = schema.example()
    assert set(example) == set(schema.names)
    assert isinstance(Schema.from_dict(json.loads(json.dumps(schema.to_dict()))), Schema)


def test_id_columns_can_be_given_explicitly_and_errors_are_clear():
    df = messy_binary()
    schema, X, *_ = build_schema(df, "Churn", id_columns=[])
    assert "Customer ID" not in X.columns or any(f.name == "Customer ID" for f in schema.features)
    with pytest.raises(DatasetError, match="not found"):
        build_schema(df, "Nope")
    with pytest.raises(DatasetError, match="at least 50"):
        build_schema(df.head(20), "Churn")
    tiny_class = df.copy()
    tiny_class["Churn"] = ["Yes"] * 3 + ["No"] * (len(df) - 3)
    with pytest.raises(DatasetError, match="smallest class"):
        build_schema(tiny_class, "Churn")
    with pytest.raises(DatasetError, match="numeric"):
        build_schema(df, "Churn", task="regression")


def test_suggest_id_columns_ignores_continuous_floats():
    df = pd.DataFrame({"id": range(100), "x": np.random.default_rng(0).normal(size=100),
                       "g": ["a", "b"] * 50, "y": [0, 1] * 50})
    assert suggest_id_columns(df, "y") == ["id"]


# ---- training and export, per task ---------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_dir(tmp_path_factory):
    r = train_dataset(messy_binary(), "Churn", models=("logistic_regression", "random_forest"))
    assert r.table.loc[0, "model"] == r.best and r.selection_metric == "f1"
    return export_result(r, tmp_path_factory.mktemp("bin"), "messy churn", version=2)


@pytest.fixture(scope="module")
def multiclass_dir(tmp_path_factory):
    r = train_dataset(wine_df(), "target", models=("logistic_regression", "xgboost"))
    assert r.schema.task == "multiclass" and r.selection_metric == "f1_macro"
    assert r.test_metrics["accuracy"] > 0.8
    return export_result(r, tmp_path_factory.mktemp("multi"), "wine")


@pytest.fixture(scope="module")
def regression_dir(tmp_path_factory):
    r = train_dataset(diabetes_df(), "target", models=("ridge", "random_forest"))
    assert r.schema.task == "regression" and r.selection_metric == "rmse"
    assert r.table["cv_rmse"].is_monotonic_increasing      # lower RMSE is better, so best first
    return export_result(r, tmp_path_factory.mktemp("reg"), "diabetes")


def test_binary_metadata_carries_the_schema(binary_dir):
    meta = json.loads((binary_dir / "metadata.json").read_text())
    assert meta["task"] == "binary" and meta["registry_version"] == 2
    assert meta["input_columns"] == [f["name"] for f in meta["schema"]["features"]]
    assert meta["threshold"] == 0.5 and "example" in meta
    assert not any(isinstance(v, str) and v.startswith("U0") for v in meta["example"].values())


def test_tuning_runs_for_every_task():
    for df, target, models in ((messy_binary(150), "Churn", ("logistic_regression",)),
                               (wine_df(), "target", ("logistic_regression",)),
                               (diabetes_df(), "target", ("ridge",))):
        r = train_dataset(df, target, models=models, tune_trials=2, tune_seconds=20)
        assert any("(tuned)" in m for m in r.table["model"]) and len(r.table) == 2


def test_unknown_model_for_task_is_rejected():
    with pytest.raises(ValueError, match="not available"):
        train_dataset(diabetes_df(), "target", models=("logistic_regression",))


def test_promotion_metric_is_higher_is_better_for_every_task():
    assert promotion_metric("binary") == "roc_auc"
    assert promotion_metric("multiclass") == "balanced_accuracy"
    assert promotion_metric("regression") == "r2"


# ---- API built from the model ----------------------------------------------------------------------

ROW = {"Age": 40, "Support Calls": 3, "Plan Type": "Basic", "Is Active": "True",
       "Total Spend": "120.5", "Region": "N"}


def test_binary_api_predicts_validates_and_logs(binary_dir, tmp_path):
    log_path = tmp_path / "p.jsonl"
    with TestClient(create_app(binary_dir, prediction_log=log_path)) as c:
        assert c.get("/health").json() == {"status": "ok", "model_loaded": True}
        info = c.get("/model").json()
        assert info["task"] == "binary" and info["registry_version"] == 2
        r = c.post("/predict", json=ROW)
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0 <= body["probability"] <= 1 and body["prediction"] in ("Yes", "No")
        assert "churn_probability" not in body and body["model_version"] == 2
        # blank text for the numbers-as-text column, and a missing nullable column, both work
        assert c.post("/predict", json={**ROW, "Total Spend": " ", "Region": None}).status_code == 200
        no_region = {k: v for k, v in ROW.items() if k != "Region"}
        assert c.post("/predict", json=no_region).status_code == 200
        batch = c.post("/predict/batch", json={"customers": [ROW, {**ROW, "Age": 70}]}).json()
        assert len(batch["predictions"]) == 2
        assert batch["predictions"][0]["probability"] == c.post("/predict", json=ROW).json()["probability"]
        bad = c.post("/predict", json={**ROW, "Plan Type": "Gold"})
        assert bad.status_code == 422 and bad.json()["detail"][0]["loc"][-1] == "Plan Type"
        assert c.post("/predict", json={**ROW, "Surprise": 1}).status_code == 422
        nan_body = json.dumps({**ROW, "Age": 0}).replace('"Age": 0', '"Age": NaN')
        assert c.post("/predict", content=nan_body,
                      headers={"Content-Type": "application/json"}).status_code == 422
        assert c.post("/predict/batch", json={"customers": []}).status_code == 422
        spec = c.get("/schema").json()
        assert spec["task"] == "binary" and spec["fields"][0]["name"] == "Age"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert rows and "score" in rows[0] and rows[0]["features"]["Plan Type"] == "Basic"
    logged = read_prediction_log(log_path)
    assert {"Age", "Plan Type", "score"} <= set(logged.columns) and len(logged) == len(rows)


def test_logged_inputs_feed_the_drift_check(binary_dir, tmp_path):
    log_path = tmp_path / "p.jsonl"
    train = messy_binary(400)
    with TestClient(create_app(binary_dir, prediction_log=log_path)) as c:
        rows = [{"Age": int(a), "Support Calls": int(s), "Plan Type": p, "Is Active": "True",
                 "Total Spend": "100", "Region": "N"}
                for a, s, p in zip(np.full(150, 78), np.full(150, 9), ["Basic"] * 150, strict=True)]
        for i in range(0, 150, 50):
            assert c.post("/predict/batch", json={"customers": rows[i:i + 50]}).status_code == 200
    current = read_prediction_log(log_path)[["Age", "Support Calls", "Plan Type"]]
    report = detect_drift(train[["Age", "Support Calls", "Plan Type"]], current)
    assert report.verdict == "drift"


def test_multiclass_api_returns_class_and_probabilities(multiclass_dir):
    row = {c: float(v) for c, v in wine_df().drop(columns="target").iloc[0].items()}
    with TestClient(create_app(multiclass_dir)) as c:
        body = c.post("/predict", json=row).json()
        assert body["prediction"] in ("class_a", "class_b", "class_c")
        assert set(body["probabilities"]) == {"class_a", "class_b", "class_c"}
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
        assert "probability" not in body and "threshold" not in body
        assert c.post("/predict/batch", json={"customers": [row, row]}).status_code == 200
        text = c.get("/metrics").text
        assert 'ael_predictions_total{prediction="' in text
        assert "ael_churn_probability_count" not in text or "ael_churn_probability_count 0.0" in text


def test_regression_api_returns_a_number(regression_dir):
    row = {c: float(v) for c, v in diabetes_df().drop(columns="target").iloc[0].items()}
    with TestClient(create_app(regression_dir)) as c:
        body = c.post("/predict", json=row).json()
        assert isinstance(body["prediction"], float)
        assert set(body) == {"prediction", "model_version"}
        assert 'ael_predictions_total{prediction="value"}' in c.get("/metrics").text


# ---- web page built from the model -------------------------------------------------------------------

def test_page_form_is_generated_from_the_schema(binary_dir):
    with TestClient(create_app(binary_dir)) as c:
        html = c.get("/app").text
    for name in ("Age", "Support_Calls", "Plan_Type", "Is_Active", "Total_Spend", "Region"):
        assert f'id="f-{name}"' in html
    assert "Premium" in html and "Churn" in html
    assert "Churn risk checker" not in html      # the Telco wording is only for the Telco demo


def test_page_for_other_tasks_mentions_the_target(multiclass_dir, regression_dir):
    with TestClient(create_app(multiclass_dir)) as c:
        assert "classes" in c.get("/app").text
    with TestClient(create_app(regression_dir)) as c:
        spec = c.get("/schema").json()
        assert spec["task"] == "regression" and "number" in spec["notice"]


def test_page_escapes_hostile_column_names(tmp_path):
    df = messy_binary().rename(columns={"Plan Type": "<script>alert(1)</script>"})
    r = train_dataset(df, "Churn", models=("logistic_regression",))
    out = export_result(r, tmp_path / "m", "x")
    with TestClient(create_app(out)) as c:
        html = c.get("/app").text
    assert "<script>alert(1)</script>" not in html.split("<script>")[0]
    assert html.count("<script>") == 1


# ---- retraining policy for other tasks ------------------------------------------------------------------

def test_retraining_policy_works_for_multiclass_and_regression():
    from ai_experiment_lab.retraining import decide_promotion, retrain_candidate, score

    for df, target, task, models in ((wine_df(), "target", "multiclass", ("logistic_regression",)),
                                     (diabetes_df(), "target", "regression", ("ridge",))):
        r = train_dataset(df, target, models=models)
        _, X, y, _, _ = build_schema(df, target)
        old, new = X.iloc[:120], X.iloc[120:]
        cand = retrain_candidate(r.pipeline, old, y.iloc[:120], new, y.iloc[120:])
        metric = promotion_metric(task)
        champ_s, cand_s = score(r.pipeline, new, y.iloc[120:], task), score(cand, new, y.iloc[120:], task)
        champ_n, cand_n = score(r.pipeline, old, y.iloc[:120], task), score(cand, old, y.iloc[:120], task)
        decision = decide_promotion(champ_s, cand_s, champ_n, cand_n, metric=metric)
        assert isinstance(decision.promote, bool) and metric in decision.reasons[0]


def test_check_drift_script_works_for_any_model(binary_dir, tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "check_drift", Path(__file__).resolve().parents[1] / "scripts" / "check_drift.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    csv = tmp_path / "train.csv"
    messy_binary().to_csv(csv, index=False)
    log_path = tmp_path / "p.jsonl"
    with TestClient(create_app(binary_dir, prediction_log=log_path)) as c:
        calm = messy_binary(150, seed=5)
        rows = [{"Age": int(r["Age"]), "Support Calls": int(r["Support Calls"]),
                 "Plan Type": r["Plan Type"], "Is Active": str(r["Is Active"]),
                 "Total Spend": r["Total Spend"],
                 "Region": r["Region"] if isinstance(r["Region"], str) else None}
                for _, r in calm.iterrows()]
        for i in range(0, 150, 50):
            assert c.post("/predict/batch", json={"customers": rows[i:i + 50]}).status_code == 200
    out = tmp_path / "drift.json"
    code = mod.main(["--data", str(csv), "--log", str(log_path), "--model-dir", str(binary_dir),
                     "--out", str(out), "--fail-on-drift"])
    report = json.loads(out.read_text())
    assert code == 0 and report["verdict"] == "no_drift"


def test_export_check_flags_missing_xgboost_in_the_image(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "check_model_export", Path(__file__).resolve().parents[1] / "scripts" / "check_model_export.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    r = train_dataset(diabetes_df(), "target", models=("xgboost",))
    out = export_result(r, tmp_path / "m", "diabetes")
    docker = tmp_path / "Dockerfile"
    import sklearn
    docker.write_text(f"ARG SKLEARN_VERSION={sklearn.__version__}\nARG XGBOOST_VERSION=\n")
    meta = json.loads((out / "metadata.json").read_text())
    assert any("XGBoost" in p for p in check.xgboost_problems(meta, docker))
    docker.write_text(f"ARG SKLEARN_VERSION={sklearn.__version__}\nARG XGBOOST_VERSION={meta['libraries']['xgboost']}\n")
    assert check.xgboost_problems(meta, docker) == []
    assert check.main([str(out), "--dockerfile", str(docker)]) == 0


def test_mlflow_logging_and_registry_for_any_dataset(tmp_path):
    pytest.importorskip("mlflow")
    from ai_experiment_lab.automl.tracking import log_and_register
    from ai_experiment_lab.registry.mlflow_registry import alias_version, load_by_alias

    r = train_dataset(diabetes_df(), "target", models=("ridge",))
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    info = log_and_register(r, "Diabetes Demo", "abc123", tracking_uri=uri)
    assert info["version"] == 1 and info["name"] == "diabetes-demo"
    assert int(alias_version("diabetes-demo", tracking_uri=uri).version) == 1
    loaded = load_by_alias("diabetes-demo", tracking_uri=uri)
    X = diabetes_df().drop(columns="target").head(5)
    assert len(loaded.predict(X)) == 5
    again = log_and_register(r, "Diabetes Demo", "abc123", tracking_uri=uri)
    assert again["version"] == 2          # a new run registers a new version and moves the alias
