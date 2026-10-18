"""Tests for the prediction API, using a small model exported from synthetic Telco-shaped data."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("skops")

from fastapi.testclient import TestClient

from ai_experiment_lab.preprocessing import split_data
from ai_experiment_lab.registry import export_pipeline
from ai_experiment_lab.serving.app import create_app
from ai_experiment_lab.serving.schemas import CustomerFeatures
from ai_experiment_lab.training.trainer import make_model_pipeline

PRE = {"coerce_numeric": ["TotalCharges"]}
EXAMPLE = CustomerFeatures.model_config["json_schema_extra"]["example"]


def _synthetic_telco(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    yn = ["Yes", "No"]
    yni = ["Yes", "No", "No internet service"]
    tenure = rng.integers(0, 72, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    monthly = rng.uniform(20, 110, n).round(2)
    logit = -1.0 + 2.0 * (contract == "Month-to-month") - 0.04 * tenure
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(yn, n),
            "Dependents": rng.choice(yn, n),
            "tenure": tenure,
            "PhoneService": rng.choice(yn, n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(yni, n),
            "OnlineBackup": rng.choice(yni, n),
            "DeviceProtection": rng.choice(yni, n),
            "TechSupport": rng.choice(yni, n),
            "StreamingTV": rng.choice(yni, n),
            "StreamingMovies": rng.choice(yni, n),
            "Contract": contract,
            "PaperlessBilling": rng.choice(yn, n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                 "Credit card (automatic)"], n),
            "MonthlyCharges": monthly,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    df["TotalCharges"] = (tenure * monthly).round(2).astype(str)
    df.loc[df.tenure == 0, "TotalCharges"] = " "
    return df


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    split = split_data(_synthetic_telco(), "Churn", "Yes", id_columns=["customerID"], seed=1)
    pipe = make_model_pipeline(
        "logistic_regression", split, {"max_iter": 300, "class_weight": "balanced"}, PRE, 1
    )
    pipe.fit(split.X_train, split.y_train)
    out = tmp_path_factory.mktemp("champion")
    export_pipeline(pipe, out, {"registered_name": "churn-model", "registry_version": 3,
                                "alias": "champion", "positive_label": "Yes", "threshold": 0.5,
                                "metrics": {"test_f1": 0.6}, "libraries": {}})
    return out


@pytest.fixture()
def client(model_dir):
    with TestClient(create_app(model_dir)) as c:
        yield c


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model").json()
    assert info["registry_version"] == 3 and len(info["input_columns"]) == 19


def test_predict_returns_valid_probability_and_consistent_label(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1 and body["model_version"] == 3
    assert body["prediction"] == ("Yes" if body["churn_probability"] >= 0.5 else "No")


@pytest.mark.parametrize("total", [None, "", " ", 0, "123.4", 99.5])
def test_total_charges_accepts_blank_text_number_and_null(client, total):
    assert client.post("/predict", json={**EXAMPLE, "TotalCharges": total}).status_code == 200


def test_batch_matches_single_predictions(client):
    other = {**EXAMPLE, "Contract": "Two year", "tenure": 60}
    singles = [client.post("/predict", json=c).json()["churn_probability"] for c in (EXAMPLE, other)]
    batch = client.post("/predict/batch", json={"customers": [EXAMPLE, other]}).json()
    assert [p["churn_probability"] for p in batch["predictions"]] == singles


@pytest.mark.parametrize(
    "bad",
    [
        {**{k: v for k, v in EXAMPLE.items() if k != "tenure"}},          # missing field
        {**EXAMPLE, "Contract": "Weekly"},                                # unknown category
        {**EXAMPLE, "tenure": -1},                                        # out of range
        {**EXAMPLE, "MonthlyCharges": "abc"},                             # wrong type
        {**EXAMPLE, "extra_field": 1},                                    # misspelt/extra field
    ],
)
def test_invalid_requests_are_rejected_with_422(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    assert client.post("/predict/batch", json={"customers": [EXAMPLE] * 1001}).status_code == 422


@pytest.mark.parametrize("bad_total", ["n/a", "12abc", -5, "-1"])
def test_bad_total_charges_is_a_422_not_a_crash(client, bad_total):
    assert client.post("/predict", json={**EXAMPLE, "TotalCharges": bad_total}).status_code == 422


def test_missing_model_gives_degraded_health_and_503(tmp_path):
    with TestClient(create_app(tmp_path / "nope")) as c:
        assert c.get("/health").json() == {"status": "degraded", "model_loaded": False}
        assert c.post("/predict", json=EXAMPLE).status_code == 503
        assert c.get("/model").status_code == 503


def test_prediction_log_records_features_and_version(model_dir, tmp_path):
    log_path = tmp_path / "preds.jsonl"
    with TestClient(create_app(model_dir, prediction_log=log_path)) as c:
        c.post("/predict", json=EXAMPLE)
        c.post("/predict/batch", json={"customers": [EXAMPLE, EXAMPLE]})
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["model_version"] == 3
    assert rows[0]["features"]["Contract"] == "Month-to-month" and "ts" in rows[0]


def test_schema_mismatch_with_model_is_reported_as_degraded(tmp_path):
    split = split_data(_synthetic_telco(), "Churn", "Yes", id_columns=["customerID"], seed=1)
    X = split.X_train.drop(columns=["gender"])
    from sklearn.pipeline import Pipeline

    from ai_experiment_lab.models import build_classifier
    from ai_experiment_lab.preprocessing import build_preprocessing_pipeline

    pipe = Pipeline([("preprocess", build_preprocessing_pipeline(X, coerce_numeric=["TotalCharges"])),
                     ("model", build_classifier("logistic_regression", {"max_iter": 200}))])
    pipe.fit(X, split.y_train)
    export_pipeline(pipe, tmp_path / "m", {})
    with TestClient(create_app(tmp_path / "m")) as c:
        assert c.get("/health").json()["status"] == "degraded"


# ---- Prometheus metrics (Phase 6) ------------------------------------------------------------

def _metric(client, name, labels=None):
    """Value of one sample from /metrics, or None if absent."""
    from prometheus_client.parser import text_string_to_metric_families

    for fam in text_string_to_metric_families(client.get("/metrics").text):
        for sample in fam.samples:
            if sample.name == name and all(sample.labels.get(k) == v for k, v in (labels or {}).items()):
                return sample.value
    return None


def test_metrics_count_requests_by_route_template_and_status(client):
    client.post("/predict", json=EXAMPLE)
    client.post("/predict", json={**EXAMPLE, "Contract": "Weekly"})
    client.get("/does-not-exist")
    ok = {"method": "POST", "path": "/predict", "status": "200"}
    bad = {"method": "POST", "path": "/predict", "status": "422"}
    assert _metric(client, "ael_http_requests_total", ok) == 1
    assert _metric(client, "ael_http_requests_total", bad) == 1
    assert _metric(client, "ael_http_requests_total", {"path": "unmatched", "status": "404"}) == 1
    assert _metric(client, "ael_http_request_duration_seconds_count", {"path": "/predict"}) == 2


def test_metrics_endpoint_itself_is_not_counted(client):
    client.get("/metrics")
    client.get("/metrics")
    assert _metric(client, "ael_http_requests_total", {"path": "/metrics"}) is None


def test_prediction_metrics_count_each_prediction_in_a_batch(client):
    client.post("/predict/batch", json={"customers": [EXAMPLE, EXAMPLE, EXAMPLE]})
    yes = _metric(client, "ael_predictions_total", {"prediction": "Yes"}) or 0
    no = _metric(client, "ael_predictions_total", {"prediction": "No"}) or 0
    assert yes + no == 3
    assert _metric(client, "ael_churn_probability_count") == 3
    assert _metric(client, "ael_churn_probability_bucket", {"le": "1.0"}) == 3


def test_model_info_and_loaded_gauges(client, tmp_path):
    assert _metric(client, "ael_model_loaded") == 1
    assert _metric(client, "ael_model_info", {"version": "3", "registered_name": "churn-model"}) == 1
    with TestClient(create_app(tmp_path / "nope")) as c:
        assert _metric(c, "ael_model_loaded") == 0
        assert _metric(c, "ael_model_info") is None


def test_each_app_has_its_own_registry(model_dir):
    with TestClient(create_app(model_dir)) as a, TestClient(create_app(model_dir)) as b:
        a.post("/predict", json=EXAMPLE)
        assert _metric(a, "ael_predictions_total", {"prediction": "Yes"}) is not None or \
            _metric(a, "ael_predictions_total", {"prediction": "No"}) is not None
        assert _metric(b, "ael_http_requests_total", {"path": "/predict"}) is None


def test_unwritable_prediction_log_does_not_break_predictions(model_dir, tmp_path):
    blocker = tmp_path / "file_not_dir"
    blocker.write_text("x")            # a file where the log's parent folder should be
    with TestClient(create_app(model_dir, prediction_log=blocker / "preds.jsonl")) as c:
        assert c.post("/predict", json=EXAMPLE).status_code == 200


def test_root_redirects_to_the_web_page(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"].endswith("app")
    assert client.get("/", follow_redirects=True).status_code == 200


def test_web_page_is_served_and_covers_every_schema_field(client):
    r = client.get("/app")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    for field in CustomerFeatures.model_fields:
        assert f'id="f-{field}"' in r.text
    assert client.get("/docs").status_code == 200  # API docs still reachable


def test_web_page_options_come_from_the_schema(client):
    page = client.get("/app").text
    for value in ("Fiber optic", "No internet service", "Bank transfer (automatic)"):
        assert value in page


def test_web_page_is_not_in_the_api_schema(client):
    assert "/app" not in client.get("/openapi.json").json()["paths"]
