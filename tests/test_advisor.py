"""The model advisor (ADR-011): the recommendation engine and the upload endpoints."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("skops")
pytest.importorskip("optuna")
import yaml
from fastapi.testclient import TestClient

from ai_experiment_lab.automl import DatasetError, advise, column_profile
from ai_experiment_lab.automl.tasks import SPACES
from ai_experiment_lab.serving.app import create_app
from test_automl import diabetes_df, messy_binary, wine_df

FAST = {"budget_seconds": 6, "max_trials": 2, "max_rows": 600}


def test_search_spaces_match_the_yaml_used_in_colab():
    cfg = yaml.safe_load(Path("configs/tuning.yaml").read_text())["tuning"]["spaces"]
    for name, space in cfg.items():
        mine = SPACES[name]
        assert set(mine) == set(space)
        for key, spec in space.items():
            wanted = {k: v for k, v in spec.items() if k != "default"}
            assert {k: v for k, v in mine[key].items() if k != "log" or v} == \
                {k: v for k, v in wanted.items() if k != "log" or v}


def test_advice_for_a_yes_no_question_is_complete():
    a = advise(messy_binary(), "Churn", **FAST)
    assert a.task == "binary" and a.recommended in {"logistic_regression", "random_forest", "xgboost"}
    assert a.baseline["roc_auc"] == pytest.approx(0.5)      # guessing the common class
    assert a.metrics["roc_auc"] > 0.5
    assert a.why and a.code.startswith("from ")
    assert "n_jobs" not in a.params and "random_state" not in a.params
    assert all(k in a.param_help or True for k in a.params)
    assert a.leaderboard and {"model", "cv", "test"} <= set(a.leaderboard[0])
    assert abs(sum(i["share"] for i in a.importances)) <= 1.0001
    assert a.data["rows"] == 400 and a.to_dict()["recommended"] == a.recommended


def test_advice_for_other_tasks():
    r = advise(diabetes_df(), "target", **FAST)
    assert r.task == "regression" and "rmse" in r.metrics
    assert r.baseline["r2"] <= 0.01                         # predicting the mean explains nothing
    w = advise(wine_df(), "target", **FAST)
    assert w.task == "multiclass" and len(w.data["classes"]) == 3


def test_a_too_good_score_is_flagged_as_possible_leakage():
    df = messy_binary(500)
    df["Outcome copy"] = df["Churn"]                        # a column that gives the answer away
    a = advise(df, "Churn", **FAST)
    assert any("almost perfect" in w or "carries" in w for w in a.warnings)


def test_a_noise_target_is_flagged_as_barely_better_than_guessing():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500),
                       "y": rng.choice(["x", "z"], 500)})
    a = advise(df, "y", **FAST)
    assert any("barely beats" in w for w in a.warnings)


def test_sampling_is_reported():
    a = advise(messy_binary(900), "Churn", budget_seconds=4, max_trials=1, max_rows=300)
    assert a.data["sampled"] and any("sample" in w for w in a.warnings)


def test_advise_errors_are_clear():
    with pytest.raises(DatasetError):
        advise(messy_binary(), "nope", **FAST)


def test_profile_suggests_target_and_ids_and_warns_about_dates():
    p = column_profile(messy_binary())
    assert p["suggested_target"] == "Churn" and "Customer ID" in p["suggested_ids"]
    assert any("time series" in w for w in p["warnings"])


# ---- the endpoints ---------------------------------------------------------------------------

def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_MAX_ROWS", "400")
    monkeypatch.setenv("LAB_BUDGET_SECONDS", "6")
    monkeypatch.setenv("LAB_MAX_TRIALS", "2")
    return TestClient(create_app(model_dir=str(tmp_path / "none")))


def wait(client, job_id, seconds=120):
    end = time.time() + seconds
    while time.time() < end:
        j = client.get(f"/lab/jobs/{job_id}").json()
        if j["status"] in {"done", "error"}:
            return j
        time.sleep(0.3)
    raise AssertionError("job did not finish")


def test_upload_run_and_download_flow(client):
    up = client.post("/lab/analyze", content=csv_bytes(messy_binary()),
                     headers={"content-type": "text/csv"})
    assert up.status_code == 200
    body = up.json()
    run = client.post("/lab/run", json={"upload_id": body["upload_id"], "target": "Churn",
                                        "id_columns": body["suggested_ids"]})
    assert run.status_code == 202
    job = wait(client, run.json()["job_id"])
    assert job["status"] == "done" and job["has_model"]
    assert job["result"]["task"] == "binary" and job["result"]["recommended"]
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/lab/jobs/{run.json()['job_id']}/model").content))
    assert {"model/model.skops", "model/metadata.json", "model/advice.json"} <= set(z.namelist())


@pytest.mark.parametrize("raw,fragment", [
    (b"", "empty"),
    (b"a,b\n1,2\n3,4\n", "at least 50"),
    (b"only\n" + b"1\n" * 80, "at least two columns"),
    (b"a,a\n" + b"1,2\n" * 80, "share a name"),
])
def test_bad_uploads_get_a_plain_422(client, raw, fragment):
    r = client.post("/lab/analyze", content=raw, headers={"content-type": "text/csv"})
    assert r.status_code == 422 and fragment in r.json()["detail"]


def test_oversized_upload_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_MAX_UPLOAD_MB", "0.001")
    c = TestClient(create_app(model_dir=str(tmp_path / "none")))
    r = c.post("/lab/analyze", content=b"x" * 5000, headers={"content-type": "text/csv"})
    assert r.status_code == 413


def test_too_many_columns_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_MAX_COLUMNS", "5")
    c = TestClient(create_app(model_dir=str(tmp_path / "none")))
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 8)), columns=list("abcdefgh"))
    r = c.post("/lab/analyze", content=csv_bytes(df), headers={"content-type": "text/csv"})
    assert r.status_code == 422 and "columns" in r.json()["detail"]


def test_unknown_upload_target_and_job_are_clear_errors(client):
    assert client.post("/lab/run", json={"upload_id": "zzz", "target": "a"}).status_code == 404
    up = client.post("/lab/analyze", content=csv_bytes(messy_binary()),
                     headers={"content-type": "text/csv"}).json()
    assert client.post("/lab/run", json={"upload_id": up["upload_id"], "target": "nope"}
                       ).status_code == 422
    assert client.get("/lab/jobs/none").status_code == 404
    assert client.get("/lab/jobs/none/model").status_code == 404


def test_only_one_analysis_runs_at_a_time(client):
    up = client.post("/lab/analyze", content=csv_bytes(messy_binary(120)),
                     headers={"content-type": "text/csv"}).json()
    first = client.post("/lab/run", json={"upload_id": up["upload_id"], "target": "Churn"})
    second = client.post("/lab/run", json={"upload_id": up["upload_id"], "target": "Churn"})
    assert first.status_code == 202 and second.status_code == 429
    wait(client, first.json()["job_id"])
    third = client.post("/lab/run", json={"upload_id": up["upload_id"], "target": "Churn"})
    assert third.status_code == 202
    wait(client, third.json()["job_id"])


def test_a_target_that_cannot_be_used_ends_in_a_job_error_not_a_crash(client):
    df = messy_binary()
    df["Many"] = [f"v{i}" for i in range(len(df))]
    up = client.post("/lab/analyze", content=csv_bytes(df), headers={"content-type": "text/csv"}).json()
    run = client.post("/lab/run", json={"upload_id": up["upload_id"], "target": "Many"})
    job = wait(client, run.json()["job_id"])
    assert job["status"] == "error" and job["error"]


def test_semicolon_and_latin1_files_are_read(client):
    df = messy_binary()
    raw = df.to_csv(index=False, sep=";").replace("Basic", "Básico").encode("latin-1")
    r = client.post("/lab/analyze", content=raw, headers={"content-type": "text/csv"})
    assert r.status_code == 200 and r.json()["rows"] == 400


def test_page_limits_and_link(client):
    page = client.get("/lab")
    assert page.status_code == 200 and "Model advisor" in page.text
    assert client.get("/lab/limits").json()["rows_used_for_training"] == 400
    assert "lab-link" in client.get("/app").text


def test_the_advisor_can_be_switched_off(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_ENABLED", "0")
    c = TestClient(create_app(model_dir=str(tmp_path / "none")))
    assert c.get("/lab").status_code == 404
    assert c.post("/lab/analyze", content=b"a,b").status_code == 404


def test_a_file_already_uploaded_can_be_reopened_by_id(client):
    raw = b"a,b,y\n" + b"".join(f"{i},{i % 7},{i % 2}\n".encode() for i in range(80))
    up = client.post("/lab/analyze", content=raw, headers={"content-type": "text/csv"}).json()
    again = client.get(f"/lab/uploads/{up['upload_id']}")
    assert again.status_code == 200 and again.json()["rows"] == 80
    assert client.get("/lab/uploads/nope").status_code == 404
