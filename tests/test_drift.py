"""Tests for drift detection and the prediction log reader."""

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_experiment_lab.monitoring import (
    DRIFT,
    INSUFFICIENT,
    NO_DRIFT,
    detect_drift,
    read_prediction_log,
)
from ai_experiment_lab.monitoring.drift import categorical_stats, infer_kind, numeric_psi

GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "load_generator.py"


def frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.normal(65, 25, n).round(2),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(["A", "B", "C", "D"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "gender": rng.choice(["Male", "Female"], n),
        }
    )


REF = frame(4000, 0)


def test_infer_kind():
    assert infer_kind(REF["tenure"]) == "numeric"
    assert infer_kind(REF["SeniorCitizen"]) == "categorical"      # 0/1 flag
    assert infer_kind(REF["Contract"]) == "categorical"


@pytest.mark.parametrize("seed", range(1, 21))
def test_no_false_alarm_when_the_distribution_is_unchanged(seed):
    rep = detect_drift(REF, frame(500, seed))
    assert rep.verdict == NO_DRIFT
    assert rep.drifted_features == []


def test_numeric_shift_is_flagged():
    cur = frame(500, 1)
    cur["tenure"] = (cur["tenure"] * 0.3).astype(int)
    rep = detect_drift(REF, cur)
    f = {x.feature: x for x in rep.features}["tenure"]
    assert f.drifted and f.psi > 0.2 and f.p_value < 0.05 and f.test == "ks"


def test_categorical_shift_and_unseen_category_are_flagged():
    cur = frame(500, 1)
    cur["Contract"] = "Month-to-month"
    cur["PaymentMethod"] = "E"                                    # never seen in the reference
    rep = detect_drift(REF, cur)
    assert {"Contract", "PaymentMethod"} <= set(rep.drifted_features)


def test_dataset_verdict_needs_enough_drifted_features():
    one = frame(500, 1)
    one["tenure"] = (one["tenure"] * 0.3).astype(int)             # 1 of 10 columns
    assert detect_drift(REF, one).verdict == NO_DRIFT

    many = one.copy()
    many["Contract"] = "Month-to-month"
    many["InternetService"] = "Fiber optic"
    many["MonthlyCharges"] = many["MonthlyCharges"] + 40          # 4 of 10 columns
    rep = detect_drift(REF, many)
    assert rep.verdict == DRIFT and rep.drifted_share == pytest.approx(0.4)


def test_too_few_current_rows_is_insufficient_data_not_drift():
    cur = frame(50, 1)
    cur["Contract"] = "Month-to-month"
    cur["tenure"] = 0
    cur["InternetService"] = "Fiber optic"
    cur["MonthlyCharges"] = 200.0
    assert detect_drift(REF, cur).verdict == INSUFFICIENT


def test_tiny_shift_with_huge_sample_is_not_drift_even_though_p_is_tiny():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 200_000)})
    cur = pd.DataFrame({"x": rng.normal(0.02, 1, 200_000)})
    f = detect_drift(ref, cur).features[0]
    assert f.p_value < 0.05 and f.psi < 0.2 and not f.drifted


def test_psi_properties():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert numeric_psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert numeric_psi(a, rng.normal(1, 1, 5000)) > 0.2
    assert numeric_psi(np.full(100, 3.0), np.full(100, 3.0)) == pytest.approx(0, abs=1e-9)
    assert numeric_psi(np.full(100, 3.0), np.full(100, 9.0)) > 0.2    # constant reference handled
    _, p, psi = categorical_stats(pd.Series(["a"] * 50), pd.Series(["a"] * 50))
    assert p == 1.0 and psi == pytest.approx(0, abs=1e-9)


def test_missing_values_do_not_crash():
    cur = frame(300, 1).astype({"MonthlyCharges": float})
    cur.loc[:150, "MonthlyCharges"] = np.nan
    cur["Partner"] = None
    rep = detect_drift(REF, cur)
    assert len(rep.features) == 10
    allnan = frame(300, 1)
    allnan["MonthlyCharges"] = np.nan
    assert detect_drift(REF, allnan).features is not None


def test_prediction_drift_is_reported_but_does_not_change_the_verdict():
    rng = np.random.default_rng(0)
    ref_scores = rng.beta(2, 5, 4000)
    cur_scores = rng.beta(5, 2, 500)
    rep = detect_drift(REF, frame(500, 1), reference_scores=ref_scores, current_scores=cur_scores)
    assert rep.prediction.drifted and rep.verdict == NO_DRIFT


def test_input_validation_and_serialisation():
    with pytest.raises(ValueError):
        detect_drift(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))
    with pytest.raises(ValueError):
        detect_drift(REF, frame(200, 1), columns=["nope"])
    rep = detect_drift(REF, frame(200, 1))
    json.dumps(rep.to_dict())                                      # JSON-serialisable
    assert len(rep.to_frame()) == 10 and "DRIFT VERDICT" in rep.summary()


# ---- integration with the traffic generator ---------------------------------------------

def _generator():
    spec = importlib.util.spec_from_file_location("load_generator", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _customers(drift: bool, n: int, seed: int) -> pd.DataFrame:
    gen = _generator()
    rng = random.Random(seed)
    df = pd.DataFrame([gen.customer(rng, drift) for _ in range(n)])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def test_generator_normal_vs_normal_is_no_drift_and_normal_vs_drifted_is_drift():
    ref = _customers(False, 3000, 0)
    assert detect_drift(ref, _customers(False, 500, 1)).verdict == NO_DRIFT
    rep = detect_drift(ref, _customers(True, 500, 2))
    assert rep.verdict == DRIFT
    assert {"Contract", "tenure", "InternetService", "MonthlyCharges"} <= set(rep.drifted_features)


# ---- prediction log ----------------------------------------------------------------------

def test_read_prediction_log_skips_bad_lines_and_filters_versions(tmp_path):
    path = tmp_path / "preds.jsonl"
    good = lambda v, p: json.dumps(
        {"ts": "2026-10-16T01:00:00+00:00", "model_version": v, "churn_probability": p,
         "features": {"tenure": 5, "Contract": "One year"}})
    path.write_text("\n".join([good(1, 0.2), "", "{not json", good(2, 0.9),
                               json.dumps({"ts": "x"}), good(1, 0.4)]) + "\n")
    df = read_prediction_log(path)
    assert len(df) == 3 and set(df.columns) >= {"tenure", "Contract", "ts", "churn_probability"}
    assert str(df["ts"].dt.tz) == "UTC"
    assert len(read_prediction_log(path, model_version=1)) == 2
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert read_prediction_log(empty).empty


def test_numbers_stored_as_text_are_compared_as_numbers():
    rng = np.random.default_rng(0)
    ref = pd.Series([str(round(x, 2)) for x in rng.uniform(20, 8000, 3000)] + [" "] * 10)
    assert infer_kind(ref) == "numeric"
    assert infer_kind(pd.Series(["a", "b", "c"] * 50)) == "categorical"
    assert infer_kind(pd.Series(["Yes", "No"] * 50)) == "categorical"
    cur = pd.Series([str(round(x, 2)) for x in rng.uniform(20, 8000, 500)])
    shifted = pd.Series([str(round(x, 2)) for x in rng.uniform(5000, 9000, 500)])
    same = detect_drift(pd.DataFrame({"TotalCharges": ref}), pd.DataFrame({"TotalCharges": cur}))
    moved = detect_drift(pd.DataFrame({"TotalCharges": ref}), pd.DataFrame({"TotalCharges": shifted}))
    assert not same.features[0].drifted and moved.features[0].drifted


def _load_script(name: str):
    path = GENERATOR.with_name(name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_drift_script_end_to_end(tmp_path):
    gen, check = _load_script("load_generator.py"), _load_script("check_drift.py")
    rng = random.Random(5)
    df = pd.DataFrame([gen.customer(rng, False) for _ in range(1500)])
    df.insert(0, "customerID", [f"C{i}" for i in range(len(df))])
    df["Churn"] = [rng.choice(["Yes", "No"]) for _ in range(len(df))]
    data = tmp_path / "data.csv"
    df.to_csv(data, index=False)

    def write_log(path, drift):
        with open(path, "w", encoding="utf-8") as f:
            for _ in range(200):
                rec = {"ts": "2026-10-17T10:00:00+00:00", "model_version": 1,
                       "churn_probability": 0.3, "features": gen.customer(rng, drift)}
                f.write(json.dumps(rec) + "\n")

    ok, bad, out = tmp_path / "ok.jsonl", tmp_path / "bad.jsonl", tmp_path / "r" / "out.json"
    write_log(ok, False)
    write_log(bad, True)
    root = GENERATOR.parents[1]
    base = ["--data", str(data), "--config", str(root / "configs/default.yaml"),
            "--monitoring-config", str(root / "configs/monitoring.yaml"), "--fail-on-drift"]
    assert check.main([*base, "--log", str(ok)]) == 0
    assert check.main([*base, "--log", str(bad), "--out", str(out)]) == 1
    assert json.loads(out.read_text())["verdict"] == "drift"
    (tmp_path / "empty.jsonl").write_text("")
    assert check.main([*base, "--log", str(tmp_path / "empty.jsonl")]) == 0
