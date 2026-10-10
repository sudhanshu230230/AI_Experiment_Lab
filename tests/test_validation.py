"""Tests for data validation. Each rule has a passing and a failing case."""

import numpy as np
import pandas as pd
import pytest

from ai_experiment_lab.data import ValidationError, validate_dataframe, validation_thresholds
from ai_experiment_lab.utils.config import load_config


@pytest.fixture
def good():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(n)],
            "tenure": rng.integers(0, 72, n),
            "plan": rng.choice(["a", "b", "c"], n),
            "Churn": rng.choice(["No", "Yes"], n, p=[0.6, 0.4]),
        }
    )


def status(res, name):
    return next(c.status for c in res.checks if c.name == name)


def run(df, **kw):
    return validate_dataframe(df, target="Churn", id_columns=["customerID"], **kw)


def test_good_data_accepted(good):
    res = run(good)
    assert res.accepted
    assert not res.failures
    res.raise_if_rejected()


def test_missing_target_rejected(good):
    res = validate_dataframe(good.drop(columns="Churn"), target="Churn")
    assert not res.accepted
    assert status(res, "target_exists") == "FAIL"


def test_target_with_nan_rejected(good):
    good.loc[0, "Churn"] = None
    assert status(run(good), "target_valid") == "FAIL"


def test_single_class_rejected(good):
    good["Churn"] = "No"
    assert status(run(good), "target_valid") == "FAIL"


def test_regression_target_must_be_numeric(good):
    assert status(run(good, task="regression"), "target_valid") == "FAIL"


def test_unknown_task_raises(good):
    with pytest.raises(ValueError):
        run(good, task="clustering")


def test_unsupported_dtype_rejected(good):
    good["when"] = pd.date_range("2024-01-01", periods=len(good))
    res = run(good)
    assert status(res, "supported_dtypes") == "FAIL"
    with pytest.raises(ValidationError):
        res.raise_if_rejected()


def test_missing_limit(good):
    good.loc[:99, "plan"] = None  # 50% missing
    assert status(run(good), "missing_values") == "FAIL"
    assert status(run(good, thresholds={"max_missing_fraction": 0.6}), "missing_values") == "PASS"


def test_blank_strings_count_as_missing(good):
    good.loc[:99, "plan"] = " "
    assert status(run(good), "missing_values") == "FAIL"


def test_duplicate_limit(good):
    dup = pd.concat([good, good.iloc[:50]], ignore_index=True)
    assert status(run(dup), "duplicates") == "FAIL"
    assert status(run(good), "duplicates") == "PASS"


def test_class_balance_levels(good):
    good["Churn"] = ["Yes"] * 100 + ["No"] * 100
    assert status(run(good), "class_balance") == "PASS"
    good["Churn"] = ["Yes"] * 60 + ["No"] * 140  # ratio 2.33 -> moderate imbalance
    assert status(run(good), "class_balance") == "WARN"
    good["Churn"] = ["Yes"] * 4 + ["No"] * 196
    res = run(good)
    assert status(res, "class_balance") == "WARN"
    assert res.accepted  # imbalance warns, never rejects


def test_numeric_like_text_warns(good):
    good["total"] = good["tenure"].astype(str)
    good.loc[0, "total"] = " "
    res = run(good)
    assert status(res, "numeric_like_text") == "WARN"
    assert res.accepted


def test_id_column_not_flagged_when_configured(good):
    assert status(run(good), "id_columns") == "PASS"
    res = validate_dataframe(good, target="Churn")  # id not configured
    assert status(res, "id_like_columns") == "WARN"


def test_missing_configured_id_warns(good):
    res = validate_dataframe(good, target="Churn", id_columns=["nope"])
    assert status(res, "id_columns") == "WARN"


def test_constant_column_warns(good):
    good["const"] = 1
    assert status(run(good), "constant_columns") == "WARN"


def test_schema_check(good):
    cols = ["customerID", "tenure", "plan"]
    assert status(run(good, expected_columns=cols), "schema") == "PASS"
    assert status(run(good, expected_columns=cols + ["gone"]), "schema") == "FAIL"
    assert status(run(good, expected_columns=cols[:2]), "schema") == "WARN"


def test_report_text(good):
    text = run(good).report()
    assert text.startswith("DATA VALIDATION")
    assert "Dataset accepted" in text


def test_thresholds_from_real_config():
    cfg = load_config("configs/default.yaml")
    th = validation_thresholds(cfg)
    assert th["max_missing_fraction"] == 0.30
