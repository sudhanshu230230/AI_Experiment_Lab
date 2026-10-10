"""Tests for ingestion and profiling, using small synthetic frames."""

import numpy as np
import pandas as pd
import pytest

from ai_experiment_lab.data import file_sha256, format_report, load_csv, profile_dataframe
from ai_experiment_lab.utils.config import get_data_dir, load_config


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "age": [20, 30, 40, 50, 60, 1000],          # 1000 is an outlier
            "tenure": [1, 2, 3, 4, 5, 6],
            "double_tenure": [2, 4, 6, 8, 10, 12],      # perfectly correlated with tenure
            "plan": ["x", "y", "x", "x", None, "y"],
            "total": ["1.5", "2.5", " ", "4.0", "5.5", "6.0"],  # numeric text with a blank
            "churn": ["No", "No", "No", "No", "Yes", "No"],
        }
    )


# ---- ingestion ----
def test_load_csv_roundtrip(tmp_path, df):
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    _, info = load_csv(p)
    assert info.n_rows == 6 and info.n_cols == 7
    assert info.name == "d"
    assert info.sha256 == file_sha256(p)
    assert len(info.sha256) == 64


def test_hash_changes_with_content(tmp_path, df):
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    df.to_csv(p1, index=False)
    df.iloc[:-1].to_csv(p2, index=False)
    assert file_sha256(p1) != file_sha256(p2)


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nope.csv")


def test_load_empty_file(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("a,b\n")
    with pytest.raises(ValueError):
        load_csv(p)


# ---- profiling ----
def test_shape_and_types(df):
    p = profile_dataframe(df, target="churn")
    assert (p["n_rows"], p["n_cols"]) == (6, 7)
    assert set(p["numeric_features"]) == {"age", "tenure", "double_tenure"}
    assert "churn" not in p["categorical_features"]
    assert "plan" in p["categorical_features"]


def test_missing_and_blank_reported_separately(df):
    p = profile_dataframe(df, target="churn")
    assert p["missing"]["plan"] == {"nan": 1, "blank": 0}
    assert p["missing"]["total"] == {"nan": 0, "blank": 1}


def test_numeric_like_text_detected(df):
    p = profile_dataframe(df, target="churn")
    assert p["numeric_like_text_features"] == ["total"]
    assert "id" not in p["numeric_like_text_features"]
    assert "plan" not in p["numeric_like_text_features"]


def test_duplicates(df):
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert profile_dataframe(dup)["n_duplicate_rows"] == 1
    assert profile_dataframe(df)["n_duplicate_rows"] == 0


def test_outliers_and_stats(df):
    p = profile_dataframe(df)
    assert p["numeric_stats"]["age"]["iqr_outliers"] == 1
    assert p["numeric_stats"]["tenure"]["iqr_outliers"] == 0
    assert p["numeric_stats"]["tenure"]["mean"] == pytest.approx(3.5)


def test_high_correlation(df):
    p = profile_dataframe(df, corr_threshold=0.99)
    pairs = {(h["a"], h["b"]) for h in p["high_correlations"]}
    assert ("tenure", "double_tenure") in pairs


def test_target_imbalance(df):
    t = profile_dataframe(df, target="churn")["target"]
    assert t["distribution"] == {"No": 5, "Yes": 1}
    assert t["minority_fraction"] == pytest.approx(1 / 6, abs=1e-3)
    assert t["imbalance_ratio"] == 5.0


def test_missing_target_raises(df):
    with pytest.raises(KeyError):
        profile_dataframe(df, target="nope")


def test_constant_numeric_column_ok():
    p = profile_dataframe(pd.DataFrame({"x": [1, 1, 1, 1]}))
    assert p["numeric_stats"]["x"]["iqr_outliers"] == 0


def test_all_nan_numeric_column_ok():
    p = profile_dataframe(pd.DataFrame({"x": [np.nan, np.nan], "y": [1, 2]}))
    assert "x" not in p["numeric_stats"]


def test_report_mentions_key_findings(df):
    text = format_report(profile_dataframe(df, target="churn"))
    assert "Imbalance ratio" in text and "total" in text and "WARNING" in text


# ---- config ----
def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n")
    assert load_config(p) == {"a": 1}
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(TypeError):
        load_config(bad)


def test_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AEL_DATA_DIR", str(tmp_path))
    assert get_data_dir() == tmp_path
    monkeypatch.delenv("AEL_DATA_DIR")
    assert str(get_data_dir()).replace("\\", "/") == "data/raw"
