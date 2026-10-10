"""Tests for preprocessing: correctness, leakage prevention and train/serve consistency."""

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from ai_experiment_lab.preprocessing import (
    NumericCoercer,
    build_preprocessing_pipeline,
    encode_target,
    infer_feature_columns,
    split_data,
)


@pytest.fixture
def raw():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(n)],
            "tenure": rng.integers(0, 72, n).astype(float),
            "MonthlyCharges": rng.uniform(20, 110, n).round(2),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Churn": rng.choice(["No", "Yes"], n, p=[0.73, 0.27]),
        }
    )
    df["TotalCharges"] = (df.tenure * df.MonthlyCharges).round(2).astype(str)
    df.loc[df.tenure == 0, "TotalCharges"] = " "
    return df


COERCE = ["TotalCharges"]


def split(raw, **kw):
    return split_data(raw, "Churn", "Yes", id_columns=["customerID"], **kw)


# ---- NumericCoercer ----
def test_coercer_converts_and_blanks_become_nan():
    df = pd.DataFrame({"t": ["1.5", " ", "3", None]})
    before = df.copy()
    out = NumericCoercer(["t"]).fit_transform(df)
    assert out["t"].tolist()[0] == 1.5 and out["t"].tolist()[2] == 3.0
    assert out["t"].isna().tolist() == [False, True, False, True]
    pd.testing.assert_frame_equal(df, before)  # input not mutated


def test_coercer_rejects_garbage():
    with pytest.raises(ValueError, match="non-numeric"):
        NumericCoercer(["t"]).fit_transform(pd.DataFrame({"t": ["1", "abc"]}))


def test_coercer_missing_column():
    with pytest.raises(KeyError):
        NumericCoercer(["nope"]).fit_transform(pd.DataFrame({"t": ["1"]}))


# ---- columns and target ----
def test_infer_columns(raw):
    X = raw.drop(columns=["Churn", "customerID"])
    num, cat = infer_feature_columns(X, COERCE)
    assert set(num) == {"tenure", "MonthlyCharges", "TotalCharges"}
    assert set(cat) == {"Contract", "Partner"}


def test_encode_target():
    y = encode_target(pd.Series(["No", "Yes", "No"]), "Yes")
    assert y.tolist() == [0, 1, 0]


def test_encode_target_errors():
    with pytest.raises(ValueError):
        encode_target(pd.Series(["a", "b"]), "Yes")          # positive label absent
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Yes", "No", "Maybe"]), "Yes")  # three classes
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Yes", "No", None]), "Yes")      # missing target


# ---- split ----
def test_split_excludes_id_and_target(raw):
    s = split(raw)
    for X in (s.X_train, s.X_test):
        assert "customerID" not in X.columns and "Churn" not in X.columns
    assert len(s.X_train) + len(s.X_test) == len(raw)
    assert not set(s.X_train.index) & set(s.X_test.index)


def test_split_stratified_and_reproducible(raw):
    a, b = split(raw, seed=7), split(raw, seed=7)
    assert a.X_train.index.equals(b.X_train.index)
    assert a.y_train.mean() == pytest.approx(a.y_test.mean(), abs=0.03)
    assert not split(raw, seed=8).X_train.index.equals(a.X_train.index)


# ---- pipeline behaviour ----
def test_output_has_no_nan_and_expected_width(raw):
    s = split(raw)
    pipe = build_preprocessing_pipeline(s.X_train, COERCE).fit(s.X_train)
    out = pipe.transform(s.X_test)
    assert not np.isnan(out).any()
    assert out.shape[1] == 3 + 3 + 2  # 3 numeric + 3 Contract + 2 Partner levels
    assert len(pipe.get_feature_names_out()) == out.shape[1]


def test_scaler_is_fitted_on_train_only(raw):
    """Leakage check: learned statistics must equal training-split statistics."""
    s = split(raw)
    pipe = build_preprocessing_pipeline(s.X_train, COERCE).fit(s.X_train)
    scaler = pipe.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    cols = ["tenure", "MonthlyCharges"]  # numeric order: tenure, MonthlyCharges, TotalCharges
    train_means = s.X_train[cols].mean().to_numpy()
    all_means = pd.concat([s.X_train, s.X_test])[cols].mean().to_numpy()
    assert np.allclose(scaler.mean_[:2], train_means)
    assert not np.allclose(scaler.mean_[:2], all_means, atol=1e-9)


def test_imputer_uses_training_median():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan], "c": ["a", "a", "b", "a"]})
    test = pd.DataFrame({"x": [np.nan, 1000.0], "c": ["a", "b"]})
    pipe = build_preprocessing_pipeline(train).fit(train)
    imputer = pipe.named_steps["columns"].named_transformers_["num"].named_steps["impute"]
    assert imputer.statistics_[0] == 2.0  # median of the training values only
    assert not np.isnan(pipe.transform(test)).any()


def test_unseen_category_is_ignored(raw):
    s = split(raw)
    pipe = build_preprocessing_pipeline(s.X_train, COERCE).fit(s.X_train)
    new = s.X_test.iloc[:2].copy()
    new["Contract"] = "Five year"  # never seen in training
    out = pipe.transform(new)
    names = list(pipe.get_feature_names_out())
    contract_cols = [i for i, n in enumerate(names) if "Contract" in n]
    assert out[:, contract_cols].sum() == 0


def test_blank_total_charges_imputed(raw):
    s = split(raw)
    pipe = build_preprocessing_pipeline(s.X_train, COERCE).fit(s.X_train)
    blank_rows = s.X_test[s.X_test["TotalCharges"].str.strip() == ""]
    if len(blank_rows):
        assert not np.isnan(pipe.transform(blank_rows)).any()


def test_serving_consistency(raw):
    """Same rows give identical output in a batch, one at a time, reordered, and after reload."""
    s = split(raw)
    pipe = build_preprocessing_pipeline(s.X_train, COERCE).fit(s.X_train)
    batch = pipe.transform(s.X_test)
    single = np.vstack([pipe.transform(s.X_test.iloc[[i]]) for i in range(5)])
    assert np.allclose(batch[:5], single)
    shuffled = s.X_test[list(reversed(s.X_test.columns))]
    assert np.allclose(batch, pipe.transform(shuffled))
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    buf.seek(0)
    assert np.allclose(batch, joblib.load(buf).transform(s.X_test))


def test_extra_columns_ignored_missing_column_fails(raw):
    s = split(raw)
    pipe = build_preprocessing_pipeline(s.X_train, COERCE).fit(s.X_train)
    extra = s.X_test.copy()
    extra["customerID"] = "x"
    assert np.allclose(pipe.transform(extra), pipe.transform(s.X_test))
    with pytest.raises((ValueError, KeyError)):
        pipe.transform(s.X_test.drop(columns=["Contract"]))


def test_no_features_raises():
    with pytest.raises(ValueError):
        build_preprocessing_pipeline(pd.DataFrame(index=range(3)))
