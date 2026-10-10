"""Leakage-safe preprocessing for tabular data.

Design rules (see docs/decisions/ADR-002):

* The result of ``build_preprocessing_pipeline`` is a scikit-learn ``Pipeline`` that accepts
  the *raw* feature columns, so the exact same object is used in training and serving.
* Everything that learns from data (imputer values, scaler mean/std, category lists) is
  inside the pipeline and is fitted on the training split only.
* ``split_data`` happens before anything is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class NumericCoercer(BaseEstimator, TransformerMixin):
    """Convert text columns that hold numbers (for example ``TotalCharges``) to floats.

    Blank or whitespace-only values become NaN so the imputer can handle them. Any other
    non-numeric value raises, so bad data is never silently turned into NaN.
    Stateless: nothing is learned, so applying it before the split is leak-free anyway.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns or []:
            if col not in out.columns:
                raise KeyError(f"Column '{col}' expected for numeric coercion is missing.")
            s = out[col]
            stripped = s.astype("string").str.strip()
            blank = stripped.eq("")
            parsed = pd.to_numeric(stripped.mask(blank), errors="coerce")
            bad = parsed.isna() & s.notna() & ~blank
            if bad.any():
                examples = s[bad].head(3).tolist()
                raise ValueError(f"Column '{col}' has non-numeric values, e.g. {examples}")
            out[col] = parsed.astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


def infer_feature_columns(
    X: pd.DataFrame, coerce_numeric: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) for a feature frame.

    Columns in ``coerce_numeric`` are treated as numeric; booleans and text are categorical.
    This reads column names and dtypes only, no statistics from the data.
    """
    coerce = set(coerce_numeric or [])
    numeric, categorical = [], []
    for c in X.columns:
        if c in coerce or (is_numeric_dtype(X[c]) and not is_bool_dtype(X[c])):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def build_preprocessing_pipeline(
    X: pd.DataFrame,
    coerce_numeric: list[str] | None = None,
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
) -> Pipeline:
    """Create an unfitted preprocessing pipeline for the feature frame ``X``.

    ``X`` is only used to decide which columns are numeric or categorical.
    Numeric: impute then standardise. Categorical: impute then one-hot encode, ignoring
    categories not seen in training.
    """
    numeric, categorical = infer_feature_columns(X, coerce_numeric)
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy=numeric_impute_strategy)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy=categorical_impute_strategy)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No feature columns to preprocess.")
    return Pipeline(
        [
            ("coerce", NumericCoercer(columns=list(coerce_numeric or []))),
            ("columns", ColumnTransformer(transformers, remainder="drop")),
        ]
    )


def encode_target(y: pd.Series, positive_label: str) -> pd.Series:
    """Encode a binary target as 1 (``positive_label``) and 0 (the other class)."""
    labels = set(y.dropna().unique())
    if positive_label not in labels:
        raise ValueError(f"Positive label '{positive_label}' not found in target values {labels}.")
    if len(labels) != 2:
        raise ValueError(f"Expected exactly 2 target classes, found {len(labels)}: {labels}.")
    if y.isna().any():
        raise ValueError("Target contains missing values.")
    return (y == positive_label).astype(int)


@dataclass
class SplitData:
    """Train/test split of features and encoded target."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_data(
    df: pd.DataFrame,
    target: str,
    positive_label: str,
    id_columns: list[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> SplitData:
    """Split raw data into train and test sets. Run this before fitting anything.

    Identifier columns and the target are removed from the features. The split is
    stratified on the target and reproducible via ``seed``.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")
    y = encode_target(df[target], positive_label)
    X = df.drop(columns=[target, *[c for c in (id_columns or []) if c in df.columns]])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if stratify else None
    )
    return SplitData(X_train, X_test, y_train, y_test)
