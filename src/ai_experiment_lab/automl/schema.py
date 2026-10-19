"""Describe any tabular dataset: the task, the target and every input column (ADR-010).

The result is stored inside the model export (``metadata.json``), so the API and the web page can
build their validation and forms from the model itself instead of from hand-written code.

The schema holds ranges and allowed values, never real data rows. The one example row is
synthetic: the median of each numeric column and the most common value of each categorical one.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

TASKS = ("binary", "multiclass", "regression")
MAX_CATEGORY_LEVELS = 50        # more levels than this: free text in the API, no dropdown
MAX_CLASSES = 50
MAX_CHOICE_INTEGERS = 8         # an integer column with this few values gets a dropdown in the form
POSITIVE_HINTS = ("yes", "true", "1", "y", "t", "churn", "churned", "positive", "default", "fraud")


class DatasetError(ValueError):
    """The data cannot be used; the message says what to change."""


@dataclass
class Feature:
    name: str
    kind: Literal["numeric", "categorical"]
    nullable: bool = False
    integer: bool = False
    numeric_text: bool = False           # numbers stored as text (blank means missing)
    min: float | None = None
    max: float | None = None
    allowed: list[str] | None = None     # categorical: the values seen in training
    choices: list[float] | None = None   # integer column with few values: shown as a dropdown
    typical: Any = None                  # median or mode, used for the example row

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Feature:
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass
class Schema:
    task: str
    target: str
    features: list[Feature]
    classes: list[str] = field(default_factory=list)   # classification: label of each encoded class
    positive_label: str | None = None                  # binary only
    negative_label: str | None = None
    excluded: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task, "target": self.target, "classes": self.classes,
            "positive_label": self.positive_label, "negative_label": self.negative_label,
            "excluded": self.excluded, "features": [f.to_dict() for f in self.features],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Schema:
        return cls(
            task=d["task"], target=d["target"],
            features=[Feature.from_dict(f) for f in d["features"]],
            classes=list(d.get("classes") or []),
            positive_label=d.get("positive_label"), negative_label=d.get("negative_label"),
            excluded=list(d.get("excluded") or []),
        )

    def example(self) -> dict[str, Any]:
        """A synthetic typical row (medians and most common values), safe to publish."""
        return {f.name: f.typical for f in self.features}


# ---- helpers -------------------------------------------------------------------------------------

def _blank_to_nan(s: pd.Series) -> pd.Series:
    if is_numeric_dtype(s) or is_bool_dtype(s):
        return s
    txt = s.astype("string")
    return s.where(~txt.str.strip().eq("").fillna(False), np.nan)


def _is_numeric_text(s: pd.Series) -> bool:
    """Text column whose every non-blank value is a number (for example TotalCharges)."""
    if is_numeric_dtype(s) or is_bool_dtype(s):
        return False
    vals = _blank_to_nan(s).dropna().astype("string").str.strip()
    if vals.empty:
        return False
    return bool(pd.to_numeric(vals, errors="coerce").notna().all())


def infer_task(y: pd.Series) -> str:
    """Binary, multiclass or regression from the target column."""
    y = y.dropna()
    n_unique = y.nunique()
    if n_unique < 2:
        raise DatasetError("the target has fewer than 2 distinct values")
    if n_unique == 2:
        return "binary"
    if is_bool_dtype(y) or not is_numeric_dtype(y):
        if n_unique > MAX_CLASSES:
            raise DatasetError(
                f"the target has {n_unique} distinct text values; pick a different target or "
                f"reduce it to at most {MAX_CLASSES} classes")
        return "multiclass"
    integral = bool(np.all(np.mod(y.to_numpy(dtype=float), 1) == 0))
    if integral and n_unique <= 20:
        return "multiclass"
    return "regression"


def choose_positive_label(y: pd.Series, given: Any = None) -> Any:
    """The class to detect: the user's choice, else a 'yes'-like value, else the rarer class."""
    values = list(y.dropna().unique())
    if given is not None:
        for v in values:
            if v == given or str(v) == str(given):
                return v
        raise DatasetError(f"positive label {given!r} is not a value of the target ({values})")
    for v in values:
        if str(v).strip().lower() in POSITIVE_HINTS:
            return v
    counts = y.value_counts()
    return counts.idxmin()


def suggest_id_columns(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """Columns that look like row identifiers: nearly every value is different."""
    out = []
    n = len(df)
    if n < 20:
        return out
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        ratio = s.nunique(dropna=True) / n
        is_float = is_numeric_dtype(s) and not pd.api.types.is_integer_dtype(s)
        if is_float:
            continue
        name_hint = bool(re.search(r"(^|[^a-z])(id|uuid|guid|index)([^a-z]|$)", c.lower()))
        if _is_numeric_text(s) and not name_hint:
            continue        # amounts stored as text (Telco's TotalCharges) are features, not ids
        if ratio >= 0.98 or (ratio > 0.9 and not is_numeric_dtype(s)) or (name_hint and ratio > 0.5):
            out.append(c)
    return out


# ---- inference -----------------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame, target: str, exclude: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Clean the feature columns. Returns ``(X, coerce_numeric, notes)``.

    * excluded, constant and entirely empty columns are dropped
    * datetime columns are dropped (a plain date is not a usable feature as it is)
    * booleans and other non-numeric columns become text categories; blank text becomes missing
    * text columns that only hold numbers are kept as text and listed in ``coerce_numeric``
    """
    notes: list[str] = []
    X = df.drop(columns=[target, *[c for c in exclude if c in df.columns]]).copy()
    coerce: list[str] = []
    for c in list(X.columns):
        s = X[c]
        if s.isna().all():
            X = X.drop(columns=c)
            notes.append(f"dropped '{c}': it is empty")
        elif is_datetime64_any_dtype(s):
            X = X.drop(columns=c)
            notes.append(f"dropped '{c}': dates are not used directly (add a numeric column such as age in days)")
        elif s.nunique(dropna=True) < 2:
            X = X.drop(columns=c)
            notes.append(f"dropped '{c}': it has a single value")
        elif is_bool_dtype(s):
            X[c] = s.map({True: "True", False: "False"}).astype(object)
        elif is_numeric_dtype(s):
            continue
        elif _is_numeric_text(s):
            X[c] = _blank_to_nan(s).astype("string").str.strip().astype(object)
            X[c] = X[c].where(X[c].notna(), np.nan)
            coerce.append(c)
        else:
            X[c] = _blank_to_nan(s).astype("string").str.strip().astype(object)
            X[c] = X[c].where(X[c].notna(), np.nan)
    if X.shape[1] == 0:
        raise DatasetError("no usable feature columns are left")
    return X, coerce, notes


def infer_features(X: pd.DataFrame, coerce_numeric: list[str]) -> list[Feature]:
    feats: list[Feature] = []
    for c in X.columns:
        s = X[c]
        nullable = bool(s.isna().any())
        if c in coerce_numeric:
            num = pd.to_numeric(s, errors="coerce")
            feats.append(Feature(c, "numeric", nullable=True, numeric_text=True,
                                 min=float(num.min()), max=float(num.max()),
                                 typical=float(f"{float(num.median()):.6g}")))
        elif is_numeric_dtype(s):
            vals = s.dropna().astype(float)
            integer = bool(np.all(np.mod(vals.to_numpy(), 1) == 0))
            uniq = sorted(vals.unique().tolist())
            choices = uniq if integer and len(uniq) <= MAX_CHOICE_INTEGERS else None
            typical = float(f"{float(vals.median()):.6g}")     # no float noise in the form
            if integer:
                typical = round(typical)
                if choices:
                    typical = int(min(choices, key=lambda v: abs(v - typical)))
            feats.append(Feature(
                c, "numeric", nullable=nullable, integer=integer, min=float(vals.min()),
                max=float(vals.max()), choices=[int(v) for v in choices] if choices else None,
                typical=typical))
        else:
            counts = s.dropna().astype(str).value_counts()
            allowed = sorted(counts.index.tolist()) if len(counts) <= MAX_CATEGORY_LEVELS else None
            feats.append(Feature(c, "categorical", nullable=nullable, allowed=allowed,
                                 typical=str(counts.index[0])))
    return feats


def build_schema(
    df: pd.DataFrame, target: str, task: str | None = None, id_columns: list[str] | None = None,
    positive_label: Any = None,
) -> tuple[Schema, pd.DataFrame, pd.Series, list[str], list[str]]:
    """Infer everything from a DataFrame.

    Returns ``(schema, X, y_encoded, coerce_numeric, notes)`` where ``y_encoded`` is 0/1 for
    binary, 0..k-1 for multiclass and the float values for regression.
    """
    if target not in df.columns:
        raise DatasetError(f"target column '{target}' not found; columns are: {list(df.columns)[:20]}")
    notes: list[str] = []
    if df[target].isna().any():
        n_bad = int(df[target].isna().sum())
        df = df[df[target].notna()].reset_index(drop=True)
        notes.append(f"removed {n_bad} rows with a missing target")
    if len(df) < 50:
        raise DatasetError(f"only {len(df)} usable rows; at least 50 are needed")
    y_raw = df[target]
    task = task or infer_task(y_raw)
    if task not in TASKS:
        raise DatasetError(f"task must be one of {TASKS}")
    if task == "regression":
        if not is_numeric_dtype(y_raw) or is_bool_dtype(y_raw):
            raise DatasetError("a regression target must be numeric")
        if not np.isfinite(y_raw.to_numpy(dtype=float)).all():
            raise DatasetError("the target has infinite values")
    exclude = list(id_columns) if id_columns is not None else suggest_id_columns(df, target)
    if id_columns is None and exclude:
        notes.append(f"treated {exclude} as identifier columns and left them out "
                     "(pass id columns explicitly to change this)")
    X, coerce, prep_notes = prepare_features(df, target, exclude)
    notes += prep_notes

    positive = negative = None
    if task == "regression":
        y = y_raw.astype(float)
        classes: list[str] = []
    elif task == "binary":
        if y_raw.nunique() != 2:
            raise DatasetError(f"a binary task needs exactly 2 target values, found {y_raw.nunique()}")
        pos = choose_positive_label(y_raw, positive_label)
        neg = next(v for v in y_raw.dropna().unique() if v != pos)
        y = (y_raw == pos).astype(int)
        positive, negative = str(pos), str(neg)
        classes = [negative, positive]
    else:
        values = sorted(y_raw.dropna().unique().tolist(), key=lambda v: (str(type(v)), v))
        if len(values) > MAX_CLASSES:
            raise DatasetError(f"{len(values)} classes; the limit is {MAX_CLASSES}")
        if len(values) < 3:
            raise DatasetError("a multiclass task needs 3 or more classes")
        index = {v: i for i, v in enumerate(values)}
        y = y_raw.map(index).astype(int)
        classes = [str(v) for v in values]
    if task != "regression":
        smallest = int(y.value_counts().min())
        if smallest < 6:
            raise DatasetError(f"the smallest class has {smallest} rows; at least 6 are needed")
    schema = Schema(task, target, infer_features(X, coerce), classes, positive, negative, exclude)
    return schema, X, y, coerce, notes


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "model"


def element_ids(names: list[str]) -> list[str]:
    """Unique, HTML-safe ids for the form fields (``f-`` plus the column name)."""
    used: set[str] = set()
    out = []
    for i, n in enumerate(names):
        base = re.sub(r"[^A-Za-z0-9_]+", "_", n).strip("_") or f"col{i}"
        cand, k = base, 2
        while cand in used:
            cand, k = f"{base}_{k}", k + 1
        used.add(cand)
        out.append(cand)
    return out
