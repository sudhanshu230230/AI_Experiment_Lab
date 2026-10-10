"""Data validation: decide whether a dataset may enter the training pipeline.

Each check returns PASS, WARN or FAIL. A dataset is accepted when nothing FAILs.
Warnings are things to act on or acknowledge (for example class imbalance); failures
block training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ai_experiment_lab.data.profiling import profile_dataframe

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_SYMBOL = {PASS: "✓", WARN: "⚠", FAIL: "✗"}

DEFAULTS = {
    "max_missing_fraction": 0.30,
    "max_duplicate_fraction": 0.05,
    "min_class_fraction": 0.05,
    "max_id_like_unique_ratio": 0.9,
}


class ValidationError(Exception):
    """Raised when a dataset is rejected."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    message: str


@dataclass
class ValidationResult:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: str, message: str) -> None:
        self.checks.append(CheckResult(name, status, message))

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == WARN]

    @property
    def accepted(self) -> bool:
        return not self.failures

    def raise_if_rejected(self) -> None:
        if not self.accepted:
            details = "; ".join(f"{c.name}: {c.message}" for c in self.failures)
            raise ValidationError(f"Dataset rejected: {details}")

    def report(self) -> str:
        lines = ["DATA VALIDATION"]
        lines += [f"{_SYMBOL[c.status]} {c.name}: {c.message}" for c in self.checks]
        verdict = "Dataset accepted" if self.accepted else "Dataset REJECTED"
        if self.accepted and self.warnings:
            verdict += f" with {len(self.warnings)} warning(s)"
        lines.append(verdict)
        return "\n".join(lines)


def validate_dataframe(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    thresholds: dict[str, float] | None = None,
    id_columns: list[str] | None = None,
    expected_columns: list[str] | None = None,
) -> ValidationResult:
    """Run all validation checks on ``df``.

    ``thresholds`` takes the keys of ``configs/default.yaml -> validation``.
    ``expected_columns`` (optional) is the feature schema a model was trained on; use it
    to validate new data against an existing model.
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"Unknown task '{task}'.")
    th = {**DEFAULTS, **(thresholds or {})}
    id_columns = id_columns or []
    res = ValidationResult()

    # 1. Target exists (everything target-related depends on this)
    if target not in df.columns:
        res.add("target_exists", FAIL, f"column '{target}' not found")
        return res
    res.add("target_exists", PASS, f"column '{target}' present")

    prof = profile_dataframe(df, target=target)
    n_rows = prof["n_rows"]

    # 2. Target validity
    n_target_missing = prof["target"]["missing"]
    if n_target_missing:
        res.add("target_valid", FAIL, f"{n_target_missing} missing target values")
    elif task == "classification" and prof["target"]["n_classes"] < 2:
        res.add("target_valid", FAIL, "classification target has fewer than 2 classes")
    elif task == "regression" and not is_numeric_dtype(df[target]):
        res.add("target_valid", FAIL, "regression target is not numeric")
    else:
        res.add("target_valid", PASS, "target has no missing values and suits the task")

    # 3. Supported dtypes (v1 supports numeric, boolean and text/categorical)
    unsupported = [
        c
        for c in df.columns
        if not (is_numeric_dtype(df[c]) or is_bool_dtype(df[c]) or _is_text_like(df[c]))
    ]
    if unsupported:
        res.add("supported_dtypes", FAIL, f"unsupported dtypes in: {', '.join(unsupported)}")
    else:
        res.add("supported_dtypes", PASS, "all columns are numeric, boolean or text")

    # 4. Missing values (NaN plus blank strings) per column
    too_missing = []
    for c, v in prof["missing"].items():
        frac = (v["nan"] + v["blank"]) / n_rows
        if frac > th["max_missing_fraction"]:
            too_missing.append(f"{c} ({frac:.0%})")
    if too_missing:
        res.add("missing_values", FAIL, "above limit: " + ", ".join(too_missing))
    else:
        some = [c for c, v in prof["missing"].items() if v["nan"] or v["blank"]]
        note = f"; present in {', '.join(some)}" if some else ""
        res.add("missing_values", PASS, f"all columns within {th['max_missing_fraction']:.0%}{note}")

    # 5. Duplicates
    dup_frac = prof["n_duplicate_rows"] / n_rows
    if dup_frac > th["max_duplicate_fraction"]:
        res.add("duplicates", FAIL, f"{dup_frac:.1%} duplicate rows exceeds limit")
    else:
        res.add("duplicates", PASS, f"{prof['n_duplicate_rows']} duplicate rows")

    # 6. Class balance (classification only)
    if task == "classification" and prof["target"]["n_classes"] >= 2:
        t = prof["target"]
        if t["minority_fraction"] < th["min_class_fraction"]:
            res.add(
                "class_balance",
                WARN,
                f"severe imbalance: minority {t['minority_fraction']:.1%} "
                f"({t['imbalance_ratio']}:1)",
            )
        elif t["imbalance_ratio"] > 1.5:
            res.add(
                "class_balance",
                WARN,
                f"imbalance detected: minority {t['minority_fraction']:.1%} "
                f"({t['imbalance_ratio']}:1); use stratified splits and F1/PR-AUC",
            )
        else:
            res.add("class_balance", PASS, "classes are roughly balanced")

    # 7. Numeric-looking text columns need type conversion before modelling
    nlt = [c for c in prof["numeric_like_text_features"] if c not in id_columns]
    if nlt:
        res.add("numeric_like_text", WARN, "convert to numeric in preprocessing: " + ", ".join(nlt))
    else:
        res.add("numeric_like_text", PASS, "no text columns that look numeric")

    # 8. Identifier columns
    absent_ids = [c for c in id_columns if c not in df.columns]
    if absent_ids:
        res.add("id_columns", WARN, f"configured id columns not found: {', '.join(absent_ids)}")
    id_like = [
        c
        for c in prof["categorical_features"]
        if c not in id_columns
        and n_rows > 1
        and df[c].nunique(dropna=True) / n_rows > th["max_id_like_unique_ratio"]
    ]
    if id_like:
        res.add("id_like_columns", WARN, "nearly unique text, maybe identifiers: " + ", ".join(id_like))
    elif not absent_ids:
        res.add("id_columns", PASS, "no unexpected identifier-like columns")

    # 9. Constant features carry no information
    constant = [c for c in df.columns if c != target and df[c].nunique(dropna=False) <= 1]
    if constant:
        res.add("constant_columns", WARN, "single-valued columns: " + ", ".join(constant))
    else:
        res.add("constant_columns", PASS, "no constant columns")

    # 10. Schema (only when an expected schema is supplied)
    if expected_columns is not None:
        feats = set(df.columns) - {target}
        missing_cols = [c for c in expected_columns if c not in feats]
        extra_cols = sorted(feats - set(expected_columns))
        if missing_cols:
            res.add("schema", FAIL, f"missing expected columns: {', '.join(missing_cols)}")
        elif extra_cols:
            res.add("schema", WARN, f"unexpected extra columns: {', '.join(extra_cols)}")
        else:
            res.add("schema", PASS, "columns match the expected schema")

    return res


def validation_thresholds(cfg: dict[str, Any]) -> dict[str, float]:
    """Pick the validation thresholds out of a loaded default.yaml."""
    return dict(cfg.get("validation", {}))


def _is_text_like(s: pd.Series) -> bool:
    return s.dtype == object or str(s.dtype) in ("string", "str", "category")
