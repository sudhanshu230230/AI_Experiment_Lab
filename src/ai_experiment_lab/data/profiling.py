"""Dataset profiling: summary statistics used for reports and validation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

NUMERIC_LIKE_THRESHOLD = 0.9  # share of non-blank values that must parse as numbers


def _blank_mask(s: pd.Series) -> pd.Series:
    """True where a text value is empty or whitespace only."""
    if s.dtype == object or str(s.dtype) in ("string", "str"):
        return s.notna() & s.astype(str).str.strip().eq("")
    return pd.Series(False, index=s.index)


def _is_numeric_like_text(s: pd.Series) -> bool:
    """True for text columns whose non-blank values are almost all numbers."""
    if is_numeric_dtype(s) or is_bool_dtype(s):
        return False
    values = s.dropna()
    values = values[~_blank_mask(values)]
    if values.empty:
        return False
    parsed = pd.to_numeric(values, errors="coerce")
    return parsed.notna().mean() >= NUMERIC_LIKE_THRESHOLD


def profile_dataframe(
    df: pd.DataFrame,
    target: str | None = None,
    corr_threshold: float = 0.8,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute a profile of ``df``. Returns a plain, JSON-friendly dict.

    The target column, if given, is profiled separately and excluded from the
    numeric / categorical feature lists.
    """
    if target is not None and target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    n_rows, n_cols = df.shape
    feature_cols = [c for c in df.columns if c != target]

    numeric, categorical, numeric_like_text = [], [], []
    for c in feature_cols:
        s = df[c]
        if is_numeric_dtype(s) and not is_bool_dtype(s):
            numeric.append(c)
        else:
            categorical.append(c)
            if _is_numeric_like_text(s):
                numeric_like_text.append(c)

    # Missing values: true NaN plus blank strings, reported separately.
    missing = {
        c: {
            "nan": int(df[c].isna().sum()),
            "blank": int(_blank_mask(df[c]).sum()),
        }
        for c in df.columns
    }
    cells_missing = sum(v["nan"] + v["blank"] for v in missing.values())

    numeric_stats: dict[str, dict[str, float]] = {}
    for c in numeric:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        numeric_stats[c] = {
            "mean": float(s.mean()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "q1": float(q1),
            "median": float(s.median()),
            "q3": float(q3),
            "max": float(s.max()),
            "iqr_outliers": outliers,
        }

    categorical_stats = {
        c: {
            "n_unique": int(df[c].nunique(dropna=True)),
            "top": {str(k): int(v) for k, v in df[c].value_counts().head(top_k).items()},
        }
        for c in categorical
    }

    high_corr: list[dict[str, Any]] = []
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        for i, a in enumerate(numeric):
            for b in numeric[i + 1 :]:
                r = corr.loc[a, b]
                if pd.notna(r) and abs(r) >= corr_threshold:
                    high_corr.append({"a": a, "b": b, "corr": round(float(r), 4)})

    target_info = None
    if target is not None:
        counts = df[target].value_counts(dropna=False)
        total = int(counts.sum())
        target_info = {
            "name": target,
            "n_classes": int(df[target].nunique(dropna=True)),
            "missing": int(df[target].isna().sum()),
            "distribution": {str(k): int(v) for k, v in counts.items()},
            "fractions": {str(k): round(float(v) / total, 4) for k, v in counts.items()},
            "minority_fraction": round(float(counts.min()) / total, 4),
            "imbalance_ratio": round(float(counts.max()) / float(counts.min()), 2),
        }

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_duplicate_rows": int(df.duplicated().sum()),
        "missing_cell_fraction": round(cells_missing / (n_rows * n_cols), 6),
        "missing": missing,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "numeric_like_text_features": numeric_like_text,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "high_correlations": high_corr,
        "target": target_info,
    }


def format_report(p: dict[str, Any]) -> str:
    """Render a profile dict as a short, readable text report."""
    lines = [
        "DATASET PROFILE",
        f"Rows: {p['n_rows']:,}   Columns: {p['n_cols']}",
        (
            f"Numeric features: {len(p['numeric_features'])}   "
            f"Categorical features: {len(p['categorical_features'])}"
        ),
        f"Missing cells: {p['missing_cell_fraction']:.2%}   Duplicate rows: {p['n_duplicate_rows']}",
    ]
    cols_missing = {c: v for c, v in p["missing"].items() if v["nan"] or v["blank"]}
    if cols_missing:
        lines.append("Columns with missing or blank values:")
        for c, v in cols_missing.items():
            lines.append(f"  - {c}: {v['nan']} NaN, {v['blank']} blank")
    if p["numeric_like_text_features"]:
        lines.append(
            "WARNING: text columns that look numeric (likely need type conversion): "
            + ", ".join(p["numeric_like_text_features"])
        )
    if p["high_correlations"]:
        lines.append("Highly correlated numeric pairs:")
        for h in p["high_correlations"]:
            lines.append(f"  - {h['a']} / {h['b']}: {h['corr']}")
    t = p["target"]
    if t:
        lines.append(f"Target '{t['name']}': {t['n_classes']} classes, distribution {t['fractions']}")
        lines.append(f"  Minority fraction: {t['minority_fraction']:.2%}   "
                     f"Imbalance ratio: {t['imbalance_ratio']}:1")
    return "\n".join(lines)
