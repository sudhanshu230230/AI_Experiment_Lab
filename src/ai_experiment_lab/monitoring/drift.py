"""Data and prediction drift detection (Phase 7, ADR-008).

Compares a *reference* sample (the data the model was trained on) with a *current* window of
production inputs. Each feature is tested on its own:

- numeric feature    : two-sample Kolmogorov-Smirnov test + Population Stability Index (PSI)
                       with bins taken from the reference quantiles
- categorical feature: chi-square test on the category counts + PSI over the category shares
                       (categories unseen in the reference are included)

A feature counts as **drifted** only if BOTH the effect size (PSI >= ``psi_threshold``) and the
significance (p < ``p_threshold``) agree. With many rows a p-value alone flags trivial shifts;
PSI alone ignores sample size. Requiring both keeps false alarms down.

The dataset has **drifted** when the share of drifted features reaches ``share_threshold``.
Fewer than ``min_current_rows`` current rows gives the verdict ``insufficient_data``: tests on a
handful of rows are noise and must not trigger retraining.

Drift is not the same as the model getting worse: it says the inputs changed. Measuring accuracy
needs the true outcome (churn labels), which arrive late (see docs/monitoring-and-drift.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4
NUMERIC_TEXT_SHARE = 0.95   # share of non-blank text values that must parse as numbers
MAX_CATEGORICAL_UNIQUE = 10   # numeric columns with this few distinct values are treated as categories

NO_DRIFT, DRIFT, INSUFFICIENT = "no_drift", "drift", "insufficient_data"


@dataclass
class FeatureDrift:
    feature: str
    kind: str                  # "numeric" | "categorical"
    test: str                  # "ks" | "chi2"
    statistic: float
    p_value: float
    psi: float
    drifted: bool
    ref_n: int
    cur_n: int


@dataclass
class DriftReport:
    verdict: str                        # NO_DRIFT | DRIFT | INSUFFICIENT
    n_reference: int
    n_current: int
    drifted_share: float
    thresholds: dict[str, float]
    features: list[FeatureDrift] = field(default_factory=list)
    prediction: FeatureDrift | None = None

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["drifted_features"] = self.drifted_features
        return d

    def to_frame(self) -> pd.DataFrame:
        rows = [asdict(f) for f in self.features]
        if self.prediction is not None:
            rows.append(asdict(self.prediction))
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            f"DRIFT VERDICT: {self.verdict.upper()}",
            f"reference rows: {self.n_reference:,}   current rows: {self.n_current:,}",
            (
                f"drifted features: {len(self.drifted_features)}/{len(self.features)} "
                f"({self.drifted_share:.0%}; threshold {self.thresholds['share_threshold']:.0%})"
            ),
        ]
        if self.drifted_features:
            lines.append("  " + ", ".join(self.drifted_features))
        if self.prediction is not None:
            p = self.prediction
            lines.append(
                f"prediction drift: {'YES' if p.drifted else 'no'} (PSI {p.psi:.3f}, KS p={p.p_value:.2g})"
            )
        return "\n".join(lines)


# ---- building blocks ----------------------------------------------------------------------

def psi_from_shares(ref: np.ndarray, cur: np.ndarray) -> float:
    """Population Stability Index between two share vectors (each sums to 1)."""
    r = np.clip(np.asarray(ref, dtype=float), EPS, None)
    c = np.clip(np.asarray(cur, dtype=float), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def numeric_psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI with bin edges from the reference quantiles; outer bins are open-ended."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # (almost) constant reference
        # one split just above the constant: values <= it fall left, anything larger falls right
        edges = np.array([-np.inf, float(np.nextafter(np.median(ref), np.inf)), np.inf])
    else:
        edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))

    def shares(x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(edges, x, side="right") - 1
        counts = np.bincount(np.clip(idx, 0, len(edges) - 2), minlength=len(edges) - 1)
        return counts / counts.sum()

    return psi_from_shares(shares(ref), shares(cur))


def categorical_stats(ref: pd.Series, cur: pd.Series) -> tuple[float, float, float]:
    """(chi2 statistic, p value, PSI) over the union of categories."""
    cats = sorted(set(ref.dropna().astype(str)) | set(cur.dropna().astype(str)))
    r = ref.dropna().astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()
    c = cur.dropna().astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()
    if r.sum() == 0 or c.sum() == 0:        # a side is entirely missing: nothing to compare
        return 0.0, 1.0, 0.0
    psi = psi_from_shares(r / r.sum(), c / c.sum())
    table = np.vstack([r, c])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0, 1.0, psi
    chi2, p, _, _ = stats.chi2_contingency(table)
    return float(chi2), float(p), psi


def infer_kind(reference: pd.Series) -> str:
    """"numeric" for numeric columns with many distinct values, else "categorical".

    Numbers stored as text (Telco's ``TotalCharges``) count as numeric when almost every
    non-blank value parses, so they are compared as numbers, not as thousands of categories.
    """
    if pd.api.types.is_bool_dtype(reference):
        return "categorical"
    values = reference
    if not pd.api.types.is_numeric_dtype(reference):
        non_blank = reference.dropna().astype(str).str.strip()
        non_blank = non_blank[non_blank != ""]
        parsed = pd.to_numeric(non_blank, errors="coerce")
        if len(non_blank) == 0 or parsed.notna().mean() < NUMERIC_TEXT_SHARE:
            return "categorical"
        values = parsed
    return "categorical" if values.nunique(dropna=True) <= MAX_CATEGORICAL_UNIQUE else "numeric"


def _feature_drift(name: str, ref: pd.Series, cur: pd.Series, kind: str,
                   psi_threshold: float, p_threshold: float) -> FeatureDrift:
    if kind == "numeric":
        r = pd.to_numeric(ref, errors="coerce").dropna().to_numpy(dtype=float)
        c = pd.to_numeric(cur, errors="coerce").dropna().to_numpy(dtype=float)
        if len(r) == 0 or len(c) == 0:
            return FeatureDrift(name, kind, "ks", float("nan"), 1.0, 0.0, False, len(r), len(c))
        ks = stats.ks_2samp(r, c)
        statistic, p, psi, test = float(ks.statistic), float(ks.pvalue), numeric_psi(r, c), "ks"
    else:
        statistic, p, psi = categorical_stats(ref, cur)
        test = "chi2"
    return FeatureDrift(name, kind, test, statistic, p, psi,
                        bool(psi >= psi_threshold and p < p_threshold),
                        int(ref.notna().sum()), int(cur.notna().sum()))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    psi_threshold: float = 0.1,
    p_threshold: float = 0.05,
    share_threshold: float = 0.3,
    min_current_rows: int = 100,
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
) -> DriftReport:
    """Compare ``current`` with ``reference`` column by column and return a ``DriftReport``.

    ``columns`` defaults to the columns present in both frames. Pass model scores (predicted
    probabilities) for both windows to also test prediction drift; it is reported but does not
    count towards the dataset verdict.
    """
    cols = columns or [c for c in reference.columns if c in current.columns]
    if not cols:
        raise ValueError("reference and current share no columns")
    missing = [c for c in cols if c not in reference.columns or c not in current.columns]
    if missing:
        raise ValueError(f"columns missing from reference or current: {missing}")

    thresholds = {"psi_threshold": psi_threshold, "p_threshold": p_threshold,
                  "share_threshold": share_threshold, "min_current_rows": float(min_current_rows)}
    n_ref, n_cur = len(reference), len(current)

    features = [
        _feature_drift(c, reference[c], current[c], infer_kind(reference[c]),
                       psi_threshold, p_threshold)
        for c in cols
    ]
    share = sum(f.drifted for f in features) / len(features)

    prediction = None
    if reference_scores is not None and current_scores is not None:
        prediction = _feature_drift("predicted_probability", pd.Series(reference_scores),
                                    pd.Series(current_scores), "numeric", psi_threshold, p_threshold)

    if n_cur < min_current_rows:
        verdict = INSUFFICIENT
    else:
        verdict = DRIFT if share >= share_threshold else NO_DRIFT
    return DriftReport(verdict, n_ref, n_cur, share, thresholds, features, prediction)
