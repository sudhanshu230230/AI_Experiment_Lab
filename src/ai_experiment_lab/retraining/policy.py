"""When to retrain and whether to promote the result (ADR-009).

Pure functions, no MLflow or I/O, so the rules are unit-tested and the notebook, scripts and
CI all apply the same ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone

from ai_experiment_lab.evaluation import classification_metrics
from ai_experiment_lab.monitoring import DRIFT, INSUFFICIENT, NO_DRIFT

PRIMARY_METRIC = "roc_auc"       # not F1: F1 moves with the churn rate (Phase 7 finding)
MIN_GAIN = 0.005                 # candidate must beat the champion by this much on shifted data
TOLERANCE = 1e-9                 # so a gain of exactly MIN_GAIN is not lost to float rounding
MAX_GUARD_DROP = 0.01            # ... and may not lose more than this on the natural data


def should_retrain(drift_verdict: str, force: bool = False) -> tuple[bool, str]:
    """Retrain on a ``drift`` verdict (or when forced). Never on too little data."""
    if force:
        return True, "forced by the operator"
    if drift_verdict == DRIFT:
        return True, "drift verdict: the inputs have moved away from the training data"
    if drift_verdict == INSUFFICIENT:
        return False, "insufficient data: too few recent rows to decide anything"
    if drift_verdict == NO_DRIFT:
        return False, "no drift: the training data still describes the incoming data"
    raise ValueError(f"unknown drift verdict: {drift_verdict!r}")


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str] = field(default_factory=list)
    gain_shifted: float = 0.0
    change_natural: float = 0.0

    def summary(self) -> str:
        head = "PROMOTE the candidate" if self.promote else "KEEP the current champion"
        return head + "\n" + "\n".join(f"  - {r}" for r in self.reasons)


def decide_promotion(
    champion_shifted: dict[str, float],
    candidate_shifted: dict[str, float],
    champion_natural: dict[str, float],
    candidate_natural: dict[str, float],
    metric: str = PRIMARY_METRIC,
    min_gain: float = MIN_GAIN,
    max_guard_drop: float = MAX_GUARD_DROP,
) -> PromotionDecision:
    """Promote only if the candidate is clearly better on the new (shifted) data AND has not
    got meaningfully worse on the natural data (a guard against forgetting the old population).

    Each argument is a metrics dict from ``classification_metrics``. All four come from data the
    candidate was not trained on.
    """
    gain = candidate_shifted[metric] - champion_shifted[metric]
    change = candidate_natural[metric] - champion_natural[metric]
    reasons = [
        (f"{metric} on shifted data: champion {champion_shifted[metric]:.4f} -> "
         f"candidate {candidate_shifted[metric]:.4f} (gain {gain:+.4f}, need >= {min_gain:+.4f})"),
        (f"{metric} on natural data: champion {champion_natural[metric]:.4f} -> "
         f"candidate {candidate_natural[metric]:.4f} (change {change:+.4f}, allowed >= {-max_guard_drop:+.4f})"),
    ]
    ok_gain = gain >= min_gain - TOLERANCE
    ok_guard = change >= -max_guard_drop - TOLERANCE
    if not ok_gain:
        reasons.append("gain on shifted data is below the required margin")
    if not ok_guard:
        reasons.append("candidate is worse on natural data than the guard allows")
    return PromotionDecision(ok_gain and ok_guard, reasons, float(gain), float(change))


def retrain_candidate(champion, X_train: pd.DataFrame, y_train, X_new: pd.DataFrame, y_new):
    """Refit an unfitted clone of the champion pipeline (same preprocessing, same
    hyperparameters) on the original training data plus newly labelled rows.

    Re-using the champion's hyperparameters is deliberate: this is *retraining*, not a new
    search. A new search is a new experiment (Phase 3), run when the model family itself is in
    doubt.
    """
    candidate = clone(champion)
    X = pd.concat([X_train, X_new], ignore_index=True)
    y = np.concatenate([np.asarray(y_train), np.asarray(y_new)])
    return candidate.fit(X, y)


def score(pipeline, X: pd.DataFrame, y, task: str = "binary") -> dict[str, float]:
    """Metrics of a fitted pipeline on (X, y). Binary uses the default 0.5 threshold.

    For other tasks pass ``task`` ("multiclass" or "regression") and compare with
    ``promotion_metric(task)`` (balanced accuracy or R^2, both higher-is-better).
    """
    if task == "binary":
        return classification_metrics(y, pipeline.predict(X), pipeline.predict_proba(X)[:, 1])
    from ai_experiment_lab.automl import evaluate

    return evaluate(task, pipeline, X, y)


def shifted_sample(X: pd.DataFrame, strength: float, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Resample ``X`` toward month-to-month, fiber, short-tenure customers (the profile that
    churns most). Row labels are kept in the index, so ``y.loc[sample.index]`` finds them.
    Used to simulate drifted data; ``strength`` 0 is a plain bootstrap."""
    risky = ((X["Contract"] == "Month-to-month").astype(int)
             + (X["InternetService"] == "Fiber optic").astype(int)
             + (X["tenure"] < 12).astype(int))
    return X.sample(n, replace=True, weights=1 + strength * risky, random_state=seed)
