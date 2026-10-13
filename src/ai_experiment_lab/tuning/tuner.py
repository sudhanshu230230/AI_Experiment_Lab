"""Optuna tuning of one model, selected by cross-validated score on the TRAINING split.

The test split is touched exactly once, after the search, to score the refit best model
(``tune_model`` does this). It never influences the search, so the test score stays an
honest estimate. The CV score of the best trial is optimistically biased (it was picked as
the maximum of many noisy scores), which is why the test score is the number to trust.

The CV folds are identical to the ones ``run_baselines`` uses (same seed), so the baseline's
CV score and the tuned CV score are directly comparable, and the baseline configuration is
enqueued as trial 0 (parameters the baseline does not set use the ``default`` in the search
space), so for a fully expressible baseline the tuned CV score cannot fall below it. A baseline
value outside the space (for example Random Forest ``max_depth=None``) is left to the sampler.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ai_experiment_lab.evaluation import classification_metrics, confusion_counts
from ai_experiment_lab.preprocessing import SplitData
from ai_experiment_lab.training.trainer import make_model_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    """Output of ``tune_model``."""

    model: str
    best_params: dict[str, Any]       # searched params only
    full_params: dict[str, Any]       # fixed + searched, as passed to the classifier
    best_cv_f1: float
    cv_fold_scores: list[float]       # per-fold F1 of the best trial
    trials: pd.DataFrame              # one row per trial: number, state, cv_f1, params, seconds
    pipeline: Pipeline                # refit on the full training split
    test_metrics: dict[str, float]
    test_confusion: dict[str, int]
    n_complete: int
    n_pruned: int
    seconds: float


def suggest_params(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Turn a YAML search-space spec into sampled values."""
    out: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            out[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=bool(spec.get("log", False))
            )
        elif kind == "int":
            out[name] = trial.suggest_int(
                name, spec["low"], spec["high"], log=bool(spec.get("log", False))
            )
        elif kind == "categorical":
            out[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unknown search-space type '{kind}' for '{name}'")
    return out




def tune_model(
    name: str,
    split: SplitData,
    space: dict[str, dict[str, Any]],
    fixed_params: dict[str, Any] | None,
    preprocessing_cfg: dict[str, Any],
    seed: int = 42,
    cv_folds: int = 5,
    n_trials: int = 40,
    timeout_seconds: float | None = None,
    startup_trials: int = 5,
    warmup_folds: int = 2,
) -> TuningResult:
    """Search ``space`` for ``name``; return the best refit pipeline and its test score.
    """
    fixed = dict(fixed_params or {})
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    folds = list(cv.split(split.X_train, split.y_train))
    fold_scores_by_trial: dict[int, list[float]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed, **suggest_params(trial, space)}
        scores: list[float] = []
        for step, (tr, va) in enumerate(folds):
            pipe = make_model_pipeline(name, split, params, preprocessing_cfg, seed)
            pipe.fit(split.X_train.iloc[tr], split.y_train.iloc[tr])
            pred = pipe.predict(split.X_train.iloc[va])
            scores.append(float(f1_score(split.y_train.iloc[va], pred)))
            trial.report(float(np.mean(scores)), step)
            if step + 1 < len(folds) and trial.should_prune():
                fold_scores_by_trial[trial.number] = scores
                raise optuna.TrialPruned()
        fold_scores_by_trial[trial.number] = scores
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=startup_trials, n_warmup_steps=warmup_folds
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # Trial 0 = the baseline configuration, where it can be expressed in the space.
    baseline = {}
    for k, spec in space.items():
        v = fixed.get(k, spec.get("default"))
        if v is not None:
            baseline[k] = v
    if baseline:
        study.enqueue_trial(baseline, skip_if_exists=True)

    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
    search_seconds = time.perf_counter() - start

    best = study.best_trial
    full_params = {**fixed, **best.params}
    pipe = make_model_pipeline(name, split, full_params, preprocessing_cfg, seed)
    pipe.fit(split.X_train, split.y_train)
    pred = pipe.predict(split.X_test)
    proba = pipe.predict_proba(split.X_test)[:, 1]

    rows = []
    for t in study.trials:
        dur = (t.duration.total_seconds() if t.duration is not None else float("nan"))
        rows.append(
            {
                "number": t.number,
                "state": t.state.name,
                "cv_f1": t.value if t.state.name == "COMPLETE" else np.nan,
                "seconds": dur,
                **{f"p_{k}": v for k, v in t.params.items()},
            }
        )
    trials = pd.DataFrame(rows)
    states = trials["state"].value_counts()
    return TuningResult(
        model=name,
        best_params=dict(best.params),
        full_params=full_params,
        best_cv_f1=float(best.value),
        cv_fold_scores=fold_scores_by_trial.get(best.number, []),
        trials=trials,
        pipeline=pipe,
        test_metrics=classification_metrics(split.y_test, pred, proba),
        test_confusion=confusion_counts(split.y_test, pred),
        n_complete=int(states.get("COMPLETE", 0)),
        n_pruned=int(states.get("PRUNED", 0)),
        seconds=search_seconds,
    )
