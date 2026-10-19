"""Train and export a model for any tabular dataset (ADR-010).

The same rules as the Telco work apply to every dataset: split first, fit all preprocessing
inside one Pipeline on the training split only, choose the model by cross-validation on the
training split, and score the test split once.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from ai_experiment_lab.automl.schema import (
    DatasetError,
    Schema,
    build_schema,
    slugify,
)
from ai_experiment_lab.automl.tasks import (
    CV_SCORING,
    DEFAULT_PARAMS,
    MODELS,
    SELECTION,
    SPACES,
    build_estimator,
    cv_splitter,
    evaluate,
    max_folds,
    suggest_params,
)
from ai_experiment_lab.preprocessing import build_preprocessing_pipeline
from ai_experiment_lab.registry.export import export_pipeline

TRACKED = ("scikit-learn", "pandas", "numpy", "xgboost", "optuna", "skops")
MIN_TEST_ROWS = 10


@dataclass
class TrainResult:
    schema: Schema
    table: pd.DataFrame                 # one row per model, best first
    pipelines: dict[str, Pipeline]      # every fitted candidate
    best: str
    notes: list[str]
    n_rows: int
    selection_metric: str
    test_metrics: dict[str, float]      # of the best model, scored once on the test split
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    seconds: float = 0.0
    split: dict[str, Any] = field(default_factory=dict)
    X_test: pd.DataFrame | None = None  # kept so callers can compare against a baseline
    y_test: Any = None
    y_train: Any = None

    @property
    def pipeline(self) -> Pipeline:
        return self.pipelines[self.best]


def _versions() -> dict[str, str]:
    out = {}
    for lib in TRACKED:
        try:
            out[lib] = importlib_metadata.version(lib)
        except importlib_metadata.PackageNotFoundError:
            continue
    return out


def _make_pipeline(task: str, name: str, params: dict[str, Any], X: pd.DataFrame,
                   coerce: list[str], seed: int, y_train) -> Pipeline:
    pre = build_preprocessing_pipeline(X, coerce_numeric=coerce)
    return Pipeline([("preprocess", pre),
                     ("model", build_estimator(task, name, params, seed, y_train))])


def _tune(task: str, name: str, X, y, coerce, base_params, folds, seed, trials, seconds):
    """Optuna search of one model on the training split. Returns ``(best_params, best_cv)``."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = {k: dict(v) for k, v in SPACES[name].items()}
    if name == "logistic_regression" and task == "multiclass":
        space.pop("solver", None)       # liblinear cannot fit more than two classes
    metric, _ = SELECTION[task]
    scoring = CV_SCORING[task][metric]
    cv = cv_splitter(task, y, folds, seed)

    def objective(trial: optuna.Trial) -> float:
        params = {**base_params, **suggest_params(trial, space)}
        pipe = _make_pipeline(task, name, params, X, coerce, seed, y)
        res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        value = float(np.mean(res["test_score"]))
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, timeout=seconds, catch=(ValueError,))
    if not study.trials or all(t.state.name != "COMPLETE" for t in study.trials):
        return dict(base_params), None
    return {**base_params, **study.best_params}, float(study.best_value)


def train_dataset(
    df: pd.DataFrame,
    target: str,
    *,
    task: str | None = None,
    id_columns: list[str] | None = None,
    positive_label: Any = None,
    models: tuple[str, ...] | None = None,
    cv_folds: int = 5,
    test_size: float = 0.2,
    seed: int = 42,
    tune_trials: int = 0,
    tune_seconds: float = 60,
    max_rows: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> TrainResult:
    """Infer the task and schema, compare models by cross-validation, return the fitted best.

    ``tune_trials`` > 0 also runs an Optuna search per model; a tuned score is the best of many
    trials and is optimistic, so the result notes say to compare the test columns.
    """
    say = progress or (lambda _m: None)
    start = time.perf_counter()
    schema, X, y, coerce, notes = build_schema(df, target, task, id_columns, positive_label)
    task = schema.task
    n_rows = len(X)
    if max_rows and n_rows > max_rows:
        keep = X.sample(max_rows, random_state=seed).index
        X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
        notes.append(f"used a random sample of {max_rows:,} of {n_rows:,} rows")
        n_rows = max_rows
    say(f"Task: {task}; {n_rows:,} rows; {X.shape[1]} features")

    stratify = y if task != "regression" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify)
    if len(X_test) < MIN_TEST_ROWS:
        raise DatasetError("too few rows for a test split; add more data")

    chosen = tuple(models) if models else MODELS[task]
    available = []
    for m in chosen:
        if m not in MODELS[task]:
            raise ValueError(f"model '{m}' is not available for {task}; choose from {MODELS[task]}")
        if m == "xgboost":
            try:
                import xgboost  # noqa: F401
            except ImportError:
                notes.append("XGBoost is not installed here and was skipped.")
                continue
        available.append(m)
    if not available:
        raise ValueError("no model left to train")

    folds = max_folds(task, y_train, cv_folds)
    if folds < cv_folds:
        notes.append(f"cross-validation reduced to {folds} folds because the smallest class is small")
    cv = cv_splitter(task, y_train, folds, seed)
    sel, higher = SELECTION[task]
    scoring = CV_SCORING[task]

    rows, pipelines, params_used = [], {}, {}
    for name in available:
        variants = [("", dict(DEFAULT_PARAMS[task][name]))]
        if tune_trials > 0:
            say(f"Tuning {name} ({tune_trials} trials) ...")
            tuned, _ = _tune(task, name, X_train, y_train, coerce, DEFAULT_PARAMS[task][name],
                             folds, seed, int(tune_trials), float(tune_seconds))
            variants.append((" (tuned)", tuned))
        for suffix, params in variants:
            label = name + suffix
            say(f"Cross-validating {label} ...")
            pipe = _make_pipeline(task, name, params, X_train, coerce, seed, y_train)
            res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - t0
            row: dict[str, Any] = {"model": label}
            for k in scoring:
                vals = res[f"test_{k}"]
                vals = -vals if scoring[k].startswith("neg_") else vals
                row[f"cv_{k}"] = float(np.mean(vals))
            sel_vals = res[f"test_{sel}"]
            row["cv_selection_std"] = float(np.std(-sel_vals if scoring[sel].startswith("neg_") else sel_vals))
            row.update({f"test_{k}": v for k, v in evaluate(task, pipe, X_test, y_test).items()})
            row["fit_seconds"] = fit_seconds
            rows.append(row)
            pipelines[label] = pipe
            params_used[label] = params
    table = pd.DataFrame(rows).sort_values(f"cv_{sel}", ascending=not higher).reset_index(drop=True)
    best = str(table.loc[0, "model"])
    if tune_trials > 0:
        notes.append("A tuned model's CV score is the best of many trials and is optimistic; "
                     "compare the test columns before trusting a tuned win.")
    if len(table) > 1:
        gap = abs(float(table.loc[0, f"cv_{sel}"] - table.loc[1, f"cv_{sel}"]))
        spread = float(table.loc[0, "cv_selection_std"])
        if gap < spread:
            notes.append(f"the top two models are within one fold-to-fold spread ({gap:.3f} < "
                         f"{spread:.3f}): treat them as tied and prefer the simpler one")
    test = evaluate(task, pipelines[best], X_test, y_test)
    return TrainResult(
        schema=schema, table=table, pipelines=pipelines, best=best, notes=notes, n_rows=n_rows,
        selection_metric=sel, test_metrics=test, params=params_used,
        seconds=time.perf_counter() - start,
        split={"train_rows": len(X_train), "test_rows": len(X_test), "cv_folds": folds,
               "seed": seed, "test_size": test_size},
        X_test=X_test, y_test=y_test, y_train=y_train,
    )


def build_metadata(result: TrainResult, name: str, version: int = 1, dataset_sha256: str | None = None,
                   title: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``metadata.json`` content: identity, metrics and the schema the API builds itself from."""
    s = result.schema
    meta: dict[str, Any] = {
        "registered_name": slugify(name),
        "registry_version": int(version),
        "alias": "champion",
        "model_name": result.best,
        "task": s.task,
        "target": s.target,
        "title": title or f"{s.target} predictor",
        "classes": s.classes,
        "positive_label": s.positive_label,
        "negative_label": s.negative_label,
        "threshold": 0.5,
        "selection_metric": result.selection_metric,
        "metrics": {f"test_{k}": v for k, v in result.test_metrics.items()},
        "n_rows": result.n_rows,
        "split": result.split,
        "hyperparameters": result.params.get(result.best, {}),
        "dataset_name": name,
        "dataset_sha256": dataset_sha256,
        "schema": s.to_dict(),
        "example": s.example(),
        "libraries": _versions(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta.update(extra or {})
    return meta


def export_result(result: TrainResult, out_dir: str | Path, name: str, version: int = 1,
                  dataset_sha256: str | None = None, title: str | None = None,
                  extra: dict[str, Any] | None = None) -> Path:
    """Write ``model.skops`` and ``metadata.json`` (the folder the API serves)."""
    meta = build_metadata(result, name, version, dataset_sha256, title, extra)
    return export_pipeline(result.pipeline, out_dir, meta)
