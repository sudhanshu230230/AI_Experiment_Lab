"""The model advisor: upload a table, get a recommended algorithm and its settings (ADR-011).

``advise`` wraps ``train_dataset`` with a time budget and then turns the numbers into something a
person can act on: which algorithm won and why, the settings to use, how much better it is than
guessing, what matters most in the data, and what to double-check.

Everything honest about the data stays honest here: the winner is chosen by cross-validation on
the training split only, the test split is scored once, and a score that looks too good is flagged
instead of celebrated.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance

from ai_experiment_lab.automl.schema import (
    MAX_CLASSES,
    DatasetError,
    infer_task,
    suggest_id_columns,
)
from ai_experiment_lab.automl.tasks import CV_SCORING, SELECTION, evaluate
from ai_experiment_lab.automl.train import TrainResult, train_dataset

TARGET_HINTS = ("target", "label", "class", "outcome", "churn", "default", "fraud", "price",
                "survived", "species", "y")
HIDDEN_PARAMS = {"n_jobs", "random_state", "eval_metric", "scale_pos_weight", "max_iter"}

FAMILY = {
    "logistic_regression": "Logistic regression",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost (gradient-boosted trees)",
}
WHY = {
    "logistic_regression": "a linear model: it works well when the pattern is mostly additive "
                           "(each factor pushes the answer up or down), it is fast, and its "
                           "behaviour is easy to explain",
    "ridge": "a linear model with a penalty that keeps the weights small: it works well when the "
             "answer moves roughly in a straight line with the inputs, it is fast, and it is "
             "easy to explain",
    "random_forest": "many decision trees voting together: it picks up interactions and "
                     "non-linear patterns without much tuning and copes well with messy columns",
    "xgboost": "trees built one after another, each correcting the last: usually the strongest "
               "choice on tables with interactions, at the cost of more tuning and less "
               "transparency",
}
PARAM_HELP = {
    "n_estimators": "number of trees; more is steadier but slower",
    "max_depth": "how deep each tree may grow; deeper fits finer detail but can overfit",
    "min_samples_leaf": "smallest group a tree may split off; larger values smooth the model",
    "min_samples_split": "smallest group that may still be split",
    "max_features": "share of columns each split may look at",
    "learning_rate": "how big each correction step is; smaller is safer but needs more trees",
    "subsample": "share of rows used to build each tree",
    "colsample_bytree": "share of columns used to build each tree",
    "reg_lambda": "penalty on large leaf values; higher is more cautious",
    "reg_alpha": "penalty that pushes small leaf values to zero",
    "min_child_weight": "smallest total weight a leaf must hold",
    "gamma": "minimum gain needed to make a split",
    "C": "inverse penalty strength; smaller means a simpler, more cautious model",
    "penalty": "type of penalty on the weights",
    "solver": "the optimisation routine (does not change what is learned, only how)",
    "alpha": "penalty strength; larger means a simpler, more cautious model",
    "class_weight": "how much to up-weight the rarer classes",
}


@dataclass
class Advice:
    """The advisor's answer. ``to_dict`` is what the web page receives."""

    task: str
    target: str
    recommended: str
    label: str
    why: list[str]
    params: dict[str, Any]
    param_help: dict[str, str]
    metrics: dict[str, float]
    baseline: dict[str, float]
    selection_metric: str
    leaderboard: list[dict[str, Any]]
    importances: list[dict[str, Any]]
    warnings: list[str]
    notes: list[str]
    code: str
    data: dict[str, Any]
    seconds: float
    result: TrainResult | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "result"}


def column_profile(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """What the upload looks like, for the column picker. Reads no more than the frame."""
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({"name": str(c), "dtype": str(s.dtype), "unique": int(s.nunique(dropna=True)),
                     "missing": int(s.isna().sum()),
                     "sample": [str(v) for v in s.dropna().astype(str).unique()[:3]]})
    warns = []
    if any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns):
        warns.append("A date column was found. Dates are ignored: forecasting over time (time "
                     "series) is not supported, only tables where each row stands alone.")
    return {"rows": len(df), "columns": cols, "suggested_target": suggest_target(df),
            "suggested_ids": suggest_id_columns(df, target or suggest_target(df)),
            "warnings": warns}


def suggest_target(df: pd.DataFrame) -> str:
    """A guess for the column to predict: a well-known name, otherwise the last column."""
    lowered = {str(c).lower().strip(): c for c in df.columns}
    for hint in TARGET_HINTS:
        if hint in lowered:
            return str(lowered[hint])
    return str(df.columns[-1])


def _baseline(task: str, y_train_like, X_test, y_test) -> dict[str, float]:
    """Always guess the most common class (or the average value): the bar to beat."""
    X0 = pd.DataFrame({"x": np.zeros(len(y_test))})
    if task == "regression":
        model = DummyRegressor(strategy="mean")
    else:
        model = DummyClassifier(strategy="prior")
    model.fit(pd.DataFrame({"x": np.zeros(len(y_train_like))}), y_train_like)
    return evaluate(task, model, X0, y_test)


def _importances(result: TrainResult, seed: int, limit: int = 1000) -> list[dict[str, Any]]:
    X, y = result.X_test, np.asarray(result.y_test)
    if X is None or len(X) == 0:
        return []
    if len(X) > limit:
        idx = np.random.default_rng(seed).choice(len(X), limit, replace=False)
        X, y = X.iloc[idx], y[idx]
    scoring = CV_SCORING[result.schema.task][result.selection_metric]
    try:
        r = permutation_importance(result.pipeline, X, y, scoring=scoring, n_repeats=3,
                                   random_state=seed, n_jobs=1)
    except Exception:  # noqa: BLE001 - an importance chart is a bonus; never fail the advice
        return []
    order = np.argsort(-r.importances_mean)
    total = float(np.clip(r.importances_mean, 0, None).sum()) or 1.0
    out = []
    for i in order[:10]:
        share = max(float(r.importances_mean[i]), 0.0) / total
        out.append({"feature": str(X.columns[i]), "share": round(share, 4)})
    return [o for o in out if o["share"] > 0]


def _snippet(task: str, name: str, params: dict[str, Any]) -> str:
    shown = {k: v for k, v in params.items() if k not in {"n_jobs"}}
    args = ", ".join(f"{k}={v!r}" for k, v in sorted(shown.items()))
    cls = {
        ("logistic_regression", False): ("sklearn.linear_model", "LogisticRegression"),
        ("ridge", True): ("sklearn.linear_model", "Ridge"),
        ("random_forest", False): ("sklearn.ensemble", "RandomForestClassifier"),
        ("random_forest", True): ("sklearn.ensemble", "RandomForestRegressor"),
        ("xgboost", False): ("xgboost", "XGBClassifier"),
        ("xgboost", True): ("xgboost", "XGBRegressor"),
    }[(name, task == "regression")]
    return (f"from {cls[0]} import {cls[1]}\n\n"
            f"model = {cls[1]}({args}{', ' if args else ''}random_state=42)\n"
            "# fit it on your prepared table: model.fit(X_train, y_train)\n"
            "# fill in missing values and encode text columns first (a scikit-learn Pipeline\n"
            "# with SimpleImputer + OneHotEncoder + StandardScaler does this).")


def _metric_name(metric: str) -> str:
    return {"f1": "F1", "f1_macro": "macro-F1", "rmse": "RMSE", "roc_auc": "ROC AUC",
            "r2": "R²"}.get(metric, metric)


def _reasons(task, name, table, sel, higher, metrics, baseline, data) -> list[str]:
    why = [f"{FAMILY[name]} is {WHY[name]}."]
    best = float(table.loc[0, f"cv_{sel}"])
    std = float(table.loc[0, "cv_selection_std"])
    mn = _metric_name(sel)
    why.append(f"It scored {mn} {best:.3f} (±{std:.3f}) in cross-validation on the training "
               f"rows, the highest of the {len(table)} candidates tried."
               if higher else
               f"It had the lowest {mn} in cross-validation ({best:.3f}, ±{std:.3f}) of the "
               f"{len(table)} candidates tried.")
    others = table[table["model"].str.split(" ").str[0] != name]
    if len(others):
        r = others.iloc[0]
        gap = abs(best - float(r[f"cv_{sel}"]))
        why.append(f"The best of the other families ({FAMILY.get(str(r['model']).split(' ')[0], r['model'])}) "
                   f"was {gap:.3f} behind"
                   + (" - close enough that the simpler model is a fair pick too." if gap < std else "."))
    if baseline:
        bm = baseline.get(sel)
        tm = metrics.get(sel)
        if bm is not None and tm is not None:
            why.append(f"On rows it never saw, {mn} was {tm:.3f}; always guessing the "
                       f"{'average' if task == 'regression' else 'most common class'} gives {bm:.3f}.")
    if data["rows"] < 1000:
        why.append("With few rows, simple models are less likely to memorise noise.")
    return why


def _warnings(task, table, sel, metrics, baseline, importances, data) -> list[str]:
    w: list[str] = []
    near = {"binary": metrics.get("roc_auc"), "multiclass": metrics.get("f1_macro"),
            "regression": metrics.get("r2")}[task]
    if near is not None and near >= 0.99:
        w.append("The score is almost perfect. Real-world data rarely is. Check that no column "
                 "gives the answer away (a copy of the target, an outcome recorded after the "
                 "fact, an ID that encodes it) and that the rows are not duplicated.")
    if importances and importances[0]["share"] >= 0.6 and len(importances) > 1:
        w.append(f"One column, '{importances[0]['feature']}', carries "
                 f"{importances[0]['share']:.0%} of the model's skill. If that column would not "
                 "be known at the moment you want to predict, remove it and run again.")
    # Skill over guessing, on a metric whose guessing level is fixed (ROC AUC 0.5, R2 0).
    # F1 is no good here: it depends on which class counts as positive.
    key, floor = {"binary": ("roc_auc", 0.05), "multiclass": ("balanced_accuracy", 0.03),
                  "regression": ("r2", 0.05)}[task]
    got, base = metrics.get(key), (baseline or {}).get(key)
    if got is not None and base is not None and got - base < floor:
        w.append("The model barely beats the always-guess baseline. The columns may not "
                 "carry much signal for this target; more or different data would help more "
                 "than a different algorithm.")
    if task != "regression" and data.get("minority_share") is not None and data["minority_share"] < 0.1:
        w.append(f"The rarest class is only {data['minority_share']:.0%} of rows. Judge the model "
                 "by F1/recall for that class, not by accuracy.")
    if data["rows"] < 500:
        w.append("Under 500 rows: the test score is a noisy estimate. Treat the ranking as "
                 "indicative.")
    if data.get("sampled"):
        w.append(f"Only a sample of {data['used_rows']:,} of {data['rows']:,} rows was used to "
                 "stay within this server's limits. The full data may favour a bigger model.")
    family = str(table.loc[0, "model"]).split(" ")[0]
    others = table[table["model"].str.split(" ").str[0] != family]
    if len(others):
        gap = abs(float(table.loc[0, f"cv_{sel}"] - others.iloc[0][f"cv_{sel}"]))
        if gap < float(table.loc[0, "cv_selection_std"]):
            w.append("The best two algorithms are tied within normal fold-to-fold noise; the "
                     "simpler one is a fair choice.")
    return w


def advise(
    df: pd.DataFrame,
    target: str,
    *,
    task: str | None = None,
    id_columns: list[str] | None = None,
    budget_seconds: float = 60.0,
    max_rows: int = 5000,
    max_trials: int = 15,
    seed: int = 42,
    models: tuple[str, ...] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Advice:
    """Pick an algorithm and settings for ``target``, within ``budget_seconds`` of tuning."""
    start = time.perf_counter()
    if target not in df.columns:
        raise DatasetError(f"column '{target}' is not in the file")
    if task is None and df[target].nunique(dropna=True) > MAX_CLASSES and not pd.api.types.is_numeric_dtype(df[target]):
        raise DatasetError("the target has too many different text values to be a category; "
                           "pick a column with fewer distinct values, or a numeric one")
    kind = task or infer_task(df[target].dropna())
    n_models = len(models) if models else 3
    per_model = max(2.0, budget_seconds / (n_models * 1.5))
    rows_in = int(df[target].notna().sum())
    try:
        import optuna  # noqa: F401
    except ImportError:
        max_trials = 0
    folds = 3 if min(rows_in, max_rows) > 3000 else 5
    result = train_dataset(df, target, task=kind, id_columns=id_columns, models=models,
                           cv_folds=folds, seed=seed, tune_trials=max_trials,
                           tune_seconds=per_model, max_rows=max_rows, progress=progress)
    task = result.schema.task
    sel, higher = SELECTION[task]
    baseline = _baseline(task, result.y_train, result.X_test, result.y_test)
    importances = _importances(result, seed)

    best_label = result.best
    family = best_label.split(" ")[0]
    params = {k: _plain(v) for k, v in result.params.get(best_label, {}).items()
              if k not in HIDDEN_PARAMS}
    helps = {k: PARAM_HELP[k] for k in params if k in PARAM_HELP}
    metrics = {k: float(v) for k, v in result.test_metrics.items()}
    minority = None
    if task != "regression":
        counts = np.asarray(pd.Series(result.y_test).value_counts())
        minority = float(counts.min() / counts.sum()) if len(counts) else None
    data = {"rows": rows_in, "used_rows": result.n_rows, "sampled": result.n_rows < rows_in,
            "features": len(result.schema.features), "classes": result.schema.classes,
            "minority_share": minority,
            "categorical": sum(f.kind == "categorical" for f in result.schema.features),
            "excluded": result.schema.excluded}
    board = []
    for _, r in result.table.iterrows():
        board.append({"model": str(r["model"]), "cv": float(r[f"cv_{sel}"]),
                      "cv_std": float(r["cv_selection_std"]), "test": float(r.get(f"test_{sel}", np.nan)),
                      "fit_seconds": round(float(r["fit_seconds"]), 2)})
    reasons = _reasons(task, family, result.table, sel, higher, metrics, baseline, data)
    warns = _warnings(task, result.table, sel, metrics, baseline, importances, data)
    return Advice(
        task=task, target=target, recommended=family, label=best_label, why=reasons,
        params=params, param_help=helps, metrics=metrics, baseline=baseline,
        selection_metric=sel, leaderboard=board, importances=importances, warnings=warns,
        notes=result.notes, code=_snippet(task, family, params), data=data,
        seconds=round(time.perf_counter() - start, 1), result=result)


def _plain(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float):
        return float(f"{v:.6g}")
    return v
