import numpy as np
import pandas as pd
import pytest
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from ai_experiment_lab.retraining import (
    decide_promotion,
    retrain_candidate,
    score,
    shifted_sample,
    should_retrain,
)


def test_should_retrain_rules():
    assert should_retrain("drift")[0] is True
    assert should_retrain("no_drift")[0] is False
    assert should_retrain("insufficient_data")[0] is False
    assert should_retrain("insufficient_data", force=True)[0] is True
    with pytest.raises(ValueError):
        should_retrain("maybe")


def m(auc):
    return {"roc_auc": auc, "f1": 0.5}


def test_promotion_requires_gain_and_guard():
    ok = decide_promotion(m(0.76), m(0.78), m(0.84), m(0.84))
    assert ok.promote and ok.gain_shifted == pytest.approx(0.02)
    small = decide_promotion(m(0.76), m(0.762), m(0.84), m(0.84))
    assert not small.promote and "below the required margin" in small.summary()
    forgot = decide_promotion(m(0.76), m(0.80), m(0.84), m(0.80))
    assert not forgot.promote and "worse on natural data" in forgot.summary()
    edge = decide_promotion(m(0.76), m(0.765), m(0.84), m(0.83))     # exactly at both limits
    assert edge.promote


def test_promotion_ignores_f1_by_default():
    # candidate has a far better F1 but the same ROC AUC: must not be promoted
    better_f1 = {"roc_auc": 0.76, "f1": 0.9}
    assert not decide_promotion(m(0.76), better_f1, m(0.84), m(0.84)).promote


def frame(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(shift, 1, n)
    c = rng.choice(["a", "b"], n)
    y = (x + (c == "a") * 0.5 + rng.normal(0, 1, n) > shift + 0.5).astype(int)
    return pd.DataFrame({"x": x, "c": c}), y


def test_retrain_candidate_is_an_unfitted_clone_refit_on_all_data():
    X, y = frame(400, 0)
    champ = make_pipeline(make_column_transformer((OneHotEncoder(), ["c"]), ("passthrough", ["x"])),
                          LogisticRegression(C=0.37)).fit(X, y)
    Xn, yn = frame(200, 1, shift=1.0)
    cand = retrain_candidate(champ, X, y, Xn, yn)
    assert cand is not champ
    assert cand[-1].C == 0.37                                  # same hyperparameters
    assert not np.allclose(cand[-1].coef_, champ[-1].coef_)    # but learned from more data
    assert len(champ[-1].coef_[0]) == len(cand[-1].coef_[0])
    s = score(cand, Xn, yn)
    assert 0 <= s["roc_auc"] <= 1


def test_candidate_beats_champion_on_shifted_data_end_to_end():
    X, y = frame(1500, 0)
    champ = make_pipeline(make_column_transformer((OneHotEncoder(), ["c"]), ("passthrough", ["x"])),
                          LogisticRegression()).fit(X, y)
    Xn, yn = frame(1500, 1, shift=3.0)
    Xe, ye = frame(1500, 2, shift=3.0)
    cand = retrain_candidate(champ, X, y, Xn, yn)
    gain = score(cand, Xe, ye)["accuracy"] - score(champ, Xe, ye)["accuracy"]
    assert gain > 0


def test_shifted_sample_moves_toward_risky_customers_and_keeps_labels_reachable():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({
        "Contract": rng.choice(["Month-to-month", "Two year"], n),
        "InternetService": rng.choice(["Fiber optic", "DSL"], n),
        "tenure": rng.integers(0, 72, n),
    })
    y = pd.Series(rng.integers(0, 2, n), index=X.index)
    s = shifted_sample(X, 30, n=1000)
    assert (s["Contract"] == "Month-to-month").mean() > (X["Contract"] == "Month-to-month").mean() + 0.12
    assert len(y.loc[s.index]) == 1000
    flat = shifted_sample(X, 0, n=1000)
    assert abs((flat["Contract"] == "Month-to-month").mean() - 0.5) < 0.06


def _script(name):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_model_export_script(tmp_path):
    pytest.importorskip("skops")
    import json

    from ai_experiment_lab.registry import export_pipeline

    check = _script("check_model_export.py")
    X, y = frame(200, 0)
    pipe = make_pipeline(make_column_transformer((OneHotEncoder(), ["c"]), ("passthrough", ["x"])),
                         LogisticRegression()).fit(X, y)
    out = tmp_path / "champion"
    meta = {"registered_name": "m", "registry_version": 7, "libraries": {"scikit-learn": "1.6.1"}}
    export_pipeline(pipe, out, meta)
    docker = tmp_path / "Dockerfile"
    docker.write_text("FROM python\nARG SKLEARN_VERSION=1.6.1\n")
    assert check.main([str(out), "--dockerfile", str(docker)]) == 0
    assert check.main([str(out), "--print-version"]) == 0
    docker.write_text("FROM python\nARG SKLEARN_VERSION=1.5.0\n")
    assert check.main([str(out), "--dockerfile", str(docker)]) == 1            # version mismatch
    assert check.main([str(tmp_path / "nothing"), "--dockerfile", str(docker)]) == 1
    (out / "metadata.json").write_text(json.dumps({"registry_version": 1}))
    assert check.main([str(out), "--dockerfile", str(docker)]) == 1            # incomplete metadata
