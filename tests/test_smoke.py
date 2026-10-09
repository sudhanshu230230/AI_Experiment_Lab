"""Smoke test: the package imports and exposes a version."""

import ai_experiment_lab


def test_version_present():
    assert isinstance(ai_experiment_lab.__version__, str)
