"""Configuration loading and environment-aware paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config {path} must contain a mapping at the top level.")
    return cfg


def get_data_dir() -> Path:
    """Return the data directory.

    Uses the AEL_DATA_DIR environment variable (set by the Colab setup cell to the
    Drive folder); falls back to ./data/raw for local work.
    """
    env = os.environ.get("AEL_DATA_DIR")
    return Path(env) if env else Path("data") / "raw"
