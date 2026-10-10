"""Dataset ingestion: load a CSV and record its identity for reproducibility."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetInfo:
    """Identity of a loaded dataset. Logged with every experiment (Phase 2)."""

    name: str
    path: str
    sha256: str
    n_rows: int
    n_cols: int

    def to_dict(self) -> dict:
        return asdict(self)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes. Used as the dataset version identifier."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: str | Path, name: str | None = None) -> tuple[pd.DataFrame, DatasetInfo]:
    """Load a CSV exactly as stored (no cleaning) and return it with its DatasetInfo.

    Cleaning and type fixes belong to validation and preprocessing, so that the raw
    data stays inspectable and the raw-to-clean steps are explicit and testable.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Dataset is empty: {p}")
    info = DatasetInfo(
        name=name or p.stem,
        path=str(p),
        sha256=file_sha256(p),
        n_rows=len(df),
        n_cols=df.shape[1],
    )
    return df, info
