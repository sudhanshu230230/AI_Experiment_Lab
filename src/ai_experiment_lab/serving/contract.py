"""What the API accepts and returns, derived from the model's own metadata (ADR-010).

A model exported by ``automl`` carries its schema, so the request validation, the form and the
response shape follow the model. An older export without a schema (the Telco demo) falls back to
the hand-written ``CustomerFeatures`` and ``Prediction`` models, unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from ai_experiment_lab.automl.schema import Feature, Schema
from ai_experiment_lab.registry.export import METADATA_FILE
from ai_experiment_lab.serving.schemas import CustomerFeatures

MAX_BATCH = 1000
MAX_TEXT = 200


class GenericPrediction(BaseModel):
    """One prediction for any task. Fields that do not apply to the task are left out."""

    prediction: Any = Field(description="Predicted class label, or the predicted number")
    probability: float | None = Field(default=None, ge=0, le=1,
                                      description="Binary: probability of the positive class")
    probabilities: dict[str, float] | None = Field(default=None,
                                                   description="Multiclass: probability per class")
    threshold: float | None = None
    model_version: int | None = None


@dataclass
class Contract:
    legacy: bool
    task: str
    columns: list[str]
    row_model: type[BaseModel]
    batch_model: type[BaseModel]
    schema: Schema | None = None
    title: str = "Prediction API"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def classes(self) -> list[str]:
        return self.schema.classes if self.schema else ["No", "Yes"]


def _numeric_validator(name: str):
    def clean(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
            try:
                v = float(v)
            except ValueError:
                raise ValueError("must be a number or blank") from None
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("must be a finite number")
        return v
    return field_validator(name, mode="before")(classmethod(clean))


def _field_for(f: Feature) -> tuple[Any, Any]:
    if f.kind == "numeric":
        ann: Any = float
        if f.nullable or f.numeric_text:
            return (float | None, Field(default=None, alias=f.name, allow_inf_nan=False))
        return (ann, Field(alias=f.name, allow_inf_nan=False))
    if f.allowed:
        ann = Literal[tuple(f.allowed)]  # type: ignore[valid-type]
    else:
        ann = str
    if f.nullable:
        return (ann | None, Field(default=None, alias=f.name, max_length=None if f.allowed else MAX_TEXT))
    return (ann, Field(alias=f.name, **({} if f.allowed else {"max_length": MAX_TEXT})))


def build_row_model(schema: Schema) -> type[BaseModel]:
    """A pydantic model with one validated field per input column.

    Python attribute names are positional (``f0``, ``f1`` ...) and the real column name is the
    alias, so columns such as "Usage Frequency" work and error locations show the real name.
    """
    fields: dict[str, Any] = {}
    validators: dict[str, Any] = {}
    for i, f in enumerate(schema.features):
        fields[f"f{i}"] = _field_for(f)
        if f.kind == "numeric":
            validators[f"clean_f{i}"] = _numeric_validator(f"f{i}")
    config = ConfigDict(extra="forbid", populate_by_name=False,
                        json_schema_extra={"example": schema.example()})
    return create_model("Row", __config__=config, __validators__=validators, **fields)


def _batch_model(row: type[BaseModel]) -> type[BaseModel]:
    return create_model("Batch", customers=(list[row], Field(min_length=1, max_length=MAX_BATCH)))  # type: ignore[valid-type]


def load_contract(model_dir: str | Path | None) -> Contract:
    """Read only ``metadata.json``. No schema (or no readable file) means the legacy Telco contract."""
    meta: dict[str, Any] = {}
    try:
        meta = json.loads((Path(model_dir) / METADATA_FILE).read_text())
    except (OSError, ValueError, TypeError):
        pass
    if isinstance(meta.get("schema"), dict) and meta.get("task"):
        schema = Schema.from_dict(meta["schema"])
        row = build_row_model(schema)
        return Contract(False, schema.task, schema.names, row, _batch_model(row), schema,
                        str(meta.get("title") or f"{schema.target} predictor"), meta)
    from ai_experiment_lab.serving.schemas import BatchRequest
    return Contract(True, "binary", list(CustomerFeatures.model_fields), CustomerFeatures,
                    BatchRequest, None, "Churn prediction", meta)
