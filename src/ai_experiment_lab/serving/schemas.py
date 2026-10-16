"""Request and response models for the prediction API (Telco churn features)."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    """One customer, with the raw columns of the Telco dataset (no id, no target).

    ``TotalCharges`` may be a number, numeric text, or blank/null (new customers have none);
    blank becomes null and the pipeline imputes it exactly as in training. ``extra="forbid"`` rejects misspelled
    field names instead of silently ignoring them.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                "StreamingMovies": "No", "Contract": "Month-to-month",
                "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5, "TotalCharges": "420.35",
            }
        },
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float | None = None

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def _clean_total_charges(cls, v):
        """Blank text means 'no value yet' (new customers); other text must be a number."""
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
            try:
                v = float(v)
            except ValueError:
                raise ValueError("TotalCharges must be a number or blank") from None
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("TotalCharges must be finite")
        if v is not None and v < 0:
            raise ValueError("TotalCharges must be >= 0")
        return v


class Prediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    prediction: Literal["Yes", "No"]
    threshold: float
    model_version: int | None = None


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
