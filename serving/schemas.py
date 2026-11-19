"""
schemas.py
Pydantic input/output schemas for the fraud detection REST API.
Validates all 30 features before they reach the model.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Literal


class TransactionInput(BaseModel):
    """
    One credit card transaction. V1–V28 are PCA components (pre-anonymised).
    Time and Amount are raw features that the serving layer will scale.
    """
    Time:   float = Field(..., description="Seconds elapsed since first transaction in dataset")
    Amount: float = Field(..., description="Transaction amount in USD")

    V1:  float; V2:  float; V3:  float; V4:  float
    V5:  float; V6:  float; V7:  float; V8:  float
    V9:  float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

    @model_validator(mode="after")
    def check_no_nulls(self):
        for field, value in self.__dict__.items():
            if value is None:
                raise ValueError(f"Field '{field}' must not be null.")
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "Time": 406.0, "Amount": 149.62,
            "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
            "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
            "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
            "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
            "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
            "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
            "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
        }
    }}


class PredictionOutput(BaseModel):
    """Model prediction response."""
    fraud_probability: float = Field(..., description="Model's estimated probability of fraud (0–1)")
    prediction:        Literal["FRAUD", "LEGIT"]
    threshold_used:    float = Field(..., description="Decision threshold applied")
    model_version:     str   = Field(default="lgbm-v1")