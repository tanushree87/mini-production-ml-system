"""
test_schemas.py
Unit tests for Pydantic input/output schemas and basic inference integrity.
Run with: pytest tests/
"""

import pytest
from pydantic import ValidationError
from serving.schemas import TransactionInput, PredictionOutput

# ── Shared valid payload ───────────────────────────────────────────────────────
VALID_PAYLOAD = {
    "Time": 406.0, "Amount": 149.62,
    "V1": -1.36, "V2": -0.07, "V3":  2.54, "V4":  1.38,
    "V5": -0.34, "V6":  0.46, "V7":  0.24, "V8":  0.10,
    "V9":  0.36, "V10": 0.09, "V11":-0.55, "V12":-0.62,
    "V13":-0.99, "V14":-0.31, "V15": 1.47, "V16":-0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21":-0.02, "V22": 0.28, "V23":-0.11, "V24": 0.07,
    "V25": 0.13, "V26":-0.19, "V27": 0.13, "V28":-0.02,
}


class TestTransactionInput:

    def test_valid_payload_accepted(self):
        tx = TransactionInput(**VALID_PAYLOAD)
        assert tx.Amount == 149.62
        assert tx.Time   == 406.0

    def test_all_30_features_present(self):
        tx     = TransactionInput(**VALID_PAYLOAD)
        fields = tx.model_fields.keys()
        assert "Time"   in fields
        assert "Amount" in fields
        for i in range(1, 29):
            assert f"V{i}" in fields

    def test_missing_field_raises(self):
        payload = VALID_PAYLOAD.copy()
        del payload["V14"]
        with pytest.raises(ValidationError):
            TransactionInput(**payload)

    def test_missing_amount_raises(self):
        payload = VALID_PAYLOAD.copy()
        del payload["Amount"]
        with pytest.raises(ValidationError):
            TransactionInput(**payload)

    def test_non_numeric_field_raises(self):
        payload = {**VALID_PAYLOAD, "Amount": "not-a-number"}
        with pytest.raises(ValidationError):
            TransactionInput(**payload)

    def test_null_field_raises(self):
        payload = {**VALID_PAYLOAD, "V1": None}
        with pytest.raises(ValidationError):
            TransactionInput(**payload)

    def test_zero_amount_accepted(self):
        """Zero-value transactions are legitimate (e.g. auth checks)."""
        payload = {**VALID_PAYLOAD, "Amount": 0.0}
        tx      = TransactionInput(**payload)
        assert tx.Amount == 0.0

    def test_large_amount_accepted(self):
        payload = {**VALID_PAYLOAD, "Amount": 99999.99}
        tx      = TransactionInput(**payload)
        assert tx.Amount == 99999.99


class TestPredictionOutput:

    def test_fraud_label(self):
        out = PredictionOutput(
            fraud_probability=0.97,
            prediction="FRAUD",
            threshold_used=0.5,
        )
        assert out.prediction == "FRAUD"

    def test_legit_label(self):
        out = PredictionOutput(
            fraud_probability=0.03,
            prediction="LEGIT",
            threshold_used=0.5,
        )
        assert out.prediction == "LEGIT"

    def test_invalid_label_raises(self):
        with pytest.raises(ValidationError):
            PredictionOutput(
                fraud_probability=0.5,
                prediction="UNKNOWN",
                threshold_used=0.5,
            )

    def test_default_model_version(self):
        out = PredictionOutput(
            fraud_probability=0.1,
            prediction="LEGIT",
            threshold_used=0.5,
        )
        assert out.model_version == "lgbm-v1"