"""
app.py
FastAPI REST service exposing /health and /predict endpoints.

Run locally:
    uvicorn serving.app:app --reload --port 8000

Endpoints:
    GET  /health   → liveness check
    POST /predict  → fraud probability + decision for one transaction
"""

import os
import sys
import json
import time
import joblib
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.schemas import TransactionInput, PredictionOutput
from src.data_pipeline import ARTIFACTS_DIR

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Artifact paths ─────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib")
SCALER_PATH    = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.json")

# ── Global model state ─────────────────────────────────────────────────────────
_model     = None
_scaler    = None
_threshold = None
SCALE_COLS = ["Amount", "Time"]
FEATURE_ORDER = (
    ["Time", "Amount"] +
    [f"V{i}" for i in range(1, 29)]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts once at startup."""
    global _model, _scaler, _threshold
    logger.info("Loading model artifacts...")
    _model  = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    with open(THRESHOLD_PATH) as f:
        _threshold = json.load(f)["threshold"]
    logger.info(f"Model loaded   : {MODEL_PATH}")
    logger.info(f"Threshold      : {_threshold:.4f}")
    yield
    logger.info("Shutting down — releasing model resources.")


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud scoring — BITS WILP MLME Assignment",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Ops"])
def health():
    """Liveness check — returns service status and threshold in use."""
    return {
        "status":    "ok",
        "model":     "lgbm-v1",
        "threshold": _threshold,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(transaction: TransactionInput):
    """
    Score one transaction and return fraud probability + decision.

    - Accepts all 30 features (Time, Amount, V1–V28)
    - Applies the same scaler fitted on training data
    - Returns probability, FRAUD/LEGIT label, and threshold used
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # 1. Build DataFrame in the exact feature order the model was trained on
        row = {col: getattr(transaction, col) for col in FEATURE_ORDER}
        df  = pd.DataFrame([row])

        # 2. Scale Amount and Time (same scaler as training)
        df[SCALE_COLS] = _scaler.transform(df[SCALE_COLS])

        # 3. Score
        fraud_prob = float(_model.predict_proba(df)[:, 1][0])
        label      = "FRAUD" if fraud_prob >= _threshold else "LEGIT"

        logger.info(
            f"Scored | prob={fraud_prob:.4f} | label={label} "
            f"| amount={transaction.Amount:.2f}"
        )

        return PredictionOutput(
            fraud_probability=round(fraud_prob, 6),
            prediction=label,
            threshold_used=_threshold,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))