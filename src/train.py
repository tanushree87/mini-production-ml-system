"""
train.py
1. Trains a DummyClassifier baseline (stratified prior)
2. Trains a LightGBM model with scale_pos_weight for class imbalance
3. Tunes classification threshold on the validation set via PR curve
4. Serializes the trained model to artifacts/
"""

import os
import json
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    classification_report,
)
import lightgbm as lgb

from data_pipeline import load_and_split, ARTIFACTS_DIR

# ── Artifact paths ─────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.json")


def compute_best_threshold(y_true, y_proba) -> float:
    """
    Find threshold maximising F-beta (beta=2) on the validation set.
    Beta=2 weights recall twice as heavily as precision — appropriate
    for fraud detection where missing a fraud costs more than a false alert.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    beta = 2
    fb_scores = np.where(
        (beta**2 * precision + recall) == 0,
        0,
        (1 + beta**2) * precision * recall / (beta**2 * precision + recall),
    )
    best_idx       = np.argmax(fb_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_fb        = float(fb_scores[best_idx])
    best_f1        = float(
        2 * precision[best_idx] * recall[best_idx]
        / (precision[best_idx] + recall[best_idx] + 1e-9)
    )
    print(f"[train] Best val threshold : {best_threshold:.4f}")
    print(f"[train] Val F2-score       : {best_fb:.4f}")
    print(f"[train] Val F1 at threshold: {best_f1:.4f}")
    print(f"[train] Val Precision      : {precision[best_idx]:.4f}")
    print(f"[train] Val Recall         : {recall[best_idx]:.4f}")
    return best_threshold


def train_baseline(X_train, y_train, X_val, y_val):
    """Stratified DummyClassifier — sets the floor for all metrics."""
    print("\n[train] ── Baseline: DummyClassifier (stratified) ──")
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_val)
    print(classification_report(y_val, y_pred,
                                 target_names=["Legit", "Fraud"],
                                 digits=4))
    pr_auc = average_precision_score(y_val, dummy.predict_proba(X_val)[:, 1])
    print(f"[train] Baseline Val PR-AUC: {pr_auc:.4f}")
    return dummy


def train_lgbm(X_train, y_train, X_val, y_val) -> tuple:
    """
    Train LightGBM with scale_pos_weight to handle extreme class imbalance.
    Returns (fitted model, best threshold).
    """
    # Compute class weight ratio: negatives / positives
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = np.sqrt(neg / pos)   # ~22 instead of 518 — better calibrated
    print(f"\n[train] ── LightGBM ──")
    print(f"[train] Train class counts: Legit={neg:,}  Fraud={pos:,}")
    print(f"[train] scale_pos_weight: {scale_pos_weight:.2f}")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                   lgb.log_evaluation(period=100)],
    )

    # ── Validation metrics ───────────────────────────────────────────────────
    y_proba_val = model.predict_proba(X_val)[:, 1]
    pr_auc_val  = average_precision_score(y_val, y_proba_val)
    print(f"[train] Val PR-AUC: {pr_auc_val:.4f}")

    # ── Threshold tuning on val set ──────────────────────────────────────────
    best_threshold = compute_best_threshold(y_val, y_proba_val)
    y_pred_tuned   = (y_proba_val >= best_threshold).astype(int)
    print("\n[train] Val classification report (tuned threshold):")
    print(classification_report(y_val, y_pred_tuned,
                                 target_names=["Legit", "Fraud"],
                                 digits=4))

    return model, best_threshold


def save_artifacts(model, threshold: float):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold}, f)
    print(f"[train] Model saved    → {MODEL_PATH}")
    print(f"[train] Threshold saved → {THRESHOLD_PATH}")


if __name__ == "__main__":
    # ── Load data ──────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()

    # ── Baseline ───────────────────────────────────────────────────────────────
    train_baseline(X_train, y_train, X_val, y_val)

    # ── LightGBM ───────────────────────────────────────────────────────────────
    model, best_threshold = train_lgbm(X_train, y_train, X_val, y_val)

    # ── Save ───────────────────────────────────────────────────────────────────
    save_artifacts(model, best_threshold)
    print("\n[train] Training complete.")