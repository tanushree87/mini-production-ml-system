"""
evaluate.py
Runs final evaluation on the held-out test set.
Produces:
  - Classification report (baseline vs LightGBM)
  - PR-AUC comparison table
  - Precision-Recall curve plot  → artifacts/pr_curve.png
  - Confusion matrix plot        → artifacts/confusion_matrix.png
"""

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

from data_pipeline import load_and_split, ARTIFACTS_DIR

# ── Artifact paths ─────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.json")
PR_CURVE_PATH  = os.path.join(ARTIFACTS_DIR, "pr_curve.png")
CM_PATH        = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")


def load_artifacts():
    model     = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]
    print(f"[evaluate] Model loaded from    : {MODEL_PATH}")
    print(f"[evaluate] Threshold loaded     : {threshold:.4f}")
    return model, threshold


def evaluate_baseline(X_train, y_train, X_test, y_test):
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)
    y_proba = dummy.predict_proba(X_test)[:, 1]
    pr_auc  = average_precision_score(y_test, y_proba)
    print(f"\n[evaluate] ── Baseline (DummyClassifier) ──")
    print(f"[evaluate] Test PR-AUC : {pr_auc:.4f}")
    return dummy, pr_auc, y_proba


def evaluate_lgbm(model, threshold, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc  = average_precision_score(y_test, y_proba)
    y_pred  = (y_proba >= threshold).astype(int)

    print(f"\n[evaluate] ── LightGBM (threshold={threshold:.4f}) ──")
    print(f"[evaluate] Test PR-AUC : {pr_auc:.4f}")
    print(f"\n[evaluate] Test classification report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Legit", "Fraud"],
                                 digits=4))
    return y_proba, y_pred, pr_auc


def plot_pr_curve(y_test, lgbm_proba, lgbm_pr_auc,
                  baseline_proba, baseline_pr_auc):
    fig, ax = plt.subplots(figsize=(8, 6))

    # LightGBM curve
    p, r, _ = precision_recall_curve(y_test, lgbm_proba)
    ax.plot(r, p, color="#1f77b4", lw=2,
            label=f"LightGBM  (PR-AUC = {lgbm_pr_auc:.4f})")

    # Baseline curve
    p_b, r_b, _ = precision_recall_curve(y_test, baseline_proba)
    ax.plot(r_b, p_b, color="#ff7f0e", lw=1.5, linestyle="--",
            label=f"Baseline  (PR-AUC = {baseline_pr_auc:.4f})")

    # Random classifier line (fraud prevalence)
    prevalence = y_test.mean()
    ax.axhline(prevalence, color="grey", linestyle=":", lw=1.2,
               label=f"Random classifier (prevalence = {prevalence:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Fraud Detection (Test Set)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PR_CURVE_PATH, dpi=150)
    plt.close()
    print(f"[evaluate] PR curve saved → {PR_CURVE_PATH}")


def plot_confusion_matrix(y_test, y_pred, threshold):
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legit", "Fraud"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.4f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {CM_PATH}")


def print_summary_table(baseline_pr_auc, lgbm_pr_auc, y_test, y_pred):
    fraud_f1    = f1_score(y_test, y_pred)
    macro_f1    = f1_score(y_test, y_pred, average="macro")
    n_fraud     = int(y_test.sum())
    n_total     = len(y_test)

    print("\n" + "="*55)
    print("  FINAL TEST SET SUMMARY")
    print("="*55)
    print(f"  Test size            : {n_total:,} transactions")
    print(f"  Fraud in test        : {n_fraud} ({n_fraud/n_total*100:.4f}%)")
    print(f"  Baseline PR-AUC      : {baseline_pr_auc:.4f}")
    print(f"  LightGBM PR-AUC      : {lgbm_pr_auc:.4f}")
    print(f"  PR-AUC lift          : {lgbm_pr_auc/baseline_pr_auc:.1f}×")
    print(f"  Fraud F1 (test)      : {fraud_f1:.4f}")
    print(f"  Macro F1 (test)      : {macro_f1:.4f}")
    print("="*55)


if __name__ == "__main__":
    # ── Load data ──────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()

    # ── Load trained artifacts ─────────────────────────────────────────────────
    model, threshold = load_artifacts()

    # ── Baseline on test ───────────────────────────────────────────────────────
    dummy, baseline_pr_auc, baseline_proba = evaluate_baseline(
        X_train, y_train, X_test, y_test
    )

    # ── LightGBM on test ───────────────────────────────────────────────────────
    lgbm_proba, y_pred, lgbm_pr_auc = evaluate_lgbm(
        model, threshold, X_test, y_test
    )

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_pr_curve(y_test, lgbm_proba, lgbm_pr_auc,
                  baseline_proba, baseline_pr_auc)
    plot_confusion_matrix(y_test, y_pred, threshold)

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary_table(baseline_pr_auc, lgbm_pr_auc, y_test, y_pred)