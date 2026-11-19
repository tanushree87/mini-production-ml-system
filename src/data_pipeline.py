"""
data_pipeline.py
Loads creditcard.csv, performs a temporal (time-ordered) train/val/test split,
fits a scaler strictly on the training partition, and saves it to artifacts/.
No data leakage: val and test sets are only transformed, never fitted on.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(ROOT_DIR, "data", "creditcard.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
SCALER_PATH   = os.path.join(ARTIFACTS_DIR, "scaler.joblib")

# ── Split ratios ───────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remaining 0.15 (implicit)


def load_and_split(
    data_path: str = DATA_PATH,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
    verbose: bool      = True,
) -> tuple:
    """
    Load the dataset, sort by Time (temporal ordering), split into
    train / val / test, scale Amount and Time, and return six DataFrames.

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame  (features)
    y_train, y_val, y_test : pd.Series     (target: Class)
    """
    # 1. Load
    df = pd.read_csv(data_path)
    if verbose:
        print(f"[data_pipeline] Loaded {len(df):,} rows, {df.shape[1]} columns.")
        print(f"[data_pipeline] Fraud rate: {df['Class'].mean()*100:.4f}%  "
              f"({df['Class'].sum():,} fraudulent transactions)")

    # 2. Sort by Time — enforces temporal order, prevents look-ahead bias
    df = df.sort_values("Time").reset_index(drop=True)

    # 3. Compute split indices
    n          = len(df)
    train_end  = int(n * train_ratio)
    val_end    = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    if verbose:
        for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            fraud_n = part["Class"].sum()
            print(f"[data_pipeline] {name:5s}: {len(part):,} rows | "
                  f"fraud={fraud_n:,} ({fraud_n/len(part)*100:.4f}%)")

    # 4. Separate features and target
    TARGET   = "Class"
    # Scale Amount and Time; V1-V28 are already PCA-transformed
    SCALE_COLS = ["Amount", "Time"]

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_val   = val_df.drop(columns=[TARGET])
    y_val   = val_df[TARGET]

    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    # 5. Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])

    # 6. Apply (transform only) to val and test — no fitting
    X_val[SCALE_COLS]  = scaler.transform(X_val[SCALE_COLS])
    X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])

    # 7. Save scaler to artifacts/
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    if verbose:
        print(f"[data_pipeline] Scaler saved → {SCALER_PATH}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
    print("\n[data_pipeline] Split complete.")
    print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}  |  X_test: {X_test.shape}")