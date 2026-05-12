"""
length_leakage_test.py
======================
Diagnostic: how much of the classifier's accuracy comes from tsfresh's
`length` feature alone? Trains a Random Forest on only the three
`__length` columns from features_selected.csv, then reports test accuracy
on the same stratified split each main model used.

A high length-only accuracy means duration is doing most of the work.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)

BASE          = Path(r"C:\github\VT2")
FEATURES_PATH = BASE / "Feature_engineering" / "features_selected.csv"
LABELS_PATH   = BASE / "Feature_engineering" / "labels.csv"
RANDOM_STATE  = 42
CLASS_NAMES   = ["N", "NS", "OT", "P", "UT"]

LENGTH_FEATURES = [
    "intr_Torque (Nm)__length",
    "intr_Current (V)__length",
    "task_Robot_I (A)__length",
]
RATIO_FEATURES = [
    "intr_Torque (Nm)__ratio_value_number_to_time_series_length",
    "intr_Current (V)__ratio_value_number_to_time_series_length",
    "task_Robot_I (A)__ratio_value_number_to_time_series_length",
]


def evaluate(X, y, cols, test_size, name):
    """Train an RF on `cols` only, with a stratified split, and report accuracy."""
    X_sub = X[cols].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                 n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n{name}")
    print(f"  Features used: {len(cols)}  -> {cols}")
    print(f"  Train: {len(X_tr)},  Test: {len(X_te)}  (test_size={test_size})")
    print(f"  Test accuracy: {acc:.4f}  ({acc * 100:.2f} %)")
    return acc, y_te, y_pred


def main():
    print("=" * 70)
    print("  Length-leakage diagnostic")
    print("=" * 70)

    X = pd.read_csv(FEATURES_PATH, index_col=0)
    y = pd.read_csv(LABELS_PATH, index_col=0).values.flatten()
    print(f"\nLoaded {len(X)} samples, {X.shape[1]} total features.")

    length_cols = [c for c in LENGTH_FEATURES if c in X.columns]
    ratio_cols  = [c for c in RATIO_FEATURES  if c in X.columns]
    print(f"Length features present: {len(length_cols)}/3")
    print(f"Ratio features present:  {len(ratio_cols)}/3")

    print("\nFirst 5 rows of the 3 length features (expected to be identical):")
    print(X[length_cols].head())

    print(f"\nAre the 3 length columns numerically identical? ", end="")
    same = (X[length_cols].nunique(axis=1) == 1).all()
    print("YES" if same else "no — they differ on some rows")

    # ── Same split as FNN (test_size = 0.1) ──────────────────────────────
    print("\n" + "=" * 70)
    print("  Using the FNN test split (test_size = 0.1, random_state = 42)")
    print("=" * 70)
    acc_len, y_te, y_pred = evaluate(X, y, length_cols, 0.10,
                                     "[A] Length-only RF (3 cols)")
    acc_lr, y_te_b, y_pred_b = evaluate(X, y, length_cols + ratio_cols, 0.10,
                                       "[B] Length + ratio-of-length (6 cols)")

    # ── Same split as SVM/RF (test_size = 0.2) ───────────────────────────
    print("\n" + "=" * 70)
    print("  Using the SVM/RF test split (test_size = 0.2, random_state = 42)")
    print("=" * 70)
    evaluate(X, y, length_cols, 0.20,
             "[C] Length-only RF (3 cols)")
    evaluate(X, y, length_cols + ratio_cols, 0.20,
             "[D] Length + ratio-of-length (6 cols)")

    # ── Drop-length experiment (does the full model NEED length features?) ───
    print("\n" + "=" * 70)
    print("  Drop-length experiment (FNN split, test_size = 0.1)")
    print("=" * 70)
    all_cols    = X.columns.tolist()
    drop_cols   = set(length_cols + ratio_cols)
    kept_cols   = [c for c in all_cols if c not in drop_cols]
    evaluate(X, y, all_cols,  0.10,
             f"[E] Full features ({len(all_cols)} cols)  — baseline")
    evaluate(X, y, kept_cols, 0.10,
             f"[F] Full minus the 6 length/ratio ({len(kept_cols)} cols)")

    # ── Per-class breakdown for the FNN-split length-only model ──────────
    print("\n" + "=" * 70)
    print("  Per-class detail for [A] length-only on FNN split")
    print("=" * 70)
    print(classification_report(y_te, y_pred, target_names=CLASS_NAMES,
                                 zero_division=0))

    cm = confusion_matrix(y_te, y_pred)
    print("Confusion matrix (rows = true, cols = predicted):")
    header = "          " + "   ".join(f"{c:>3}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  true {CLASS_NAMES[i]:<3}  " +
              "   ".join(f"{v:3d}" for v in row))

    # ── Per-class detail for [B] length + ratio model ────────────────────
    print("\n" + "=" * 70)
    print("  Per-class detail for [B] length + ratio-of-length on FNN split")
    print(f"  Overall accuracy: {acc_lr * 100:.2f} %")
    print("=" * 70)
    print(classification_report(y_te_b, y_pred_b, target_names=CLASS_NAMES,
                                 zero_division=0))

    cm_b = confusion_matrix(y_te_b, y_pred_b)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(header)
    for i, row in enumerate(cm_b):
        print(f"  true {CLASS_NAMES[i]:<3}  " +
              "   ".join(f"{v:3d}" for v in row))

    # Random baseline (stratified majority): the largest class
    counts = pd.Series(y).value_counts()
    majority_acc = counts.max() / counts.sum()
    print(f"\nReference points:")
    print(f"  Random guess (5 classes, uniform):           20.00 %")
    print(f"  Majority-class baseline ({CLASS_NAMES[counts.idxmax()]}): "
          f"{majority_acc * 100:.2f} %")
    print(f"  Length-only RF accuracy:                     {acc_len * 100:.2f} %")
    print(f"\nIf length-only is close to your full-model accuracy, the model is "
          f"mostly using duration.")


if __name__ == "__main__":
    main()
