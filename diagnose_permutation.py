"""
diagnose_permutation.py
=======================
Quick check why permutation importance returned all-zeros for SVM and RF.

For each model:
  1. Load model + reconstruct the same train/test split + scaler used to train it
  2. Print baseline accuracy + class distribution of predictions
  3. Manually shuffle ONE feature and print the accuracy delta
If predictions collapse to a single class, the scaling pipeline is broken.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE          = Path(r"C:\github\VT2")
FEATURES_PATH = BASE / "Feature_engineering" / "features_selected.csv"
LABELS_PATH   = BASE / "Feature_engineering" / "labels.csv"

MODELS = [
    ("RF",  BASE / "RandomForest" / "trained_rf.joblib",  0.2),
    ("SVM", BASE / "SVM"          / "trained_svm.joblib", 0.2),
]

RANDOM_STATE = 42

X_df = pd.read_csv(FEATURES_PATH, index_col=0)
y    = pd.read_csv(LABELS_PATH, index_col=0).values.flatten()
feature_names = X_df.columns.tolist()
print(f"Loaded {len(X_df)} samples, {X_df.shape[1]} features.\n")

for name, path, test_size in MODELS:
    print("=" * 60)
    print(f"  {name}  (model: {path.name},  test_size={test_size})")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y, test_size=test_size,
        random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = joblib.load(path)
    print(f"  Model type:           {type(model).__name__}")
    print(f"  Test samples:         {len(X_test)}")
    print(f"  True class counts:    {np.bincount(y_test).tolist()}")

    y_pred   = model.predict(X_test_scaled)
    baseline = (y_pred == y_test).mean()
    classes  = np.unique(y_pred).tolist()
    distrib  = np.bincount(y_pred, minlength=5).tolist()

    print(f"  Baseline accuracy:    {baseline:.4f} ({baseline*100:.2f}%)")
    print(f"  Distinct predicted:   {classes}")
    print(f"  Predicted class cnt:  {distrib}")

    # Manual permutation: shuffle the first feature, see what changes
    rng = np.random.default_rng(RANDOM_STATE)
    X_perm = X_test_scaled.copy()
    X_perm[:, 0] = rng.permutation(X_perm[:, 0])
    y_pred_perm = model.predict(X_perm)
    n_changed = (y_pred_perm != y_pred).sum()
    acc_perm  = (y_pred_perm == y_test).mean()
    print(f"  After shuffling '{feature_names[0][:50]}...':")
    print(f"    predictions changed: {n_changed} / {len(X_test)}")
    print(f"    accuracy:            {acc_perm:.4f}  (delta {acc_perm-baseline:+.4f})")

    # Also try shuffling a couple of obviously-important features
    for feat_name in ["intr_Torque (Nm)__length",
                       "intr_Current (V)__standard_deviation"]:
        if feat_name not in feature_names:
            continue
        col = feature_names.index(feat_name)
        X_perm = X_test_scaled.copy()
        X_perm[:, col] = rng.permutation(X_perm[:, col])
        y_pred_perm = model.predict(X_perm)
        n_changed = (y_pred_perm != y_pred).sum()
        acc_perm  = (y_pred_perm == y_test).mean()
        print(f"  After shuffling '{feat_name}':")
        print(f"    predictions changed: {n_changed} / {len(X_test)}")
        print(f"    accuracy:            {acc_perm:.4f}  (delta {acc_perm-baseline:+.4f})")
    print()
