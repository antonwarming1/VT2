"""
permutation_importance_svm.py
=============================
Permutation importance for the trained SVM (RBF kernel).

The RBF kernel makes the SVM's decision function non-linear in the
original feature space, so there's no `coef_`-based feature importance.
Permutation importance is the standard model-agnostic alternative:
shuffle each feature in the test set and measure the accuracy drop.

Reproduces the exact train/test split used to train trained_svm.joblib
(TEST_SIZE=0.2, random_state=42).

Output:
  SVM/permutation_importance.csv  — full ranking
  console                         — top 20 summary
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────
BASE          = Path(r"C:\github\VT2")
FEATURES_PATH = BASE / "Feature_engineering" / "features_selected.csv"
LABELS_PATH   = BASE / "Feature_engineering" / "labels.csv"
MODEL_PATH    = BASE / "SVM" / "trained_svm.joblib"
OUT_DIR       = BASE / "SVM"

TEST_SIZE     = 0.2      # must match SVM/SVM_All_Features.py
RANDOM_STATE  = 42
N_REPEATS     = 5
TOP_N         = 20


def main():
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    X_df  = pd.read_csv(FEATURES_PATH, index_col=0)
    y     = pd.read_csv(LABELS_PATH, index_col=0).values.flatten()
    feature_names = X_df.columns.tolist()
    print(f"  {len(X_df)} samples, {X_df.shape[1]} features")

    # Reproduce the exact split + scaler used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  Test set: {len(X_test)} samples")

    # Baseline accuracy
    baseline = (model.predict(X_test_scaled) == y_test).mean()
    print(f"  Baseline test accuracy: {baseline:.4f} ({baseline*100:.2f}%)")

    print(f"\nComputing permutation importance "
          f"(n_repeats={N_REPEATS}) — this may take ~5 minutes")
    result = permutation_importance(
        model, X_test_scaled, y_test,
        n_repeats=N_REPEATS, random_state=RANDOM_STATE,
        n_jobs=-1, scoring='accuracy',
    )

    full_idx = np.argsort(result.importances_mean)[::-1]
    df_out = pd.DataFrame({
        "feature":          np.array(feature_names)[full_idx],
        "importance_mean":  result.importances_mean[full_idx],
        "importance_std":   result.importances_std[full_idx],
    })
    csv_path = OUT_DIR / "permutation_importance.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved full ranking -> {csv_path}")

    print(f"\nTop {TOP_N} features by permutation importance:")
    print(df_out.head(TOP_N).to_string(index=False))


if __name__ == "__main__":
    main()
