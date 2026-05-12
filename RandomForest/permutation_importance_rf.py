"""
permutation_importance_rf.py
============================
Permutation importance for the trained Random Forest.

Reproduces the exact train/test split used to train trained_rf.joblib
(TEST_SIZE=0.2, random_state=42), then measures how much test accuracy
drops when each feature is shuffled.

Unlike RF's built-in Gini importance, permutation is model-agnostic and
not biased toward high-cardinality features — so it's directly comparable
with the SVM and FNN versions of this script.

Output:
  RandomForest/permutation_importance.csv  — full ranking
  console                                  — top 20 summary
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
MODEL_PATH    = BASE / "RandomForest" / "trained_rf.joblib"
OUT_DIR       = BASE / "RandomForest"

TEST_SIZE     = 0.2      # must match RandomForest/RandomForest.py
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

    # Baseline accuracy for context
    baseline = (model.predict(X_test_scaled) == y_test).mean()
    print(f"  Baseline test accuracy: {baseline:.4f} ({baseline*100:.2f}%)")

    print(f"\nComputing permutation importance "
          f"(n_repeats={N_REPEATS}) — this may take ~1 minute")
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
