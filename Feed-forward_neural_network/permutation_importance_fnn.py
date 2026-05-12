"""
permutation_importance_fnn.py
=============================
Permutation importance for the trained Feed-Forward Neural Network.

Keras models don't expose a sklearn-compatible `.predict()` that returns
class labels (Keras's `.predict()` returns the softmax probability vector),
so we wrap the model in a thin adapter to make it compatible with
sklearn's `permutation_importance`.

Reproduces the exact train/test split used to train trained_model.keras
(TEST_SIZE=0.1, random_state=42).

Output:
  Feed-forward_neural_network/permutation_importance.csv  — full ranking
  console                                                 — top 20 summary
"""

import os
import warnings
from pathlib import Path

# Quiet TF startup messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# ── Config ────────────────────────────────────────────────────────────
BASE          = Path(r"C:\github\VT2")
FEATURES_PATH = BASE / "Feature_engineering" / "features_selected.csv"
LABELS_PATH   = BASE / "Feature_engineering" / "labels.csv"
MODEL_PATH    = BASE / "Feed-forward_neural_network" / "trained_model.keras"
OUT_DIR       = BASE / "Feed-forward_neural_network"

TEST_SIZE     = 0.1      # must match FNN.py Config.TEST_SIZE
RANDOM_STATE  = 42
N_REPEATS     = 5
TOP_N         = 20


class KerasSklearnAdapter:
    """
    Make a Keras classifier look like a sklearn estimator for
    permutation_importance: expose .predict() returning hard class
    labels (argmax of softmax) instead of probabilities.
    """
    def __init__(self, model):
        self.model = model
        self.classes_ = np.arange(model.output_shape[-1])

    def predict(self, X):
        probs = self.model.predict(X, verbose=0, batch_size=256)
        return np.argmax(probs, axis=1)

    # Required by sklearn's clone() check in some versions
    def fit(self, X, y):
        return self


def main():
    print("Loading model and data...")
    keras_model = keras.models.load_model(MODEL_PATH)
    model = KerasSklearnAdapter(keras_model)
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

    # n_jobs=1 because TensorFlow doesn't fork cleanly across processes
    print(f"\nComputing permutation importance "
          f"(n_repeats={N_REPEATS}) — this may take ~15 minutes")
    result = permutation_importance(
        model, X_test_scaled, y_test,
        n_repeats=N_REPEATS, random_state=RANDOM_STATE,
        n_jobs=1, scoring='accuracy',
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
