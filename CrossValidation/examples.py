"""
examples.py
===========
Usage examples showing clean separation of concerns.

These examples demonstrate:
1. How to prepare data separately
2. How to define model factories separately
3. How to call cross-validation (pure orchestration)
4. How to launch hyperparameter tuning separately

Key principle: Each layer knows only about its own domain.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Suppress warnings
logging.getLogger("sklearn").setLevel(logging.WARNING)

from core import CrossValidator, ModelFactory

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================================
# LAYER 1: DATA LOADING (EXTERNAL - Not part of CV module)
# ============================================================================

def load_features_and_labels(
    feature_file: str = r"C:\github\VT2\Feature_engineering\features_selected.csv",
    label_file: str = r"C:\github\VT2\Feature_engineering\labels.csv",
) -> tuple:
    """
    Load precomputed features and labels from CSV.
    
    This is EXTERNAL to the CV module.
    Features can come from:
    - CSV file (here)
    - Database query
    - API call
    - In-memory cache
    - Pickle/numpy files
    
    Returns:
        (X, y): Feature matrix and labels
    """
    features = pd.read_csv(feature_file)
    labels = pd.read_csv(label_file).iloc[:, 0].values
    
    # Basic validation
    assert features.shape[0] == len(labels), "Shape mismatch"
    
    # Remove NaN
    mask = ~(features.isna().any(axis=1) | pd.isna(labels))
    X = features[mask].reset_index(drop=True).values
    y = labels[mask]
    
    print(f"✓ Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Class distribution: {np.bincount(y)}\n")
    
    return X, y


def preprocess_features(
    X: np.ndarray, 
    preprocess_fn=None,
) -> np.ndarray:
    """
    Apply preprocessing to features.
    
    This is EXTERNAL preprocessing - happens before CV.
    Preprocessing is pluggable via factory pattern.
    
    Args:
        X: Feature matrix
        preprocess_fn: Callable that takes X and returns preprocessed X
                      If None, uses StandardScaler (default)
    
    Returns:
        Preprocessed feature matrix
    
    Example:
        from preprocessing import get_preprocessing
        
        # Use standard scaler
        X = preprocess_features(X, get_preprocessing("standard"))
        
        # Use MinMax scaler
        X = preprocess_features(X, get_preprocessing("minmax"))
        
        # Use custom pipeline
        X = preprocess_features(X, custom_pipeline_fn)
    """
    from preprocessing import get_preprocessing
    
    if preprocess_fn is None:
        preprocess_fn = get_preprocessing("standard")
    
    X_processed = preprocess_fn(X)
    print(f"✓ Features preprocessed\n")
    return X_processed


# ============================================================================
# LAYER 2: MODEL FACTORIES (Produces fresh models)
# ============================================================================
# These are defined in a MODEL CONFIGURATION MODULE (not here in production)
# Shown here for clarity.

def create_logistic_regression():
    """Factory: Logistic Regression with defaults."""
    return LogisticRegression(max_iter=1000, random_state=42)


def create_random_forest():
    """Factory: Random Forest with defaults."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )


def create_svm():
    """Factory: SVM with defaults."""
    return SVC(kernel='rbf', C=1.0, random_state=42)


def create_gradient_boosting():
    """Factory: Gradient Boosting with defaults."""
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )


def create_feedforward_nn():
    """Factory: Feed-Forward Neural Network (MLPClassifier)."""
    return MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        max_iter=500,
        random_state=42,
    )


# Model registry for easy switching
MODEL_FACTORIES = {
    "Logistic Regression": create_logistic_regression,
    "Random Forest": create_random_forest,
    "SVM": create_svm,
    "Gradient Boosting": create_gradient_boosting,
    "Feed-Forward NN": create_feedforward_nn,
}


# ============================================================================
# LAYER 3: METRIC FUNCTIONS (Pluggable)
# ============================================================================
# These can be defined anywhere. Here for clarity.

def f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted F1 score (accounts for class imbalance)."""
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 score."""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ROC-AUC (binary classification)."""
    return roc_auc_score(y_true, y_pred)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy."""
    return accuracy_score(y_true, y_pred)


METRICS = {
    "F1 (weighted)": f1_weighted,
    "F1 (macro)": f1_macro,
    "ROC-AUC": roc_auc,
    "Accuracy": accuracy,
}


# ============================================================================
# LAYER 4: ORCHESTRATION (Uses all above layers)
# ============================================================================

def example_1_single_model_single_metric():
    """
    Example 1: Evaluate one model with one metric.
    Demonstrates clean separation of concerns.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Model + Single Metric")
    print("="*70 + "\n")
    
    # Layer 1: Load data
    X, y = load_features_and_labels()
    X = preprocess_features(X)
    
    # Layer 2 & 3: Define model and metric
    model_factory = MODEL_FACTORIES["Random Forest"]
    metric_fn = METRICS["F1 (weighted)"]
    
    # Layer 4: Orchestrate
    validator = CrossValidator(
        cv_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=True,
    )
    
    results = validator.validate(
        X, y,
        model_factory=model_factory,
        metric_fn=metric_fn,
        model_name="Random Forest",
        metric_name="F1 (weighted)",
    )
    
    print("\n" + results.summary())
    
    return results


def example_2_same_model_different_metrics():
    """
    Example 2: Evaluate one model with multiple metrics.
    Shows how easy it is to swap metrics (metric-agnostic).
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Same Model + Different Metrics")
    print("="*70 + "\n")
    
    # Layer 1: Load data (once)
    X, y = load_features_and_labels()
    X = preprocess_features(X)
    
    # Layer 2: Define model (once)
    model_factory = MODEL_FACTORIES["Random Forest"]
    
    # Layer 4: Evaluate with different metrics
    validator = CrossValidator(verbose=False)
    
    results_dict = {}
    for metric_name, metric_fn in METRICS.items():
        print(f"Evaluating: {metric_name}...", end=" ")
        results = validator.validate(
            X, y,
            model_factory=model_factory,
            metric_fn=metric_fn,
            model_name="Random Forest",
            metric_name=metric_name,
        )
        results_dict[metric_name] = results
        print(f"Mean: {results.mean_score:.4f}")
    
    print("\nComparison across metrics:")
    for metric_name, results in results_dict.items():
        print(f"  {metric_name:20s}: {results.mean_score:.4f} ± {results.std_score:.4f}")
    
    return results_dict


def example_3_different_models_same_metric():
    """
    Example 3: Evaluate multiple models with one metric.
    Shows how easy it is to swap models (model-agnostic).
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Different Models + Same Metric")
    print("="*70 + "\n")
    
    # Layer 1: Load data (once)
    X, y = load_features_and_labels()
    X = preprocess_features(X)
    
    # Layer 3: Define metric (once)
    metric_fn = METRICS["F1 (weighted)"]
    
    # Layer 4: Compare models
    validator = CrossValidator(verbose=False)
    
    comparison = validator.compare_models(
        X, y,
        model_factories=MODEL_FACTORIES,
        metric_fn=metric_fn,
        metric_name="F1 (weighted)",
    )
    
    print("\n" + comparison.summary())
    
    return comparison


def example_4_hyperparameter_tuning_integration():
    """
    Example 4: Integrate with hyperparameter tuning.
    
    NOTE: Hyperparameter tuning happens BEFORE cross-validation.
    They are separate concerns:
    - Tuning: Find best parameters (on train set)
    - CV: Estimate generalization error (on held-out folds)
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Hyperparameter Tuning + Cross-Validation")
    print("="*70 + "\n")
    
    from sklearn.model_selection import GridSearchCV
    
    # Layer 1: Load data
    X, y = load_features_and_labels()
    X = preprocess_features(X)
    
    # Step 1: Hyperparameter tuning on full dataset
    print("Step 1: Tuning hyperparameters...")
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5],
    }
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    tuner = GridSearchCV(
        base_model, param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring="f1_weighted",
        n_jobs=-1,
    )
    tuner.fit(X, y)
    
    print(f"✓ Best parameters: {tuner.best_params_}")
    print(f"✓ Best CV F1: {tuner.best_score_:.4f}\n")
    
    # Step 2: Cross-validate with TUNED model
    print("Step 2: Cross-validating with tuned model...")
    
    def tuned_model_factory():
        """Factory that creates model with best parameters."""
        return RandomForestClassifier(**tuner.best_params_, random_state=42, n_jobs=-1)
    
    metric_fn = METRICS["F1 (weighted)"]
    
    validator = CrossValidator(verbose=False)
    results = validator.validate(
        X, y,
        model_factory=tuned_model_factory,
        metric_fn=metric_fn,
        model_name="Random Forest (Tuned)",
        metric_name="F1 (weighted)",
    )
    
    print("\n" + results.summary())
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def example_5_different_preprocessing_strategies():
    """
    Example 5: Compare same model with different preprocessing approaches.
    Shows how preprocessing is pluggable and swappable.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Same Model + Different Preprocessing")
    print("="*70 + "\n")
    
    from preprocessing import get_preprocessing
    
    # Layer 1: Load data (raw, no preprocessing yet)
    X, y = load_features_and_labels()
    
    # Layer 2 & 3: Define model and metric
    model_factory = MODEL_FACTORIES["Random Forest"]
    metric_fn = METRICS["F1 (weighted)"]
    
    # Layer 4: Try different preprocessing strategies
    preprocessing_strategies = {
        "Standard (mean=0, std=1)": get_preprocessing("standard"),
        "MinMax (0 to 1 range)": get_preprocessing("minmax"),
        "Robust (outlier-safe)": get_preprocessing("robust"),
        "None (raw features)": get_preprocessing("none"),
    }
    
    validator = CrossValidator(verbose=False)
    results_dict = {}
    
    print("Testing different preprocessing approaches:\n")
    for prep_name, prep_fn in preprocessing_strategies.items():
        print(f"Testing: {prep_name}...", end=" ")
        
        # Apply preprocessing
        X_processed = preprocess_features(X, prep_fn)
        
        # Evaluate
        results = validator.validate(
            X_processed, y,
            model_factory=model_factory,
            metric_fn=metric_fn,
            model_name="Random Forest",
            metric_name="F1 (weighted)",
        )
        
        results_dict[prep_name] = results
        print(f"Mean F1: {results.mean_score:.4f}")
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPARISON")
    print("="*70 + "\n")
    
    for prep_name, results in results_dict.items():
        print(
            f"{prep_name:30s}: "
            f"{results.mean_score:.4f} ± {results.std_score:.4f}"
        )
    
    return results_dict


if __name__ == "__main__":
    # Run examples
    # results_1 = example_1_single_model_single_metric()
    # results_2 = example_2_same_model_different_metrics()
    # results_3 = example_3_different_models_same_metric()
    # results_4 = example_4_hyperparameter_tuning_integration()
    results_5 = example_5_different_preprocessing_strategies()
    
    # Uncomment to run other examples
