"""
preprocessing.py
════════════════

Preprocessing strategies (factories for different preprocessing pipelines).

This layer is EXTERNAL to the CV module.
Different preprocessing approaches can be composed and swapped.

Key principle: Preprocessing happens BEFORE cross-validation, not during.
"""

from typing import Callable, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# Type definition
PreprocessingFn = Callable[[np.ndarray], np.ndarray]


# ============================================================================
# PREPROCESSING FACTORIES
# ============================================================================

def create_standard_scaler():
    """
    Factory: StandardScaler (mean=0, std=1).
    
    Best for: Most algorithms (SVM, Neural Networks, Logistic Regression)
    Assumes: Data is roughly normally distributed
    """
    def preprocess(X: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    return preprocess


def create_minmax_scaler():
    """
    Factory: MinMaxScaler (scales to [0, 1]).
    
    Best for: Neural Networks, algorithms sensitive to feature range
    Assumes: Need bounded output range
    """
    def preprocess(X: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    return preprocess


def create_robust_scaler():
    """
    Factory: RobustScaler (robust to outliers).
    
    Best for: Data with outliers
    Assumes: Working with median and quartiles is safer than mean/std
    """
    def preprocess(X: np.ndarray) -> np.ndarray:
        scaler = RobustScaler()
        return scaler.fit_transform(X)
    return preprocess


def create_no_preprocessing():
    """
    Factory: No preprocessing (identity).
    
    Best for: Tree-based models (RandomForest, GradientBoosting)
    Assumes: Features are already on meaningful scale
    """
    def preprocess(X: np.ndarray) -> np.ndarray:
        return X
    return preprocess


def create_custom_pipeline(scalers: list):
    """
    Factory: Custom preprocessing pipeline (chain multiple scalers).
    
    Args:
        scalers: List of preprocessing functions to apply sequentially
    
    Returns:
        Combined preprocessing function
    
    Example:
        pipeline = create_custom_pipeline([
            create_standard_scaler(),
            create_pca_reducer(n_components=10),  # if implemented
        ])
    """
    def preprocess(X: np.ndarray) -> np.ndarray:
        X_processed = X.copy()
        for scaler_fn in scalers:
            X_processed = scaler_fn(X_processed)
        return X_processed
    return preprocess


# ============================================================================
# PREPROCESSING REGISTRY
# ============================================================================

PREPROCESSING_PIPELINES = {
    "standard": create_standard_scaler,
    "minmax": create_minmax_scaler,
    "robust": create_robust_scaler,
    "none": create_no_preprocessing,
}


def get_preprocessing(name: str) -> PreprocessingFn:
    """
    Get preprocessing function by name.
    
    Args:
        name: Key from PREPROCESSING_PIPELINES
    
    Returns:
        Preprocessing function
    
    Example:
        preprocess_fn = get_preprocessing("standard")
        X_processed = preprocess_fn(X)
    """
    if name not in PREPROCESSING_PIPELINES:
        raise ValueError(
            f"Unknown preprocessing: {name}. "
            f"Available: {list(PREPROCESSING_PIPELINES.keys())}"
        )
    return PREPROCESSING_PIPELINES[name]()
