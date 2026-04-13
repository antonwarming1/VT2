"""
Cross-Validation with GridSearch for FNN
═════════════════════════════════════════

Simple, focused workflow:
1. Load features and labels
2. K-Fold Cross-Validation
3. F1 Score metric
4. GridSearchCV for hyperparameter tuning on Feed-Forward Neural Network
5. Display results
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data():
    """Load features and labels."""
    feature_file = r"C:\github\VT2\Feature_engineering\features_selected.csv"
    label_file = r"C:\github\VT2\Feature_engineering\labels.csv"
    
    features = pd.read_csv(feature_file)
    labels = pd.read_csv(label_file).iloc[:, 0].values
    
    # Remove rows with NaN
    mask = ~(features.isna().any(axis=1) | pd.isna(labels))
    X = features[mask].reset_index(drop=True).values
    y = labels[mask]
    
    print(f"✓ Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Classes: {np.bincount(y)}\n")
    
    return X, y


# ============================================================================
# STEP 2: PREPROCESSING
# ============================================================================

def preprocess(X):
    """Standardize features (important for neural networks)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# ============================================================================
# STEP 3: HYPERPARAMETER TUNING (GridSearchCV)
# ============================================================================

def tune_hyperparameters(X, y):
    """
    GridSearch to find best hyperparameters for FNN.
    
    Returns:
        best_model: MLPClassifier with best parameters
        best_params: Dictionary of best parameters
        grid_results: Full GridSearch results
    """
    print("="*70)
    print("STEP 1: HYPERPARAMETER TUNING (GridSearchCV)")
    print("="*70 + "\n")
    
    param_grid = {
        'hidden_layer_sizes': [
            (50,),
            (100,),
            (50, 50),
            (100, 50),
            (100, 100),
        ],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [300, 500],
    }
    
    base_model = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
    )
    
    grid_search.fit(X, y)
    
    print(f"\n✓ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param:25s}: {value}")
    
    print(f"\n✓ Best F1 score (CV):      {grid_search.best_score_:.4f}\n")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search


# ============================================================================
# STEP 4: K-FOLD CROSS-VALIDATION (with best parameters)
# ============================================================================

def cross_validate_model(X, y, model):
    """
    Perform K-Fold Cross-Validation with F1 score.
    
    Returns:
        results: Dictionary with CV scores
    """
    print("="*70)
    print("STEP 2: K-FOLD CROSS-VALIDATION (F1 Score)")
    print("="*70 + "\n")
    
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scoring = {
        'f1_weighted': 'f1_weighted',
        'f1_macro': 'f1_macro',
        'accuracy': 'accuracy',
        'precision_weighted': 'precision_weighted',
        'recall_weighted': 'recall_weighted',
    }
    
    cv_results = cross_validate(
        model, X, y,
        cv=kfold,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )
    
    # Print results
    print(f"Folds: {n_splits}\n")
    print("F1 Score (Weighted):")
    print(f"  Train: {cv_results['train_f1_weighted'].mean():.4f} ± {cv_results['train_f1_weighted'].std():.4f}")
    print(f"  Test:  {cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}\n")
    
    print("F1 Score (Macro):")
    print(f"  Train: {cv_results['train_f1_macro'].mean():.4f} ± {cv_results['train_f1_macro'].std():.4f}")
    print(f"  Test:  {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}\n")
    
    print("Accuracy:")
    print(f"  Train: {cv_results['train_accuracy'].mean():.4f} ± {cv_results['train_accuracy'].std():.4f}")
    print(f"  Test:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}\n")
    
    print("Per-Fold F1 Scores:")
    for i, score in enumerate(cv_results['test_f1_weighted']):
        print(f"  Fold {i+1}: {score:.4f}")
    
    return cv_results


# ============================================================================
# STEP 5: FINAL EVALUATION
# ============================================================================

def evaluate_model(X, y, model, cv_results):
    """Summary of results."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    
    f1_mean = cv_results['test_f1_weighted'].mean()
    f1_std = cv_results['test_f1_weighted'].std()
    f1_min = cv_results['test_f1_weighted'].min()
    f1_max = cv_results['test_f1_weighted'].max()
    
    print(f"F1 Score (Weighted):")
    print(f"  Mean:     {f1_mean:.4f}")
    print(f"  Std Dev:  {f1_std:.4f}")
    print(f"  Min:      {f1_min:.4f}")
    print(f"  Max:      {f1_max:.4f}")
    
    return {
        'f1_mean': f1_mean,
        'f1_std': f1_std,
        'f1_min': f1_min,
        'f1_max': f1_max,
        'cv_results': cv_results,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CROSS-VALIDATION WITH GRIDSEARCH FOR FNN")
    print("="*70 + "\n")
    
    # Load data
    print("="*70)
    print("LOADING DATA")
    print("="*70 + "\n")
    X, y = load_data()
    
    # Preprocess
    print("="*70)
    print("PREPROCESSING")
    print("="*70 + "\n")
    X = preprocess(X)
    print("✓ Features standardized (StandardScaler)\n")
    
    # Tune hyperparameters
    best_model, best_params, grid_results = tune_hyperparameters(X, y)
    
    # K-Fold CV
    cv_results = cross_validate_model(X, y, best_model)
    
    # Evaluate
    summary = evaluate_model(X, y, best_model, cv_results)
    
    print("\n" + "="*70)
    print("✓ DONE")
    print("="*70)
