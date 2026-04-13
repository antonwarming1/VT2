"""
Cross-Validation Framework with F1 Score, Error Estimation & Hyperparameter Tuning
====================================================================================

This framework provides:
  1. Cross-validation with F1 scoring
  2. True prediction error estimation
  3. Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
  4. Algorithm-agnostic design (easy to swap models)

Features:
  - k-Fold Cross-Validation
  - Train/Test split analysis
  - Hyperparameter optimization
  - Comprehensive metrics reporting

Usage:
  python CrossValidation/CrossValidation.py
"""

import warnings
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_validate, 
    GridSearchCV, 
    RandomizedSearchCV,
    train_test_split, 
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score
)

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ============================================================================
# CONFIGURATION
# ============================================================================
FEATURE_DIR = Path(r"C:\github\VT2\Feature_engineering")
FEATURE_FILE = FEATURE_DIR / "features_selected.csv"
LABEL_FILE = FEATURE_DIR / "labels.csv"

# Can also try features_extracted.csv for all features
# FEATURE_FILE = FEATURE_DIR / "features_extracted.csv"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load features and labels."""
    features = pd.read_csv(FEATURE_FILE)
    labels = pd.read_csv(LABEL_FILE)
    
    # Flatten labels if needed
    if labels.shape[1] == 1:
        y = labels.iloc[:, 0].values
    else:
        y = labels.values.flatten()
    
    # Remove any rows with NaN
    mask = ~(features.isna().any(axis=1) | pd.isna(y))
    X = features[mask].reset_index(drop=True)
    y = y[mask]
    
    print(f"✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"✓ Class distribution: {np.bincount(y)}")
    
    return X, y


# ============================================================================
# PREPROCESSING
# ============================================================================
def preprocess_data(X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize features to have mean=0, std=1.
    This is important for algorithms like SVM, Logistic Regression.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


# ============================================================================
# ALGORITHM CONFIGURATIONS
# ============================================================================
ALGORITHMS = {
    "logistic_regression": {
        "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "params": {
            "C": [0.001, 0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "tuning_type": "grid",  # "grid" or "random"
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "tuning_type": "random",  # Too many combinations for grid
    },
    "svm": {
        "model": SVC(random_state=RANDOM_STATE, probability=True),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "tuning_type": "grid",
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5],
        },
        "tuning_type": "random",
    },
}


# ============================================================================
# CROSS-VALIDATION WITH F1 SCORE
# ============================================================================
class CrossValidationFramework:
    """
    Comprehensive cross-validation framework with F1 scoring and error estimation.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5):
        """
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            cv_folds: Number of folds for k-fold cross-validation
        """
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        
    def evaluate_with_cv(self, model, algorithm_name: str) -> Dict[str, Any]:
        """
        Perform cross-validation with multiple metrics.
        
        Returns:
            Dictionary with CV results and statistics
        """
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATION: {algorithm_name.upper()}")
        print(f"{'='*70}")
        
        scoring = {
            "f1": "f1",
            "f1_weighted": "f1_weighted",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        }
        
        cv_results = cross_validate(
            model, self.X, self.y, 
            cv=self.cv, 
            scoring=scoring,
            return_train_score=True,
        )
        
        # Extract and report results
        results = self._summarize_cv_results(cv_results)
        return results
    
    def _summarize_cv_results(self, cv_results: Dict) -> Dict[str, Any]:
        """Summarize cross-validation results with statistics."""
        summary = {}
        
        for metric in ["f1", "f1_weighted", "accuracy", "precision", "recall", "roc_auc"]:
            test_key = f"test_{metric}"
            train_key = f"train_{metric}"
            
            if test_key in cv_results:
                test_scores = cv_results[test_key]
                train_scores = cv_results[train_key]
                
                summary[metric] = {
                    "test_mean": test_scores.mean(),
                    "test_std": test_scores.std(),
                    "test_scores": test_scores,
                    "train_mean": train_scores.mean(),
                    "train_std": train_scores.std(),
                    "overfitting_gap": train_scores.mean() - test_scores.mean(),
                }
                
                print(f"\n{metric.upper()}:")
                print(f"  Test:  {test_scores.mean():.4f} ± {test_scores.std():.4f}")
                print(f"  Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")
                print(f"  Overfitting Gap: {summary[metric]['overfitting_gap']:.4f}")
        
        return summary
    
    def estimate_true_error(self, model) -> Dict[str, float]:
        """
        Estimate true prediction error using train/test split.
        This is a single train/test split to estimate generalization error.
        
        Returns:
            Dictionary with error metrics
        """
        print(f"\n{'='*70}")
        print("TRUE PREDICTION ERROR ESTIMATION")
        print(f"{'='*70}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=RANDOM_STATE, stratify=self.y
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None
        
        error_metrics = {
            "train_f1": f1_score(y_train, model.predict(X_train)),
            "test_f1": f1_score(y_test, y_pred),
            "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_roc_auc": auc,
            "test_size": len(y_test),
        }
        
        print(f"\nTrain F1 Score:     {error_metrics['train_f1']:.4f}")
        print(f"Test F1 Score:      {error_metrics['test_f1']:.4f}")
        print(f"Train Accuracy:     {error_metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy:      {error_metrics['test_accuracy']:.4f}")
        print(f"Test Precision:     {error_metrics['test_precision']:.4f}")
        print(f"Test Recall:        {error_metrics['test_recall']:.4f}")
        if auc is not None:
            print(f"Test ROC-AUC:       {auc:.4f}")
        
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Under"]))
        
        return error_metrics


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================
class HyperparameterTuner:
    """Hyperparameter tuning with GridSearchCV or RandomizedSearchCV."""
    
    @staticmethod
    def tune_hyperparameters(
        X: np.ndarray, 
        y: np.ndarray,
        model,
        params: Dict[str, List],
        tuning_type: str = "grid",
        cv_folds: int = 5,
        n_iter: int = 20,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Tune hyperparameters using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            model: Scikit-learn model
            params: Parameter grid
            tuning_type: "grid" for GridSearchCV, "random" for RandomizedSearchCV
            cv_folds: Number of CV folds
            n_iter: Number of iterations for random search
            
        Returns:
            Best model and tuning results
        """
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER TUNING ({tuning_type.upper()})")
        print(f"{'='*70}")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        
        if tuning_type == "grid":
            searcher = GridSearchCV(
                model, 
                params, 
                cv=cv, 
                scoring="f1",
                n_jobs=-1,
                verbose=1,
            )
        else:  # random
            searcher = RandomizedSearchCV(
                model, 
                params, 
                cv=cv, 
                scoring="f1",
                n_jobs=-1,
                n_iter=n_iter,
                random_state=RANDOM_STATE,
                verbose=1,
            )
        
        searcher.fit(X, y)
        
        print(f"\n✓ Best parameters: {searcher.best_params_}")
        print(f"✓ Best F1 score (CV): {searcher.best_score_:.4f}")
        print(f"✓ Number of models evaluated: {len(searcher.cv_results_['params'])}")
        
        tuning_results = {
            "best_params": searcher.best_params_,
            "best_score": searcher.best_score_,
            "best_model": searcher.best_estimator_,
            "cv_results": searcher.cv_results_,
        }
        
        return searcher.best_estimator_, tuning_results


# ============================================================================
# MAIN WORKFLOW
# ============================================================================
def run_full_pipeline(algorithm_name: str = "random_forest"):
    """
    Run complete pipeline:
    1. Load and preprocess data
    2. Tune hyperparameters
    3. Cross-validate with F1 scoring
    4. Estimate true prediction error
    
    Args:
        algorithm_name: One of ALGORITHMS.keys()
    """
    if algorithm_name not in ALGORITHMS:
        print(f"Available algorithms: {list(ALGORITHMS.keys())}")
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # ========== STEP 1: Load Data ==========
    print(f"\n{'='*70}")
    print("STEP 1: LOADING DATA")
    print(f"{'='*70}")
    X, y = load_data()
    
    # ========== STEP 2: Preprocess ==========
    print(f"\n{'='*70}")
    print("STEP 2: PREPROCESSING")
    print(f"{'='*70}")
    X_scaled, y = preprocess_data(X, y)
    print(f"✓ Data standardized (mean=0, std=1)")
    
    # ========== STEP 3: Hyperparameter Tuning ==========
    algo_config = ALGORITHMS[algorithm_name]
    model, tuning_results = HyperparameterTuner.tune_hyperparameters(
        X_scaled,
        y,
        algo_config["model"],
        algo_config["params"],
        tuning_type=algo_config["tuning_type"],
        cv_folds=5,
        n_iter=20,  # For random search
    )
    
    # ========== STEP 4: Cross-Validation with F1 ==========
    cv_framework = CrossValidationFramework(X_scaled, y, cv_folds=5)
    cv_results = cv_framework.evaluate_with_cv(model, algorithm_name)
    
    # ========== STEP 5: True Error Estimation ==========
    error_metrics = cv_framework.estimate_true_error(model)
    
    # ========== SUMMARY ==========
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Algorithm:           {algorithm_name}")
    print(f"Best Hyperparameters: {tuning_results['best_params']}")
    print(f"CV F1 Score:         {cv_results['f1']['test_mean']:.4f} ± {cv_results['f1']['test_std']:.4f}")
    print(f"Test F1 Score:       {error_metrics['test_f1']:.4f}")
    print(f"Test Accuracy:       {error_metrics['test_accuracy']:.4f}")
    print(f"F1-Weighted:         {cv_results['f1_weighted']['test_mean']:.4f} ± {cv_results['f1_weighted']['test_std']:.4f}")
    
    return {
        "algorithm": algorithm_name,
        "tuning_results": tuning_results,
        "cv_results": cv_results,
        "error_metrics": error_metrics,
        "model": model,
    }


def compare_algorithms(algorithms: List[str] = None):
    """
    Compare multiple algorithms side-by-side.
    
    Args:
        algorithms: List of algorithm names to compare. 
                   If None, compares all available algorithms.
    """
    if algorithms is None:
        algorithms = list(ALGORITHMS.keys())
    
    results = {}
    for algo in algorithms:
        print(f"\n\n{'#'*70}")
        print(f"# ALGORITHM: {algo.upper()}")
        print(f"{'#'*70}\n")
        
        try:
            results[algo] = run_full_pipeline(algo)
        except Exception as e:
            print(f"✗ Error with {algo}: {e}")
            results[algo] = None
    
    # ========== COMPARISON TABLE ==========
    print(f"\n\n{'='*70}")
    print("COMPARISON - F1 SCORES")
    print(f"{'='*70}\n")
    
    comparison_data = []
    for algo, result in results.items():
        if result is not None:
            comparison_data.append({
                "Algorithm": algo,
                "CV F1 (Mean)": result["cv_results"]["f1"]["test_mean"],
                "CV F1 (Std)": result["cv_results"]["f1"]["test_std"],
                "Test F1": result["error_metrics"]["test_f1"],
                "Test Accuracy": result["error_metrics"]["test_accuracy"],
            })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values("CV F1 (Mean)", ascending=False)
    print(comparison_df.to_string(index=False))
    
    return results


if __name__ == "__main__":
    # ========== CHOOSE ONE OPTION ==========
    
    # Option 1: Test a single algorithm
    result = run_full_pipeline("random_forest")
    
    # Option 2: Compare multiple algorithms
    # results = compare_algorithms(["logistic_regression", "random_forest", "svm"])
    # results = compare_algorithms()  # Compares all
