"""
core.py
=======
Core cross-validation orchestration logic.

Model-agnostic, feature-agnostic, metric-agnostic.
Pure orchestration - no implementation details.
"""

import time
import logging
from typing import Callable, List, Tuple, Dict, Any
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from schema import CVResults

# Optional: Set up logging (non-intrusive)
logger = logging.getLogger(__name__)


# Type definitions for clarity
ModelFactory = Callable[[], Any]  # Callable that returns a fresh model
MetricFn = Callable[[np.ndarray, np.ndarray], float]  # (y_true, y_pred) -> score


class CrossValidator:
    """
    Pure cross-validation orchestrator.
    
    Does NOT:
    - Define or instantiate models
    - Engineer features
    - Know about specific metrics
    
    Does:
    - Split data according to strategy (KFold, StratifiedKFold, custom)
    - Train models via factory pattern
    - Evaluate metrics via pluggable functions
    - Return structured results
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.metrics import f1_score
        >>> 
        >>> model_factory = lambda: RandomForestClassifier(n_estimators=100)
        >>> metric_fn = lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
        >>> 
        >>> validator = CrossValidator(cv_strategy=StratifiedKFold(n_splits=5))
        >>> results = validator.validate(X, y, model_factory, metric_fn)
        >>> print(results.summary())
    """
    
    def __init__(
        self,
        cv_strategy: Any = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize cross-validator.
        
        Args:
            cv_strategy: sklearn splitter (KFold, StratifiedKFold, etc.)
                        If None, defaults to StratifiedKFold(n_splits=5)
            random_state: Seed for reproducibility
            verbose: Print per-fold progress
        """
        self.random_state = random_state
        self.verbose = verbose
        
        if cv_strategy is None:
            self.cv_strategy = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=random_state,
            )
        else:
            self.cv_strategy = cv_strategy
        
        np.random.seed(random_state)
    
    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: ModelFactory,
        metric_fn: MetricFn,
        model_name: str = "Model",
        metric_name: str = "Metric",
    ) -> CVResults:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            model_factory: Callable that returns a fresh model instance
            metric_fn: Function (y_true, y_pred) -> float
            model_name: Name for metadata (for debugging/logging)
            metric_name: Name for metadata
        
        Returns:
            CVResults: Structured results with means, stds, per-fold scores, etc.
        
        Raises:
            AssertionError: If input dimensions don't match
        """
        assert X.shape[0] == len(y), \
            f"Shape mismatch: {X.shape[0]} samples but {len(y)} labels"
        assert X.ndim == 2, f"X must be 2D, got {X.ndim}D"
        
        fold_scores = []
        fold_predictions = []
        fold_indices = []
        fit_times = []
        predict_times = []
        
        fold_id = 0
        
        # Iterate over folds
        for train_idx, test_idx in self.cv_strategy.split(X, y):
            fold_id += 1
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create fresh model instance
            model = model_factory()
            
            # Fit
            t0 = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - t0
            fit_times.append(fit_time)
            
            # Predict
            t0 = time.time()
            y_pred = model.predict(X_test)
            predict_time = time.time() - t0
            predict_times.append(predict_time)
            
            # Evaluate
            score = metric_fn(y_test, y_pred)
            fold_scores.append(score)
            fold_predictions.append(y_pred)
            fold_indices.append((train_idx, test_idx))
            
            if self.verbose:
                logger.info(
                    f"Fold {fold_id}: {metric_name}={score:.4f} "
                    f"(fit: {fit_time:.4f}s, pred: {predict_time:.4f}s)"
                )
        
        # Aggregate
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        min_score = np.min(fold_scores)
        max_score = np.max(fold_scores)
        
        # Package results
        metadata = {
            "model_name": model_name,
            "metric_name": metric_name,
            "cv_strategy": str(self.cv_strategy),
            "random_state": self.random_state,
        }
        
        results = CVResults(
            fold_scores=fold_scores,
            fold_predictions=fold_predictions,
            fold_indices=fold_indices,
            mean_score=mean_score,
            std_score=std_score,
            min_score=min_score,
            max_score=max_score,
            fit_times=fit_times,
            predict_times=predict_times,
            metadata=metadata,
        )
        
        if self.verbose:
            logger.info(
                f"\nCross-validation summary:\n"
                f"  {metric_name} mean: {mean_score:.4f} ± {std_score:.4f}\n"
                f"  Range: [{min_score:.4f}, {max_score:.4f}]"
            )
        
        return results
    
    def compare_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factories: Dict[str, ModelFactory],
        metric_fn: MetricFn,
        metric_name: str = "Metric",
    ) -> "ComparisonResults":
        """
        Compare multiple models via cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            model_factories: Dict mapping model_name -> factory
            metric_fn: Metric function
            metric_name: Name for metadata
        
        Returns:
            ComparisonResults: Ranking and individual results for each model
        """
        from schema import ComparisonResults
        
        results = {}
        for model_name, factory in model_factories.items():
            if self.verbose:
                logger.info(f"\nEvaluating: {model_name}")
            
            results[model_name] = self.validate(
                X, y, factory, metric_fn,
                model_name=model_name,
                metric_name=metric_name,
            )
        
        best_model = max(results.items(), key=lambda x: x[1].mean_score)[0]
        
        comparison = ComparisonResults(
            results=results,
            best_model_name=best_model,
        )
        
        if self.verbose:
            logger.info(f"\n{comparison.summary()}")
        
        return comparison
