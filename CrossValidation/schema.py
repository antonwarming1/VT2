"""
schema.py
=========
Data classes for structured cross-validation results.

Designed for clarity and extensibility across teams.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np


@dataclass
class CVResults:
    """
    Structured results from cross-validation.
    
    Attributes:
        fold_scores: F1 score (or user metric) for each fold
        fold_predictions: Predictions for each fold's test set
        fold_indices: (train_idx, test_idx) for each fold (for reproducibility)
        
        mean_score: Mean metric across folds
        std_score: Standard deviation across folds
        min_score: Minimum fold score
        max_score: Maximum fold score
        
        fit_times: Time to fit model on each fold (seconds)
        predict_times: Time to predict on each fold (seconds)
        
        metadata: Experiment info (model_factory name, metric name, cv_strategy, etc.)
    """
    
    # Per-fold results
    fold_scores: List[float]
    fold_predictions: List[np.ndarray]
    fold_indices: List[Tuple[np.ndarray, np.ndarray]]
    
    # Aggregate statistics
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    
    # Timing
    fit_times: List[float] = field(default_factory=list)
    predict_times: List[float] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate consistency."""
        n_folds = len(self.fold_scores)
        assert len(self.fold_predictions) == n_folds, \
            f"Mismatch: {len(self.fold_scores)} scores but {len(self.fold_predictions)} predictions"
        assert len(self.fold_indices) == n_folds, \
            f"Mismatch: {len(self.fold_scores)} scores but {len(self.fold_indices)} index tuples"
    
    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return len(self.fold_scores)
    
    @property
    def mean_fit_time(self) -> float:
        """Average fitting time (seconds)."""
        return np.mean(self.fit_times) if self.fit_times else 0.0
    
    @property
    def mean_predict_time(self) -> float:
        """Average prediction time (seconds)."""
        return np.mean(self.predict_times) if self.predict_times else 0.0
    
    def summary(self) -> str:
        """
        Return human-readable summary.
        
        Returns:
            str: Formatted summary of results
        """
        lines = [
            "=" * 70,
            "CROSS-VALIDATION RESULTS",
            "=" * 70,
            f"Folds:                  {self.n_folds}",
            f"Metric:                 {self.metadata.get('metric_name', 'Unknown')}",
            "",
            "SCORE STATISTICS:",
            f"  Mean:                 {self.mean_score:.4f}",
            f"  Std Dev:              {self.std_score:.4f}",
            f"  Min:                  {self.min_score:.4f}",
            f"  Max:                  {self.max_score:.4f}",
            "",
            "PER-FOLD SCORES:",
        ]
        
        for i, score in enumerate(self.fold_scores):
            lines.append(f"  Fold {i+1}: {score:.4f}")
        
        if self.fit_times:
            lines.extend([
                "",
                "TIMING:",
                f"  Mean fit time:        {self.mean_fit_time:.4f}s",
                f"  Mean predict time:    {self.mean_predict_time:.4f}s",
            ])
        
        if self.metadata:
            lines.extend([
                "",
                "METADATA:",
            ])
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization (JSON, etc).
        
        Returns:
            dict: All results as native Python types (numpy arrays converted)
        """
        return {
            "fold_scores": [float(s) for s in self.fold_scores],
            "mean_score": float(self.mean_score),
            "std_score": float(self.std_score),
            "min_score": float(self.min_score),
            "max_score": float(self.max_score),
            "fit_times": self.fit_times,
            "predict_times": self.predict_times,
            "metadata": self.metadata,
            "n_folds": self.n_folds,
        }


@dataclass
class ComparisonResults:
    """
    Results from comparing multiple models via cross-validation.
    
    Attributes:
        results: Dict mapping model_name -> CVResults
        best_model_name: Name of model with highest mean_score
        ranking: List of (model_name, mean_score) sorted by score
    """
    
    results: Dict[str, CVResults]
    best_model_name: str
    ranking: List[Tuple[str, float]] = field(default_factory=list)
    
    def __post_init__(self):
        """Sort ranking."""
        if not self.ranking:
            self.ranking = sorted(
                [(name, res.mean_score) for name, res in self.results.items()],
                key=lambda x: x[1],
                reverse=True,
            )
    
    def summary(self) -> str:
        """Human-readable comparison summary."""
        lines = [
            "=" * 70,
            "MODEL COMPARISON (Cross-Validation)",
            "=" * 70,
            "",
            "RANKING (by mean score):",
            "",
        ]
        
        for rank, (model_name, score) in enumerate(self.ranking, 1):
            result = self.results[model_name]
            lines.append(
                f"{rank}. {model_name:20s} | "
                f"Mean: {score:.4f} | "
                f"Std: {result.std_score:.4f} | "
                f"Range: [{result.min_score:.4f}, {result.max_score:.4f}]"
            )
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
