"""
CrossValidation Module
======================

Clean, production-grade cross-validation framework.

Key principle: Separation of Concerns

- No feature engineering happens here
- No models are defined here
- No metrics are defined here
- Pure orchestration of existing components

Public API:
-----------
    - CrossValidator: Main class for CV orchestration
    - CVResults: Structured results dataclass
    - ComparisonResults: Results from model comparison

Example:
--------
    from CrossValidation import CrossValidator, CVResults
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    
    # Prepare data externally
    X, y = load_data()
    
    # Define factory and metric
    model_factory = lambda: RandomForestClassifier(n_estimators=100)
    metric_fn = lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
    
    # Run CV
    validator = CrossValidator()
    results = validator.validate(X, y, model_factory, metric_fn)
    
    # Access results
    print(f"Mean F1: {results.mean_score:.4f}")
    print(results.summary())

For detailed examples, see examples.py
For architecture notes, see ARCHITECTURE.md
"""

try:
    from core import CrossValidator, ModelFactory, MetricFn
    from schema import CVResults, ComparisonResults
except ImportError:
    # Support both relative and absolute imports
    from .core import CrossValidator, ModelFactory, MetricFn
    from .schema import CVResults, ComparisonResults

__all__ = [
    "CrossValidator",
    "CVResults",
    "ComparisonResults",
    "ModelFactory",
    "MetricFn",
]
