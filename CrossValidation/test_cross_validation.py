"""
test_cross_validation.py
════════════════════════

Unit tests for the cross-validation module.
Demonstrates testability from separation of concerns.

Run: pytest test_cross_validation.py -v
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold

from core import CrossValidator
from schema import CVResults, ComparisonResults


# ============================================================================
# FIXTURES (Reusable test data and components)
# ============================================================================

@pytest.fixture
def synthetic_data():
    """Create synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def model_factory():
    """Simple model factory for testing."""
    return lambda: RandomForestClassifier(n_estimators=10, random_state=42)


@pytest.fixture
def metric_function():
    """Simple metric for testing."""
    return lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)


@pytest.fixture
def validator():
    """Default validator for testing."""
    return CrossValidator(verbose=False)


# ============================================================================
# UNIT TESTS: CrossValidator
# ============================================================================

class TestCrossValidatorBasic:
    """Test basic cross-validation functionality."""
    
    def test_validate_returns_cvresults(self, validator, synthetic_data, model_factory, metric_function):
        """Test that validate() returns a CVResults object."""
        X, y = synthetic_data
        results = validator.validate(X, y, model_factory, metric_function)
        
        assert isinstance(results, CVResults)
    
    def test_validate_correct_number_of_folds(self, synthetic_data, model_factory, metric_function):
        """Test that results contain correct number of folds."""
        X, y = synthetic_data
        validator = CrossValidator(
            cv_strategy=StratifiedKFold(n_splits=5),
            verbose=False,
        )
        results = validator.validate(X, y, model_factory, metric_function)
        
        assert results.n_folds == 5
        assert len(results.fold_scores) == 5
        assert len(results.fold_predictions) == 5
        assert len(results.fold_indices) == 5
    
    def test_validate_predictions_shapes(self, synthetic_data, model_factory, metric_function):
        """Test that per-fold predictions have correct shapes."""
        X, y = synthetic_data
        validator = CrossValidator(cv_strategy=StratifiedKFold(n_splits=5), verbose=False)
        results = validator.validate(X, y, model_factory, metric_function)
        
        total_samples = 0
        for pred in results.fold_predictions:
            assert pred.ndim == 1, "Predictions should be 1D"
            total_samples += len(pred)
        
        assert total_samples == len(y), "All samples should be predicted exactly once"
    
    def test_validate_scores_in_valid_range(self, validator, synthetic_data, model_factory, metric_function):
        """Test that F1 scores are in [0, 1]."""
        X, y = synthetic_data
        results = validator.validate(X, y, model_factory, metric_function)
        
        for score in results.fold_scores:
            assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"
        
        assert 0.0 <= results.mean_score <= 1.0
        assert 0.0 <= results.min_score <= 1.0
        assert 0.0 <= results.max_score <= 1.0
    
    def test_validate_statistics_consistency(self, validator, synthetic_data, model_factory, metric_function):
        """Test that mean/std/min/max are consistent with fold_scores."""
        X, y = synthetic_data
        results = validator.validate(X, y, model_factory, metric_function)
        
        expected_mean = np.mean(results.fold_scores)
        expected_std = np.std(results.fold_scores)
        expected_min = np.min(results.fold_scores)
        expected_max = np.max(results.fold_scores)
        
        assert np.isclose(results.mean_score, expected_mean)
        assert np.isclose(results.std_score, expected_std)
        assert np.isclose(results.min_score, expected_min)
        assert np.isclose(results.max_score, expected_max)
    
    def test_validate_timing_recorded(self, validator, synthetic_data, model_factory, metric_function):
        """Test that fit and predict times are recorded."""
        X, y = synthetic_data
        results = validator.validate(X, y, model_factory, metric_function)
        
        assert len(results.fit_times) == 5
        assert len(results.predict_times) == 5
        
        for fit_time in results.fit_times:
            assert fit_time > 0.0
        
        for predict_time in results.predict_times:
            assert predict_time >= 0.0  # Predicting fast samples might be ~0


# ============================================================================
# UNIT TESTS: Metric Agnosticism
# ============================================================================

class TestMetricAgnosticism:
    """Test that different metrics work interchangeably."""
    
    def test_validate_with_different_metrics(self, validator, synthetic_data, model_factory):
        """Test CV works with different metric functions."""
        X, y = synthetic_data
        
        f1_metric = lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0)
        acc_metric = lambda yt, yp: accuracy_score(yt, yp)
        
        f1_results = validator.validate(X, y, model_factory, f1_metric, metric_name="F1")
        acc_results = validator.validate(X, y, model_factory, acc_metric, metric_name="Accuracy")
        
        # Both should be valid
        assert isinstance(f1_results, CVResults)
        assert isinstance(acc_results, CVResults)
        
        # Different metrics should give different scores
        assert not np.isclose(f1_results.mean_score, acc_results.mean_score)
    
    def test_validate_metric_metadata(self, validator, synthetic_data, model_factory):
        """Test that metric name is stored in metadata."""
        X, y = synthetic_data
        custom_metric = lambda yt, yp: accuracy_score(yt, yp)
        
        results = validator.validate(
            X, y,
            model_factory=model_factory,
            metric_fn=custom_metric,
            metric_name="Custom Accuracy"
        )
        
        assert results.metadata["metric_name"] == "Custom Accuracy"


# ============================================================================
# UNIT TESTS: Model Agnosticism
# ============================================================================

class TestModelAgnosticism:
    """Test that different models work interchangeably."""
    
    def test_validate_with_different_models(self, validator, synthetic_data, metric_function):
        """Test CV works with different model types."""
        X, y = synthetic_data
        
        rf_factory = lambda: RandomForestClassifier(n_estimators=10, random_state=42)
        lr_factory = lambda: LogisticRegression(max_iter=1000, random_state=42)
        
        rf_results = validator.validate(X, y, rf_factory, metric_function, model_name="RF")
        lr_results = validator.validate(X, y, lr_factory, metric_function, model_name="LR")
        
        # Both should be valid
        assert isinstance(rf_results, CVResults)
        assert isinstance(lr_results, CVResults)
        
        # Different models likely give different scores
        # (not guaranteed, but very likely)
        assert rf_results.metadata["model_name"] == "RF"
        assert lr_results.metadata["model_name"] == "LR"
    
    def test_validate_model_metadata(self, validator, synthetic_data, metric_function):
        """Test that model name is stored in metadata."""
        X, y = synthetic_data
        model_factory = lambda: RandomForestClassifier(n_estimators=10, random_state=42)
        
        results = validator.validate(
            X, y,
            model_factory=model_factory,
            metric_fn=metric_function,
            model_name="Test Model"
        )
        
        assert results.metadata["model_name"] == "Test Model"


# ============================================================================
# UNIT TESTS: Reproducibility
# ============================================================================

class TestReproducibility:
    """Test that results are reproducible with same random_state."""
    
    def test_same_random_state_same_results(self, synthetic_data, model_factory, metric_function):
        """Test that same random_state produces same results."""
        X, y = synthetic_data
        
        validator1 = CrossValidator(random_state=42, verbose=False)
        validator2 = CrossValidator(random_state=42, verbose=False)
        
        results1 = validator1.validate(X, y, model_factory, metric_function)
        results2 = validator2.validate(X, y, model_factory, metric_function)
        
        # Scores should be identical
        for s1, s2 in zip(results1.fold_scores, results2.fold_scores):
            assert np.isclose(s1, s2)
    
    def test_different_random_state_different_results(self, synthetic_data, model_factory, metric_function):
        """Test that different random_states can produce different splits."""
        X, y = synthetic_data
        
        validator1 = CrossValidator(random_state=42, verbose=False)
        validator2 = CrossValidator(random_state=123, verbose=False)
        
        results1 = validator1.validate(X, y, model_factory, metric_function)
        results2 = validator2.validate(X, y, model_factory, metric_function)
        
        # Fold indices should be different
        indices1 = results1.fold_indices[0]
        indices2 = results2.fold_indices[0]
        
        assert not np.array_equal(indices1[0], indices2[0])  # Different train indices


# ============================================================================
# UNIT TESTS: CVResults dataclass
# ============================================================================

class TestCVResults:
    """Test CVResults data structure."""
    
    def test_cvresults_to_dict(self, synthetic_data, model_factory, metric_function):
        """Test conversion to dictionary (for serialization)."""
        X, y = synthetic_data
        validator = CrossValidator(verbose=False)
        results = validator.validate(X, y, model_factory, metric_function)
        
        results_dict = results.to_dict()
        
        assert isinstance(results_dict, dict)
        assert "fold_scores" in results_dict
        assert "mean_score" in results_dict
        assert "metadata" in results_dict
        
        # Check that numpy arrays converted to lists (JSON-serializable)
        assert isinstance(results_dict["fold_scores"], list)
        assert all(isinstance(s, float) for s in results_dict["fold_scores"])
    
    def test_cvresults_summary(self, synthetic_data, model_factory, metric_function):
        """Test that summary generates readable output."""
        X, y = synthetic_data
        validator = CrossValidator(verbose=False)
        results = validator.validate(X, y, model_factory, metric_function)
        
        summary = results.summary()
        
        assert isinstance(summary, str)
        assert "CROSS-VALIDATION RESULTS" in summary
        assert f"Folds: 5" in summary
        assert f"Mean:" in summary


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test full workflow integration."""
    
    def test_compare_models(self, synthetic_data, metric_function):
        """Test comparing multiple models."""
        X, y = synthetic_data
        
        models = {
            "RandomForest": lambda: RandomForestClassifier(n_estimators=10, random_state=42),
            "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        }
        
        validator = CrossValidator(verbose=False)
        comparison = validator.compare_models(X, y, models, metric_function)
        
        assert isinstance(comparison, ComparisonResults)
        assert len(comparison.results) == 2
        assert comparison.best_model_name in ["RandomForest", "LogisticRegression"]
        assert len(comparison.ranking) == 2
    
    def test_full_workflow(self, synthetic_data):
        """Test a complete workflow: load -> validate -> compare."""
        X, y = synthetic_data
        
        # Define components (as separate engineers would)
        def f1_metric(yt, yp):
            return f1_score(yt, yp, average='weighted', zero_division=0)
        
        models = {
            "Simple RF": lambda: RandomForestClassifier(n_estimators=5, random_state=42),
            "Simple LR": lambda: LogisticRegression(max_iter=1000, random_state=42),
        }
        
        # Orchestrate (as ML Scientist would)
        validator = CrossValidator(
            cv_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            verbose=False,
        )
        
        comparison = validator.compare_models(X, y, models, f1_metric, metric_name="F1")
        
        # Validate results
        assert comparison.best_model_name in models.keys()
        print("\n" + comparison.summary())


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_validate_input_validation(self, model_factory, metric_function):
        """Test that mismatched X, y dimensions raise error."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 50)  # Wrong size
        
        validator = CrossValidator(verbose=False)
        
        with pytest.raises(AssertionError):
            validator.validate(X, y, model_factory, metric_function)
    
    def test_validate_x_wrong_dimensions(self, metric_function):
        """Test that 1D X raises error."""
        X = np.random.randn(100)  # 1D instead of 2D
        y = np.random.randint(0, 2, 100)
        
        validator = CrossValidator(verbose=False)
        model_factory = lambda: RandomForestClassifier(n_estimators=10, random_state=42)
        
        with pytest.raises(AssertionError):
            validator.validate(X, y, model_factory, metric_function)


if __name__ == "__main__":
    # Run: pytest test_cross_validation.py -v
    pytest.main([__file__, "-v"])
