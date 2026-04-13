"""
QUICK START GUIDE
=================

This module is designed for multi-developer ML projects.
Each role knows only about its own layer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLE 1: DATA ENGINEER
─────────────────────

Responsibility: Load and preprocess data

Code lives: data_loading.py or preprocessing.py (NOT in CV module)

You provide: X (n_samples, n_features), y (n_samples,)

Example:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    def load_data():
        features = pd.read_csv("features.csv")
        labels = pd.read_csv("labels.csv").iloc[:, 0].values
        return features.values, labels
    
    def preprocess(X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    X, y = load_data()
    X = preprocess(X)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLE 2: PREPROCESSING ENGINEER
──────────────────────────────

Responsibility: Define preprocessing strategies

Code lives: preprocessing.py (NOT in CV module)

You provide: Callable that transforms X (array) -> X_transformed (array)

Example:
    from sklearn.preprocessing import StandardScaler
    
    def create_standard_scaler():
        def preprocess(X):
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        return preprocess
    
    preprocess_fn = create_standard_scaler()
    X_processed = preprocess_fn(X)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLE 3: MODEL ENGINEER
──────────────────────

Responsibility: Define model factory functions

Code lives: model_config.py or models.py (NOT in CV module)

You provide: Callable that returns fresh model instance

Example:
    from sklearn.ensemble import RandomForestClassifier
    
    def create_random_forest():
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )
    
    model_factory = create_random_forest


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLE 3: METRICS ENGINEER
────────────────────────

Responsibility: Define metric functions

Code lives: metrics.py (NOT in CV module)

You provide: Callable (y_true, y_pred) -> float

Example:
    from sklearn.metrics import f1_score, roc_auc_score
    
    def f1_weighted(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')
    
    metric_fn = f1_weighted


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLE 4: ML SCIENTIST (ORCHESTRATOR)
────────────────────────────────────

Responsibility: Run experiments (assemble all layers)

Code lives: experiments.py or scripts/

You consume: Data, models, metrics
You run: CV orchestration
You get: Structured results

Example:
    from CrossValidation import CrossValidator
    from data_loading import load_data, preprocess
    from model_config import MODEL_FACTORIES
    from metrics import METRICS
    
    # Get components from other roles
    X, y = load_data()
    X = preprocess(X)
    
    model_factory = MODEL_FACTORIES["Random Forest"]
    metric_fn = METRICS["F1 (weighted)"]
    
    # Orchestrate
    validator = CrossValidator()
    results = validator.validate(X, y, model_factory, metric_fn)
    
    print(results.summary())


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPICAL WORKFLOW
━━━━━━━━━━━━━━━━

1. Data Engineer prepares X, y
2. Model Engineer defines factories
3. Metrics Engineer defines metric functions
4. ML Scientist chains them together via CrossValidator

Result: Clean separation, easy to test, easy to swap components


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API CHEAT SHEET
═══════════════

Single Model Evaluation:
    from CrossValidation import CrossValidator
    
    validator = CrossValidator(cv_strategy=StratifiedKFold(n_splits=5))
    results = validator.validate(X, y, model_factory, metric_fn)
    
    print(results.mean_score)
    print(results.std_score)
    print(results.summary())


Compare Multiple Models:
    comparison = validator.compare_models(
        X, y,
        model_factories={
            "Model A": factory_a,
            "Model B": factory_b,
        },
        metric_fn=metric_fn,
    )
    
    print(comparison.summary())


Access Per-Fold Results:
    for i, (score, pred) in enumerate(zip(results.fold_scores, results.fold_predictions)):
        print(f"Fold {i+1}: Score={score:.4f}, Predictions shape={pred.shape}")


Get All Results as Dictionary (for serialization):
    results_dict = results.to_dict()
    # Save to JSON, pickle, etc.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT: What NOT to Do in CV Module
═══════════════════════════════════════

❌ Don't add feature engineering
❌ Don't import specific models (RandomForest, SVM, etc.)
❌ Don't define metrics (F1, AUC, etc.)
❌ Don't add data loading logic
❌ Don't add hyperparameter tuning (that's separate!)

✅ DO:
✅ Accept any model via factory pattern
✅ Accept any metric function
✅ Accept pre-computed X, y
✅ Orchestrate CV workflow
✅ Return structured results


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

See examples.py for complete working examples!
"""
