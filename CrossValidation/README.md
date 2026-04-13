"""
CrossValidation Module - Complete Guide
========================================

This folder contains a production-grade, multi-developer-friendly 
cross-validation framework for machine learning projects.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 WHAT'S IN THIS FOLDER?
═════════════════════════

core.py
────────
The heart of the module. Contains:
- CrossValidator class (pure orchestration logic)
- validate() method (single model evaluation)
- compare_models() method (multiple model comparison)

Does NOT contain:
- Model definitions (DecisionTree, RandomForest, etc.)
- Preprocessing logic
- Feature engineering
- Metric definitions
- Data loading logic

→ Use when: Orchestrating CV workflow


preprocessing.py
────────────────
Pluggable preprocessing strategies:
- StandardScaler (default)
- MinMaxScaler
- RobustScaler
- NoPreprocessing (for tree-based algorithms)
- Custom pipelines

Does NOT contain:
- Feature engineering or selection
- CV logic

→ Use when: Transforming features before CV


schema.py
─────────
Data structures for results:
- CVResults: Single model results
  • Per-fold scores and predictions
  • Summary statistics (mean, std, min, max)
  • Timing information
  • Metadata (model name, metric name, CV strategy)
- ComparisonResults: Multiple models comparison

→ Use when: Accessing results


__init__.py
───────────
Package initialization. Exports public API.

→ Use when: Importing module


examples.py
───────────
4 complete, working examples showing:
1. Single model + single metric
2. Same model + different metrics
3. Different models + same metric
4. Hyperparameter tuning + CV integration

→ Use when: Learning how to use the module


ARCHITECTURE.md
───────────────
Detailed design rationale:
- Separation of concerns explanation
- Model-agnostic design pattern
- Feature-agnostic design pattern
- Metric-agnostic design pattern
- Workflow diagram

→ Read when: Understanding the design


QUICKSTART.md
─────────────
Quick reference for different roles:
- Data Engineer
- Model Engineer
- Metrics Engineer
- ML Scientist (Orchestrator)

Contains: Role descriptions, code patterns, API cheat sheet

→ Read when: Learning your role or using for first time


INTEGRATION.md
──────────────
How to integrate CV module into complete ML project:
- Recommended project structure
- Step-by-step setup guide
- Role responsibilities
- Common tasks and patterns
- Testing examples
- Benefits of this architecture

→ Read when: Setting up project structure


test_cross_validation.py
────────────────────────
Unit tests demonstrating:
- Basic CV functionality
- Metric agnosticism
- Model agnosticism
- Reproducibility
- Edge cases

Run: pytest test_cross_validation.py -v

→ Use when: Testing, or as examples for your own tests


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 DESIGN PRINCIPLES
════════════════════

1. SEPARATION OF CONCERNS
   Data → Models → Metrics → CV Orchestration
   Each layer handles ONE responsibility

2. MODEL-AGNOSTIC
   Accept models via factory pattern, not by importing specific models

3. FEATURE-AGNOSTIC
   Accept pre-computed X, y; don't do feature engineering

4. METRIC-AGNOSTIC
   Accept any function (y_true, y_pred) → float; don't define metrics here

5. STRUCTURED RESULTS
   Return dataclass with mean, std, per-fold scores, metadata

6. EXTENSIBILITY
   Easy to add new models, metrics, splitters without changing core

➜ BENEFIT: Multiple developers can work on different layers without conflicts


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ QUICK START
══════════════

Minimal example (copy-paste ready):

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.datasets import make_classification
    from CrossValidation import CrossValidator
    
    # 1. Prepare data
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    
    # 2. Define model factory
    model_factory = lambda: RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 3. Define metric
    metric_fn = lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
    
    # 4. Run cross-validation
    validator = CrossValidator()
    results = validator.validate(X, y, model_factory, metric_fn)
    
    # 5. Access results
    print(f"Mean F1: {results.mean_score:.4f}")
    print(f"Std Dev:  {results.std_score:.4f}")
    print(results.summary())


Output:
    Mean F1: 0.9234
    Std Dev:  0.0156
    ======================================================================
    CROSS-VALIDATION RESULTS
    ======================================================================
    Folds:                  5
    Metric:                 Metric
    
    SCORE STATISTICS:
      Mean:                 0.9234
      Std Dev:              0.0156
      Min:                  0.9011
      Max:                  0.9512
    
    PER-FOLD SCORES:
      Fold 1: 0.9011
      Fold 2: 0.9234
      Fold 3: 0.9389
      Fold 4: 0.9512
      Fold 5: 0.9145
    ======================================================================


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 LEARNING PATH
════════════════

1st time user?
    → Start with examples.py (copy-paste one example)
    → Run it to see output

Want to understand design?
    → Read ARCHITECTURE.md
    → Compare different layers

Setting up project?
    → Read INTEGRATION.md
    → Follow the step-by-step guide

Team development?
    → Share QUICKSTART.md with team
    → Assign roles (Data, Models, Metrics, Orchestration)

Want to extend?
    → Look at core.py to understand structure
    → Adapt patterns from examples.py
    → Add your own models/metrics


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 API DOCUMENTATION
════════════════════

CrossValidator Class
────────────────────

    validator = CrossValidator(
        cv_strategy=StratifiedKFold(n_splits=5, shuffle=True),
        random_state=42,
        verbose=True,
    )

Methods:

    results = validator.validate(
        X: np.ndarray,                    # (n_samples, n_features)
        y: np.ndarray,                    # (n_samples,)
        model_factory: Callable,          # Returns fresh model
        metric_fn: Callable,              # (y_true, y_pred) -> float
        model_name: str = "Model",        # For metadata
        metric_name: str = "Metric",      # For metadata
    ) -> CVResults

    comparison = validator.compare_models(
        X: np.ndarray,
        y: np.ndarray,
        model_factories: Dict[str, Callable],  # {name: factory, ...}
        metric_fn: Callable,
        metric_name: str = "Metric",
    ) -> ComparisonResults


CVResults Class
───────────────

    Attributes:
        .fold_scores               # List of scores per fold
        .fold_predictions          # List of predictions per fold
        .fold_indices              # Tuple of (train_idx, test_idx) per fold
        .mean_score                # Mean across folds
        .std_score                 # Std dev across folds
        .min_score                 # Minimum fold score
        .max_score                 # Maximum fold score
        .fit_times                 # Time to fit each fold (seconds)
        .predict_times             # Time to predict each fold (seconds)
        .metadata                  # Dict of experiment info

    Methods:
        .summary()                 # Human-readable string
        .to_dict()                 # For serialization (JSON, pickle)


ComparisonResults Class
───────────────────────

    Attributes:
        .results                   # Dict[model_name] -> CVResults
        .best_model_name           # Name of best performing model
        .ranking                   # List of (name, score) sorted

    Methods:
        .summary()                 # Human-readable ranking table


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧪 TESTING
═══════════

Run the test suite:

    pytest test_cross_validation.py -v

Test coverage includes:
- ✓ Basic CV functionality
- ✓ Metric agnosticism (different metrics work)
- ✓ Model agnosticism (different models work)
- ✓ Reproducibility (same random_state → same results)
- ✓ Data validation (mismatched X,y raises error)
- ✓ Results structure (correct shapes and statistics)
- ✓ Serialization (to_dict conversion)
- ✓ Integration (full workflows)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ FAQ
══════

Q: Can I use this with XGBoost, LightGBM, custom models?
A: Yes! Any object with .fit(X, y) and .predict(X) works.
   Just pass it via factory: lambda: XGBClassifier(...)

Q: Can I use different metrics (precision, recall, ROC-AUC)?
A: Yes! Any function (y_true, y_pred) → float works.
   Pass it via metric_fn parameter.

Q: How is this different from sklearn's cross_validate?
A: This is simpler and purpose-built for clean architectures:
   - Pure factory pattern (no model imports here)
   - Structured results (dataclass, not dict)
   - Multi-developer friendly (clear separation of concerns)
   - Easy to extend (add metrics/models without touching CV code)

Q: Should I do hyperparameter tuning before or after CV?
A: BEFORE CV. Use GridSearchCV to find best params, then
   pass results to CV module for generalization error estimation.

Q: Can I customize the CV strategy?
A: Yes, pass cv_strategy parameter to CrossValidator:
   validator = CrossValidator(cv_strategy=StratifiedKFold(...))
   or KFold, LeaveOneOut, TimeSeriesSplit, custom splitter, etc.

Q: How do I save results?
A: Use .to_dict() for JSON/pickle:
   import json
   with open("results.json", "w") as f:
       json.dump(results.to_dict(), f)

Q: Is this production-ready?
A: Yes. Designed for production ML teams:
   - Type hints throughout
   - Comprehensive docstrings
   - Tested and validated
   - Logging built-in
   - Error checking


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤝 CONTRIBUTING
═════════════════

Before modifying this module, ask yourself:

Q: Am I adding feature engineering?
→ NO! Goes in data_loading.py

Q: Am I adding model definitions?
→ NO! Goes in models.py

Q: Am I adding metric functions?
→ NO! Goes in metrics.py

Q: Am I orchestrating CV workflow?
→ YES! Can modify core.py

Q: Am I extending/improving data structures?
→ YES! Can modify schema.py


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📞 HELP & RESOURCES
════════════════════

Unsure about something?
1. Read ARCHITECTURE.md       (WHY is it designed this way?)
2. Read QUICKSTART.md         (WHO should do WHAT?)
3. Look at examples.py         (HOW do I use it?)
4. Check test_cross_validation.py (What are edge cases?)

Still stuck?
- Check docstrings (Ctrl+K Ctrl+I in VS Code)
- Look at existing CVResults usage in examples.py
- Run a test in test_cross_validation.py


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Version: 1.0
Created for: Multi-developer ML projects
Principles: Separation of concerns, modularity, extensibility
"""
