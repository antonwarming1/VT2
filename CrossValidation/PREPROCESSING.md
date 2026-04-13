"""
PREPROCESSING FRAMEWORK GUIDE
═════════════════════════════

How to use the pluggable preprocessing layer.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY EXTERNAL PREPROCESSING?
═════════════════════════════

Problem:
  Different algorithms need different preprocessing:
  - StandardScaler for SVM, Neural Networks, Logistic Regression
  - MinMaxScaler for neural networks
  - RobustScaler for data with outliers
  - No preprocessing for tree-based algorithms

Solution:
  Make preprocessing external and pluggable, just like models and metrics


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARCHITECTURE
════════════

    Data Layer              Preprocessing Layer        CV Module
    ──────────             ──────────────────         ─────────
    
    load_data()            get_preprocessing()       CrossValidator()
         ↓                         ↓                         ↓
    X, y                  preprocess_fn(X) → X'     X', y → results
    (raw)                  (transformed)


Key principle: Preprocessing happens BEFORE CV, not during


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AVAILABLE PREPROCESSING STRATEGIES
══════════════════════════════════

1. STANDARD SCALER (default)
   ────────────────
   • Formula: (X - mean) / std
   • Range: -∞ to +∞ (typically -3 to +3)
   • Best for: Most algorithms (SVM, Neural Networks, Logistic Regression)
   • Assumes: Data roughly normally distributed
   
   Use: get_preprocessing("standard")


2. MINMAX SCALER
   ──────────────
   • Formula: (X - min) / (max - min)
   • Range: [0, 1]
   • Best for: Neural Networks, bounded output needed
   • Assumes: Need known range limits
   
   Use: get_preprocessing("minmax")


3. ROBUST SCALER
   ──────────────
   • Formula: (X - median) / IQR (interquartile range)
   • Range: -∞ to +∞ (centered on median)
   • Best for: Data with outliers
   • Assumes: Median and quartiles more important than mean/std
   
   Use: get_preprocessing("robust")


4. NO PREPROCESSING
   ─────────────────
   • Formula: X (identity, no change)
   • Best for: Tree-based algorithms (RandomForest, GradientBoosting)
   • Assumes: Features already on meaningful scale
   
   Use: get_preprocessing("none")


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASIC USAGE
═════════════

Example 1: Use default (StandardScaler)
────────────────────────────────────────

    from CrossValidation import CrossValidator
    from preprocessing import get_preprocessing
    
    # Load data
    X, y = load_data()
    
    # Apply preprocessing (default: standard)
    preprocess_fn = get_preprocessing("standard")
    X = preprocess_fn(X)
    
    # Cross-validate
    validator = CrossValidator()
    results = validator.validate(X, y, model_factory, metric_fn)


Example 2: Try different scalers
─────────────────────────────────

    preprocessing_strategies = {
        "standard": get_preprocessing("standard"),
        "minmax": get_preprocessing("minmax"),
        "robust": get_preprocessing("robust"),
        "none": get_preprocessing("none"),
    }
    
    for name, prep_fn in preprocessing_strategies.items():
        X_processed = prep_fn(X)
        results = validator.validate(X_processed, y, model_factory, metric_fn)
        print(f"{name}: {results.mean_score:.4f}")


Example 3: Custom preprocessing
────────────────────────────────

    def my_custom_preprocessing():
        def preprocess(X):
            # Your custom logic here
            X = X / 100  # Scale down
            X = X - X.mean()  # Center
            return X
        return preprocess
    
    prep_fn = my_custom_preprocessing()
    X = prep_fn(X)


Example 4: Preprocessing pipeline
──────────────────────────────────

    from preprocessing import create_custom_pipeline
    
    # Chain multiple preprocessing steps
    pipeline = create_custom_pipeline([
        get_preprocessing("standard"),
        # Could add feature selection here
        # Could add dimensionality reduction here
    ])
    
    X = pipeline(X)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN TO USE WHICH SCALER
═════════════════════════

Algorithm                      Recommended Scaler
─────────────────────────      ──────────────────────────────
Logistic Regression            Standard or MinMax
SVM (RBF/sigmoid)              Standard (important!)
Neural Networks                Standard or MinMax
DecisionTree/RandomForest      None (not needed)
GradientBoosting               None (not needed)
KNN                            Standard or MinMax (important!)
Naive Bayes                    None or MinMax (depends on model)
Data with outliers             Robust (better than Standard)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CREATING CUSTOM PREPROCESSING
══════════════════════════════

Template:

    def my_preprocessing():
        \"\"\"Factory: My custom preprocessing.\"\"\"
        def preprocess(X: np.ndarray) -> np.ndarray:
            # Your logic here
            X_processed = X.copy()
            # Process...
            return X_processed
        return preprocess
    
    # Use it
    prep_fn = my_preprocessing()
    X = prep_fn(X)


Example: Log transformation

    def log_preprocessing():
        \"\"\"Log transform features (for positive data).\"\"\"
        def preprocess(X):
            return np.log(X + 1)  # +1 to avoid log(0)
        return preprocess


Example: Z-score normalization

    def zscore_preprocessing():
        \"\"\"Custom z-score normalization.\"\"\"
        def preprocess(X):
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            return (X - mean) / (std + 1e-8)  # +1e-8 to avoid division by zero
        return preprocess


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WORKFLOW: COMPARING PREPROCESSING IMPACT
═════════════════════════════════════════

1. Load raw data
    X, y = load_data()

2. Define your model and metric (fixed)
    model_factory = MODEL_FACTORIES["RandomForest"]
    metric_fn = METRICS["F1"]

3. Try different preprocessing
    for prep_name in ["standard", "minmax", "robust", "none"]:
        prep_fn = get_preprocessing(prep_name)
        X_processed = prep_fn(X)
        results = validator.validate(X_processed, y, model_factory, metric_fn)
        print(f"{prep_name}: {results.mean_score:.4f}")

4. See which preprocessing works best
    Standard: 0.92
    MinMax:   0.88
    Robust:   0.91
    None:     0.85
    
    → Use Standard for this model+data combination


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FILE STRUCTURE
═══════════════

preprocessing.py
────────────────
Contains:
  • PreprocessingFn type definition
  • create_standard_scaler()
  • create_minmax_scaler()
  • create_robust_scaler()
  • create_no_preprocessing()
  • create_custom_pipeline()
  • get_preprocessing() function
  • PREPROCESSING_PIPELINES registry


Usage:
  from preprocessing import get_preprocessing
  prep_fn = get_preprocessing("standard")


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTENDING PREPROCESSING
═══════════════════════

To add a new preprocessing strategy:

1. Create factory function in preprocessing.py:

    def create_my_preprocessing():
        \"\"\"Factory: My preprocessing.\"\"\"
        def preprocess(X):
            # Your logic
            return X_processed
        return preprocess

2. Add to PREPROCESSING_PIPELINES:

    PREPROCESSING_PIPELINES = {
        ...existing...
        "my_preprocessing": create_my_preprocessing,
    }

3. Use it:

    prep_fn = get_preprocessing("my_preprocessing")


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT NOTES
═══════════════

✅ DO:
  • Apply preprocessing BEFORE CV
  • Use same preprocessing _strategy_ for all folds (but fitted on training data each time)
  • Try different strategies to find best for your data
  • Document which preprocessing you used in results

❌ DON'T:
  • Fit preprocessing on entire dataset then use in CV (data leakage!)
  • Put preprocessing inside CV loop (handled by sklearn internally)
  • Mix preprocessing strategies between models in comparison
  • Forget to scale features for distance-based algorithms


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE: FULL WORKFLOW
══════════════════════

See examples.py example_5_different_preprocessing_strategies() for complete working code.

    from preprocessing import get_preprocessing
    
    # Load data
    X, y = load_features_and_labels()
    
    # Try different preprocessing
    strategies = {
        "Standard": get_preprocessing("standard"),
        "MinMax": get_preprocessing("minmax"),
        "Robust": get_preprocessing("robust"),
    }
    
    for name, prep_fn in strategies.items():
        X_processed = prep_fn(X)
        results = validate(X_processed, y, model, metric)
        print(f"{name}: {results.mean_score:.4f}")
"""
