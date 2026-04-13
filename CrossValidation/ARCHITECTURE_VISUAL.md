"""
VISUAL ARCHITECTURE GUIDE
═════════════════════════

This document shows the data flow and design patterns visually.


╔════════════════════════════════════════════════════════════════════════════╗
║                     LAYER ARCHITECTURE (SEPARATION OF CONCERNS)            ║
╚════════════════════════════════════════════════════════════════════════════╝

        ┌─────────────────────────────────────────────────────────────────┐
        │                    ORCHESTRATION LAYER                          │
        │           (CrossValidation/core.py)                            │
        │                                                                 │
        │  CrossValidator.validate(X, y, model_factory, metric_fn)       │
        │                                                                 │
        │  ↑ INPUT                                      ↓ OUTPUT          │
        │  factory() → model             CVResults(mean, std, per-fold)   │
        │             ↓ fit & predict    ↓ metadata    ↓ timing           │
        │  metric_fn(y_true, y_pred)     ↓ predictions ↓ indices          │
        └──────────────────────────────────────────────────────────────────┘
                ↑                    ↑                        ↑
                │                    │                        │
        ┌───────┴──────┐    ┌────────┴──────┐    ┌────────────┴─────────┐
        │               │    │               │    │                      │
    ┌───┴────────────┐ │ ┌──┴─────────────┐ │ ┌──┴──────────────────┐  │
    │  MODEL LAYER   │ │ │  METRIC LAYER  │ │ │   DATA LAYER        │  │
    │   (models.py)  │ │ │ (metrics.py)   │ │ │ (data_loading.py)   │  │
    │                │ │ │                │ │ │                     │  │
    │ RandomForest() │ │ │  f1_weighted() │ │ │  load_data()        │  │
    │      ↓         │ │ │      ↓         │ │ │      ↓              │  │
    │ returns model  │ │ │  returns       │ │ │  returns            │  │
    │ with fit(),    │ │ │  float score   │ │ │  X, y arrays        │  │
    │ predict()      │ │ │                │ │ │                     │  │
    └───────────────── │ └─────────────────┘ │ └──────────────────────  │
                    │        │        │
                    └────────┼────────┘
                             │
                    THIS MODULE DOESN'T
                    DEFINE THESE THINGS
                             


╔════════════════════════════════════════════════════════════════════════════╗
║                        DATA FLOW THROUGH CV                                ║
╚════════════════════════════════════════════════════════════════════════════╝

    Load Data              Define Components          Run CV
    ─────────────         ──────────────────         ──────
    
    X, y = load()         model_factory()            validation_loop()
      ↓                         ↓                         ↓
    Array[n, d]           fn: ∅ → Model[0]          for fold in splits:
                          (fresh each time)              ↓
                                                      model = factory()
                                                         ↓
                          metric_fn()                model.fit(X_train, y_train)
                               ↓                        ↓
                          fn: (ŷ, y) → f1            ŷ = model.predict(X_test)
                                                         ↓
                                                      score = metric(y, ŷ)
                                                         ↓
                                                      collect results
                                                         ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  CVResults                                                      │
    │  ├─ fold_scores: [0.92, 0.89, 0.91, 0.93, 0.90]               │
    │  ├─ mean_score: 0.91                                           │
    │  ├─ std_score: 0.015                                           │
    │  ├─ fold_predictions: [array(...), ...]                        │
    │  ├─ fit_times: [0.15, 0.12, 0.14, 0.13, 0.14]                 │
    │  └─ metadata: {model: "RF", metric: "F1", ...}                │
    └─────────────────────────────────────────────────────────────────┘


╔════════════════════════════════════════════════════════════════════════════╗
║                   FACTORY PATTERN (MODEL-AGNOSTIC)                        ║
╚════════════════════════════════════════════════════════════════════════════╝

    ❌ WRONG (Tightly Coupled)
    ────────────────────────
    
    from sklearn.ensemble import RandomForestClassifier
    
    def validate(X, y):
        model = RandomForestClassifier(n_estimators=100)  ← Import specific model
        ...


    ✅ CORRECT (Loosely Coupled)
    ──────────────────────────
    
    def validate(X, y, model_factory):
        model = model_factory()                            ← Call factory
        ...
    
    # Caller decides what model
    validator.validate(X, y, lambda: RandomForestClassifier(...))
    validator.validate(X, y, lambda: SVC(...))
    validator.validate(X, y, lambda: GradientBoostingClassifier(...))


╔════════════════════════════════════════════════════════════════════════════╗
║                   METRICS AGNOSTICISM (PLUGGABLE)                         ║
╚════════════════════════════════════════════════════════════════════════════╝

    ❌ WRONG (Hardcoded Metric)
    ───────────────────────────
    
    def validate(X, y):
        score = f1_score(y_test, y_pred)  ← Only F1 works
        ...


    ✅ CORRECT (Pluggable Metrics)
    ──────────────────────────────
    
    def validate(X, y, metric_fn):
        score = metric_fn(y_test, y_pred)  ← Any metric works
        ...
    
    # Caller decides what metric
    validator.validate(X, y, f1_weighted)
    validator.validate(X, y, roc_auc_score)
    validator.validate(X, y, accuracy_score)


╔════════════════════════════════════════════════════════════════════════════╗
║                    STRUCTURED RESULTS (vs Dict)                           ║
╚════════════════════════════════════════════════════════════════════════════╝

    ❌ WRONG (Unstructured Dict)
    ────────────────────────
    
    results = {
        "scores": [0.92, 0.89, ...],
        "mean": 0.91,
        "std": 0.015,
        ...
    }
    
    # Unclear what's required, easy to misspell keys
    mean = results["mean"]
    mean = results["Mean"]  ← Oops! KeyError


    ✅ CORRECT (Structured Dataclass)
    ────────────────────────────────
    
    @dataclass
    class CVResults:
        fold_scores: List[float]
        mean_score: float
        std_score: float
        ...
    
    results = CVResults(...)
    
    # Clear structure, type hints, IDE autocomplete
    mean = results.mean_score   ← IDE shows all options
    # Or: results.m<TAB> → mean_score


╔════════════════════════════════════════════════════════════════════════════╗
║                   MULTI-DEVELOPER WORKFLOW                                ║
╚════════════════════════════════════════════════════════════════════════════╝

    Person A (Data Engineer)      Person B (Model Engineer)
    ────────────────────────      ────────────────────────
    
    load_data()                   create_random_forest()
         ↓                              ↓
    features.csv → X              returns RandomForest
    labels.csv   → y              with fit(), predict()
         ↓                              ↓
    preprocess()                  (no merge conflict!)
         ↓
    X_scaled, y
         ↘                          Person C (Metrics Engineer)
           ↘                        ─────────────────────────
             ╲                      
              ╲                     f1_weighted()
               ╲                         ↓
                ╲                    returns float
                 ╲                   (no merge conflict!)
                  ╲                       ↓
                   ↘                 Person D (Scientist)
                     ↘                ─────────────────
                       ╲              Orchestrates all layers:
                        ╲             
    Experiment Runner:    ╲           validator.validate(
    ─────────────────     ╲            X, y,
                           ╲           factory_from_B,
    validator = CV()        ╲          metric_from_C
    results = validator.    ╲          )
        validate(            ↘              ↓
            X_from_A              ║   CVResults
            y_from_A      ────────╫──────────╫────────
            factory_B,  ║ Pure Orchestration Module ║
            metric_C    ║ (Nobody modifies this!)    ║
        )               ║ (No conflicts!)            ║
                        ╚════════════════════════════╝
    
    ✅ BENEFITS:
    • No merge conflicts in CV module
    • Each person works independently
    • Clear responsibilities
    • Easy code review


╔════════════════════════════════════════════════════════════════════════════╗
║                    SEPARATION OF CONCERNS                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

    Module            Responsibility              Do Not Include
    ──────────────    ─────────────────────       ──────────────────────
    
    data_loading      Load & preprocess data      Models, metrics, CV logic
    models            Define model factories      Feature engineering, CV
    metrics           Define metric functions     Models, data loading
    CV module         Orchestrate CV workflow     Models, metrics, data prep
    experiments       Chain everything together   None! (just orchestrate)


    KEY INSIGHT:
    Each layer knows WHAT to do with inputs, not WHERE they come from
    
    Example:
    • CV doesn't care WHERE the model comes from (factory pattern)
    • CV doesn't care WHERE the metric comes from (pluggable fn)
    • CV doesn't care WHERE X, y come from (accepts arrays)


╔════════════════════════════════════════════════════════════════════════════╗
║                    RESULT STRUCTURE                                       ║
╚════════════════════════════════════════════════════════════════════════════╝

    CVResults
    ─────────
    
    [ Fold 1 ]
    X_train  X_test
       ↓        ↓
    [train:test indices stored] → fold_indices[0]
       
    model.fit(X_train) 
    ŷ = model.predict(X_test)
       
    score = metric(y_test, ŷ) = 0.92 → fold_scores[0]
    predictions = ŷ            → fold_predictions[0]
    fit_time = 0.15s            → fit_times[0]
    
    [ Fold 2 ] ... [Fold 5 ]
    
    ──────────────────────────
    
    After all folds:
    
    mean_score = mean([0.92, 0.89, 0.91, 0.93, 0.90]) = 0.91
    std_score  = std([...]) = 0.015
    min_score  = 0.89
    max_score  = 0.93
    
    metadata = {
        "model_name": "Random Forest",
        "metric_name": "F1 (weighted)",
        "cv_strategy": "StratifiedKFold(5)",
        ...
    }


╔════════════════════════════════════════════════════════════════════════════╗
║                    COMPARISON (Multiple Models)                           ║
╚════════════════════════════════════════════════════════════════════════════╝

    Model A              Model B              Model C
    (factory_a)          (factory_b)          (factory_c)
    ↓                    ↓                    ↓
    
    validate(X, y, factory_a, ...)    
    ↓
    CVResults_A {mean: 0.91, std: 0.02}
    
    validate(X, y, factory_b, ...)
    ↓
    CVResults_B {mean: 0.88, std: 0.03}
    
    validate(X, y, factory_c, ...)
    ↓
    CVResults_C {mean: 0.94, std: 0.01}
    
    ──────────────────────────────────
    
    ComparisonResults {
        results: {
            "Model A": CVResults_A,
            "Model B": CVResults_B,
            "Model C": CVResults_C,
        },
        best_model_name: "Model C",
        ranking: [
            ("Model C", 0.94),
            ("Model A", 0.91),
            ("Model B", 0.88),
        ]
    }


╔════════════════════════════════════════════════════════════════════════════╗
║                      KEY DESIGN DECISIONS                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

    DECISION #1: Factory Pattern (not direct models)
    ────────────────────────────────────────────────
    ✅ MODEL_AGNOSTIC
    ✅ Fresh instance per fold (no state leakage)
    ✅ Easy to pass hyperparameters
    ✅ Works with any library


    DECISION #2: Pluggable Metric Functions
    ────────────────────────────────────────
    ✅ METRIC_AGNOSTIC
    ✅ Simple interface: (y_true, y_pred) → float
    ✅ Easy to experiment with different metrics
    ✅ No dependencies on specific libraries


    DECISION #3: Accept Pre-computed X, y
    ──────────────────────────────────────
    ✅ FEATURE_AGNOSTIC
    ✅ Data engineering is external
    ✅ CV module stays focused
    ✅ Easier to test


    DECISION #4: Structured Results (CVResults dataclass)
    ──────────────────────────────────────────────────────
    ✅ TYPE-SAFE (type hints, IDE autocomplete)
    ✅ SERIALIZABLE (.to_dict() for JSON, pickle)
    ✅ EXTENSIBLE (easy to add new fields)
    ✅ READABLE (clear structure, docstrings)


    DECISION #5: Per-fold Data in Results
    ──────────────────────────────────────
    ✅ REPRODUCIBLE (can recompute if needed)
    ✅ DEBUGGABLE (see which fold was problematic)
    ✅ FLEXIBLE (user can calculate their own stats)
    ✅ INFORMATIVE (fold indices matter for debugging)


╔════════════════════════════════════════════════════════════════════════════╗
║                   WHEN TO EXTEND EACH FILE                               ║
╚════════════════════════════════════════════════════════════════════════════╝

    core.py
    ───────
    Add when:
      • Need new CV strategy type
      • Want to parallelize training
      • Need new aggregation statistic
      • Improving logging/monitoring
    
    Don't add:
      • Model definitions
      • Metric implementations


    schema.py
    ─────────
    Add when:
      • Need to track new result type
      • Want different serialization format
      • Need new summary/report format
    
    Don't add:
      • CV logic
      • Model training
      • Data loading


    examples.py
    ───────────
    Add when:
      • Need new usage pattern
      • Demonstrating different workflow
      • Showing integration with other tools
    
    Don't add:
      • New CV functionality (→ core.py)
      • New data structures (→ schema.py)


    At project level (not in CV module):
    ─────────────────────────────────────
    Add when:
      • Working with specific models
      • Creating domain metrics
      • Handling your data format
      • Creating experiment orchestration
    
    Never add to CV module:
      • Model code
      • Feature engineering
      • Domain-specific logic


═════════════════════════════════════════════════════════════════════════════

SUMMARY: This architecture enables multiple developers to work on an ML 
         project without conflicts, with clear roles and responsibilities,
         while keeping the CV module stable, tested, and general-purpose.
"""
