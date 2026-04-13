"""
MODULE STRUCTURE & FILE GUIDE
═════════════════════════════

CrossValidation/
├── __init__.py                    [Public API • Module initialization]
│   └─ Exports: CrossValidator, CVResults, ComparisonResults, ModelFactory, MetricFn
│
├── core.py                        [CORE LOGIC • Orchestration only]
│   ├─ CrossValidator class
│   │  ├─ __init__(cv_strategy, random_state, verbose)
│   │  ├─ validate() method
│   │  └─ compare_models() method
│   ├─ ModelFactory type hint
│   └─ MetricFn type hint
│
├── preprocessing.py               [PREPROCESSING LAYER • Pluggable strategies]
│   ├─ create_standard_scaler()
│   ├─ create_minmax_scaler()
│   ├─ create_robust_scaler()
│   ├─ create_no_preprocessing()
│   ├─ create_custom_pipeline()
│   ├─ get_preprocessing() function
│   └─ PREPROCESSING_PIPELINES registry
│
├── schema.py                      [DATA STRUCTURES • Results containers]
│   ├─ CVResults dataclass
│   │  ├─ fold_scores, fold_predictions, fold_indices
│   │  ├─ mean_score, std_score, min_score, max_score
│   │  ├─ fit_times, predict_times
│   │  ├─ metadata
│   │  ├─ .summary() method
│   │  └─ .to_dict() method (serialization)
│   └─ ComparisonResults dataclass
│      ├─ results, best_model_name, ranking
│      └─ .summary() method
│
├── examples.py                    [USAGE EXAMPLES • 5 real workflows]
│   ├─ load_features_and_labels()
│   ├─ preprocess_features() (uses preprocessing.py)
│   ├─ MODEL_FACTORIES dict
│   ├─ METRICS dict
│   ├─ example_1_single_model_single_metric()
│   ├─ example_2_same_model_different_metrics()
│   ├─ example_3_different_models_same_metric()
│   ├─ example_4_hyperparameter_tuning_integration()
│   └─ example_5_different_preprocessing_strategies()
│
├── test_cross_validation.py       [UNIT TESTS • Comprehensive testing]
│   ├─ Fixtures: synthetic_data, model_factory, metric_function, validator
│   ├─ TestCrossValidatorBasic
│   ├─ TestMetricAgnosticism
│   ├─ TestModelAgnosticism
│   ├─ TestReproducibility
│   ├─ TestCVResults
│   ├─ TestIntegration
│   └─ TestEdgeCases
│
├── README.md                      [Getting started guide]
│   ├─ What's in this folder
│   ├─ Design principles
│   ├─ Quick start
│   ├─ Learning path
│   ├─ API documentation
│   ├─ Testing guide
│   ├─ FAQ
│   ├─ Contributing guidelines
│   └─ Help & resources
│
├── ARCHITECTURE.md                [Design rationale]
│   ├─ Design principles explained
│   ├─ Separation of concerns
│   ├─ Model-agnostic pattern
│   ├─ Feature-agnostic pattern
│   ├─ Metric-agnostic pattern
│   ├─ Preprocessing-agnostic pattern
│   ├─ Structured results pattern
│   └─ Workflow diagram
│
├── ARCHITECTURE_VISUAL.md         [Visual diagrams]
│   ├─ Layer architecture visualization
│   ├─ Data flow diagrams
│   ├─ Factory pattern illustrations
│   ├─ Results structure
│   └─ Design decisions
│
├── PREPROCESSING.md               [Preprocessing guide]
│   ├─ Why external preprocessing
│   ├─ Available strategies
│   ├─ When to use which scaler
│   ├─ Custom preprocessing examples
│   ├─ Preprocessing workflow
│   └─ Extending preprocessing
│
├── QUICKSTART.md                  [Role-based reference]
│   ├─ Role 1: Data Engineer
│   ├─ Role 2: Preprocessing Engineer
│   ├─ Role 3: Model Engineer
│   ├─ Role 4: Metrics Engineer
│   ├─ Role 5: ML Scientist (Orchestrator)
│   ├─ Typical workflow
│   ├─ API cheat sheet
│   └─ What NOT to do
│
├── INTEGRATION.md                 [Project setup guide]
│   ├─ Recommended project structure
│   ├─ Step-by-step setup
│   ├─ Roles and responsibilities
│   ├─ Common tasks and patterns
│   ├─ Testing examples
│   ├─ Key benefits
│   └─ Integration examples
│
└── MODULE_STRUCTURE.md            [This file]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY FILES TO START WITH
═══════════════════════

If you...                          Read this first
─────────────────────────────────  ─────────────────────────────
Are new to the module             README.md → examples.py
Want to understand design         ARCHITECTURE.md
Are setting up a project          INTEGRATION.md
Work on a team                    QUICKSTART.md
Need to extend/modify             core.py → schema.py
Want to ensure quality            test_cross_validation.py


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FILE RESPONSIBILITIES
═════════════════════

✅ Modify these files when...

core.py
  - Adding new CV strategies
  - Improving orchestration logic
  - Adding performance optimizations
  - Fixing bugs in validation workflow

preprocessing.py
  - Adding new preprocessing strategies
  - Extending scaler options
  - Creating preprocessing pipelines

schema.py
  - Adding new result attributes
  - Improving result summary/serialization
  - Changing result data structure

examples.py
  - Adding new example workflows
  - Updating existing examples
  - Documenting usage patterns


❌ DON'T modify CV module for...

Adding models
  → Create models.py outside CV module
  → Pass via factory pattern

Adding preprocessing strategies
  → Add to preprocessing.py (part of this module)
  → Or define custom preprocessing externally
  → Call preprocess_fn before CV

Adding feature engineering
  → Create data_loading.py outside CV module
  → Preprocess data before passing to CV

Adding metrics
  → Create metrics.py outside CV module
  → Pass via metric_fn parameter

Data loading
  → Create data_loading.py outside CV module
  → Load and prepare X, y before CV


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPENDENCY GRAPH
════════════════

Examples.py imports:
  → core.py (CrossValidator)
  → (Not schema.py directly, but core uses it internally)

core.py imports:
  → schema.py (CVResults, ComparisonResults)

schema.py imports:
  → Nothing from this module (only stdlib + numpy)

test_cross_validation.py imports:
  → core.py (CrossValidator)
  → schema.py (CVResults, ComparisonResults)

__init__.py imports:
  → core.py
  → schema.py


Key insight: Minimal dependencies, clear data flow
  inputs (X, y) → CV orchestration → outputs (CVResults)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON TASKS
═════════════

Task: Run a single experiment with Random Forest
─────────────────────────────────────────────────
1. Prepare X, y
2. Create model factory (lambda)
3. Create metric function (lambda)
4. Create validator
5. Call validator.validate()
→ See: examples.py example_1_single_model_single_metric()

Task: Compare multiple models
──────────────────────────────
1. Prepare X, y
2. Create model factories dict
3. Create metric function
4. Create validator
5. Call validator.compare_models()
→ See: examples.py example_3_different_models_same_metric()

Task: Try different metrics
───────────────────────────
1. Prepare X, y
2. Create model factory
3. Create multiple metric functions
4. Create validator
5. Call validator.validate() for each metric
→ See: examples.py example_2_same_model_different_metrics()

Task: Tune hyperparameters then CV
──────────────────────────────────
1. Use GridSearchCV/RandomizedSearchCV externally
2. Get best parameters
3. Create model factory with best params
4. Call validator.validate()
→ See: examples.py example_4_hyperparameter_tuning_integration()


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TESTING STRATEGY
═════════════════

Unit tests (test_cross_validation.py):
  ✓ Test basic CV functionality
  ✓ Test model agnosticism
  ✓ Test metric agnosticism
  ✓ Test data validation
  ✓ Test result structure
  ✓ Test reproducibility
  ✓ Test edge cases

Your tests should:
  ✓ Test models.py (do factories work?)
  ✓ Test metrics.py (do metrics compute correctly?)
  ✓ Test data_loading.py (is X, y valid?)
  ✓ DON'T test CV module (we already tested it!)

Run tests:
  pytest test_cross_validation.py -v


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LINE COUNT SUMMARY
══════════════════

File                          Lines        Purpose
──────────────────────────    ───────      ─────────────────────
__init__.py                    ~50         Module initialization
core.py                        ~280        Pure orchestration
schema.py                      ~180        Data structures
examples.py                    ~400        Usage patterns
test_cross_validation.py       ~600        Comprehensive testing
README.md                      ~400        Getting started
ARCHITECTURE.md                ~200        Design explanation
QUICKSTART.md                  ~300        Role reference
INTEGRATION.md                 ~400        Project setup
MODULE_STRUCTURE.md            ~200        This file

Total: ~3000 lines of well-documented, tested code


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUALITY ATTRIBUTES
═══════════════════

Testability
  ✓ Unit tests for all major functionality
  ✓ Fixtures for common test components
  ✓ Edge case coverage
  ✓ Reproducible tests (deterministic with random_state)

Maintainability
  ✓ Clear separation of concerns
  ✓ Type hints throughout
  ✓ Comprehensive docstrings
  ✓ Examples for common patterns

Extensibility
  ✓ Factory pattern for models
  ✓ Pluggable metrics
  ✓ Configurable CV strategies
  ✓ Easy to add new result attributes

Scalability
  ✓ Works with 1 developer or 10
  ✓ Clear role separation
  ✓ Minimal conflicts in version control
  ✓ Can be used in production

Security
  ✓ No eval() or exec() calls
  ✓ Type checking with type hints
  ✓ Input validation in core.py
  ✓ Safe JSON serialization via to_dict()


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEXT STEPS
═══════════

1. Read README.md (high-level overview)
2. Read ARCHITECTURE.md (understand design)
3. Run examples.py (hands-on learning)
4. Read INTEGRATION.md (project setup)
5. Look at core.py (implementation details)
6. Run tests (validate everything)
7. Create your own experiment scripts


Questions? See README.md FAQ section.
"""
