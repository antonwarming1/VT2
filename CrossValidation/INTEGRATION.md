"""
INTEGRATION GUIDE
=================

How to integrate this CV module into a complete ML project.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDED PROJECT STRUCTURE
═════════════════════════════

project/
├── data/                           # Data layer (not tracked in git usually)
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── notebooks/                      # Exploratory work
│   ├── 01_eda.ipynb
│   └── 02_feature_engineering.ipynb
│
├── src/                            # Main source code
│   ├── __init__.py
│   │
│   ├── data_loading.py            # Role: Data Engineer
│   │   ├── load_features_and_labels()
│   │   ├── preprocess_features()
│   │   └── split_data()
│   │
│   ├── models.py                  # Role: Model Engineer
│   │   ├── create_random_forest()
│   │   ├── create_svm()
│   │   └── MODEL_FACTORIES = {...}
│   │
│   ├── metrics.py                 # Role: Metrics Engineer
│   │   ├── f1_weighted()
│   │   ├── roc_auc()
│   │   └── METRICS = {...}
│   │
│   ├── hyperparameter_tuning.py   # Role: Model Engineer (advanced)
│   │   └── tune_hyperparameters()
│   │
│   └── CrossValidation/           # THIS MODULE (not modified by roles!)
│       ├── __init__.py
│       ├── core.py               # Pure orchestration
│       ├── schema.py             # Data structures
│       ├── examples.py            # Usage patterns
│       ├── ARCHITECTURE.md
│       ├── QUICKSTART.md
│       └── INTEGRATION.md (this file)
│
├── experiments/                    # Role: ML Scientist (orchestration)
│   ├── exp_01_baseline.py
│   ├── exp_02_tuned_rf.py
│   ├── exp_03_model_comparison.py
│   └── results/
│       └── exp_01_baseline_results.json
│
├── tests/                         # Testing
│   ├── test_data_loading.py
│   ├── test_models.py
│   ├── test_cross_validation.py
│   └── test_metrics.py
│
├── requirements.txt
└── README.md


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP-BY-STEP SETUP
══════════════════

1. Create src/data_loading.py
   ──────────────────────────
   
   import pandas as pd
   from sklearn.preprocessing import StandardScaler
   
   def load_features_and_labels():
       features = pd.read_csv("data/features/features_selected.csv")
       labels = pd.read_csv("data/features/labels.csv").iloc[:, 0].values
       mask = ~(features.isna().any(axis=1) | pd.isna(labels))
       X = features[mask].reset_index(drop=True).values
       y = labels[mask]
       return X, y
   
   def preprocess_features(X):
       scaler = StandardScaler()
       return scaler.fit_transform(X)


2. Create src/models.py
   ────────────────────
   
   from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
   from sklearn.svm import SVC
   from sklearn.linear_model import LogisticRegression
   
   def create_logistic_regression():
       return LogisticRegression(max_iter=1000, random_state=42)
   
   def create_random_forest():
       return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
   
   def create_gradient_boosting():
       return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
   
   MODEL_FACTORIES = {
       "logistic_regression": create_logistic_regression,
       "random_forest": create_random_forest,
       "gradient_boosting": create_gradient_boosting,
   }


3. Create src/metrics.py
   ──────────────────────
   
   from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
   
   def f1_weighted(y_true, y_pred):
       return f1_score(y_true, y_pred, average='weighted', zero_division=0)
   
   def roc_auc(y_true, y_pred):
       return roc_auc_score(y_true, y_pred)
   
   def accuracy(y_true, y_pred):
       return accuracy_score(y_true, y_pred)
   
   METRICS = {
       "f1_weighted": f1_weighted,
       "roc_auc": roc_auc,
       "accuracy": accuracy,
   }


4. Create experiments/exp_01_baseline.py
   ─────────────────────────────────────
   
   from src.data_loading import load_features_and_labels, preprocess_features
   from src.models import MODEL_FACTORIES
   from src.metrics import METRICS
   from src.CrossValidation import CrossValidator
   from sklearn.model_selection import StratifiedKFold
   import json
   
   # Load data
   X, y = load_features_and_labels()
   X = preprocess_features(X)
   
   # Create CV validator
   validator = CrossValidator(
       cv_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
       verbose=True,
   )
   
   # Evaluate
   results = validator.validate(
       X, y,
       model_factory=MODEL_FACTORIES["random_forest"],
       metric_fn=METRICS["f1_weighted"],
       model_name="Random Forest",
       metric_name="F1 (weighted)",
   )
   
   # Save results
   with open("experiments/results/exp_01_baseline_results.json", "w") as f:
       json.dump(results.to_dict(), f, indent=2)
   
   print(results.summary())


5. Create experiments/exp_02_model_comparison.py
   ──────────────────────────────────────────────
   
   from src.data_loading import load_features_and_labels, preprocess_features
   from src.models import MODEL_FACTORIES
   from src.metrics import METRICS
   from src.CrossValidation import CrossValidator
   
   X, y = load_features_and_labels()
   X = preprocess_features(X)
   
   validator = CrossValidator(verbose=True)
   
   comparison = validator.compare_models(
       X, y,
       model_factories=MODEL_FACTORIES,
       metric_fn=METRICS["f1_weighted"],
       metric_name="F1 (weighted)",
   )
   
   print(comparison.summary())


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLES AND RESPONSIBILITIES
══════════════════════════

Data Engineer
─────────────
✓ Maintains src/data_loading.py
✓ Handles data validation and preprocessing
✓ Provides clean X, y to other roles
✗ Never touches CV module
✗ Never defines models or metrics

Model Engineer
──────────────
✓ Maintains src/models.py
✓ Manages hyperparameter tuning (separate script)
✓ Provides model factories
✗ Never touches CV module
✗ Never defines metrics

Metrics Engineer
────────────────
✓ Maintains src/metrics.py
✓ Defines evaluation functions
✗ Never touches CV module
✗ Never defines models

ML Scientist
────────────
✓ Maintains experiments/ directory
✓ Chains components together
✓ Uses CrossValidator for orchestration
✓ Analyzes and interprets results
✗ Never adds logic to CV module


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON TASKS AND PATTERNS
═════════════════════════

Task 1: Add a new model
────────────────────────
1. Data Engineer: No change
2. Model Engineer: Add to src/models.py
   
   def create_xgboost():
       return XGBClassifier(n_estimators=100, random_state=42)
   
   MODEL_FACTORIES["xgboost"] = create_xgboost

3. Metrics Engineer: No change
4. ML Scientist: Can immediately use in experiments


Task 2: Add a new metric
────────────────────────
1. Data Engineer: No change
2. Model Engineer: No change
3. Metrics Engineer: Add to src/metrics.py
   
   def precision_weighted(y_true, y_pred):
       return precision_score(y_true, y_pred, average='weighted')
   
   METRICS["precision_weighted"] = precision_weighted

4. ML Scientist: Can immediately use in experiments


Task 3: Change preprocessing
────────────────────────────
1. Data Engineer: Modify src/data_loading.py
2. Model Engineer: No change
3. Metrics Engineer: No change
4. ML Scientist: Runs experiments again with new data


Task 4: Optimize hyperparameters
─────────────────────────────────
1. Model Engineer: Create separate script (hyperparameter_tuning.py)
2. Use GridSearchCV or RandomizedSearchCV
3. Update best parameters in MODEL_FACTORIES
4. ML Scientist: Re-runs experiments

After tuning:

def create_random_forest_tuned():
    return RandomForestClassifier(
        n_estimators=200,        # Tuned
        max_depth=15,            # Tuned
        min_samples_split=5,     # Tuned
        random_state=42,
        n_jobs=-1,
    )

MODEL_FACTORIES["random_forest_tuned"] = create_random_forest_tuned


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TESTING THE SETUP
═════════════════

Create tests/test_cross_validation.py:

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.CrossValidation import CrossValidator
from src.models import MODEL_FACTORIES
from src.metrics import METRICS

def test_cross_validation_basic():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    validator = CrossValidator(verbose=False)
    results = validator.validate(
        X, y,
        model_factory=MODEL_FACTORIES["random_forest"],
        metric_fn=METRICS["f1_weighted"],
    )
    
    assert results.mean_score > 0.0
    assert len(results.fold_scores) == 5
    assert results.std_score >= 0.0

Run: pytest tests/


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY BENEFITS OF THIS SETUP
══════════════════════════

✅ No merge conflicts in CV module (it's stable, rarely touched)
✅ Easy onboarding (each role has clear domain)
✅ Easy testing (each layer tested independently)
✅ Easy refactoring (changes isolated to one layer)
✅ Easy experimentation (just add experiment file)
✅ Reproducible (all components versioned, seeded)
✅ Scalable (works with 1 person or 10 people)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Questions?
See ARCHITECTURE.md for design rationale
See QUICKSTART.md for quick API reference
See examples.py for working code
"""
