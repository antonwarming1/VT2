# Cross-Validation Module Architecture

## Design Principles

### 1. **Separation of Concerns**
- **Data Layer**: Handles X, y (external, pre-computed)
- **Preprocessing Layer**: Transforms features (external, pluggable)
- **Model Layer**: Abstracts via factory pattern (no models defined here)
- **Metric Layer**: Pluggable metric functions (F1, AUC, etc.)
- **CV Orchestration**: Combines above layers without coupling

### 2. **Model-Agnostic Design**
Instead of instantiating models:
```python
# ❌ Bad - tightly coupled
model = RandomForest(n_estimators=100)

# ✅ Good - decoupled via factory pattern
model_factory = lambda: RandomForest(n_estimators=100)
```

**Benefit**: Same CV module works with any model:
- sklearn models
- XGBoost, LightGBM
- Custom models (any object with `fit()` and `predict()`)

### 3. **Feature-Agnostic Design**
No feature engineering happens in the CV module:
```python
# External: Load/engineer features
X, y = load_features_and_labels()

# CV module: Accept pre-computed data
results = cross_validate(X, y, model_factory, metric_fn)
```

**Benefit**: Features can come from anywhere (CSV, database, API)

### 4. **Metric-Agnostic Design**
Any function `f(y_true, y_pred) -> float` works:
```python
# Pluggable metrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

results_f1 = cross_validate(X, y, model_factory, f1_score)
results_auc = cross_validate(X, y, model_factory, roc_auc_score)
```

**Benefit**: Easy to experiment with different metrics

### 5. **Preprocessing-Agnostic Design**
Preprocessing is external and pluggable:
```python
# ❌ Bad - preprocessing inside CV
def cross_validate(X, y, model_factory):
    X = StandardScaler().fit_transform(X)  # Tightly coupled
    ...

# ✅ Good - preprocessing before CV
preprocess_fn = get_preprocessing("standard")
X = preprocess_fn(X)
results = cross_validate(X, y, model_factory, metric_fn)
```

**Benefit**: 
- Different algorithms need different preprocessing
- Easy to experiment with preprocessing strategies
- Preprocessing is reproducible and testable independently

### 6. **Structured Results**
Return dataclass with:
- Per-fold scores and predictions
- Summary statistics (mean, std, min, max)
- Timing information
- Metadata (n_folds, model_name, metric_name)

```python
@dataclass
class CVResults:
    fold_scores: List[float]           # Score per fold
    fold_predictions: List[np.ndarray] # Predictions per fold
    fold_indices: List[Tuple]          # Train/test indices per fold
    
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    
    fit_times: List[float]
    predict_times: List[float]
    
    metadata: Dict[str, Any]
```

---

## Module Structure

```
CrossValidation/
├── __init__.py                 # Public API
├── core.py                     # Core CV logic
├── schema.py                   # Data classes (CVResults, etc.)
├── splitters.py                # Pre-configured CV strategies
├── examples.py                 # Usage examples
└── ARCHITECTURE.md             # This file
```

---

## Key Interfaces

### 1. **Model Factory**
```python
ModelFactory = Callable[[], Any]

# Must have .fit(X, y) and .predict(X)
factory = lambda: RandomForestClassifier(n_estimators=100)
```

### 2. **Metric Function**
```python
MetricFn = Callable[[np.ndarray, np.ndarray], float]

# Examples
f1_metric = lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
```

### 3. **CV Strategy**
```python
# sklearn splitters or custom
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## Workflow Example

```
┌─────────────────────────────────┐
│ External Layer                  │
├─────────────────────────────────┤
│ • Load features: X (n, d)       │
│ • Load labels: y (n,)           │
│ • Define model factory          │
│ • Choose metric function        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ CV Module: cross_validate()     │
├─────────────────────────────────┤
│ 1. Split X, y into k folds      │
│ 2. For each fold:               │
│    • model = factory()          │
│    • model.fit(X_train, y_train)│
│    • y_pred = model.predict()   │
│    • score = metric(y_test, ...)│
│ 3. Aggregate results            │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Results: CVResults dataclass    │
├─────────────────────────────────┤
│ • fold_scores: [0.92, 0.89...]  │
│ • mean_score: 0.91              │
│ • std_score: 0.02               │
│ • fold_predictions: [...]       │
│ • metadata: {...}               │
└─────────────────────────────────┘
```

---

## Integration Guidelines

### For Single-Developer
Use `examples.py` as a template for your experiments.

### For Multi-Developer Teams
1. **Data Engineer** →  Prepares X, y (imports from data module)
2. **ML Engineer** → Calls `cross_validate(X, y, model_factory, metric_fn)`
3. **Results Consumer** → Gets `CVResults` with structured data
4. No coupling between roles

### Extending the Module
- Add new splitter? → Add to `splitters.py`
- Add new result type? → Update `schema.py`
- Add new logging? → Import and use in `core.py`
- **Never** add feature engineering or model definitions!

---

## Best Practices Implemented

✅ Type hints (Python 3.7+)
✅ Dataclasses for structured results
✅ Factory pattern for model instantiation
✅ Pluggable metric functions
✅ Clear separation: orchestration ≠ implementation
✅ Comprehensive docstrings
✅ Logging (optional, non-intrusive)
✅ Testing-friendly (deterministic, seeded)
