"""
Random Forest classification with selectable hyperparameter optimization:
- Grid Search
- Randomized Search
- Bayesian Optimization (Optuna)
- Or default model (no search)

Selection is made via terminal input.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from scipy.stats import randint
import optuna


# ==================================================
# Utility functions
# ==================================================

def load_csv(filepath):
    print(f"Loading CSV file: {filepath}")
    return pd.read_csv(filepath)


def display_confusion_matrix(cm, title="Confusion Matrix"):
    labels = ['N', 'NS', 'OT', 'P', 'UT']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ==================================================
# Model training functions
# ==================================================

def train_default_random_forest(X_train, y_train):
    print("\nTraining default Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def grid_search_rf(X_train, y_train):
    print("\n=== Grid Search CV (Random Forest) ===")

    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"]
    }

    search = GridSearchCV(
        model,
        param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    return search.best_estimator_


def random_search_rf(X_train, y_train):
    print("\n=== Randomized Search CV (Random Forest) ===")

    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_dist = {
        "n_estimators": randint(100, 800),
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"]
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    return search.best_estimator_


# ==================================================
# Bayesian optimization (Optuna)
# ==================================================

def bayesian_objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

    return scores.mean()


def bayesian_search_rf(X_train, y_train, n_trials=30):
    print(f"\n=== Bayesian Optimization (Random Forest, {n_trials} trials) ===")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda trial: bayesian_objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return RandomForestClassifier(
        **study.best_params,
        random_state=42,
        n_jobs=-1
    )


# ==================================================
# Main
# ==================================================

def main():

    start_time = time.time()

    feature_path = os.path.join(os.path.dirname(__file__), "..", "Feature_engineering", "features_selected.csv")
    label_path = os.path.join(os.path.dirname(__file__), "..", "Feature_engineering", "labels.csv")

    X = load_csv(feature_path)
    y = np.array(load_csv(label_path)["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scaling kept for consistency (not required for RF)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nChoose hyperparameter search method:")
    print("  0 - No search (default model)")
    print("  1 - Grid Search")
    print("  2 - Randomized Search")
    print("  3 - Bayesian Optimization")

    while True:
        choice = input("Enter choice (0/1/2/3): ").strip()
        if choice in {"0", "1", "2", "3"}:
            break
        print("Invalid choice.")

    if choice == "0":
        model = train_default_random_forest(X_train, y_train)
        title = "Random Forest (Default)"
    elif choice == "1":
        model = grid_search_rf(X_train, y_train)
        title = "Random Forest (Grid Search)"
    elif choice == "2":
        model = random_search_rf(X_train, y_train)
        title = "Random Forest (Random Search)"
    else:
        model = bayesian_search_rf(X_train, y_train)
        title = "Random Forest (Bayesian)"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['N', 'NS', 'OT', 'P', 'UT']))

    cm = confusion_matrix(y_test, y_pred)
    display_confusion_matrix(cm, title)

    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()