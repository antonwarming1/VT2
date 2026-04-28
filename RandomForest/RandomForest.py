"""
Random Forest classification with optional grid search.
Grid search can be enabled or disabled via a Y/N prompt in the terminal.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

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


# --------------------------------------------------
# Model training functions
# --------------------------------------------------

def train_default_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier using default hyperparameters.
    """
    print("\nTraining default Random Forest classifier...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def perform_grid_search(X_train, y_train):
    """
    Perform grid search with cross-validation to find optimal
    Random Forest hyperparameters.
    """
    print("\nRunning grid search for Random Forest hyperparameters...")

    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"]
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\nBest hyperparameters found:")
    print(grid_search.best_params_)

    return grid_search.best_estimator_


# --------------------------------------------------
# Main execution
# --------------------------------------------------

def main():

    start_time = time.time()

    # -------------------------
    # Load data
    # -------------------------
    feature_path = os.path.join(
        os.path.dirname(__file__),
        "..", "Feature_engineering", "features_selected.csv"
    )
    label_path = os.path.join(
        os.path.dirname(__file__),
        "..", "Feature_engineering", "labels.csv"
    )

    X = load_csv(feature_path)
    labels_df = load_csv(label_path)
    y = np.array(labels_df["label"])

    print(f"Total samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Label distribution: {np.bincount(y)}")

    # -------------------------
    # Train / test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------
    # Feature scaling
    # -------------------------
    # (Not strictly required for Random Forest, but kept for consistency)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------
    # Grid-search switch (Y/N)
    # -------------------------
    while True:
        user_choice = input("Perform grid search? (y/n): ").strip().lower()
        if user_choice in ("y", "n"):
            break
        print("Invalid input. Please type 'y' or 'n'.")

    use_grid_search = (user_choice == "y")

    if use_grid_search:
        model = perform_grid_search(X_train_scaled, y_train)
        title = "Random Forest Confusion Matrix (Grid Search Optimized)"
    else:
        model = train_default_random_forest(X_train_scaled, y_train)
        title = "Random Forest Confusion Matrix (Default Hyperparameters)"

    # -------------------------
    # Evaluation
    # -------------------------
    y_pred = model.predict(X_test_scaled)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['N', 'NS', 'OT', 'P', 'UT']
    ))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    display_confusion_matrix(cm, title)

    elapsed_time = time.time() - start_time
    print(f"\nExecution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()