"""
SVM — Multi-Class Classification

Runs four hyperparameter strategies (base CV, Grid, Random, Bayesian),
compares their cross-validation scores, retrains the best configuration
on the full training set, and saves the model.
"""

import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
from scipy.stats import uniform

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                      StratifiedKFold, cross_val_score, train_test_split)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Configuration ────────────────────────────────────────────────────────────


class Config:
    BASE_DIR = Path(__file__).resolve().parents[1]

    FEATURES_PATH = BASE_DIR / "Feature_engineering" / "features_selected.csv"
    LABELS_PATH = BASE_DIR / "Feature_engineering" / "labels.csv"
    MODEL_SAVE_PATH = BASE_DIR / "SVM" / "trained_svm.joblib"

    # SVM training is closed-form per fit, so no validation split is needed.
    TRAIN_SIZE = 0.8
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Hyperparameter defaults (overwritten by the best search result before final training)
    KERNEL = "rbf"
    C = 1.0
    GAMMA = "scale"
    CLASS_WEIGHT = None

    CLASS_LABELS = {0: "N", 1: "NS", 2: "OT", 3: "P", 4: "UT"}


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(features_path, labels_path):
    print("Loading data...")
    class_names = list(Config.CLASS_LABELS.values())
    X = pd.read_csv(features_path, index_col=0).values
    labels_df = pd.read_csv(labels_path, index_col=0)
    y = labels_df.values.flatten()
    print(f"  X: {X.shape},  y: {y.shape}")

    label_col = labels_df.columns[0]
    target_counts = [(t, (labels_df[label_col] == t).sum()) for t in sorted(labels_df[label_col].unique())]

    plt.figure(figsize=(10, 7))
    sns.countplot(x=label_col, data=labels_df)
    plt.title('Multiclass Fault Distribution')
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
    for i, (target, count) in enumerate(target_counts):
        plt.text(i, count + 5, str(count), ha='center', fontsize=12)
    plt.show()

    return X, y


def split_and_normalize(X, y, config):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ── Hyperparameter search ─────────────────────────────────────────────────────

def base_model_cv(X_train, y_train, config):
    print("\nBase model cross-validation...")
    model = SVC(kernel="rbf", C=1.0, gamma="scale",
                decision_function_shape="ovr", random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"  CV f1_macro: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean(), None


def grid_search(X_train, y_train, config):
    print("\n=== Grid Search CV (SVM) ===")
    model = SVC(decision_function_shape="ovr", random_state=config.RANDOM_STATE)

    param_grid = [
        {
            "kernel": ["linear"],
            "C": [0.1, 1, 10, 100],
            "class_weight": [None, "balanced"],
        },
        {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", 0.01, 0.1, 1],
            "class_weight": [None, "balanced"],
        },
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    search = GridSearchCV(model, param_grid, scoring="f1_macro", cv=skf, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    print(f"  Best score:  {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_score_, search.best_params_


def random_search(X_train, y_train, config):
    print("\n=== Randomized Search CV (SVM) ===")
    model = SVC(decision_function_shape="ovr", random_state=config.RANDOM_STATE)

    param_dist = {
        "kernel": ["rbf"],
        "C": uniform(0.01, 100),
        "gamma": uniform(0.0001, 1),
        "class_weight": [None, "balanced"],
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=30,
                                 scoring="f1_macro", cv=skf, n_jobs=-1,
                                 random_state=config.RANDOM_STATE, verbose=1)
    search.fit(X_train, y_train)

    print(f"  Best score:  {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_score_, search.best_params_


def objective(trial, X_train, y_train, config):
    params = {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        "kernel": "rbf",
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "decision_function_shape": "ovr",
        "random_state": config.RANDOM_STATE,
    }
    model = SVC(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    return scores.mean()


def bayesian_search(X_train, y_train, config, n_trials=30):
    print(f"\n=== Bayesian Optimization (SVM, {n_trials} trials) ===")

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_train, config),
                   n_trials=n_trials, show_progress_bar=True)

    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best CV f1_macro: {study.best_value:.4f}")
    print("  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Reshape into the same key set Grid/Random produce so a single param_map applies later.
    best = study.best_params
    best_params = {
        "C": best["C"],
        "gamma": best["gamma"],
        "kernel": "rbf",
        "class_weight": best["class_weight"],
    }
    return study.best_value, best_params


# ── Final model ───────────────────────────────────────────────────────────────

def build_final_model(config):
    kwargs = dict(
        kernel=config.KERNEL,
        C=config.C,
        class_weight=config.CLASS_WEIGHT,
        decision_function_shape="ovr",
        random_state=config.RANDOM_STATE,
    )
    # gamma is only meaningful for non-linear kernels
    if config.KERNEL != "linear":
        kwargs["gamma"] = config.GAMMA
    model = SVC(**kwargs)
    print(f"\nFinal model: {model}")
    return model


def train_model(model, X_train, y_train):
    print("\nTraining final model...")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, config):
    y_pred = model.predict(X_test)
    class_names = list(config.CLASS_LABELS.values())

    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    return y_test, y_pred, confusion_matrix(y_test, y_pred)


# ── Plots ─────────────────────────────────────────────────────────────────────

_SVM_DIR = Path(__file__).resolve().parent


def plot_search_comparison(scores_by_method):
    methods = list(scores_by_method.keys())
    scores = [scores_by_method[m] for m in methods]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=methods, y=scores, hue=methods, palette='Blues_d', legend=False)
    plt.ylim(0, 1.1)
    plt.ylabel('CV f1_macro')
    plt.title('Hyperparameter Search Comparison')
    for i, s in enumerate(scores):
        plt.text(i, s + 0.01, f"{s:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(_SVM_DIR / "cv_comparison.png", dpi=300)
    plt.show()


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(_SVM_DIR / "confusion_matrix.png", dpi=300)
    plt.show()


def solo_model(X_train, X_test, y_train, y_test, config):
    print("\nTraining solo model with default Config params...")
    model = build_final_model(config)
    model = train_model(model, X_train, y_train)
    _, _, cm = evaluate_model(model, X_test, y_test, config)
    plot_confusion_matrix(cm, list(config.CLASS_LABELS.values()))
    return model


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("SVM — Multi-Class Classification\n")
    start_time = time.time()

    X, y = load_data(Config.FEATURES_PATH, Config.LABELS_PATH)
    X_train, X_test, y_train, y_test = split_and_normalize(X, y, Config)

    print("\n--- Hyperparameter Search ---")
    base_score, _ = base_model_cv(X_train, y_train, Config)
    grid_score, grid_params = grid_search(X_train, y_train, Config)
    random_score, random_params = random_search(X_train, y_train, Config)
    bayes_score, bayes_params = bayesian_search(X_train, y_train, Config)

    results = {
        'Base Model': (base_score, None),
        'Grid Search': (grid_score, grid_params),
        'Rand Search': (random_score, random_params),
        'Bayesian': (bayes_score, bayes_params),
    }
    plot_search_comparison({name: score for name, (score, _) in results.items()})

    best_name, (best_score, best_params) = max(results.items(), key=lambda x: x[1][0])
    print(f"\nBest method: {best_name}  (CV f1_macro {best_score:.4f})")
    print(f"Params: {best_params}")

    if best_params:
        param_map = {
            'kernel': 'KERNEL',
            'C': 'C',
            'gamma': 'GAMMA',
            'class_weight': 'CLASS_WEIGHT',
        }
        for search_key, config_attr in param_map.items():
            if search_key in best_params:
                setattr(Config, config_attr, best_params[search_key])

    print(f"\n--- Final Training [{best_name} params] ---")
    model = build_final_model(Config)
    model = train_model(model, X_train, y_train)

    _, _, cm = evaluate_model(model, X_test, y_test, Config)
    plot_confusion_matrix(cm, list(Config.CLASS_LABELS.values()))

    joblib.dump(model, Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")

    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")

    # ── Quick solo run (comment out the block above and uncomment below) ──────
    # solo_model(X_train, X_test, y_train, y_test, Config)


if __name__ == "__main__":
    main()
