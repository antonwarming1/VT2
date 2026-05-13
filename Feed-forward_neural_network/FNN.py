"""
Feed-Forward Neural Network — Multi-Class Classification
"""

from functools import partial
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                      StratifiedKFold, cross_val_score, train_test_split)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from scipy import signal
from scipy.stats import uniform, loguniform
from tensorflow.keras.regularizers import l2 as l2_reg
from scikeras.wrappers import KerasClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Configuration ────────────────────────────────────────────────────────────


class Config:
    BASE_DIR = Path(__file__).resolve().parents[1]

    FEATURES_PATH = BASE_DIR / "Feature_engineering" / "features_selected_audio.csv"
    LABELS_PATH = BASE_DIR / "Feature_engineering" / "labels.csv"
    MODEL_SAVE_PATH = BASE_DIR / "Feed-forward_neural_network" / "trained_model.keras"


    # Data split: 70% train / 20% val / 10% test
    TRAIN_SIZE = 0.7
    VALIDATION_SIZE = 0.2
    TEST_SIZE = 0.1
    RANDOM_STATE = 42

    # Architecture defaults (updated by best search result before final training)
    HIDDEN_LAYERS = [128, 64, 32]
    ACTIVATION_FUNCTION = 'relu'
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001
    L2_REGULARIZATION = 0.01

    BATCH_SIZE = 32
    EPOCHS = 100
    SEARCH_EPOCHS = 20   # fewer epochs during hyperparameter search
    EARLY_STOPPING_PATIENCE = 10

    CLASS_LABELS = {0: "N", 1: "NS", 2: "OT", 3: "P", 4: "UT"}


# ── Model factory ────────────────────────────────────────────────────────────
# Defined at module level so functools.partial produces a picklable object,
# which sklearn needs to clone estimators during cross-validation.

def build_keras_model(input_dim, num_classes=5,
                       hidden_layers=None, activation='relu', dropout_rate=0.2, l2=0.0):
    if hidden_layers is None:
        hidden_layers = [128, 64, 32]
    model = Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for neurons in hidden_layers:
        model.add(layers.Dense(neurons, activation=activation, kernel_regularizer=l2_reg(l2)))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model  # uncompiled — scikeras handles compilation


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(features_path, labels_path):
    print("Loading data...")
    class_names = list(Config.CLASS_LABELS.values())
    X = pd.read_csv(features_path, index_col=0).values
    labels_df = pd.read_csv(labels_path, index_col=0)
    y = labels_df.values.flatten()
    print(f"  X: {X.shape},  y: {y.shape}")

    # Visualize the distribution of fault types
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
    # Hold out test set
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    # Split remaining into train / val
    val_ratio = config.VALIDATION_SIZE / (config.TRAIN_SIZE + config.VALIDATION_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest,
        test_size=val_ratio,
        random_state=config.RANDOM_STATE,
        stratify=y_rest
    )
    # Normalize using only training statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Hyperparameter search ─────────────────────────────────────────────────────

PARAM_SPACE = {
    'model__hidden_layers': [[64], [128, 64]],
    'model__activation': ['relu', 'tanh', 'sigmoid'],
    'model__dropout_rate': [0.1, 0.2, 0.3],
    'batch_size': [16, 32],
    'optimizer__learning_rate': [0.001, 0.0001],
    'model__l2': [0.0, 0.01],
}

def make_clf(input_dim, num_classes, search_epochs, **kwargs):
    """Build a KerasClassifier ready for sklearn CV."""
    model_fn = partial(build_keras_model, input_dim=input_dim, num_classes=num_classes)
    return KerasClassifier(
        model=model_fn,
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        epochs=search_epochs,
        model__l2=Config.L2_REGULARIZATION,
        verbose=0,
        **kwargs
    )


def base_model_cv(X_train, y_labels, config):
    print("\nBase model cross-validation...")
    clf = make_clf(X_train.shape[1], len(config.CLASS_LABELS), config.SEARCH_EPOCHS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(clf, X_train, y_labels, cv=cv, scoring='accuracy')
    print(f"  CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean(), None


def grid_search(X_train, y_labels, config):
    print("\nGrid Search CV...")
    clf = make_clf(X_train.shape[1], len(config.CLASS_LABELS), config.SEARCH_EPOCHS)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    search = GridSearchCV(clf, PARAM_SPACE, cv=skf, n_jobs=1, verbose=1)
    search.fit(X_train, y_labels)

    print(f"  Best score:  {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_score_, search.best_params_


def random_search(X_train, y_labels, config, scoring='accuracy'):
    # Perform Randomized Search CV for hyperparameter tuning of the FNN
    print("\nRandom Search CV...")

    # Define parameter distributions for the FNN
    param_distributions = {
        'model__hidden_layers':     [[64], [128], [128, 64], [256, 128], [128, 64, 32]],
        'model__activation':        ['relu', 'tanh', 'sigmoid'],
        'model__dropout_rate':      uniform(loc=0.1, scale=0.3),     # [0.1, 0.4)
        'model__l2':                loguniform(1e-4, 1e-1),
        'batch_size':               [16, 32, 64],
        'optimizer__learning_rate': loguniform(1e-4, 1e-2),
    }
    # setup model
    model = make_clf(X_train.shape[1], len(config.CLASS_LABELS), config.SEARCH_EPOCHS)

    # Use StratifiedKFold due to imbalanced data
    skf = StratifiedKFold(n_splits=5, random_state=config.RANDOM_STATE, shuffle=True)
    # do the random search with stratified kfold
    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=50,
        cv=skf,
        n_jobs=1,                 # keep at 1 — Keras/TF doesn't parallelize cleanly across processes
        scoring=scoring,
        verbose=1,
        random_state=config.RANDOM_STATE,
    )
    # fit the random search
    search.fit(X_train, y_labels)

    print(f"  Best score:  {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_score_, search.best_params_


def objective(trial, X_train, y_labels, config):
    # Select hyperparameters to tune
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3, step=0.1)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Build and evaluate model via cross-validation
    clf = make_clf(X_train.shape[1], len(config.CLASS_LABELS), config.SEARCH_EPOCHS,
                   model__hidden_layers=[hidden_dim] * num_layers,
                   model__activation=activation,
                   model__dropout_rate=dropout_rate,
                   optimizer__learning_rate=lr,
                   batch_size=batch_size)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(clf, X_train, y_labels, cv=cv, scoring='f1_macro')
    return scores.mean()


def bayesian_search(X_train, y_labels, config, n_trials=20):
    print(f"\n=== RUNNING BAYESIAN OPTIMIZATION ({n_trials} trials) ===")

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_labels, config),
                   n_trials=n_trials, show_progress_bar=True)

    print(f"\n=== OPTIMIZATION COMPLETE ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation F1 score (macro): {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best = study.best_params
    best_params = {
        'model__hidden_layers': [best['hidden_dim']] * best['num_layers'],
        'model__activation': best['activation'],
        'model__dropout_rate': best['dropout_rate'],
        'batch_size': best['batch_size'],
        'optimizer__learning_rate': best['lr'],
    }
    return study.best_value, best_params


# ── Final model ───────────────────────────────────────────────────────────────

def build_final_model(config, input_dim, num_classes):
    model = build_keras_model(input_dim, num_classes,
                               config.HIDDEN_LAYERS,
                               config.ACTIVATION_FUNCTION,
                               config.DROPOUT_RATE,
                           config.L2_REGULARIZATION)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val, config):
    print("\nTraining final model...")
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test, config):
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    class_names = list(config.CLASS_LABELS.values())

    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    return y_test, y_pred, confusion_matrix(y_test, y_pred)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_search_comparison(scores_by_method):
    methods = list(scores_by_method.keys())
    scores = [scores_by_method[m] for m in methods]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=methods, y=scores, palette='Blues_d')
    plt.ylim(0, 1.1)
    plt.ylabel('CV Accuracy')
    plt.title('Hyperparameter Search Comparison')
    for i, s in enumerate(scores):
        plt.text(i, s + 0.01, f"{s:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(r"Feed-forward_neural_network\cv_comparison.png", dpi=300)
    plt.show()


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Val')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Val')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(r"Feed-forward_neural_network\training_history.png", dpi=300)
    plt.show()


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(r"Feed-forward_neural_network\confusion_matrix.png", dpi=300)
    plt.show()


def solo_model(X_train, y_train, X_val, y_val, X_test, y_test, config):
    print("\nTraining solo model with default Config params...")
    model = build_final_model(config, X_train.shape[1], len(config.CLASS_LABELS))
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    cm = evaluate_model(model, X_test, y_test, config)[2]
    plot_confusion_matrix(cm, list(config.CLASS_LABELS.values()))
    return model, history
# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    """
    print("Feed-Forward Neural Network — Multi-Class Classification\n")

    # Load and prepare data
    X, y = load_data(Config.FEATURES_PATH, Config.LABELS_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_normalize(X, y, Config)

    # Run all four search methods
    print("\n--- Hyperparameter Search ---")
    base_score, _ = base_model_cv(X_train, y_train, Config)
    grid_score, grid_params = grid_search(X_train, y_train, Config)
    random_score, random_params = random_search(X_train, y_train, Config)
    bayes_score, bayes_params = bayesian_search(X_train, y_train, Config)

    # Compare and pick the best
    results = {
        'Base Model': (base_score, None),
        'Grid Search': (grid_score, grid_params),
        'Rand Search': (random_score, random_params),
        'Bayesian': (bayes_score, bayes_params),
    }
    plot_search_comparison({name: score for name, (score, _) in results.items()})

    best_name, (best_score, best_params) = max(results.items(), key=lambda x: x[1][0])
    print(f"\nBest method: {best_name}  (CV acc {best_score:.4f})")
    print(f"Params: {best_params}")

    # Apply best params to Config before final training
    if best_params:
        param_map = {
            'model__hidden_layers': 'HIDDEN_LAYERS',
            'model__activation': 'ACTIVATION_FUNCTION',
            'model__dropout_rate': 'DROPOUT_RATE',
            'model__l2': 'L2_REGULARIZATION',
            'batch_size': 'BATCH_SIZE',
            'optimizer__learning_rate': 'LEARNING_RATE',
        }
        for search_key, config_attr in param_map.items():
            if search_key in best_params:
                setattr(Config, config_attr, best_params[search_key])

    # Train final model on train/val splits (no cross-validation)
    print(f"\n--- Final Training [{best_name} params] ---")
    model = build_final_model(Config, X_train.shape[1], len(Config.CLASS_LABELS))
    history = train_model(model, X_train, y_train, X_val, y_val, Config)

    # Evaluate and plot
    _, _, cm = evaluate_model(model, X_test, y_test, Config)
    plot_training_history(history)
    plot_confusion_matrix(cm, list(Config.CLASS_LABELS.values()))

    # Save
    model.save(Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")
    
    # For quick testing without running the full search, you can comment out the search methods and directly train with default Config params:
    """
    X, y = load_data(Config.FEATURES_PATH, Config.LABELS_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_normalize(X, y, Config)
    solo_model(X_train, y_train, X_val, y_val, X_test, y_test, Config)
    

if __name__ == "__main__":
    main()
