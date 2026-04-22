"""
Feed-Forward Neural Network — Multi-Class Classification
"""

import os
import numpy as np
import pandas as pd
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
import librosa


# ── Configuration ────────────────────────────────────────────────────────────

class Config:
    FEATURES_PATH = r"C:\github\VT2\Feature_engineering\features_selected.csv"
    LABELS_PATH = r"C:\github\VT2\Feature_engineering\labels.csv"
    MODEL_SAVE_PATH = r"Feed-forward_neural_network\trained_model.keras"

    # Data split: 70% train / 20% val / 10% test
    TRAIN_SIZE = 0.7
    VALIDATION_SIZE = 0.2
    TEST_SIZE = 0.1
    RANDOM_STATE = 42
    NORMALIZATION = True  # StandardScaler normalization
    
    # NEURAL NETWORK ARCHITECTURE
    HIDDEN_LAYERS = [128, 64, 32]  # MODIFY: Number of neurons in each hidden layer
    ACTIVATION_FUNCTION = 'sigmoid'  # Output activation for multi-class
    OPTIMIZER = Adam(learning_rate=0.001)
    LOSS_FUNCTION = 'categorical_crossentropy'
    
    # TRAINING CONFIGURATION
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
    'model__activation': ['relu', 'tanh'],
    'model__dropout_rate': [ 0.2],
    'batch_size': [16],
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

    # 3×2×2×2×2 = 48 combos × 5 folds — keeps runtime manageable
    param_grid = {k: v for k, v in PARAM_SPACE.items() if k != 'model__dropout_rate'}
    param_grid['model__dropout_rate'] = [0.1, 0.3]
    param_grid['batch_size'] = [16, 32]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    search = GridSearchCV(clf, param_grid, cv=skf, n_jobs=1, verbose=1)
    search.fit(X_train, y_labels)

    print(f"  Best score:  {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_score_, search.best_params_


def random_search(X_train, y_labels, config):
    print("\nRandom Search CV...")
    clf = make_clf(X_train.shape[1], len(config.CLASS_LABELS), config.SEARCH_EPOCHS)
    search = RandomizedSearchCV(clf, PARAM_SPACE, n_iter=10, cv=5,
                                 n_jobs=1, verbose=1, random_state=config.RANDOM_STATE)
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
    print("Training history plot saved")


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(r"Feed-forward_neural_network\confusion_matrix.png", dpi=300)
    plt.show()
    print("Confusion matrix plot saved")


def solo_model(X_train, y_train, X_val, y_val, X_test, y_test, config):
    print("\nTraining solo model with default Config params...")
    model = build_final_model(config, X_train.shape[1], len(config.CLASS_LABELS))
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    cm = evaluate_model(model, X_test, y_test, config)[2]
    plot_confusion_matrix(cm, list(config.CLASS_LABELS.values()))
    return model, history
# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    """Main pipeline: Load data, train model, evaluate"""
    
    print("="*70)
    print("FEED-FORWARD NEURAL NETWORK - MULTI-CLASS CLASSIFICATION")
    print("="*70)
    
    # Step 1: Load data
    loader = DataLoader(Config)
    X, y = loader.load_from_folders()
    
    if len(X) == 0:
        print("ERROR: No data loaded. Please check your data folder paths.")
        return
    
    # Step 2: Encode labels to one-hot
    y_encoded = keras.utils.to_categorical(y, num_classes=5)
    
    # Step 3: Split data - First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=np.argmax(y_encoded, axis=1)
    )
    
    # Step 4: Split temp into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=Config.VALIDATION_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=np.argmax(y_temp, axis=1)
    )
    
    print(f"\nData split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Step 5: Normalize data
    if Config.NORMALIZATION:
        X_train, X_val, X_test, scaler = normalize_data(X_train, X_val, X_test, Config)
    
    # Step 6: Build and train model
    nn = FeedForwardNN(Config, input_dim=X_train.shape[1], num_classes=5)
    nn.build_model()
    nn.train(X_train, y_train, X_val, y_val)
    
    # Step 7: Evaluate model
    results = nn.evaluate(X_test, y_test)
    
    # Step 8: Visualize results
    plot_training_history(nn.history, Config)
    plot_confusion_matrix(results['confusion_matrix'], Config, 
                         class_names=list(Config.CLASS_FOLDERS.values()))
    
    # Step 9: Save model
    nn.save_model()
    print("\nModel training and evaluation completed successfully!")
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()
