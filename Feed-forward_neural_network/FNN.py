"""
Feed-Forward Neural Network for Multi-Class Classification
Loads data from CSV files and trains the model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier


# =====================================================================
# CONFIGURATION
# =====================================================================

class Config:
    """Configuration class for easy customization"""
    
    # CSV FILE PATHS
    FEATURES_PATH = r"C:\github\VT2\Feature_engineering\features_selected.csv"
    LABELS_PATH = r"C:\github\VT2\Feature_engineering\labels.csv"
    
    # DATA PREPROCESSING (Total split: 70% train, 20% validation, 10% test)
    TRAIN_SIZE = 0.7
    VALIDATION_SIZE = 0.2
    TEST_SIZE = 0.1
    RANDOM_STATE = 42
    NORMALIZATION = True
    
    # NEURAL NETWORK ARCHITECTURE
    HIDDEN_LAYERS = [128, 64, 32]
    ACTIVATION_FUNCTION = 'relu'
    OPTIMIZER = Adam(learning_rate=0.001)
    LOSS_FUNCTION = 'categorical_crossentropy'
    
    # TRAINING CONFIGURATION
    BATCH_SIZE = 32
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
    # GRID SEARCH (Set to True to perform hyperparameter tuning)
    USE_GRID_SEARCH = False  # Change to True to enable grid search
    
    # MODEL SAVING
    MODEL_SAVE_PATH = r"Feed-forward_neural_network\trained_model.keras"
    
    # OUTPUT
    PLOT_HISTORY = True
    PLOT_CONFUSION_MATRIX = True
    VERBOSE = 1
    
    # CLASS LABELS
    CLASS_LABELS = {0: "N", 1: "NS", 2: "OT", 3: "P", 4: "UT"}


# =====================================================================
# DATA LOADING
# =====================================================================

class DataLoader:
    """Load features and labels from CSV files"""
    
    def __init__(self, features_path, labels_path):
        self.features_path = features_path
        self.labels_path = labels_path
    
    def load_data(self):
        """Load and return features and labels"""
        print("Loading data from CSV files...")
        
        features = pd.read_csv(self.features_path, index_col=0)
        print(f"✓ Features loaded: {features.shape}")
        
        labels = pd.read_csv(self.labels_path, index_col=0)
        print(f"✓ Labels loaded: {labels.shape}")
        
        return features.values, labels.values.flatten()



# =====================================================================
# NEURAL NETWORK MODEL
# =====================================================================

class FeedForwardNN:
    """Feed-Forward Neural Network for multi-class classification"""
    
    def __init__(self, config, input_dim, num_classes=5):
        self.config = config
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        self.scaler = None
        self.history = None
    
    def build_model(self, hidden_layers=None, activation=None):
        """Build the neural network architecture"""
        # Use provided parameters or fall back to config
        hidden_layers = hidden_layers or self.config.HIDDEN_LAYERS
        activation = activation or self.config.ACTIVATION_FUNCTION
        
        self.model = Sequential()
        
        self.model.add(layers.Input(shape=(self.input_dim,)))
        
        for neurons in hidden_layers:
            self.model.add(layers.Dense(neurons, activation=activation))
            self.model.add(layers.Dropout(0.2))
        
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        self.model.compile(
            optimizer=self.config.OPTIMIZER,
            loss=self.config.LOSS_FUNCTION,
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        self.model.summary()
        return self.model
    
    def random_search_cv(self, X_train, y_train_labels):
        """Perform Randomized Search CV using KerasClassifier"""
        print("\nPerforming Randomized Search CV with KerasClassifier...")
        param={
            'model__hidden_layers': [[64], [128, 64], [256, 128, 64]],
            'model__activation': ['relu', 'tanh'],
            'batch_size': [16, 32, 64],
            'model__dropout_rate': [0.1, 0.2, 0.3],
            'model__optimizer': [Adam(learning_rate=0.001), Adam(learning_rate=0.0001)]
        }
        # Create KerasClassifier wrapper using build_model
        keras_clf = KerasClassifier(
            model=self.build_model,
            loss=self.config.LOSS_FUNCTION,
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy'],
            batch_size=16,
            epochs=self.config.EPOCHS,
            verbose=0
        )
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=keras_clf,
            param_distributions=param,
            n_iter=10,  # Number of parameter settings sampled
            cv=5,  # Number of folds for cross-validation
            n_jobs=1,  # Set to 1 for TensorFlow compatibility
            verbose=1
        )
        
        random_search.fit(X_train, y_train_labels)
        
        print(f"✓ Best params: {random_search.best_params_}")
        print(f"✓ Best score: {random_search.best_score_:.4f}")
        print(f"\nTop 5 Results:")
        results_df = pd.DataFrame(random_search.cv_results_)
        print(results_df[['param_model__hidden_layers', 'param_model__activation', 'param_batch_size', 'mean_test_score']].head())
        
        return random_search
    def objective(self, X_train, y_train_labels):
        """Perform Bayesian Search CV using KerasClassifier"""
        print("\nPerforming Bayesian Search CV with KerasClassifier...")
        param={
            'model__hidden_layers': [[64], [128, 64], [256, 128, 64]],
            'model__activation': ['relu', 'tanh'],
            'batch_size': [16, 32, 64],
            'model__dropout_rate': [0.1, 0.2, 0.3],
            'model__optimizer': [Adam(learning_rate=0.001), Adam(learning_rate=0.0001)],
        }
        # Create KerasClassifier wrapper using build_model
        keras_clf = KerasClassifier(
            model=self.build_model,
            loss=self.config.LOSS_FUNCTION,
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy'],
            batch_size=16,
            epochs=self.config.EPOCHS,
            verbose=0
        )
        StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        score=cross_val_score(keras_clf, X_train, y_train_labels, cv=5, scoring='accuracy')
        print(f"✓ Cross-validation score: {score.mean():.4f}")
        return score.mean()
    
    def bayesian_optimization(self, X_train, y_train_labels):
        """Perform Bayesian Optimization for hyperparameter tuning using Optuna"""
        import optuna
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(X_train, y_train_labels), n_trials=50)
        
        print(f"✓ Best params: {study.best_params}")
        print(f"✓ Best score: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    


    def grid_search_cv(self, X_train, y_train_labels):
        """Perform Grid Search CV using KerasClassifier"""
        print("\nPerforming Grid Search CV with KerasClassifier...")
        
        # Create KerasClassifier wrapper using build_model
        keras_clf = KerasClassifier(
            model=self.build_model,
            loss=self.config.LOSS_FUNCTION,
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy'],
            batch_size=16,
            epochs=self.config.EPOCHS,
            verbose=0
        )
        
        # Define parameter grid
        param_grid = {
            'model__hidden_layers': [[64], [128, 64], [256, 128, 64]],
            'model__activation': ['relu', 'tanh'],
            'batch_size': [16, 32, 64],
            'model__dropout_rate': [0.1, 0.2, 0.3],
            'model__optimizer': [Adam(learning_rate=0.001), Adam(learning_rate=0.0001)]
        
        }
        
        # Use StratifiedKFold for imbalanced data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.RANDOM_STATE)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=keras_clf,
            param_grid=param_grid,
            cv=skf,
            n_jobs=1,  # Set to 1 for TensorFlow compatibility
            verbose=1
        )
        
        grid_search.fit(X_train, y_train_labels)
        
        
        
        return grid_search.best_params_, grid_search.best_score_
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the neural network"""
        print("\nTraining the model...")
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=self.config.VERBOSE
        )
        
        print("✓ Training completed!")
        return self.history
    
       

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        print("\nEvaluating model on test set...")
        
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_test_labels, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred))
        
        cm = confusion_matrix(y_test_labels, y_pred)
        
        return {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_test': y_test_labels,
            'confusion_matrix': cm
        }
    
    def save_model(self):
        """Save trained model"""
        if self.model is not None:
            self.model.save(self.config.MODEL_SAVE_PATH)
            print(f"✓ Model saved to: {self.config.MODEL_SAVE_PATH}")

    def evaluate_models(self, X_train, y_train_labels, X_test, y_test):
        #train and evaluate grid search, bayesian optimization models and randoom search model
        best_params_grid, best_score_grid = self.grid_search_cv(X_train, y_train_labels)
        best_params_bayesian, best_score_bayesian = self.bayesian_optimization(X_train, y_train_labels)




# =====================================================================
# UTILITIES
# =====================================================================




def normalize_data(X_train, X_val, X_test, config):
    """Normalize data using StandardScaler"""
    if not config.NORMALIZATION:
        return X_train, X_val, X_test, None
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print("✓ Data normalized using StandardScaler")
    return X_train, X_val, X_test, scaler


def plot_training_history(history, config):
    """Plot training and validation loss/accuracy"""
    if not config.PLOT_HISTORY or history is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(r"Feed-forward_neural_network\training_history.png", dpi=300)
    plt.show()
    print("✓ Training history plot saved")


def plot_confusion_matrix(cm, config, class_names=None):
    """Plot confusion matrix"""
    if not config.PLOT_CONFUSION_MATRIX:
        return
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(r"Feed-forward_neural_network\confusion_matrix.png", dpi=300)
    plt.show()
    print("✓ Confusion matrix plot saved")



# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    """Main pipeline: Load data, train model, evaluate"""
    
    print("="*70)
    print("FEED-FORWARD NEURAL NETWORK - MULTI-CLASS CLASSIFICATION")
    print("="*70)
    
    # Step 1: Load data
    loader = DataLoader(Config.FEATURES_PATH, Config.LABELS_PATH)
    
    X, y = loader.load_data()
    
    if len(X) == 0 or len(y) == 0:
        print("ERROR: No data loaded!")
        return
    
    print(f"Data shapes: X={X.shape}, y={y.shape}\n")
    
    # Step 2: Encode labels to one-hot
    y_encoded = keras.utils.to_categorical(y, num_classes=5)
    
    # Step 3: Split data - Separate test set first (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=np.argmax(y_encoded, axis=1)
    )
    
    # Step 4: Split remaining into train and validation
    # Calculate validation split size from remaining 90%
    # (20% of total) / (90% of total) = validation_ratio
    validation_ratio = Config.VALIDATION_SIZE / (Config.TRAIN_SIZE + Config.VALIDATION_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=validation_ratio,
        random_state=Config.RANDOM_STATE,
        stratify=np.argmax(y_temp, axis=1)
    )
    
    print(f"Data split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples\n")
    
    # Step 5: Normalize data
    X_train, X_val, X_test, scaler = normalize_data(X_train, X_val, X_test, Config)
    
    # Step 6: Optional - Perform Grid Search for hyperparameter tuning
    if Config.USE_GRID_SEARCH:
        print("\n" + "="*70)
        print("STARTING GRID SEARCH FOR HYPERPARAMETER TUNING")
        print("="*70)
        nn = FeedForwardNN(Config, input_dim=X_train.shape[1], num_classes=5)
        y_train_labels = np.argmax(y_train, axis=1)  # Convert one-hot to class labels
        grid_search = nn.grid_search_cv(X_train, y_train, y_train_labels)
        
        # Extract best parameters
        best_params = grid_search.best_params_
        Config.HIDDEN_LAYERS = best_params['model__hidden_layers']
        Config.ACTIVATION_FUNCTION = best_params['model__activation']
        Config.BATCH_SIZE = best_params['batch_size']
        
        print(f"\nApplied best params from grid search:")
        print(f"  Hidden Layers: {Config.HIDDEN_LAYERS}")
        print(f"  Activation: {Config.ACTIVATION_FUNCTION}")
        print(f"  Batch Size: {Config.BATCH_SIZE}\n")
    
    # Step 7: Build and train model (with best params if grid search was used)
    print("="*70)
    print("BUILDING AND TRAINING FINAL MODEL")
    print("="*70)
    nn = FeedForwardNN(Config, input_dim=X_train.shape[1], num_classes=5)
    nn.build_model()
    nn.train(X_train, y_train, X_val, y_val)
    
    # Step 8: Evaluate model
    results = nn.evaluate(X_test, y_test)
    
    # Step 9: Visualize results
    plot_training_history(nn.history, Config)
    plot_confusion_matrix(results['confusion_matrix'], Config, 
                         class_names=list(Config.CLASS_LABELS.values()))
    
    # Step 10: Save model
    nn.save_model()
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
