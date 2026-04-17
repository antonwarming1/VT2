"""
Feed-Forward Neural Network for Multi-Class Classification
Loads data from CSV files and trains the model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam


# =====================================================================
# CONFIGURATION
# =====================================================================

class Config:
    """Configuration class for easy customization"""
    
    # CSV FILE PATHS
    FEATURES_PATH = r"C:\github\VT2\Feature_engineering\features_selected.csv"
    LABELS_PATH = r"C:\github\VT2\Feature_engineering\labels.csv"
    
    # DATA PREPROCESSING
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
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
    
    def build_model(self):
        """Build the neural network architecture"""
        self.model = Sequential()
        
        self.model.add(layers.Input(shape=(self.input_dim,)))
        
        for neurons in self.config.HIDDEN_LAYERS:
            self.model.add(layers.Dense(neurons, activation=self.config.ACTIVATION_FUNCTION))
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
    
    # Step 3: Split data - train+val vs test
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
    
    print(f"Data split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples\n")
    
    # Step 5: Normalize data
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
                         class_names=list(Config.CLASS_LABELS.values()))
    
    # Step 9: Save model
    nn.save_model()
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
