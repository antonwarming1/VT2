"""
Feed-Forward Neural Network for Multi-Class Classification
Supports JSON and CSV data from class-specific folders
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
import librosa


# =====================================================================
# CONFIGURATION - Modify these parameters for your use case
# =====================================================================

class Config:
    """Configuration class for easy customization"""
    
    # DATA CONFIGURATION
    DATA_ROOT_FOLDER = r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2\Data fra tidligere project"  # Root folder
    SUBFOLDERS = [
        r"Dataset\Extrinsic data (clean)",  # Contains .wav files
        r"Dataset\Intrinsic data",          # Contains .csv files
        r"Dataset\Task data"                # Contains .csv files
    ]
    CLASS_FOLDERS = {
        0: "N",
        1: "NS",
        2: "OT",
        3: "P",
        4: "UT"
    }
    
    # FILE EXTENSIONS TO LOAD
    LOAD_CSV = True
    LOAD_WAV = True  # Enable WAV audio file loading
    
    # COLUMNS TO IGNORE (1-indexed: column 1 = index 0)
    IGNORE_COLUMNS_INTRINSIC = [2, 5, 6]  # From Intrinsic data only
    IGNORE_COLUMNS_TASK = []  # Leave empty to keep all Task columns
    
    # DATA PREPROCESSING
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
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
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    
    # MODEL SAVING/LOADING
    MODEL_SAVE_PATH = r"Feed-forward_neural_network\trained_model.keras"
    SCALER_SAVE_PATH = r"Feed-forward_neural_network\scaler.pkl"
    
    # OUTPUT
    PLOT_HISTORY = True
    PLOT_CONFUSION_MATRIX = True
    VERBOSE = 1  # 0=silent, 1=progress bar, 2=one line per epoch


# =====================================================================
# DATA LOADING
# =====================================================================

class DataLoader:
    """Load and process data from class folders containing CSV/JSON files"""
    
    def __init__(self, config):
        self.config = config
        self.X = []
        self.y = []
        
    def load_csv(self, file_path, folder_name=None):
        """Load and flatten CSV file to feature vector
        
        Args:
            file_path: Path to CSV file
            folder_name: Name of subfolder (to determine which columns to ignore)
        """
        df = pd.read_csv(file_path)
        
        # Drop ignored columns based on data source
        if folder_name and "Intrinsic" in folder_name:
            ignore_cols = [col - 1 for col in self.config.IGNORE_COLUMNS_INTRINSIC]
            cols_to_drop = [df.columns[i] for i in ignore_cols if i < len(df.columns)]
            df = df.drop(columns=cols_to_drop, errors='ignore')
        elif folder_name and "Task" in folder_name:
            ignore_cols = [col - 1 for col in self.config.IGNORE_COLUMNS_TASK]
            cols_to_drop = [df.columns[i] for i in ignore_cols if i < len(df.columns)]
            df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Convert all columns to numeric, skip non-numeric columns
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(axis=1)  # Remove columns with NaN
        features = numeric_df.values.flatten()
        return features
    
    def load_wav(self, file_path):
        """Load WAV file and extract audio features using librosa"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            
            # Extract features
            features = []
            
            # Time-domain features
            features.append(np.mean(y))  # Mean amplitude
            features.append(np.std(y))   # Std amplitude
            features.append(np.max(y))   # Max amplitude
            features.append(np.min(y))   # Min amplitude
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # Spectral features (using original magnitude spectrogram)
            S = np.abs(librosa.stft(y))
            
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(S=S)[0]
            features.extend([np.mean(centroid), np.std(centroid)])
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(S=S)[0]
            features.extend([np.mean(rolloff), np.std(rolloff)])
            
            # MFCC (Mel-frequency cepstral coefficients) - first 13
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features.append(np.mean(mfcc[i]))
                features.append(np.std(mfcc[i]))
            
            return np.array(features, dtype=float)
        
        except Exception as e:
            print(f"    WARNING: Could not extract features from {Path(file_path).name}: {str(e)}")
            return None
    
    
    def load_from_folders(self):
        """Load all data from multiple subfolders with class-specific folders"""
        print("Loading data from folders...")
        print(f"Root folder: {self.config.DATA_ROOT_FOLDER}")
        print(f"Subfolders: {self.config.SUBFOLDERS}")
        print(f"Classes: {list(self.config.CLASS_FOLDERS.values())}\n")
        
        # Use Path for robust path handling
        root_path = Path(self.config.DATA_ROOT_FOLDER)
        
        if not root_path.exists():
            print(f"ERROR: Root folder not found: {root_path}")
            return self.X, self.y
        
        # Loop through each subfolder
        for subfolder_name in self.config.SUBFOLDERS:
            subfolder_path = root_path / subfolder_name
            
            if not subfolder_path.exists():
                print(f"WARNING: Subfolder not found: {subfolder_path}\n")
                continue
            
            print(f"Loading from subfolder: '{subfolder_name}'")
            
            # Loop through each class folder within the subfolder
            for class_id, class_name in self.config.CLASS_FOLDERS.items():
                class_folder = subfolder_path / class_name
                
                if not class_folder.exists():
                    print(f"  WARNING: Class folder not found: {class_folder}")
                    continue
                
                file_count = 0
                for csv_file in class_folder.glob("*.csv"):
                    try:
                        features = self.load_csv(str(csv_file), folder_name=subfolder_name)
                        self.X.append(features)
                        self.y.append(class_id)
                        file_count += 1
                            
                    except Exception as e:
                        print(f"    ERROR loading {csv_file.name}: {str(e)}")
                
                # Load WAV files if available
                for wav_file in class_folder.glob("*.wav"):
                    try:
                        features = self.load_wav(str(wav_file))
                        if features is not None:  # Skip if feature extraction failed
                            self.X.append(features)
                            self.y.append(class_id)
                            file_count += 1
                            
                    except Exception as e:
                        print(f"    ERROR loading {wav_file.name}: {str(e)}")
                
                if file_count > 0:
                    print(f"    Class '{class_name}' (ID: {class_id}): {file_count} files loaded")
            
            print()  # Blank line between subfolders
        
        # Convert lists to numpy arrays, padding features to same size
        if len(self.X) > 0:
            # Find maximum feature size
            max_features = max(len(x) for x in self.X)
            print(f"Max feature size across all files: {max_features}")
            
            # Pad all features to the same size
            X_padded = []
            for features in self.X:
                padded = np.pad(features, (0, max_features - len(features)), mode='constant', constant_values=0)
                X_padded.append(padded)
            
            self.X = np.array(X_padded)
            self.y = np.array(self.y)
            
            print(f"Total samples loaded: {len(self.X)}")
            print(f"Feature vector shape: {self.X.shape}")
            print(f"Class distribution: {np.bincount(self.y)}")
        else:
            print("ERROR: No data was loaded!")
        
        return self.X, self.y


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
        
        # Input layer
        self.model.add(layers.Input(shape=(self.input_dim,)))
        
        # Hidden layers with sigmoid activation
        for neurons in self.config.HIDDEN_LAYERS:
            self.model.add(layers.Dense(neurons, activation=self.config.ACTIVATION_FUNCTION))
            self.model.add(layers.Dropout(0.2))  # Dropout for regularization
        
        # Output layer - softmax for multi-class classification
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        self.model.compile(
            optimizer=self.config.OPTIMIZER,
            loss=self.config.LOSS_FUNCTION,
            metrics=['accuracy']
        )
        
        print("Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the neural network"""
        print("\nTraining the model...")
        
        # Early stopping callback
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=self.config.VERBOSE
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nEvaluating model on test set...")
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_labels, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred))
        
        # Confusion matrix
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
            print(f"Model saved to: {self.config.MODEL_SAVE_PATH}")
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.config.MODEL_SAVE_PATH):
            self.model = keras.models.load_model(self.config.MODEL_SAVE_PATH)
            print(f"Model loaded from: {self.config.MODEL_SAVE_PATH}")
            return self.model
        else:
            print(f"Model file not found: {self.config.MODEL_SAVE_PATH}")
            return None


# =====================================================================
# VISUALIZATION AND UTILITIES
# =====================================================================

def plot_training_history(history, config):
    """Plot training and validation loss/accuracy"""
    if not config.PLOT_HISTORY:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
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
    print("Training history plot saved")


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
    print("Confusion matrix plot saved")


def normalize_data(X_train, X_val, X_test, config):
    """Normalize data using StandardScaler"""
    if not config.NORMALIZATION:
        return X_train, X_val, X_test
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print("Data normalized using StandardScaler")
    return X_train, X_val, X_test, scaler


# =====================================================================
# MAIN PIPELINE
# =====================================================================

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
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()
