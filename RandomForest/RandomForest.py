"""
Random Forest Classifier for Multi-Class Classification
Supports CSV data from class-specific folders
Classes: N, NS, OT, P, UT
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# CONFIGURATION
# =====================================================================

class Config:
    """Configuration class for Random Forest"""
    
    # DATA CONFIGURATION
    DATA_ROOT_FOLDER = r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2\Data fra tidligere project"
    SUBFOLDERS = [
        r"Dataset\Extrinsic data (clean)",
        r"Dataset\Intrinsic data",
        r"Dataset\Task data"
    ]
    CLASS_FOLDERS = {
        0: "N",
        1: "NS",
        2: "OT",
        3: "P",
        4: "UT"
    }
    
    # DATA PREPROCESSING
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    NORMALIZATION = True  # StandardScaler normalization
    
    # RANDOM FOREST CONFIGURATION
    N_ESTIMATORS = 100          # Number of trees
    MAX_DEPTH = 20              # Maximum depth of trees
    MIN_SAMPLES_SPLIT = 5       # Minimum samples to split node
    MIN_SAMPLES_LEAF = 2        # Minimum samples in leaf
    MAX_FEATURES = 'sqrt'       # 'sqrt', 'log2', or int
    RANDOM_STATE_RF = 42
    N_JOBS = -1                 # Use all available processors
    
    # COLUMNS TO IGNORE (1-indexed)
    IGNORE_COLUMNS_INTRINSIC = [2, 5, 6]
    IGNORE_COLUMNS_TASK = []
    
    # OUTPUT
    PLOT_CONFUSION_MATRIX = True
    PLOT_FEATURE_IMPORTANCE = True
    VERBOSE = True


# =====================================================================
# DATA LOADING
# =====================================================================

class DataLoader:
    """Load and process data from class folders"""
    
    def __init__(self, config):
        self.config = config
        self.X = []
        self.y = []
        self.feature_names = []
        
    def load_csv(self, file_path, folder_name=None):
        """Load and flatten CSV file to feature vector"""
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
        
        # Convert to numeric
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(axis=1)
        
        # Store feature names (from first file)
        if len(self.feature_names) == 0:
            self.feature_names = numeric_df.columns.tolist()
        
        features = numeric_df.values.flatten()
        return features
    
    def load_from_folders(self):
        """Load all data from multiple subfolders with class-specific folders"""
        print("Loading data from folders...")
        print(f"Root folder: {self.config.DATA_ROOT_FOLDER}")
        print(f"Subfolders: {self.config.SUBFOLDERS}")
        print(f"Classes: {list(self.config.CLASS_FOLDERS.values())}\n")
        
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
            
            # Loop through each class folder
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
                
                if file_count > 0:
                    print(f"    Class '{class_name}' (ID: {class_id}): {file_count} files loaded")
            
            print()
        
        # Convert to numpy arrays and pad features
        if len(self.X) > 0:
            max_features = max(len(x) for x in self.X)
            print(f"Max feature size: {max_features}")
            
            X_padded = []
            for features in self.X:
                padded = np.pad(features, (0, max_features - len(features)), 
                              mode='constant', constant_values=0)
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
# RANDOM FOREST CLASSIFIER
# =====================================================================

class RandomForestModel:
    """Random Forest classifier for multi-class classification"""
    
    def __init__(self, config, num_classes=5):
        self.config = config
        self.num_classes = num_classes
        self.model = None
        self.scaler = None
        self.feature_importances = None
        self.training_time = 0
        
    def build_model(self):
        """Create Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=self.config.N_ESTIMATORS,
            max_depth=self.config.MAX_DEPTH,
            min_samples_split=self.config.MIN_SAMPLES_SPLIT,
            min_samples_leaf=self.config.MIN_SAMPLES_LEAF,
            max_features=self.config.MAX_FEATURES,
            random_state=self.config.RANDOM_STATE_RF,
            n_jobs=self.config.N_JOBS,
            verbose=1 if self.config.VERBOSE else 0
        )
        
        print("\nRandom Forest Configuration:")
        print(f"  Number of Estimators: {self.config.N_ESTIMATORS}")
        print(f"  Max Depth: {self.config.MAX_DEPTH}")
        print(f"  Min Samples Split: {self.config.MIN_SAMPLES_SPLIT}")
        print(f"  Min Samples Leaf: {self.config.MIN_SAMPLES_LEAF}")
        print(f"  Max Features: {self.config.MAX_FEATURES}\n")
        
        return self.model
    
    def train(self, X_train, y_train):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.feature_importances = self.model.feature_importances_
        
        print(f"Training completed in {self.training_time:.2f} seconds!")
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
        
        return self.model
    
    def evaluate(self, X_test, y_test, class_names=None):
        """Evaluate model performance"""
        print("Evaluating model on test set...")
        
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision (weighted): {precision:.4f}")
        print(f"  Recall (weighted): {recall:.4f}")
        print(f"  F1-Score (weighted): {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        if class_names:
            print(classification_report(y_test, y_pred, target_names=class_names))
        else:
            print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_test': y_test,
            'confusion_matrix': cm
        }
    
    def save_model(self, model_path=None):
        """Save trained model and scaler"""
        if model_path is None:
            model_path = r"RandomForest\trained_rf_model.pkl"
        
        if self.model is not None:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to: {model_path}")
        
        if self.scaler is not None:
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler saved to: {scaler_path}")
    
    def load_model(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            model_path = r"RandomForest\trained_rf_model.pkl"
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from: {model_path}")
            return self.model
        else:
            print(f"Model file not found: {model_path}")
            return None


# =====================================================================
# VISUALIZATION AND UTILITIES
# =====================================================================

def plot_confusion_matrix(cm, config, class_names=None):
    """Plot confusion matrix"""
    if not config.PLOT_CONFUSION_MATRIX:
        return
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Random Forest', fontweight='bold', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(r"RandomForest\confusion_matrix.png", dpi=300)
    plt.show()
    print("Confusion matrix plot saved")


def plot_feature_importance(feature_importances, config, top_n=20):
    """Plot feature importance"""
    if not config.PLOT_FEATURE_IMPORTANCE:
        return
    
    # Get top N features
    indices = np.argsort(feature_importances)[::-1][:top_n]
    importances = feature_importances[indices]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances, align='center')
    plt.xlabel('Feature Rank', fontsize=12)
    plt.ylabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances - Random Forest', 
             fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(r"RandomForest\feature_importance.png", dpi=300)
    plt.show()
    print(f"Feature importance plot saved (top {top_n} features)")


def normalize_data(X_train, X_test, config):
    """Normalize data using StandardScaler"""
    if not config.NORMALIZATION:
        return X_train, X_test, None
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Data normalized using StandardScaler")
    return X_train, X_test, scaler


def plot_model_comparison(results, metrics_list=['accuracy', 'precision', 'recall', 'f1']):
    """Plot model performance metrics"""
    metrics_values = [results.get(metric, 0) for metric in metrics_list]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_list, metrics_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    plt.ylabel('Score', fontsize=12)
    plt.title('Random Forest Performance Metrics', fontweight='bold', fontsize=14)
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(r"RandomForest\model_metrics.png", dpi=300)
    plt.show()
    print("Model metrics plot saved")


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    """Main pipeline: Load data, train model, evaluate"""
    
    print("="*70)
    print("RANDOM FOREST CLASSIFIER - MULTI-CLASS CLASSIFICATION")
    print("="*70)
    
    # Step 1: Load data
    loader = DataLoader(Config)
    X, y = loader.load_from_folders()
    
    if len(X) == 0:
        print("ERROR: No data loaded.")
        return
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"\nData split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Step 3: Normalize data
    if Config.NORMALIZATION:
        X_train, X_test, scaler = normalize_data(X_train, X_test, Config)
    
    # Step 4: Build model
    rf_model = RandomForestModel(Config, num_classes=5)
    rf_model.scaler = scaler
    rf_model.build_model()
    
    # Step 5: Train model
    rf_model.train(X_train, y_train)
    
    # Step 6: Evaluate model
    results = rf_model.evaluate(X_test, y_test, 
                               class_names=list(Config.CLASS_FOLDERS.values()))
    
    # Step 7: Visualize results
    plot_confusion_matrix(results['confusion_matrix'], Config,
                         class_names=list(Config.CLASS_FOLDERS.values()))
    plot_feature_importance(rf_model.feature_importances, Config, top_n=20)
    plot_model_comparison(results)
    
    # Step 8: Save model
    rf_model.save_model()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()
