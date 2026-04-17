"""
Cross-Validation Script for Feed-Forward Neural Network with Grid Search
Uses F1 score as the primary metric
Reuses the FeedForwardNN class from FNN.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import FNN module
sys.path.insert(0, str(Path(__file__).parent.parent / "Feed-forward_neural_network"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
import itertools
import warnings

# Import from FNN.py
from FNN import Config, DataLoader

warnings.filterwarnings('ignore')


# =====================================================================
# CROSS-VALIDATION WITH GRID SEARCH
# =====================================================================

class GridSearchCV:
    """Grid Search for Hyperparameter Tuning using existing FeedForwardNN"""
    
    def __init__(self, param_grid, config, n_splits=5):
        self.param_grid = param_grid
        self.config = config
        self.n_splits = n_splits
        self.results = []
        self.best_params = None
        self.best_f1_score = -np.inf
        self.best_model = None
    
    def generate_param_combinations(self):
        """Generate all parameter combinations"""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def evaluate_params(self, X_train, y_train, X_val, y_val, params):
        """Evaluate FeedForwardNN model with given parameters"""
        try:
            # Lazy import to avoid circular dependency
            from FNN import FeedForwardNN
            
            # Get NUM_CLASSES from config (handle both class and instance)
            num_classes = 5  # Default
            if isinstance(self.config, type):
                # Config is a class
                num_classes = getattr(self.config, 'NUM_CLASSES', len(getattr(self.config, 'CLASS_FOLDERS', {})))
            else:
                # Config is an instance
                num_classes = getattr(self.config, 'NUM_CLASSES', len(getattr(self.config, 'CLASS_FOLDERS', {})))
            
            if num_classes == 0:
                num_classes = 5  # Fallback
            
            # Convert to categorical for training
            y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
            y_val_cat = keras.utils.to_categorical(y_val, num_classes=num_classes)
            
            # Create a simple config object with all necessary attributes
            class TempConfig:
                def __init__(self, base_config, params, num_classes):
                    # Handle both class and instance
                    def get_attr(name, default):
                        if isinstance(base_config, type):
                            return getattr(base_config, name, default)
                        else:
                            return getattr(base_config, name, default)
                    
                    # Copy all important attributes from base config
                    self.DATA_ROOT_FOLDER = get_attr('DATA_ROOT_FOLDER', '')
                    self.SUBFOLDERS = get_attr('SUBFOLDERS', [])
                    self.CLASS_FOLDERS = get_attr('CLASS_FOLDERS', {})
                    self.LOAD_CSV = get_attr('LOAD_CSV', True)
                    self.LOAD_WAV = get_attr('LOAD_WAV', True)
                    self.IGNORE_COLUMNS_INTRINSIC = get_attr('IGNORE_COLUMNS_INTRINSIC', [])
                    self.IGNORE_COLUMNS_TASK = get_attr('IGNORE_COLUMNS_TASK', [])
                    self.ACTIVATION_FUNCTION = get_attr('ACTIVATION_FUNCTION', 'sigmoid')
                    self.LOSS_FUNCTION = get_attr('LOSS_FUNCTION', 'categorical_crossentropy')
                    self.NUM_CLASSES = num_classes
                    self.RANDOM_STATE = get_attr('RANDOM_STATE', 42)
                    self.NORMALIZATION = get_attr('NORMALIZATION', True)
                    
                    # Training configuration attributes
                    self.EPOCHS = get_attr('EPOCHS', 100)
                    self.VALIDATION_SPLIT = get_attr('VALIDATION_SPLIT', 0.2)
                    self.EARLY_STOPPING_PATIENCE = get_attr('EARLY_STOPPING_PATIENCE', 10)
                    self.PLOT_HISTORY = get_attr('PLOT_HISTORY', False)
                    self.PLOT_CONFUSION_MATRIX = get_attr('PLOT_CONFUSION_MATRIX', False)
                    self.VERBOSE = 0  # Silent mode for grid search
                    
                    # Set hyperparameters
                    self.HIDDEN_LAYERS = params['hidden_layers']
                    self.OPTIMIZER = Adam(learning_rate=params['learning_rate'])
                    self.BATCH_SIZE = params['batch_size']
            
            temp_config = TempConfig(self.config, params, num_classes)
            
            # Build model using FeedForwardNN
            nn = FeedForwardNN(temp_config, input_dim=X_train.shape[1], num_classes=self.config.NUM_CLASSES)
            nn.build_model()
            
            # Train model
            nn.train(X_train, y_train_cat, X_val, y_val_cat)
            
            # Get predictions on validation set
            y_val_pred_prob = nn.model.predict(X_val, verbose=0)
            y_val_pred = np.argmax(y_val_pred_prob, axis=1)
            
            # Calculate F1 scores
            f1_weighted = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
            f1_macro = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
            accuracy = accuracy_score(y_val, y_val_pred)
            
            return {
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'accuracy': accuracy,
                'model': nn
            }
        
        except Exception as e:
            print(f"    ERROR during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'f1_weighted': 0.0,
                'f1_macro': 0.0,
                'accuracy': 0.0,
                'model': None
            }
    
    def fit(self, X, y):
        """Perform grid search with cross-validation"""
        print("\n" + "="*70)
        print("STARTING GRID SEARCH WITH CROSS-VALIDATION")
        print("="*70)
        
        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Generate parameter combinations
        param_combinations = self.generate_param_combinations()
        print(f"Total parameter combinations to evaluate: {len(param_combinations)}")
        print(f"Number of CV folds: {self.n_splits}\n")
        
        # K-Fold Cross-Validation
        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.config.RANDOM_STATE)
        
        # Grid search over parameters
        for param_idx, params in enumerate(param_combinations):
            print(f"\n[{param_idx + 1}/{len(param_combinations)}] Evaluating parameters:")
            print(f"  Hidden layers: {params['hidden_layers']}")
            print(f"  Learning rate: {params['learning_rate']}")
            print(f"  Batch size: {params['batch_size']}")
            
            fold_f1_scores = []
            fold_accuracies = []
            
            # Cross-validation folds
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
                X_train_fold = X_scaled[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X_scaled[val_idx]
                y_val_fold = y[val_idx]
                
                # Evaluate on this fold
                fold_results = self.evaluate_params(X_train_fold, y_train_fold, X_val_fold, y_val_fold, params)
                
                if fold_results['model'] is not None:
                    fold_f1_scores.append(fold_results['f1_weighted'])
                    fold_accuracies.append(fold_results['accuracy'])
            
            # Calculate mean scores
            if fold_f1_scores:
                mean_f1 = np.mean(fold_f1_scores)
                std_f1 = np.std(fold_f1_scores)
                mean_accuracy = np.mean(fold_accuracies)
                std_accuracy = np.std(fold_accuracies)
                
                print(f"  Mean F1 Score (weighted): {mean_f1:.4f} (+/- {std_f1:.4f})")
                print(f"  Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
                
                # Store results
                result = {
                    'params': params,
                    'mean_f1_weighted': mean_f1,
                    'std_f1': std_f1,
                    'mean_accuracy': mean_accuracy,
                    'std_accuracy': std_accuracy,
                    'fold_f1_scores': fold_f1_scores,
                    'fold_accuracies': fold_accuracies
                }
                self.results.append(result)
                
                # Update best params if this is better
                if mean_f1 > self.best_f1_score:
                    self.best_f1_score = mean_f1
                    self.best_params = params
        
        print("\n" + "="*70)
        print("GRID SEARCH COMPLETED")
        print("="*70)
        
        if self.best_params is None:
            print("\nERROR: No valid parameters found! All evaluations failed.")
            print("This usually means there's an issue with the model training.")
            print("Check the error messages above for more details.\n")
        else:
            print(f"\nBest Parameters:")
            print(f"  Hidden layers: {self.best_params['hidden_layers']}")
            print(f"  Learning rate: {self.best_params['learning_rate']}")
            print(f"  Batch size: {self.best_params['batch_size']}")
            print(f"  Best F1 Score (weighted): {self.best_f1_score:.4f}\n")
        
        return self.results
    
    def get_best_params(self):
        """Return best parameters found"""
        return self.best_params, self.best_f1_score
    
    def get_results_dataframe(self):
        """Return results as pandas DataFrame"""
        if not self.results:
            print("WARNING: No results to display. Grid search may have failed.")
            return pd.DataFrame()
        
        results_list = []
        for result in self.results:
            results_list.append({
                'Hidden Layers': str(result['params']['hidden_layers']),
                'Learning Rate': result['params']['learning_rate'],
                'Batch Size': result['params']['batch_size'],
                'Mean F1 (weighted)': result['mean_f1_weighted'],
                'Std F1': result['std_f1'],
                'Mean Accuracy': result['mean_accuracy'],
                'Std Accuracy': result['std_accuracy']
            })
        
        return pd.DataFrame(results_list)


# =====================================================================
# VISUALIZATION
# =====================================================================

def plot_grid_search_results(grid_search, config):
    """Plot grid search results"""
    if not config.PLOT_RESULTS:
        return
    
    results_df = grid_search.get_results_dataframe()
    
    if len(results_df) == 0:
        print("WARNING: No results to plot. Grid search may have failed.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sort by F1 score for better visualization
    results_df_sorted = results_df.sort_values('Mean F1 (weighted)', ascending=False).head(10)
    
    # F1 Score comparison
    x_pos = np.arange(len(results_df_sorted))
    axes[0].bar(x_pos, results_df_sorted['Mean F1 (weighted)'].values, 
                yerr=results_df_sorted['Std F1'].values, capsize=5, alpha=0.7)
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('F1 Score (weighted)')
    axes[0].set_title('Top 10 Configurations - F1 Score')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"Config {i}" for i in range(len(results_df_sorted))], rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    axes[1].bar(x_pos, results_df_sorted['Mean Accuracy'].values, 
                yerr=results_df_sorted['Std Accuracy'].values, capsize=5, alpha=0.7, color='orange')
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Top 10 Configurations - Accuracy')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"Config {i}" for i in range(len(results_df_sorted))], rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r"CrossValidation\grid_search_results.png", dpi=300)
    plt.show()
    print("Grid search results plot saved")




def main():
    """Main pipeline: Load data, perform grid search with cross-validation using FeedForwardNN"""
    
    print("="*70)
    print("CROSS-VALIDATION WITH GRID SEARCH FOR FEED-FORWARD NEURAL NETWORK")
    print("Using existing FNN.py architecture")
    print("F1 Score as Primary Metric")
    print("="*70)
    
    # Step 1: Load data using the same DataLoader from FNN.py
    loader = DataLoader(Config)
    X, y = loader.load_from_folders()
    
    if len(X) == 0:
        print("ERROR: No data loaded. Please check your data folder paths.")
        return
    
    print(f"\nData loaded successfully!")
    print(f"Shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}\n")
    
    # Step 2: Define parameter grid for grid search
    param_grid = {
        'hidden_layers': [
            [128, 64],
            [128, 64, 32],
            [256, 128],
            [256, 128, 64],
            [64, 32, 16]
        ],
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64]
    }
    
    # Step 3: Perform grid search with cross-validation
    grid_search = GridSearchCV(param_grid, Config, n_splits=5)
    results = grid_search.fit(X, y)
    
    # Step 4: Display results
    results_df = grid_search.get_results_dataframe()
    print("\nGrid Search Results (Top 10 by F1 Score):")
    print("="*70)
    print(results_df.sort_values('Mean F1 (weighted)', ascending=False).head(10).to_string(index=False))
    print("="*70)
    
    # Step 5: Save results
    results_df.to_csv(r"CrossValidation\grid_search_results.csv", index=False)
    print("\nResults saved to: CrossValidation\\grid_search_results.csv")
    
    # Step 6: Visualize results
    plot_grid_search_results(grid_search, Config)
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
