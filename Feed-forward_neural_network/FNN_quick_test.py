"""
Feed-Forward Neural Network - Quick Test Version (loads only first 20 files)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam

# Try to import librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("WARNING: librosa not available")

# Configuration
DATA_ROOT = r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2\Data fra tidligere project"
SUBFOLDERS = [r"Dataset\Extrinsic data (clean)", r"Dataset\Intrinsic data", r"Dataset\Task data"]
CLASS_FOLDERS = {0: "N", 1: "NS", 2: "OT", 3: "P", 4: "UT"}
MAX_FILES_PER_CLASS = 20  # QUICK TEST: 20 files per class

def extract_wav_features(file_path):
    """Extract features from WAV file"""
    y, sr = librosa.load(file_path, sr=None)
    features = []
    
    # Time-domain
    features.extend([np.mean(y), np.std(y), np.max(y), np.min(y)])
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.extend([np.mean(zcr), np.std(zcr)])
    
    # Spectral features
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    centroid = librosa.feature.spectral_centroid(S=S_db)[0]
    features.extend([np.mean(centroid), np.std(centroid)])
    
    rolloff = librosa.feature.spectral_rolloff(S=S_db)[0]
    features.extend([np.mean(rolloff), np.std(rolloff)])
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features.extend([np.mean(mfcc[i]), np.std(mfcc[i])])
    
    return np.array(features, dtype=float)

print("="*70)
print("QUICK TEST - Loading limited dataset (all 3 data sources)")
print("="*70)

X = []
y = []
root = Path(DATA_ROOT)

for subfolder in SUBFOLDERS:
    print(f"\nLoading from: {subfolder}")
    subfolder_path = root / subfolder
    
    if not subfolder_path.exists():
        print(f"  FOLDER NOT FOUND: {subfolder_path}")
        continue
    
    for class_id, class_name in CLASS_FOLDERS.items():
        class_folder = subfolder_path / class_name
        
        if not class_folder.exists():
            print(f"  {class_name}: FOLDER NOT FOUND")
            continue
        
        # Load CSV files
        csv_files = list(class_folder.glob("*.csv"))[:MAX_FILES_PER_CLASS]
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                numeric_df = df.apply(pd.to_numeric, errors='coerce')
                numeric_df = numeric_df.dropna(axis=1)
                features = numeric_df.values.flatten()
                X.append(features)
                y.append(class_id)
            except Exception as e:
                print(f"    ERROR (CSV): {csv_file.name}")
        
        # Load WAV files
        if LIBROSA_AVAILABLE:
            wav_files = list(class_folder.glob("*.wav"))[:MAX_FILES_PER_CLASS]
            for wav_file in wav_files:
                try:
                    features = extract_wav_features(str(wav_file))
                    X.append(features)
                    y.append(class_id)
                except Exception as e:
                    print(f"    ERROR (WAV): {wav_file.name}")
            
            if csv_files or wav_files:
                print(f"  {class_name}: {len(csv_files)} CSV + {len(wav_files)} WAV = {len(csv_files) + len(wav_files)} files")
        else:
            if csv_files:
                print(f"  {class_name}: {len(csv_files)} CSV files")

print("\nProcessing data...")
X_array = np.array(X)
y_array = np.array(y)

if len(X_array) > 0:
    print(f"Total samples: {len(X_array)}")
    print(f"Feature shapes: {[X[i].shape for i in range(min(3, len(X)))]}")
    
    # Pad features to same size for stacking
    max_features = max(x.shape[0] for x in X)
    X_padded = np.array([np.pad(x, (0, max_features - len(x)), 'constant') for x in X])
    
    print(f"Padded feature shape: {X_padded.shape}")
    print(f"Classes distribution: {np.bincount(y_array)}")
    
    print("\nSUCCESS: Data loaded and processed!")
else:
    print("ERROR: No data loaded!")

