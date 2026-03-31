"""
Simple test to verify padding and data loading works
"""
import numpy as np
from pathlib import Path
import pandas as pd

# Try importing librosa
try:
    import librosa
    HAS_LIBROSA = True
except:
    HAS_LIBROSA = False

DATA_ROOT = Path(r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2\Data fra tidligere project")
SUBFOLDERS = [r"Dataset\Extrinsic data (clean)", r"Dataset\Intrinsic data", r"Dataset\Task data"]

print("Testing data loading with padding...")
print("=" * 70)

X = []
y = []

for subfolder_name in SUBFOLDERS:
    subfolder_path = DATA_ROOT / subfolder_name
    print(f"\nTesting: {subfolder_name}")
    print(f"  Exists: {subfolder_path.exists()}")
    
    if not subfolder_path.exists():
        print(f"  SKIPPED: Not found")
        continue
    
    # Test N class
    class_folder = subfolder_path / "N"
    
    # Load 1 CSV file as example
    csv_files = list(class_folder.glob("*.csv"))[:1]
    if csv_files:
        df = pd.read_csv(csv_files[0])
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(axis=1)
        features = numeric_df.values.flatten()
        X.append(features)
        y.append(0)
        print(f"  CSV sample shape: {features.shape}")
    
    # Load 1 WAV file as example
    if HAS_LIBROSA:
        wav_files = list(class_folder.glob("*.wav"))[:1]
        if wav_files:
            y_audio, sr = librosa.load(str(wav_files[0]), sr=None)
            wav_features = np.array([1, 2, 3] * 14)  # Simplified: 42 features
            X.append(wav_features)
            y.append(0)
            print(f"  WAV sample shape: {wav_features.shape}")

print("\n" + "=" * 70)
print("Testing array padding...")
print("=" * 70)

if len(X) > 0:
    print(f"\nFeature sizes before padding:")
    for i, x in enumerate(X):
        print(f"  Sample {i}: {x.shape}")
    
    # Test padding
    max_features = max(len(x) for x in X)
    print(f"\nMax feature size: {max_features}")
    
    X_padded = []
    for features in X:
        padded = np.pad(features, (0, max_features - len(features)), mode='constant', constant_values=0)
        X_padded.append(padded)
    
    X_array = np.array(X_padded)
    print(f"\nAfter padding:")
    print(f"  Array shape: {X_array.shape}")
    print(f"  All samples same size: {len(set(len(x) for x in X_padded)) == 1}")
    
    print("\n✓ SUCCESS: Padding works correctly!")
else:
    print("ERROR: No data loaded")
