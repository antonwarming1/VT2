#!/usr/bin/env python
"""Minimal test of padding logic"""
import numpy as np

print("Testing padding logic...")

# Simulate mixed feature sizes (CSV vs WAV)
features_list = [
    np.array([1.0, 2.0, 3.0, 4.0, 5.0]),           # 5 features
    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), # 7 features  
    np.array([1.0, 2.0, 3.0, 4.0]),                # 4 features
]

print(f"Features before padding:")
for i, f in enumerate(features_list):
    print(f"  {i}: shape {f.shape}")

# Find max size
max_features = max(len(x) for x in features_list)
print(f"\nMax size: {max_features}")

# Pad all
padded_list = [
    np.pad(f, (0, max_features - len(f)), mode='constant', constant_values=0) 
    for f in features_list
]

print(f"\nFeatures after padding:")
for i, f in enumerate(padded_list):
    print(f"  {i}: shape {f.shape}")

# Convert to array
X = np.array(padded_list)
print(f"\nFinal array shape: {X.shape}")
print("✓ SUCCESS - padding works!")
