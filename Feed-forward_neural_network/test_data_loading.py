"""
Test script to diagnose data loading issues
"""
import os
import pandas as pd
import numpy as np

DATA_ROOT = r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2\Data fra tidigare project"
INTRINSIC_DATA = os.path.join(DATA_ROOT, r"Dataset\Intrinsic data")
TASK_DATA = os.path.join(DATA_ROOT, r"Dataset\Task data")

print("=" * 70)
print("DATA LOADING TEST")
print("=" * 70)

# Test Intrinsic data
print("\nTesting Intrinsic data folder...")
if os.path.exists(INTRINSIC_DATA):
    n_folder = os.path.join(INTRINSIC_DATA, "N")
    if os.path.exists(n_folder):
        files = [f for f in os.listdir(n_folder) if f.endswith('.csv')]
        print(f"  Found {len(files)} CSV files in N class")
        
        if files:
            test_file = os.path.join(n_folder, files[0])
            print(f"  Testing load of: {files[0]}")
            try:
                df = pd.read_csv(test_file)
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {list(df.columns)[:5]}...")
                
                # Try to get numeric data
                numeric_df = df.apply(pd.to_numeric, errors='coerce')
                numeric_df = numeric_df.dropna(axis=1)
                features = numeric_df.values.flatten()
                print(f"    Feature vector shape: {features.shape}")
                print(f"    SUCCESS: File loaded correctly")
            except Exception as e:
                print(f"    ERROR: {e}")
    else:
        print(f"  ERROR: N folder not found at {n_folder}")
else:
    print(f"  ERROR: Intrinsic data folder not found at {INTRINSIC_DATA}")

# Test Task data
print("\nTesting Task data folder...")
if os.path.exists(TASK_DATA):
    n_folder = os.path.join(TASK_DATA, "N")
    if os.path.exists(n_folder):
        files = [f for f in os.listdir(n_folder) if f.endswith('.csv')]
        print(f"  Found {len(files)} CSV files in N class")
        
        if files:
            test_file = os.path.join(n_folder, files[0])
            print(f"  Testing load of: {files[0]}")
            try:
                df = pd.read_csv(test_file)
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {list(df.columns)[:5]}...")
                
                # Try to get numeric data
                numeric_df = df.apply(pd.to_numeric, errors='coerce')
                numeric_df = numeric_df.dropna(axis=1)
                features = numeric_df.values.flatten()
                print(f"    Feature vector shape: {features.shape}")
                print(f"    SUCCESS: File loaded correctly")
            except Exception as e:
                print(f"    ERROR: {e}")
    else:
        print(f"  ERROR: N folder not found at {n_folder}")
else:
    print(f"  ERROR: Task data folder not found at {TASK_DATA}")

# Count all files
print("\n" + "=" * 70)
print("FILE COUNT SUMMARY")
print("=" * 70)

for subfolder_name in [r"Dataset\Intrinsic data", r"Dataset\Task data"]:
    subfolder_path = os.path.join(DATA_ROOT, subfolder_name)
    print(f"\n{subfolder_name}:")
    
    for class_name in ["N", "NS", "OT", "P", "UT"]:
        class_folder = os.path.join(subfolder_path, class_name)
        if os.path.exists(class_folder):
            csv_files = [f for f in os.listdir(class_folder) if f.endswith('.csv')]
            print(f"  {class_name}: {len(csv_files)} CSV files")
        else:
            print(f"  {class_name}: FOLDER NOT FOUND")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
