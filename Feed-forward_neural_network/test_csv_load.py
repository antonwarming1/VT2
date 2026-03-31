import os
import pandas as pd

BASE = r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2\Data fra tidigare project"
INTRINSIC = os.path.join(BASE, r"Dataset\Intrinsic data\N")

print(f"Checking: {INTRINSIC}")
print(f"Exists: {os.path.exists(INTRINSIC)}")

if os.path.exists(INTRINSIC):
    files = [f for f in os.listdir(INTRINSIC) if f.endswith('.csv')]
    print(f"CSV files found: {len(files)}")
    
    if files:
        test_file = os.path.join(INTRINSIC, files[0])
        print(f"\nLoading: {files[0]}")
        
        df = pd.read_csv(test_file)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check data types
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        print(f"Numeric columns: {len(numeric_df.columns)}")
        
        # Flatten
        numeric_df = numeric_df.dropna(axis=1)
        features = numeric_df.values.flatten()
        print(f"Flattened features shape: {features.shape}")
        print("SUCCESS!")
