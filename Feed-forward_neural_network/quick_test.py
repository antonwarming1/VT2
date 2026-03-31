import os
import sys

DATA_ROOT = r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2\Data fra tidigare project"

print("Checking folder structure...")
print(f"Root: {DATA_ROOT}")
print(f"Exists: {os.path.exists(DATA_ROOT)}")

intrinsic = os.path.join(DATA_ROOT, r"Dataset\Intrinsic data\N")
task = os.path.join(DATA_ROOT, r"Dataset\Task data\N") 

print(f"\nIntrinsic N: {intrinsic}")
print(f"Exists: {os.path.exists(intrinsic)}")

if os.path.exists(intrinsic):
    files = os.listdir(intrinsic)
    print(f"Files: {len(files)}")
    print(f"First 3: {files[:3]}")

print(f"\nTask N: {task}")
print(f"Exists: {os.path.exists(task)}")

if os.path.exists(task):
    files = os.listdir(task)
    print(f"Files: {len(files)}")
    print(f"First 3: {files[:3]}")
