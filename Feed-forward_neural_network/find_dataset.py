from pathlib import Path

# Get the data folder using Path
vt2 = Path(r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2")

# Find the data folder
data_folder = None
for item in vt2.iterdir():
    if item.is_dir() and "Data fra nalazi project" in item.name:
        data_folder = item
        break

# If not found, try more flexible matching
if data_folder is None:
    for item in vt2.iterdir():
        if item.is_dir() and "Data" in item.name and "tidligere" in item.name:
            data_folder = item
            break

if data_folder:
    print(f"Found data folder: {data_folder}")
    print(f"\nChecking Dataset subfolder...")
    
    dataset = data_folder / "Dataset"
    if dataset.exists():
        print(f"Dataset folder exists!")
        print(f"Contents:")
        for item in sorted(dataset.iterdir()):
            if item.is_dir():
                print(f"  {item.name}/")
                # Count files in each subfolder
                csv_count = len(list(item.glob("**/*.csv")))
                print(f"    CSV files: {csv_count}")
    else:
        print(f"Dataset folder NOT found at {dataset}")
else:
    print("Data folder not found!")
