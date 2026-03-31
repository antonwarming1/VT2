from pathlib import Path
import os

# Get the VT2 folder
vt2 = Path(r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2")

# List all directories
print("All directories in VT2:")
for item in vt2.iterdir():
    if item.is_dir() and "Data" in item.name and "tidligere" in item.name:
        print(f"  {item.name}")
        print(f"  Full path: {item}")
        
        # Check what's inside
        print(f"  Contents:")
        for subitem in item.iterdir():
            print(f"    {subitem.name}")
            if (subitem / "Dataset").exists():
                print(f"      Found Dataset!")
                dataset = subitem / "Dataset"
                for ditem in dataset.iterdir():
                    print(f"        {ditem.name}")
