import os

# Test different path variants
paths = [
    r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2\Data fra tidligere project",
    "C:\\Users\\emil_\\OneDrive - Aalborg Universitet\\VT2\\VT2\\Data fra tidligere project",
]

for p in paths:
    print(f"Path: {p}")
    print(f"Repr: {repr(p)}")
    print(f"Exists: {os.path.exists(p)}")
    print()

# List the actual directory
base = r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2"
print(f"Contents of {base}:")
for item in os.listdir(base):
    if "Data" in item:
        print(f"  {repr(item)}")
