"""Temporary script to inspect CSV and JSON data structure."""
import pandas as pd
import json
import numpy as np

# CSV info
csv = pd.read_csv(r"C:\Users\Anton\OneDrive\Dokumenter\GitHub\VT2\data_opsamling\Normal\120320261A1.csv")
print("=== CSV ===")
print(f"Shape: {csv.shape}")
print(f"Columns: {list(csv.columns)}")
time_col = csv.columns[0]
print(f"Time range: {csv[time_col].iloc[0]} to {csv[time_col].iloc[-1]} ms")
diffs = csv[time_col].diff().dropna()
print(f"Time diffs: mean={diffs.mean():.2f}, std={diffs.std():.2f}, min={diffs.min():.2f}, max={diffs.max():.2f}")
print(csv.head(3))
print("...")
print(csv.tail(3))

# JSON info
with open(r"C:\Users\Anton\OneDrive\Dokumenter\GitHub\VT2\data_opsamling\Normal\120320261A1.json", "r") as f:
    data = json.load(f)

print("\n=== JSON ===")
vectors = data["XML_Data"]["Wsk3Vectors"]
x_axis = vectors["X_Axis"]
print(f"X_Axis: Name={x_axis['Header']['Name']}, Unit={x_axis['Header']['Unit']}")
x_vals = [float(v) for v in x_axis["Values"]["float"]]
print(f"X range: {x_vals[0]} to {x_vals[-1]}, len={len(x_vals)}")
x_diffs = [x_vals[i+1]-x_vals[i] for i in range(len(x_vals)-1)]
print(f"X diffs: mean={np.mean(x_diffs):.4f}, min={min(x_diffs):.4f}, max={max(x_diffs):.4f}")

axes = vectors["Y_AxesList"]["AxisData"]
for ax in axes:
    name = ax["Header"]["Name"]
    unit = ax["Header"]["Unit"]
    vals = [float(v) for v in ax["Values"]["float"]]
    print(f"Y Axis: {name} ({unit}), len={len(vals)}, range=[{min(vals):.3f}, {max(vals):.3f}]")

# Also check Under
csv2 = pd.read_csv(r"C:\Users\Anton\OneDrive\Dokumenter\GitHub\VT2\data_opsamling\Under\120320261A2.csv")
print("\n=== Under CSV ===")
print(f"Shape: {csv2.shape}")
time_col2 = csv2.columns[0]
print(f"Time range: {csv2[time_col2].iloc[0]} to {csv2[time_col2].iloc[-1]} ms")

with open(r"C:\Users\Anton\OneDrive\Dokumenter\GitHub\VT2\data_opsamling\Under\120320261A2.json", "r") as f:
    data2 = json.load(f)
vectors2 = data2["XML_Data"]["Wsk3Vectors"]
x_axis2 = vectors2["X_Axis"]
x_vals2 = [float(v) for v in x_axis2["Values"]["float"]]
print(f"JSON X range: {x_vals2[0]} to {x_vals2[-1]}, len={len(x_vals2)}")

# Check what Robot_I looks like at start (idle detection)
print("\n=== IDLE DETECTION (CSV - first 20 Robot_I values) ===")
print(csv["Robot_I (A)"].head(20).tolist())
print(f"\nRobot_I std in first 50 samples: {csv['Robot_I (A)'].head(50).std():.4f}")
print(f"Robot_I std in last 50 samples: {csv['Robot_I (A)'].tail(50).std():.4f}")

# Check JSON Torque for idle
torque_axis = [ax for ax in axes if ax["Header"]["Name"].lower() == "torque"][0]
torque_vals = [float(v) for v in torque_axis["Values"]["float"]]
print(f"\n=== IDLE DETECTION (JSON - first 20 Torque values) ===")
print(torque_vals[:20])
print(f"Torque std in first 50: {np.std(torque_vals[:50]):.4f}")
