import pandas as pd, json, numpy as np

base = r"C:\Users\Anton\OneDrive\Dokumenter\GitHub\VT2\data_opsamling_preprocessed\Normal\120320261A1"
csv = pd.read_csv(f"{base}.csv")
with open(f"{base}.json") as f:
    data = json.load(f)

tc = "Time (ms)"
print("=== Preprocessed CSV ===")
print(f"Shape: {csv.shape}")
print(f"Time: {csv[tc].iloc[0]:.1f} to {csv[tc].iloc[-1]:.1f} ms")
diffs = csv[tc].diff().dropna()
print(f"Time steps: mean={diffs.mean():.2f}, std={diffs.std():.4f}")
print(csv.head(3))

print("\n=== Preprocessed JSON ===")
v = data["XML_Data"]["Wsk3Vectors"]
x = [float(xv) for xv in v["X_Axis"]["Values"]["float"]]
print(f"Time: {x[0]} to {x[-1]} ms, len={len(x)}")
for ax in v["Y_AxesList"]["AxisData"]:
    vals = [float(vv) for vv in ax["Values"]["float"]]
    print(f"  {ax['Header']['Name']}: first 3 = {vals[:3]}, range=[{min(vals):.3f}, {max(vals):.3f}]")
