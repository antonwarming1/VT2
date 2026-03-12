"""Show concrete examples for each cleaning step."""
import pandas as pd
import json
import numpy as np
from pathlib import Path

DATA = Path(r"C:\github\VT2\data\020320261")

def load_json_df(fp):
    with open(fp) as f:
        data = json.load(f)
    vectors = data["XML_Data"]["Wsk3Vectors"]
    x = [float(v) for v in vectors["X_Axis"]["Values"]["float"]]
    result = {"Time (ms)": x}
    for axis in vectors["Y_AxesList"]["AxisData"]:
        name = axis["Header"]["Name"]
        unit = axis["Header"]["Unit"]
        result[f"{name} ({unit})"] = [float(v) for v in axis["Values"]["float"]]
    return pd.DataFrame(result)

# ======================================================================
print("=" * 70)
print("STEP 1: NORMALIZE TIME — shift all to start at 0")
print("=" * 70)
for name in ["020320261B7", "020320261C17"]:
    df = pd.read_csv(DATA / f"{name}.csv")
    t0 = df["Time (ms)"].iloc[0]
    t_end = df["Time (ms)"].iloc[-1]
    print(f"\n  {name}:")
    print(f"    BEFORE: time = [{t0}, ..., {t_end}] ms")
    print(f"    AFTER:  time = [0.0, ..., {t_end - t0}] ms")
    print(f"    Fix: df['Time (ms)'] -= df['Time (ms)'].iloc[0]")

# ======================================================================
print("\n" + "=" * 70)
print("STEP 2: RESAMPLE CSV to uniform interval (interpolation)")
print("=" * 70)
df = pd.read_csv(DATA / "020320261A1.csv")
diffs = df["Time (ms)"].diff().dropna()
print(f"\n  A1 time gaps (ms): min={diffs.min():.2f}, max={diffs.max():.2f}, "
      f"mean={diffs.mean():.2f}, std={diffs.std():.2f}")
print(f"  First 10 gaps: {diffs.head(10).values.tolist()}")
print(f"  Problem: gaps range from ~2ms to ~18ms (not uniform)")
print(f"  Fix: resample at fixed 4ms intervals using interpolation:")
print(f"    new_time = np.arange(0, df['Time (ms)'].max(), 4.0)")
print(f"    df_resampled = df.set_index('Time (ms)').reindex(new_time).interpolate('index')")
print(f"    Before: {len(df)} rows, After: ~{int(df['Time (ms)'].max() / 4)} rows")

# ======================================================================
print("\n" + "=" * 70)
print("STEP 3: REMOVE IDLE/STARTUP PHASE")
print("=" * 70)
df = pd.read_csv(DATA / "020320261A1.csv")
print("\n  A1 — First 10 rows of TCP_x:")
print(df[["Time (ms)", "TCP_x (mm)", "TCP_y (mm)"]].head(10).to_string(index=False))
tcp_move = df["TCP_x (mm)"].diff().abs()
first_idx = tcp_move[tcp_move > 0.05].index[0] if (tcp_move > 0.05).any() else 0
print(f"\n  TCP_x is flat at -80.9 mm for the first {first_idx} rows ({df['Time (ms)'].iloc[first_idx]:.0f} ms)")
print(f"  = robot is idle, not yet moving")
print(f"  Fix: trim rows before movement starts")
print(f"    movement_start = df['TCP_x (mm)'].diff().abs().gt(0.05).idxmax()")
print(f"    df_trimmed = df.iloc[movement_start:]")

# Also show JSON idle
dfj = load_json_df(DATA / "020320261A1.json")
torque_col = [c for c in dfj.columns if "torque" in c.lower()][0]
nset_col = [c for c in dfj.columns if "nset" in c.lower()][0]
# Find where Nset goes from 0 to 250
nset_start = (dfj[nset_col] > 0).idxmax()
print(f"\n  JSON: Nset (speed) is 0 for first {nset_start} rows = motor not spinning yet")
print(f"  Torque at those rows: {dfj[torque_col].iloc[:nset_start+2].values.tolist()[:5]}...")

# ======================================================================
print("\n" + "=" * 70)
print("STEP 4: CLIP OR SMOOTH OUTLIERS in Robot_I and Torque")
print("=" * 70)
df = pd.read_csv(DATA / "020320261A1.csv")
col = "Robot_I (A)"
q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
mask = (df[col] < lo) | (df[col] > hi)
print(f"\n  Robot_I (A1 CSV):")
print(f"    Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}")
print(f"    3x IQR bounds: [{lo:.3f}, {hi:.3f}]")
print(f"    {mask.sum()} outlier values: {df.loc[mask, col].values.tolist()}")
print(f"    Fix option A — clip:  df['Robot_I (A)'].clip({lo:.3f}, {hi:.3f})")
print(f"    Fix option B — rolling median:  df['Robot_I (A)'].rolling(5, center=True).median()")

# Torque
dfj = load_json_df(DATA / "020320261A1.json")
neg = dfj[torque_col] < 0
print(f"\n  Torque (A1 JSON):")
print(f"    {neg.sum()} negative values, min = {dfj[torque_col].min():.4f} Nm")
print(f"    Example negative torques: {dfj.loc[neg, torque_col].head(5).values.tolist()}")
print(f"    Fix: df['Torque'].clip(lower=0)  # clamp negatives to 0")

# ======================================================================
print("\n" + "=" * 70)
print("STEP 5: SEGMENT THE SCREWING PROCESS")
print("=" * 70)
dfj = load_json_df(DATA / "020320261A1.json")
nset = dfj[nset_col].values
# Find transitions
ramp_up_end = np.where(nset == 250)[0][0] if 250 in nset else 0
ramp_down_start = len(nset) - 1 - np.where(nset[::-1] == 250)[0][0] if 250 in nset else len(nset)
print(f"\n  A1 JSON — Nset (speed) profile:")
print(f"    Rows 0-{ramp_up_end}: RAMP-UP (speed goes 0 -> 250 rpm)")
print(f"    Rows {ramp_up_end}-{ramp_down_start}: ACTIVE SCREWING (constant 250 rpm)")
print(f"    Rows {ramp_down_start}-{len(nset)-1}: RAMP-DOWN (speed 250 -> 0 rpm)")
print(f"    Nset at boundaries: [{nset[0]}, {nset[ramp_up_end]}, ..., {nset[ramp_down_start]}, {nset[-1]}]")
print(f"    Fix:")
print(f"      ramp_mask = df['Nset'] < 250")
print(f"      active = df[df['Nset'] == 250]  # only the active screwing part")

# ======================================================================
print("\n" + "=" * 70)
print("STEP 6: FIX ENCODING OF ANGLE UNIT")
print("=" * 70)
with open(DATA / "020320261A1.json") as f:
    raw = json.load(f)
unit = raw["XML_Data"]["Wsk3Vectors"]["Y_AxesList"]["AxisData"][3]["Header"]["Unit"]
print(f"\n  Raw unit string: {repr(unit)}")
print(f"  Displayed as: '{unit}'")
print(f"  Problem: UTF-8 encoding artifact, should be degree symbol")
print(f"  Fix: name = name.replace('\\u00c2\\u00b0', '\\u00b0').replace('Â°', '°')")

# ======================================================================
print("\n" + "=" * 70)
print("STEP 7: NORMALIZE TO COMMON LENGTH")
print("=" * 70)
csv_lens = []
json_lens = []
for f in sorted(DATA.glob("*.csv")):
    csv_lens.append((f.stem, len(pd.read_csv(f))))
for f in sorted(DATA.glob("*.json")):
    json_lens.append((f.stem, len(load_json_df(f))))

csv_min = min(l for _, l in csv_lens)
csv_max = max(l for _, l in csv_lens)
json_min = min(l for _, l in json_lens)
json_max = max(l for _, l in json_lens)

print(f"\n  CSV row counts: {csv_min} to {csv_max} (range of {csv_max-csv_min})")
print(f"  JSON row counts: {json_min} to {json_max} (range of {json_max-json_min})")
print(f"  Shortest CSV: {min(csv_lens, key=lambda x: x[1])}")
print(f"  Longest CSV:  {max(csv_lens, key=lambda x: x[1])}")
print(f"\n  Fix option A — truncate all to shortest:")
print(f"    df = df.iloc[:{csv_min}]")
print(f"  Fix option B — normalize time to [0, 1] then resample to N points:")
print(f"    df['Time_norm'] = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())")
print(f"    new_time = np.linspace(0, 1, 1000)")
print(f"    # Then interpolate each column to these 1000 points")

print("\nDone!")
