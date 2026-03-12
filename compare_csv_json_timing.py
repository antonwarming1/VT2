"""Compare CSV vs JSON timing for the same screwing cycles."""
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
        values = [float(v) for v in axis["Values"]["float"]]
        result[name] = values
    return pd.DataFrame(result)

print(f"{'File':<18} {'CSV rows':>9} {'CSV dt(ms)':>10} {'CSV dur(ms)':>12} {'JSON rows':>10} {'JSON dt(ms)':>11} {'JSON dur(ms)':>12} {'Duration match?':>16}")
print("-" * 110)

for f in sorted(DATA.glob("*.csv")):
    base = f.stem
    json_path = DATA / f"{base}.json"
    if not json_path.exists():
        continue

    df_csv = pd.read_csv(f)
    df_json = load_json_df(json_path)

    csv_dur = df_csv["Time (ms)"].max() - df_csv["Time (ms)"].min()
    json_dur = df_json["Time (ms)"].max() - df_json["Time (ms)"].min()
    csv_dt = df_csv["Time (ms)"].diff().dropna().mean()
    json_dt = df_json["Time (ms)"].diff().dropna().mean()

    match = "YES" if abs(csv_dur - json_dur) < 100 else f"NO (diff={csv_dur - json_dur:.0f}ms)"

    print(f"{base:<18} {len(df_csv):>9} {csv_dt:>10.2f} {csv_dur:>12.1f} {len(df_json):>10} {json_dt:>11.2f} {json_dur:>12.1f} {match:>16}")

print()

# Check if JSON time is always 0,1,2,3... (integer ms)
df_j = load_json_df(DATA / "020320261A1.json")
print("JSON time first 10:", df_j["Time (ms)"].head(10).values.tolist())
print("JSON time last 5:", df_j["Time (ms)"].tail(5).values.tolist())
print(f"JSON time step: always {df_j['Time (ms)'].diff().dropna().unique()}")

print()

# CSV time is irregular
df_c = pd.read_csv(DATA / "020320261A1.csv")
print("CSV time first 10:", df_c["Time (ms)"].head(10).values.tolist())
print(f"CSV time step: min={df_c['Time (ms)'].diff().dropna().min():.3f}, max={df_c['Time (ms)'].diff().dropna().max():.3f}")
