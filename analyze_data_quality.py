"""Analyze data quality for CSV and JSON files in 020320261."""
import pandas as pd
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\github\VT2\data\020320261")

def load_json_df(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    vectors = data["XML_Data"]["Wsk3Vectors"]
    x_vals = [float(v) for v in vectors["X_Axis"]["Values"]["float"]]
    axes_data = vectors["Y_AxesList"]["AxisData"]
    result = {"Time (ms)": x_vals}
    for axis in axes_data:
        name = axis["Header"]["Name"]
        unit = axis["Header"]["Unit"]
        values = [float(v) for v in axis["Values"]["float"]]
        result[f"{name} ({unit})"] = values
    return pd.DataFrame(result)

# === CSV ANALYSIS ===
print("=" * 70)
print("CSV DATA QUALITY REPORT")
print("=" * 70)

all_csv = []
for f in sorted(DATA_DIR.glob("*.csv")):
    df = pd.read_csv(f)
    name = f.stem
    nulls = df.isnull().sum().sum()
    dups = df.duplicated().sum()
    time_col = df.columns[0]
    t_min, t_max = df[time_col].min(), df[time_col].max()
    
    # Check time monotonicity
    time_diffs = df[time_col].diff().dropna()
    non_monotonic = (time_diffs <= 0).sum()
    
    # Check for constant columns
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    
    # Check for outliers (IQR method)
    outlier_cols = []
    for c in df.columns:
        if c == time_col:
            continue
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            n_outliers = ((df[c] < q1 - 3*iqr) | (df[c] > q3 + 3*iqr)).sum()
            if n_outliers > 0:
                outlier_cols.append((c, n_outliers))
    
    print(f"\n{name}:")
    print(f"  Rows: {len(df)}, Nulls: {nulls}, Duplicates: {dups}")
    print(f"  Time range: [{t_min:.1f}, {t_max:.1f}] ms")
    print(f"  Non-monotonic time steps: {non_monotonic}")
    if constant_cols:
        print(f"  CONSTANT columns: {constant_cols}")
    if outlier_cols:
        print(f"  Outlier columns (3x IQR): {outlier_cols}")
    
    all_csv.append(df)

combined_csv = pd.concat(all_csv, ignore_index=True)
print("\n" + "-" * 70)
print("AGGREGATED CSV STATS:")
print(combined_csv.describe().round(3).to_string())

# Check time sampling regularity
print("\n\nTIME SAMPLING ANALYSIS (CSV):")
for f in sorted(DATA_DIR.glob("*.csv")):
    df = pd.read_csv(f)
    time_col = df.columns[0]
    diffs = df[time_col].diff().dropna()
    print(f"  {f.stem}: mean_dt={diffs.mean():.2f}ms, std_dt={diffs.std():.2f}ms, min_dt={diffs.min():.2f}ms, max_dt={diffs.max():.2f}ms")


# === JSON ANALYSIS ===
print("\n" + "=" * 70)
print("JSON DATA QUALITY REPORT")
print("=" * 70)

all_json = []
for f in sorted(DATA_DIR.glob("*.json")):
    df = load_json_df(f)
    name = f.stem
    nulls = df.isnull().sum().sum()
    dups = df.duplicated().sum()
    time_col = "Time (ms)"
    
    # Check for zero/flat regions
    flat_info = []
    for c in df.columns:
        if c == time_col:
            continue
        diffs = df[c].diff().dropna()
        zero_pct = (diffs == 0).sum() / len(diffs) * 100
        if zero_pct > 50:
            flat_info.append((c, f"{zero_pct:.1f}%"))
    
    # Check for NaN/Inf
    n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    
    # Range check
    print(f"\n{name}:")
    print(f"  Rows: {len(df)}, Nulls: {nulls}, Inf: {n_inf}, Duplicates: {dups}")
    for c in df.columns:
        if c == time_col:
            continue
        print(f"    {c}: min={df[c].min():.4f}, max={df[c].max():.4f}, mean={df[c].mean():.4f}, std={df[c].std():.4f}")
    if flat_info:
        print(f"  Flat regions (>50% zero-diff): {flat_info}")
    
    all_json.append(df)

# Check consistency across files
print("\n" + "-" * 70)
print("CROSS-FILE CONSISTENCY:")
csv_lens = [len(df) for df in all_csv]
json_lens = [len(df) for df in all_json]
print(f"  CSV row counts: min={min(csv_lens)}, max={max(csv_lens)}, unique={set(csv_lens)}")
print(f"  JSON row counts: min={min(json_lens)}, max={max(json_lens)}, unique={set(json_lens)}")

# Check for negative values where unexpected
print("\n\nPOTENTIAL ISSUES SUMMARY:")
combined_json = pd.concat(all_json, ignore_index=True)
for c in combined_json.columns:
    if c == "Time (ms)":
        continue
    neg = (combined_json[c] < 0).sum()
    if neg > 0:
        print(f"  {c}: {neg} negative values")

print("\nDone!")
