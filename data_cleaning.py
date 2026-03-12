"""
data_cleaning.py
================
Cleans CSV and JSON data files in C:\\github\\VT2\\data\\<subfolder>:

1. CSV: Shifts time to start at 0 (for files where it doesn't)
2. CSV: Checks for and reports NaN values
3. JSON: Clips negative Torque values to 0
4. JSON: Clips negative Current values to 0
5. JSON: Checks for and reports NaN values
6. JSON: Fixes Angle unit encoding (Â° -> °)

Cleaned files are saved to C:\\github\\VT2\\data_cleaned\\<subfolder>.
Run with: python data_cleaning.py <subfolder>
Example:  python data_cleaning.py 020320261
"""

import pandas as pd
import json
import shutil
import sys
from pathlib import Path

DATA_ROOT = Path(r"C:\github\VT2\data")
OUTPUT_ROOT = Path(r"C:\github\VT2\data_cleaned")


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f)


def clean_csv(filepath, output_path):
    """Clean a single CSV file. Returns list of actions taken."""
    actions = []
    df = pd.read_csv(filepath)

    # Check for NaN
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        actions.append(f"  WARNING: {nan_count} NaN values found — dropped rows with NaN")
        df = df.dropna()

    # Shift time to start at 0
    time_col = "Time (ms)"
    if time_col in df.columns:
        t0 = df[time_col].iloc[0]
        if t0 != 0.0:
            df[time_col] = df[time_col] - t0
            actions.append(f"  Time shifted by -{t0:.3f} ms (was starting at {t0})")

    df.to_csv(output_path, index=False)

    return actions


def clean_json(filepath, output_path):
    """Clean a single JSON file. Returns list of actions taken."""
    actions = []
    data = load_json(filepath)

    vectors = data["XML_Data"]["Wsk3Vectors"]
    axes = vectors["Y_AxesList"]["AxisData"]

    for axis in axes:
        name = axis["Header"]["Name"]
        unit = axis["Header"]["Unit"]
        values = [float(v) for v in axis["Values"]["float"]]

        # Check for NaN
        nan_count = sum(1 for v in values if v != v)  # NaN != NaN
        if nan_count > 0:
            actions.append(f"  WARNING: {name} has {nan_count} NaN values — replaced with 0")
            values = [0.0 if v != v else v for v in values]

        # Clip negative Torque to 0
        if name.lower() == "torque":
            neg_count = sum(1 for v in values if v < 0)
            if neg_count > 0:
                values = [max(0.0, v) for v in values]
                actions.append(f"  Torque: clipped {neg_count} negative values to 0")

        # Clip negative Current to 0
        if name.lower() == "current":
            neg_count = sum(1 for v in values if v < 0)
            if neg_count > 0:
                values = [max(0.0, v) for v in values]
                actions.append(f"  Current: clipped {neg_count} negative values to 0")

        # Fix encoding of Angle unit
        if "Â°" in unit:
            axis["Header"]["Unit"] = unit.replace("Â°", "°")
            actions.append(f"  Fixed Angle unit encoding: '{unit}' -> '{axis['Header']['Unit']}'")

        # Write back cleaned values as strings (matching original format)
        axis["Values"]["float"] = [str(v) for v in values]

    save_json(data, output_path)

    return actions


def main():
    if len(sys.argv) < 2:
        print("Usage: python data_cleaning.py <subfolder>")
        print("Example: python data_cleaning.py 020320261")
        print(f"\nAvailable subfolders: {[d.name for d in DATA_ROOT.iterdir() if d.is_dir()]}")
        sys.exit(1)

    subfolder = sys.argv[1]
    data_dir = DATA_ROOT / subfolder
    output_dir = OUTPUT_ROOT / subfolder

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    json_files = sorted(data_dir.glob("*.json"))
    print(f"Cleaning {len(csv_files)} CSV + {len(json_files)} JSON files")
    print(f"  Source: {data_dir}")
    print(f"  Output: {output_dir}\n")

    total_actions = 0

    # Clean CSV files
    print("--- CSV FILES ---")
    for f in csv_files:
        out = output_dir / f.name
        actions = clean_csv(f, out)
        if actions:
            print(f"{f.name}:")
            for a in actions:
                print(a)
            total_actions += len(actions)
        else:
            print(f"{f.name}: OK (no changes needed)")

    # Clean JSON files
    print("\n--- JSON FILES ---")
    for f in json_files:
        out = output_dir / f.name
        actions = clean_json(f, out)
        if actions:
            print(f"{f.name}:")
            for a in actions:
                print(a)
            total_actions += len(actions)
        else:
            print(f"{f.name}: OK (no changes needed)")

    print(f"\nDone! {total_actions} total fixes applied.")


if __name__ == "__main__":
    main()
