"""
exclude_features.py — Remove unwanted columns from preprocessed CSV and JSON files.

Reads from data_opsamling_preprocessed/, drops the columns listed in
EXCLUDE_CSV and EXCLUDE_JSON, and saves the result to data_opsamling_final/.

Configure which columns to drop below. Time (ms) is always kept.
"""

import copy
import json
import sys
from pathlib import Path

import pandas as pd


# ── Config ───────────────────────────────────────────────────────────────────

INPUT_ROOT  = Path(__file__).parent.parent / "data_old_preprocessed"
OUTPUT_ROOT = Path(__file__).parent.parent / "data_opsamling_final"

PROCESS_SUBFOLDERS = ["--all"]   # ["Normal"], ["Normal","Under"], or ["--all"]

# Columns to remove from CSV files (task + intrinsic)
# Covers both new data (mm units) and old data (rad units).
# Intrinsic columns are included here because old data stores everything as CSV.

EXCLUDE_CSV = [
    # Task CSV — new data (mm)
    
    # Task CSV — old data (rad)
    "TCP_rx (rad)",
    "TCP_ry (rad)",
    "TCP_rz (rad)",
    # Intrinsic CSV — old data
    "Nset (1/min)",
    "Angle (deg)",
    "Depth (mm)",
    "Current (V)"
]

# Columns to remove from JSON files (new dataset screwing cell data)
EXCLUDE_JSON = [
    "Nset",
    "Angle",
    "Depth",
    "Current (V)"

]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def drop_csv_columns_df(df):
    """In-memory version of drop_csv_columns: drop EXCLUDE_CSV columns from a DataFrame."""
    to_drop = [col for col in EXCLUDE_CSV if col in df.columns]
    return df.drop(columns=to_drop)


def drop_csv_columns(input_path, output_path):
    """Load a CSV, drop excluded columns, and save."""
    df = pd.read_csv(input_path)

    to_drop = [col for col in EXCLUDE_CSV if col in df.columns]
    df = df.drop(columns=to_drop)

    df.to_csv(output_path, index=False)
    return to_drop


def drop_json_columns(input_path, output_path):
    """Load a JSON, remove excluded signal channels, and save."""
    data = load_json(input_path)
    data_out = copy.deepcopy(data)

    axes = data_out["XML_Data"]["Wsk3Vectors"]["Y_AxesList"]["AxisData"]

    # Filter out excluded axes
    original_names = [a["Header"]["Name"] for a in axes]
    kept = [a for a in axes if a["Header"]["Name"] not in EXCLUDE_JSON]
    dropped = [name for name in original_names if name in EXCLUDE_JSON]

    data_out["XML_Data"]["Wsk3Vectors"]["Y_AxesList"]["AxisData"] = kept

    save_json(data_out, output_path)
    return dropped


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Resolve subfolders
    if PROCESS_SUBFOLDERS == ["--all"]:
        if not INPUT_ROOT.exists():
            sys.exit(f"Error: {INPUT_ROOT} does not exist")
        subfolders = sorted(
            d for d in INPUT_ROOT.iterdir()
            if d.is_dir() and d.name != "Extrinsic data"
        )
    else:
        subfolders = [INPUT_ROOT / name for name in PROCESS_SUBFOLDERS]

    print(f"Excluding from CSV:  {EXCLUDE_CSV}")
    print(f"Excluding from JSON: {EXCLUDE_JSON}\n")

    for folder in subfolders:
        if not folder.exists():
            sys.exit(f"Error: {folder} does not exist")

        out_dir = OUTPUT_ROOT / folder.name
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(folder.glob("*.csv"))
        json_files = sorted(folder.glob("*.json"))

        print(f"{'='*50}\n  {folder.name} — {len(csv_files)} CSV, {len(json_files)} JSON\n{'='*50}")

        for csv_file in csv_files:
            dropped = drop_csv_columns(csv_file, out_dir / csv_file.name)
            kept = pd.read_csv(out_dir / csv_file.name).columns.tolist()
            print(f"  {csv_file.stem}.csv: dropped {dropped} | kept {kept}")

        for json_file in json_files:
            dropped = drop_json_columns(json_file, out_dir / json_file.name)
            print(f"  {json_file.stem}.json: dropped {dropped}")

    print(f"\nDone! Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
