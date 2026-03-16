"""
tsfresh_features.py
===================
Extracts time-series features from cleaned CSV (robot) and JSON (screwing)
data using tsfresh, then selects relevant features for classification.

Data sources:
  - CSV: TCP_x, TCP_y, TCP_z, TCP_rx, TCP_ry, TCP_rz, Robot_I
  - JSON: Nset, Torque, Current, Angle, Depth

Labels:
  - Normal  → 0
  - Under   → 1

Usage:
  python Feature_engineering/code.py
  python Feature_engineering/code.py --no-select   (skip feature selection)

Output:
  Feature_engineering/features_extracted.csv   — all extracted features
  Feature_engineering/features_selected.csv    — relevant features only
"""

import json
import sys
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters

DATA_ROOT = Path(r"C:\github\VT2\data_opsamling_cleaned")
OUTPUT_DIR = Path(r"C:\github\VT2\Feature_engineering")
LABEL_MAP = {"Normal": 0, "Under": 1}


def load_csv_timeseries(filepath, sample_id):
    """Load a CSV file into tsfresh long format."""
    df = pd.read_csv(filepath)
    time_col = "Time (ms)"
    value_cols = [c for c in df.columns if c != time_col]
    df = df.rename(columns={time_col: "time"})
    df["id"] = sample_id
    return df[["id", "time"] + value_cols]


def load_json_timeseries(filepath, sample_id):
    """Load a JSON file into tsfresh long format."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    vectors = data["XML_Data"]["Wsk3Vectors"]
    x_vals = [float(v) for v in vectors["X_Axis"]["Values"]["float"]]
    axes = vectors["Y_AxesList"]["AxisData"]

    rows = {"id": [sample_id] * len(x_vals), "time": x_vals}
    for axis in axes:
        name = axis["Header"]["Name"]
        vals = [float(v) for v in axis["Values"]["float"]]
        rows[name] = vals

    return pd.DataFrame(rows)


def build_dataset():
    """Build combined long-format DataFrames for CSV and JSON, plus labels."""
    csv_frames = []
    json_frames = []
    labels = {}
    sample_id = 0

    for subfolder, label in LABEL_MAP.items():
        folder = DATA_ROOT / subfolder
        if not folder.exists():
            print(f"WARNING: {folder} not found, skipping")
            continue

        csv_files = sorted(folder.glob("*.csv"))
        json_files = sorted(folder.glob("*.json"))

        # Pair CSV and JSON by matching stem (e.g. 120320261A1.csv + .json)
        csv_stems = {f.stem: f for f in csv_files}
        json_stems = {f.stem: f for f in json_files}
        common_stems = sorted(set(csv_stems) & set(json_stems))

        if not common_stems:
            print(f"WARNING: No matching CSV/JSON pairs in {subfolder}")
            continue

        print(f"{subfolder} (label={label}): {len(common_stems)} paired samples")

        for stem in common_stems:
            csv_df = load_csv_timeseries(csv_stems[stem], sample_id)
            json_df = load_json_timeseries(json_stems[stem], sample_id)
            csv_frames.append(csv_df)
            json_frames.append(json_df)
            labels[sample_id] = label
            sample_id += 1

    csv_long = pd.concat(csv_frames, ignore_index=True)
    json_long = pd.concat(json_frames, ignore_index=True)
    y = pd.Series(labels, name="label")

    return csv_long, json_long, y


def main():
    do_select = "--no-select" not in sys.argv

    print("Building dataset...")
    csv_long, json_long, y = build_dataset()
    print(f"Total samples: {len(y)}  (Normal: {(y == 0).sum()}, Under: {(y == 1).sum()})")
    print(f"CSV long shape:  {csv_long.shape}")
    print(f"JSON long shape: {json_long.shape}")

    fc_params = EfficientFCParameters()
    if "--minimal" in sys.argv:
        fc_params = MinimalFCParameters()

    # Extract features from CSV (robot) data
    print("\nExtracting CSV (robot) features...")
    csv_features = extract_features(
        csv_long,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        n_jobs=0,  # single-process to avoid Windows multiprocessing issues
        show_warnings=False,
        disable_progressbar=True,
    )
    impute(csv_features)
    print(f"  CSV features: {csv_features.shape}")

    # Extract features from JSON (screwing) data
    print("Extracting JSON (screwing) features...")
    json_features = extract_features(
        json_long,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        n_jobs=0,
        show_warnings=False,
        disable_progressbar=True,
    )
    impute(json_features)
    print(f"  JSON features: {json_features.shape}")

    # Prefix to avoid column name collisions
    csv_features = csv_features.add_prefix("robot_")
    json_features = json_features.add_prefix("screw_")

    # Merge into one feature matrix
    all_features = pd.concat([csv_features, json_features], axis=1)
    all_features.index.name = "id"
    print(f"\nCombined features: {all_features.shape}")

    # Save all extracted features
    out_all = OUTPUT_DIR / "features_extracted.csv"
    all_features.to_csv(out_all)
    print(f"Saved all features to {out_all}")

    # Feature selection
    if do_select:
        print("\nSelecting relevant features...")
        selected = select_features(all_features, y)
        print(f"Selected features: {selected.shape[1]} / {all_features.shape[1]}")

        out_sel = OUTPUT_DIR / "features_selected.csv"
        selected.to_csv(out_sel)
        print(f"Saved selected features to {out_sel}")

        # Save labels alongside
        out_labels = OUTPUT_DIR / "labels.csv"
        y.to_csv(out_labels, header=True)
        print(f"Saved labels to {out_labels}")
    else:
        # Still save labels
        out_labels = OUTPUT_DIR / "labels.csv"
        y.to_csv(out_labels, header=True)
        print(f"Saved labels to {out_labels}")

    print("\nDone!")


if __name__ == "__main__":
    main()
