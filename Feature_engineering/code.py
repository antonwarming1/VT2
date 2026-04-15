"""
tsfresh_features.py
===================
Extracts time-series features from old-format preprocessed data
(Task CSV + Intrinsic CSV pairs) using tsfresh, then selects
relevant features for classification.

Data sources:
  - Task CSV (t*.csv):      TCP_rx, TCP_ry, TCP_rz, Robot_I
  - Intrinsic CSV (i*.csv): Nset, Torque, Current, Angle, Depth

Labels:
  - N  (Normal)         → 0
  - NS (Not Screwed)    → 1
  - OT (Over-Torqued)   → 2
  - P  (Pass)           → 3
  - UT (Under-Torqued)  → 4

Usage:
  python Feature_engineering/code.py
  python Feature_engineering/code.py --no-select   (skip feature selection)

Output:
  Feature_engineering/features_extracted.csv   — all extracted features
  Feature_engineering/features_selected.csv    — relevant features only
"""

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
from tsfresh.feature_extraction import EfficientFCParameters

DATA_ROOT = Path(r"C:\github\VT2\data_opsamling_final")
OUTPUT_DIR = Path(r"C:\github\VT2\Feature_engineering")
LABEL_MAP = {"N": 0, "NS": 1, "OT": 2, "P": 3, "UT": 4}


def load_csv_timeseries(filepath, sample_id):
    """Load a CSV file into tsfresh long format."""
    df = pd.read_csv(filepath)
    time_col = "Time (ms)"
    value_cols = [c for c in df.columns if c != time_col]
    df = df.rename(columns={time_col: "time"})
    df["id"] = sample_id
    return df[["id", "time"] + value_cols]





def build_dataset():
    """Build combined long-format DataFrames for Task and Intrinsic CSVs, plus labels."""
    task_frames = []
    intr_frames = []
    labels = {}
    sample_id = 0

    for subfolder, label in LABEL_MAP.items():
        folder = DATA_ROOT / subfolder
        if not folder.exists():
            print(f"WARNING: {folder} not found, skipping")
            continue

        # Pair t*.csv and i*.csv by matching the ID after the prefix letter
        task_files = {f.stem[1:]: f for f in sorted(folder.glob("t*.csv"))}
        intr_files = {f.stem[1:]: f for f in sorted(folder.glob("i*.csv"))}
        paired = sorted(set(task_files) & set(intr_files))

        if not paired:
            print(f"WARNING: No matching Task/Intrinsic pairs in {subfolder}")
            continue

        print(f"{subfolder} (label={label}): {len(paired)} paired samples")

        for base_id in paired:
            task_df = load_csv_timeseries(task_files[base_id], sample_id)
            intr_df = load_csv_timeseries(intr_files[base_id], sample_id)
            task_frames.append(task_df)
            intr_frames.append(intr_df)
            labels[sample_id] = label
            sample_id += 1

    task_long = pd.concat(task_frames, ignore_index=True)
    intr_long = pd.concat(intr_frames, ignore_index=True)
    y = pd.Series(labels, name="label")

    return task_long, intr_long, y


def main():
    do_select = "--no-select" not in sys.argv

    print("Building dataset...")
    task_long, intr_long, y = build_dataset()
    for label_name, label_val in LABEL_MAP.items():
        print(f"  {label_name}: {(y == label_val).sum()}")
    print(f"Total samples: {len(y)}")
    print(f"Task long shape:      {task_long.shape}")
    print(f"Intrinsic long shape: {intr_long.shape}")

    fc_params = EfficientFCParameters()

    # Extract features from Task (robot) data
    print("\nExtracting Task (robot) features...")
    task_features = extract_features(
        task_long,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        n_jobs=0,  # single-process to avoid Windows multiprocessing issues
        show_warnings=False,
        disable_progressbar=True,
    )
    impute(task_features)
    print(f"  Task features: {task_features.shape}")

    # Extract features from Intrinsic (screwing) data
    print("Extracting Intrinsic (screwing) features...")
    intr_features = extract_features(
        intr_long,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        n_jobs=0,
        show_warnings=False,
        disable_progressbar=True,
    )
    impute(intr_features)
    print(f"  Intrinsic features: {intr_features.shape}")

    # Prefix to avoid column name collisions
    task_features = task_features.add_prefix("task_")
    intr_features = intr_features.add_prefix("intr_")

    # Merge into one feature matrix
    all_features = pd.concat([task_features, intr_features], axis=1)
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
