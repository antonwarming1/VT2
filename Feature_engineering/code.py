"""
tsfresh_features.py
===================
Extracts time-series features from preprocessed data using tsfresh,
then selects relevant features for classification.

Supports two dataset formats (controlled by DATASET config):

Old data (Task CSV + Intrinsic CSV pairs):
  - t*.csv → keeps: Robot_I (A)
  - i*.csv → keeps: Torque (Nm), Current (V)
  - Labels: N, NS, OT, P, UT

New data (CSV + JSON pairs, same filename stem):
  - CSV  → keeps: Robot_I (A)
  - JSON → keeps: Torque, Current
  - Labels: Normal, Under

Usage:
  python Feature_engineering/code.py
  python Feature_engineering/code.py --no-select   (skip feature selection)

Output:
  Feature_engineering/features_extracted.csv   — all extracted features
  Feature_engineering/features_selected.csv    — relevant features only
  Feature_engineering/labels.csv               — class labels
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
from tsfresh.feature_extraction import EfficientFCParameters


# ── Config ───────────────────────────────────────────────────────────────────

DATASET = "old"  # "old" or "new"

OLD_DATA_ROOT = Path(r"C:\github\VT2\data_opsamling_final")
OLD_LABEL_MAP = {"N": 0, "NS": 1, "OT": 2, "P": 3, "UT": 4}

NEW_DATA_ROOT = Path(r"C:\github\VT2\data_opsamling_preprocessed")
NEW_LABEL_MAP = {"Normal": 0, "Under": 1}

OUTPUT_DIR = Path(r"C:\github\VT2\Feature_engineering")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _csv_to_long(filepath, sample_id):
    """Load a CSV into tsfresh long format. All columns except Time are used as features."""
    df = pd.read_csv(filepath)
    cols = [c for c in df.columns if c != "Time (ms)"]
    df = df.rename(columns={"Time (ms)": "time"})
    df["id"] = sample_id
    return df[["id", "time"] + cols]


def _json_to_long(filepath, sample_id):
    """Load a JSON (WSK3 format) into tsfresh long format. All channels are used as features."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    vectors = data["XML_Data"]["Wsk3Vectors"]
    time_values = [float(v) for v in vectors["X_Axis"]["Values"]["float"]]
    rows = {"time": time_values}

    for axis in vectors["Y_AxesList"]["AxisData"]:
        name = axis["Header"]["Name"]
        rows[name] = [float(v) for v in axis["Values"]["float"]]

    df = pd.DataFrame(rows)
    df["id"] = sample_id
    cols = [c for c in df.columns if c not in ("id", "time")]
    return df[["id", "time"] + cols]


# ── Dataset builders ─────────────────────────────────────────────────────────

def build_old_dataset():
    """
    Old data: pair t*.csv / i*.csv by matching ID after prefix letter.
    Returns (task_long, intr_long, labels).
    """
    task_frames, intr_frames, labels = [], [], {}
    sample_id = 0

    for subfolder, label in OLD_LABEL_MAP.items():
        folder = OLD_DATA_ROOT / subfolder
        if not folder.exists():
            print(f"  WARNING: {folder} not found, skipping")
            continue

        task_files = {f.stem[1:]: f for f in sorted(folder.glob("t*.csv"))}
        intr_files = {f.stem[1:]: f for f in sorted(folder.glob("i*.csv"))}
        paired = sorted(task_files.keys() & intr_files.keys())

        if not paired:
            print(f"  WARNING: no pairs in {subfolder}")
            continue

        print(f"  {subfolder} (label={label}): {len(paired)} samples")
        for base_id in paired:
            task_frames.append(_csv_to_long(task_files[base_id], sample_id))
            intr_frames.append(_csv_to_long(intr_files[base_id], sample_id))
            labels[sample_id] = label
            sample_id += 1

    return (pd.concat(task_frames, ignore_index=True),
            pd.concat(intr_frames, ignore_index=True),
            pd.Series(labels, name="label"))


def build_new_dataset():
    """
    New data: pair *.csv / *.json by matching filename stem.
    Returns (task_long, intr_long, labels).
    """
    task_frames, intr_frames, labels = [], [], {}
    sample_id = 0

    for subfolder, label in NEW_LABEL_MAP.items():
        folder = NEW_DATA_ROOT / subfolder
        if not folder.exists():
            print(f"  WARNING: {folder} not found, skipping")
            continue

        csv_files = {f.stem: f for f in sorted(folder.glob("*.csv"))}
        json_files = {f.stem: f for f in sorted(folder.glob("*.json"))}
        paired = sorted(csv_files.keys() & json_files.keys())

        if not paired:
            print(f"  WARNING: no pairs in {subfolder}")
            continue

        print(f"  {subfolder} (label={label}): {len(paired)} samples")
        for stem in paired:
            task_frames.append(_csv_to_long(csv_files[stem], sample_id))
            intr_frames.append(_json_to_long(json_files[stem], sample_id))
            labels[sample_id] = label
            sample_id += 1

    return (pd.concat(task_frames, ignore_index=True),
            pd.concat(intr_frames, ignore_index=True),
            pd.Series(labels, name="label"))


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_from_long(df, name):
    """Run tsfresh feature extraction on a long-format DataFrame."""
    print(f"  Extracting {name} features...")
    features = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=0,
        show_warnings=False,
        disable_progressbar=True,
    )
    impute(features)
    print(f"  {name} features: {features.shape}")
    return features


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    do_select = "--no-select" not in sys.argv
    label_map = OLD_LABEL_MAP if DATASET == "old" else NEW_LABEL_MAP

    # Build dataset
    print(f"Building dataset (DATASET={DATASET})...")
    if DATASET == "old":
        task_long, intr_long, y = build_old_dataset()
    else:
        task_long, intr_long, y = build_new_dataset()

    for name, val in label_map.items():
        print(f"  {name}: {(y == val).sum()}")
    print(f"  Total: {len(y)} samples")
    print(f"  Task shape:      {task_long.shape}")
    print(f"  Intrinsic shape: {intr_long.shape}")

    # Extract features
    print("\nExtracting features...")
    task_features = extract_from_long(task_long, "Task").add_prefix("task_")
    intr_features = extract_from_long(intr_long, "Intrinsic").add_prefix("intr_")

    # Combine
    all_features = pd.concat([task_features, intr_features], axis=1)
    all_features.index.name = "id"
    print(f"\nCombined: {all_features.shape}")

    # Save extracted features
    all_features.to_csv(OUTPUT_DIR / "features_extracted.csv")
    print(f"Saved → features_extracted.csv")

    # Feature selection
    if do_select:
        print("\nSelecting relevant features...")
        selected = select_features(all_features, y)
        print(f"Selected: {selected.shape[1]} / {all_features.shape[1]}")
        selected.to_csv(OUTPUT_DIR / "features_selected.csv")
        print(f"Saved → features_selected.csv")

    # Save labels
    y.to_csv(OUTPUT_DIR / "labels.csv", header=True)
    print(f"Saved → labels.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
