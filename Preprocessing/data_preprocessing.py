"""
data_preprocessing.py — Runs AFTER data_cleaning.py.

Steps:
  1. Remove idle phase (detected from JSON Depth derivative)
  2. Resample both CSV and JSON to uniform time steps (linear interpolation)
  3. Optional Savitzky-Golay smoothing on Robot_I
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


# ── Config ───────────────────────────────────────────────────────────────────

DATA_ROOT   = Path(r"C:\github\VT2\data_opsamling_cleaned")
OUTPUT_ROOT = Path(r"C:\github\VT2\data_opsamling_preprocessed")

PROCESS_SUBFOLDERS = ["--all"]   # ["Normal"], ["Normal","Under"], or ["--all"]

RESAMPLE_MS    = 2       # Target sample interval (ms)
IDLE_DEPTH_RATE = 0.005  # Depth rate threshold (mm/ms) to detect screwing start
IDLE_WINDOW    = 50      # Rolling window size (ms) for smoothing the depth rate
IDLE_MARGIN_MS = 100     # Keep this many ms before detected screwing start

SMOOTH_CSV     = True            # Apply Savitzky-Golay smoothing to CSV columns
SMOOTH_COLS    = ["Robot_I (A)"] # Which columns to smooth
SAVGOL_WINDOW  = 11              # Must be odd
SAVGOL_POLY    = 3


# ── JSON helpers ─────────────────────────────────────────────────────────────

def load_json(path):
    """Read a JSON file and return the parsed data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    """Write data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def json_to_df(data):
    """
    Convert the WSK3 JSON structure into a flat DataFrame.

    The JSON stores time in X_Axis and signal channels (Torque, Depth, etc.)
    in Y_AxesList. This pulls them all into columns alongside a Time (ms) column.
    """
    vectors = data["XML_Data"]["Wsk3Vectors"]

    # Time axis
    time_values = [float(v) for v in vectors["X_Axis"]["Values"]["float"]]

    # Build a dict: column name -> list of floats
    columns = {"Time (ms)": time_values}
    for axis in vectors["Y_AxesList"]["AxisData"]:
        name = axis["Header"]["Name"]
        values = [float(v) for v in axis["Values"]["float"]]
        columns[name] = values

    return pd.DataFrame(columns)


def df_to_json(df, original_json):
    """
    Write DataFrame values back into a copy of the original JSON structure.
    This preserves all metadata while updating the numerical data.
    """
    data = copy.deepcopy(original_json)
    vectors = data["XML_Data"]["Wsk3Vectors"]

    # Update time axis
    vectors["X_Axis"]["Values"]["float"] = [str(v) for v in df["Time (ms)"].tolist()]

    # Update each signal channel
    for axis in vectors["Y_AxesList"]["AxisData"]:
        name = axis["Header"]["Name"]
        if name in df.columns:
            axis["Values"]["float"] = [str(v) for v in df[name].tolist()]

    return data


# ── Step 1: Detect and remove idle phase ─────────────────────────────────────

def detect_active_start(json_df):
    """
    Find the time (ms) when the screw starts engaging.

    How it works:
      - Compute the absolute change in Depth between each sample
      - Smooth it with a rolling average (IDLE_WINDOW samples)
      - Find the first moment the smoothed rate exceeds IDLE_DEPTH_RATE
      - Subtract IDLE_MARGIN_MS to keep a small buffer before that point

    Returns None if no active screwing is detected (file should be skipped).
    """
    if "Depth" not in json_df.columns:
        return None

    depth = json_df["Depth"].values
    time = json_df["Time (ms)"].values

    # Absolute change in depth between consecutive samples
    depth_change = np.abs(np.diff(depth))

    if len(depth_change) < IDLE_WINDOW:
        return None

    # Smooth the depth change with a rolling average
    kernel = np.ones(IDLE_WINDOW) / IDLE_WINDOW
    smoothed_rate = np.convolve(depth_change, kernel, mode="valid")

    # Find where the smoothed rate first exceeds the threshold
    above_threshold = np.where(smoothed_rate > IDLE_DEPTH_RATE)[0]

    if len(above_threshold) == 0:
        return None  # Depth never changes enough — no real screwing happened

    first_active_time = time[above_threshold[0]]

    # Keep a margin before the detected start
    return max(0.0, first_active_time - IDLE_MARGIN_MS)


def trim_to_start(df, start_time_ms):
    """
    Remove all rows before start_time_ms and shift time so the new start is 0.
    """
    trimmed = df[df["Time (ms)"] >= start_time_ms].copy()
    trimmed["Time (ms)"] -= start_time_ms
    return trimmed.reset_index(drop=True)


# ── Step 2: Resample to uniform time steps ───────────────────────────────────

def resample_uniform(df, smooth=False):
    """
    Resample a DataFrame to uniform RESAMPLE_MS intervals using linear interpolation.

    CSV data comes in at irregular intervals (~2-36 ms gaps).
    JSON data comes at 1 ms (gets downsampled to RESAMPLE_MS).
    After this, both have the same uniform time grid.

    If smooth=True, applies Savitzky-Golay smoothing to the columns in SMOOTH_COLS.
    """
    time_original = df["Time (ms)"].values

    if len(time_original) < 2:
        return df

    # Create a uniform time grid from start to end
    time_uniform = np.arange(
        time_original[0],
        time_original[-1] + RESAMPLE_MS / 2,  # +half step to include the last point
        RESAMPLE_MS
    )

    resampled = {"Time (ms)": time_uniform}

    for col in df.columns:
        if col == "Time (ms)":
            continue

        # Interpolate this column onto the uniform time grid
        interpolator = interp1d(
            time_original, df[col].values,
            kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        values = interpolator(time_uniform)

        # Optionally smooth (only for columns like Robot_I, not step-like TCP positions)
        if smooth and col in SMOOTH_COLS and len(values) >= SAVGOL_WINDOW:
            values = savgol_filter(values, SAVGOL_WINDOW, SAVGOL_POLY)

        resampled[col] = values

    return pd.DataFrame(resampled)


# ── Pipeline ─────────────────────────────────────────────────────────────────

def preprocess_pair(csv_path, json_path, out_csv, out_json):
    """
    Preprocess one CSV/JSON file pair:
      1. Detect when screwing starts (from JSON Depth)
      2. Trim both files to that start point
      3. Resample both to uniform time steps
      4. Save the results
    """
    # Load both files
    csv_df = pd.read_csv(csv_path)
    json_raw = load_json(json_path)
    json_df = json_to_df(json_raw)

    # Step 1: Find where screwing actually starts
    active_start = detect_active_start(json_df)

    if active_start is None:
        print("  SKIPPED (no active screwing detected)")
        return False

    # Trim both CSV and JSON to start at the screwing event
    if active_start > 0:
        csv_df = trim_to_start(csv_df, active_start)
        json_df = trim_to_start(json_df, active_start)
        print(f"  Idle removed: first {active_start:.0f} ms")

    # Step 2: Resample both to uniform time steps
    csv_df = resample_uniform(csv_df, smooth=SMOOTH_CSV)
    json_df = resample_uniform(json_df, smooth=False)

    # Save
    csv_df.to_csv(out_csv, index=False)
    save_json(df_to_json(json_df, json_raw), out_json)

    # Print summary
    csv_end = csv_df["Time (ms)"].iloc[-1]
    json_end = json_df["Time (ms)"].iloc[-1]
    print(f"  CSV: {csv_end:.0f} ms ({len(csv_df)} pts)  "
          f"JSON: {json_end:.0f} ms ({len(json_df)} pts)")
    return True


def main():
    """Find all CSV/JSON pairs in the configured subfolders and preprocess them."""

    # Resolve which subfolders to process
    if PROCESS_SUBFOLDERS == ["--all"]:
        if not DATA_ROOT.exists():
            sys.exit(f"Error: {DATA_ROOT} does not exist")
        subfolders = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir())
    else:
        subfolders = [DATA_ROOT / name for name in PROCESS_SUBFOLDERS]

    print(f"Resample: {RESAMPLE_MS} ms | Idle threshold: {IDLE_DEPTH_RATE} mm/ms | "
          f"Smooth: {'SavGol' if SMOOTH_CSV else 'OFF'}\n")

    for folder in subfolders:
        if not folder.exists():
            sys.exit(f"Error: {folder} does not exist")

        out_dir = OUTPUT_ROOT / folder.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find file pairs (files that have both a .csv and .json with the same name)
        csv_files = {f.stem: f for f in sorted(folder.glob("*.csv"))}
        json_files = {f.stem: f for f in sorted(folder.glob("*.json"))}
        paired = sorted(csv_files.keys() & json_files.keys())

        print(f"{'='*50}\n  {folder.name} — {len(paired)} pairs\n{'='*50}")

        for base in paired:
            print(f"{base}:")
            preprocess_pair(
                csv_files[base], json_files[base],
                out_dir / f"{base}.csv", out_dir / f"{base}.json"
            )

    print(f"\nDone! Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
