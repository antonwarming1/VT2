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

DATA_ROOT   = Path(__file__).parent.parent / "data_opsamling_cleaned"
OUTPUT_ROOT = Path(__file__).parent.parent / "data_opsamling_preprocessed"

OUTPUT_ROOT.mkdir(exist_ok=True)

 
# Old dataset (from earlier project)
OLD_DATA_ROOT = Path(__file__).parent.parent / "data_old_cleaned"
OLD_OUTPUT_ROOT = Path(__file__).parent.parent / "data_old_preprocessed"

OLD_OUTPUT_ROOT.mkdir(exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
OLD_OR_NEW_DATA = ["old"]       # ["old"], ["new"], or ["old", "new"]
FOLDERS_OLD = ["Intrinsic data", "Task data", "Extrinsic data"]  # For old data: which subfolders to include

RESAMPLE_MS    = 2       # Target sample interval (ms)
IDLE_DEPTH_RATE = 0.005  # Depth rate threshold (mm/ms) to detect screwing start
IDLE_WINDOW    = 50      # Rolling window size (ms) for smoothing the depth rate
IDLE_MARGIN_MS = 0       # Keep this many ms before detected screwing start

SMOOTH_CSV     = True            # Apply Savitzky-Golay smoothing to CSV columns
SMOOTH_COLS    = ["Robot_I (A)"] # Which Task/CSV columns to smooth
SMOOTH_INTR    = True            # Apply Savitzky-Golay smoothing to Intrinsic/JSON columns
SMOOTH_COLS_INTR = ["Torque (Nm)", "Current (V)"]  # Which Intrinsic/JSON columns to smooth
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
    # show dataframe columns and last 3 values for each column
    df = pd.DataFrame(columns)
    print("JSON DataFrame columns and last 3 values:")
    print(df.columns.tolist()) # print column names
    print(df.head(3))
    print(df.tail(3))  # last 3 rows

    return df


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

def _smooth_depth_rate(df, depth_col="Depth"):
    """Compute smoothed depth change rate.
    Works for both new data (depth_col='Depth') and old data (depth_col='Depth (mm)').
    Returns (time, smoothed_rate, above_threshold) or None if not enough data."""
    if depth_col not in df.columns:
        return None
    depth = df[depth_col].values
    time = df["Time (ms)"].values
    depth_change = np.abs(np.diff(depth))
    if len(depth_change) < IDLE_WINDOW:
        return None
    kernel = np.ones(IDLE_WINDOW) / IDLE_WINDOW
    smoothed_rate = np.convolve(depth_change, kernel, mode="valid")
    above_threshold = np.where(smoothed_rate > IDLE_DEPTH_RATE)[0]
    if len(above_threshold) == 0:
        print("  SKIPPED (depth rate never exceeds threshold — no active screwing)")
        return None
    return time, smoothed_rate, above_threshold


def detect_active_start(df, depth_col="Depth"):
    """
    Find the time (ms) when the screw starts engaging.

    How it works:
      - Compute the absolute change in Depth between each sample
      - Smooth it with a rolling average (IDLE_WINDOW samples)
      - Find the first moment the smoothed rate exceeds IDLE_DEPTH_RATE
      - Subtract IDLE_MARGIN_MS to keep a small buffer before that point

    Returns None if no active screwing is detected (file should be skipped).
    """
    result = _smooth_depth_rate(df, depth_col)
    if result is None:
        return None
    time, _, above_threshold = result
    return max(0.0, time[above_threshold[0]] - IDLE_MARGIN_MS)


def detect_plateau(df, depth_col="Depth"):
    """
    Find the time (ms) when the depth plateaus after rising.

    Requires the depth rate to stay above threshold for at least
    MIN_ACTIVE_SAMPLES consecutive samples before looking for the
    plateau drop-off.

    Returns None if no plateau is detected.
    """
    result = _smooth_depth_rate(df, depth_col)
    if result is None:
        return None
    time, smoothed_rate, above_threshold = result

    # Find the end of the first sustained active region
    MIN_ACTIVE_SAMPLES = 10
    run_start = above_threshold[0]
    run_end = run_start
    for i in range(1, len(above_threshold)):
        if above_threshold[i] == above_threshold[i - 1] + 1:
            run_end = above_threshold[i]
        else:
            if run_end - run_start >= MIN_ACTIVE_SAMPLES:
                break
            run_start = above_threshold[i]
            run_end = run_start

    if run_end - run_start < MIN_ACTIVE_SAMPLES:
        return None

    search_from = run_end
    below_after = np.where(smoothed_rate[search_from:] <= IDLE_DEPTH_RATE)[0]
    if len(below_after) == 0:
        return None
    return time[search_from + below_after[0]]

def trim_to_start(df, start_time_ms):
    """
    Remove all rows before start_time_ms and shift time so the new start is 0.
    """
    trimmed = df[df["Time (ms)"] >= start_time_ms].copy()
    trimmed["Time (ms)"] -= start_time_ms
    return trimmed.reset_index(drop=True)


# ── Step 2: Resample to uniform time steps ───────────────────────────────────

def resample_uniform(df, smooth=False, smooth_cols=None):
    """
    Resample a DataFrame to uniform RESAMPLE_MS intervals using linear interpolation.

    CSV data comes in at irregular intervals (~2-36 ms gaps).
    JSON data comes at 1 ms (gets downsampled to RESAMPLE_MS).
    After this, both have the same uniform time grid.

    If smooth=True, applies Savitzky-Golay smoothing to the columns in smooth_cols.
    """
    if smooth_cols is None:
        smooth_cols = SMOOTH_COLS
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
        if smooth and col in smooth_cols and len(values) >= SAVGOL_WINDOW:
            values = savgol_filter(values, SAVGOL_WINDOW, SAVGOL_POLY)

        resampled[col] = values

    return pd.DataFrame(resampled)


# ── Pipeline ─────────────────────────────────────────────────────────────────

def preprocess_pair(csv_path, json_path, out_csv, out_json):
    """
    Preprocess one CSV/JSON file pair (new dataset):
      1. Detect when screwing starts (from JSON Depth)
      2. Trim both files to that start point
      3. Resample both to uniform time steps
      4. Save the results
    """
    # Load both files
    csv_df = pd.read_csv(csv_path)
    json_raw = load_json(json_path)
    json_df = json_to_df(json_raw)

    # Step 1: Find where depth starts increasing and trim everything before it
    active_start = detect_active_start(json_df)

    if active_start is None:
        print("  SKIPPED (no active screwing detected)")
        return False

    csv_df = trim_to_start(csv_df, active_start)
    json_df = trim_to_start(json_df, active_start)
    print(f"  Trimmed at depth increase: first {active_start:.0f} ms removed")

    # Step 2: Resample both to uniform time steps
    csv_df = resample_uniform(csv_df, smooth=SMOOTH_CSV)
    json_df = resample_uniform(json_df, smooth=SMOOTH_INTR, smooth_cols=SMOOTH_COLS_INTR)

    # Save
    csv_df.to_csv(out_csv, index=False)
    save_json(df_to_json(json_df, json_raw), out_json)

    # Print summary
    csv_end = csv_df["Time (ms)"].iloc[-1]
    json_end = json_df["Time (ms)"].iloc[-1]
    print(f"  CSV: {csv_end:.0f} ms ({len(csv_df)} pts)  "
          f"JSON: {json_end:.0f} ms ({len(json_df)} pts)")
    return True





def preprocess_old_pair(task_csv_path, intrinsic_csv_path, out_task, out_intrinsic):
    """
    Preprocess one old-dataset pair (Task CSV + Intrinsic CSV):
      1. Detect when screwing starts from Intrinsic Depth (mm)
      2. Trim both files
      3. Resample both to uniform time steps
      4. Save the results
    """
    task_df = pd.read_csv(task_csv_path)
    intr_df = pd.read_csv(intrinsic_csv_path)

    # Detect plateau from Intrinsic Depth (mm)
    plateau_time = detect_plateau(intr_df, depth_col="Depth (mm)")

    if plateau_time is None:
        print("  SKIPPED (no depth activity or plateau detected)")
        return False

    # Step 1: Trim everything before the depth plateau, keep the tail
    task_df = trim_to_start(task_df, plateau_time)
    intr_df = trim_to_start(intr_df, plateau_time)
    print(f"  Trimmed at depth plateau: first {plateau_time:.0f} ms removed")

    task_df = resample_uniform(task_df, smooth=SMOOTH_CSV)
    intr_df = resample_uniform(intr_df, smooth=SMOOTH_INTR, smooth_cols=SMOOTH_COLS_INTR)

    task_df.to_csv(out_task, index=False)
    intr_df.to_csv(out_intrinsic, index=False)

    task_end = task_df["Time (ms)"].iloc[-1]
    intr_end = intr_df["Time (ms)"].iloc[-1]
    print(f"  Task: {task_end:.0f} ms ({len(task_df)} pts)  "
          f"Intrinsic: {intr_end:.0f} ms ({len(intr_df)} pts)")
    return True


def preprocess_old_pair_df(task_df, intr_df):
    """
    In-memory version of preprocess_old_pair. Returns (task_df, intr_df) or None
    if no depth plateau detected (instance should be skipped).
    """
    plateau_time = detect_plateau(intr_df, depth_col="Depth (mm)")
    if plateau_time is None:
        return None

    task_df = trim_to_start(task_df, plateau_time)
    intr_df = trim_to_start(intr_df, plateau_time)

    task_df = resample_uniform(task_df, smooth=SMOOTH_CSV)
    intr_df = resample_uniform(intr_df, smooth=SMOOTH_INTR, smooth_cols=SMOOTH_COLS_INTR)

    return task_df, intr_df


def main():
    """Find all CSV/JSON pairs in the configured subfolders and preprocess them."""

    print(f"Resample: {RESAMPLE_MS} ms | Idle threshold: {IDLE_DEPTH_RATE} mm/ms | "
          f"Smooth: {'SavGol' if SMOOTH_CSV else 'OFF'}\n")

    # ── Process new dataset ──
    if "new" in OLD_OR_NEW_DATA:
        if not DATA_ROOT.exists():
            sys.exit(f"Error: {DATA_ROOT} does not exist")
        subfolders = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir())

        for folder in subfolders:
            if not folder.exists():
                sys.exit(f"Error: {folder} does not exist")

            out_dir = OUTPUT_ROOT / folder.name
            out_dir.mkdir(parents=True, exist_ok=True)

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

    # ── Process old dataset ──
    if "old" in OLD_OR_NEW_DATA:
        print(f"\n{'#'*50}")
        print(f"  OLD DATASET")
        print(f"{'#'*50}")
        intrinsic_root = OLD_DATA_ROOT / "Intrinsic data"
        task_root = OLD_DATA_ROOT / "Task data"
        if not intrinsic_root.exists() or not task_root.exists():
            print("  Old dataset not found, skipping.")
        else:
            for label_dir in sorted(intrinsic_root.iterdir()):
                if not label_dir.is_dir():
                    continue
                label = label_dir.name
                task_label_dir = task_root / label
                if not task_label_dir.exists():
                    print(f"  {label}: no matching Task data folder, skipping.")
                    continue

                out_dir = OLD_OUTPUT_ROOT / label
                out_dir.mkdir(parents=True, exist_ok=True)

                intr_files = {f.stem[1:]: f for f in sorted(label_dir.glob("*.csv"))}
                task_files = {f.stem[1:]: f for f in sorted(task_label_dir.glob("*.csv"))}
                paired = sorted(intr_files.keys() & task_files.keys())

                print(f"\n{'='*50}\n  {label} — {len(paired)} pairs\n{'='*50}")

                for base_id in paired:
                    print(f"i{base_id} / t{base_id}:")
                    preprocess_old_pair(
                        task_files[base_id], intr_files[base_id],
                        out_dir / f"t{base_id}.csv", out_dir / f"i{base_id}.csv"
                    )

    print(f"\nDone! Output:", end="")
    if "new" in OLD_OR_NEW_DATA:
        print(f" {OUTPUT_ROOT}", end="")
    if "old" in OLD_OR_NEW_DATA:
        print(f" {OLD_OUTPUT_ROOT}", end="")
    print()

    # ── Visualize first instance after preprocessing ──
    from Preprocessing.visualize_preprocessing import visualize_old_first_instance
    if "old" in OLD_OR_NEW_DATA and OLD_OUTPUT_ROOT.exists():
        print("\n── Visualizing first old-dataset sample ──")
        visualize_old_first_instance(label="N", preprocessed_dir=OLD_OUTPUT_ROOT)


if __name__ == "__main__":
    main()
