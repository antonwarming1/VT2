"""
data_preprocessing.py
=====================
Advanced preprocessing: aligns CSV/JSON, removes idle phase, resamples CSV.

Runs AFTER data_cleaning.py (uses cleaned data from data_opsamling_cleaned).

Steps performed:
  1. Align CSV and JSON in time
     - CSV starts recording ~200-230 ms before JSON
     - Offset estimated from duration difference (both end at same event)
     - CSV is trimmed so t=0 matches JSON t=0

  2. Remove idle phase
     - Detects when actual screwing begins using JSON Depth derivative
     - Trims both CSV and JSON to start at the active screwing event
     - Both signals shifted so active phase starts at t=0

  3. Resample both CSV and JSON to uniform 2 ms time steps
     - CSV is irregularly sampled (~2-36 ms gaps)
     - JSON is sampled at 1 ms (downsampled to 2 ms)
     - Both resampled via linear interpolation
     - Optional Savitzky-Golay smoothing on CSV to reduce jitter/spikes

Note: CSV and JSON may have different end times after preprocessing.
      JSON typically runs ~450 ms longer (captures the final tightening phase
      with Torque/Current peaks). This is intentional — trimming would remove
      the most discriminative part of the signal.

Output: data_opsamling_preprocessed/<subfolder>/

Usage:
  python data_preprocessing.py Normal
  python data_preprocessing.py Under
  python data_preprocessing.py --all
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"C:\Users\Anton\OneDrive\Dokumenter\GitHub\VT2\data_opsamling_cleaned")
OUTPUT_ROOT = Path(r"C:\Users\Anton\OneDrive\Dokumenter\GitHub\VT2\data_opsamling_preprocessed")

# ── Parameters ───────────────────────────────────────────────────────────────
RESAMPLE_MS = 2              # Uniform sample interval for both CSV and JSON (ms)
IDLE_DEPTH_RATE = 0.005      # Depth change rate threshold (mm/ms) for active detection
IDLE_WINDOW = 50             # Rolling window (ms) for depth rate smoothing
IDLE_MARGIN_MS = 100         # Keep this many ms before detected active start
SMOOTH_CSV = True            # Apply Savitzky-Golay smoothing to CSV
SMOOTH_COLS = ["Robot_I (A)"]  # Only smooth continuous signals (not step-like TCP positions)
SAVGOL_WINDOW = 11           # Savitzky-Golay window length (must be odd)
SAVGOL_POLY = 3              # Savitzky-Golay polynomial order


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f)


def json_to_df(data):
    """Convert JSON screwing data to a DataFrame with Time (ms) column."""
    vectors = data["XML_Data"]["Wsk3Vectors"]
    x_vals = [float(v) for v in vectors["X_Axis"]["Values"]["float"]]
    axes = vectors["Y_AxesList"]["AxisData"]

    result = {"Time (ms)": x_vals}
    for axis in axes:
        name = axis["Header"]["Name"]
        values = [float(v) for v in axis["Values"]["float"]]
        result[name] = values

    return pd.DataFrame(result)


def df_to_json(df, original_data):
    """Write DataFrame values back into the JSON structure."""
    data = json.loads(json.dumps(original_data))  # deep copy
    vectors = data["XML_Data"]["Wsk3Vectors"]

    # Update X axis (Time)
    vectors["X_Axis"]["Values"]["float"] = [str(v) for v in df["Time (ms)"].tolist()]

    # Update Y axes
    axes = vectors["Y_AxesList"]["AxisData"]
    for axis in axes:
        name = axis["Header"]["Name"]
        if name in df.columns:
            axis["Values"]["float"] = [str(v) for v in df[name].tolist()]

    return data


# ── Step 1: Compute alignment offset ────────────────────────────────────────

def compute_offset(csv_df, json_df):
    """
    Estimate the time offset between CSV and JSON recordings.
    Both recordings end at the same screwing event, but CSV starts earlier.
    offset = CSV_duration - JSON_duration (ms).
    """
    csv_duration = csv_df["Time (ms)"].iloc[-1] - csv_df["Time (ms)"].iloc[0]
    json_duration = json_df["Time (ms)"].iloc[-1] - json_df["Time (ms)"].iloc[0]
    offset = csv_duration - json_duration
    return max(0.0, offset)


def align_csv_to_json(csv_df, offset):
    """
    Trim the beginning of CSV by `offset` ms so it starts at the same time as JSON.
    Returns a new DataFrame with adjusted time.
    """
    t0_csv = csv_df["Time (ms)"].iloc[0]
    new_start = t0_csv + offset

    # Keep rows from new_start onwards
    aligned = csv_df[csv_df["Time (ms)"] >= new_start].copy()
    # Shift time so it starts at 0
    aligned["Time (ms)"] = aligned["Time (ms)"] - new_start
    aligned = aligned.reset_index(drop=True)
    return aligned


# ── Step 2: Detect and remove idle phase ────────────────────────────────────

def detect_active_start_json(json_df):
    """
    Detect when the screw actually engages by looking at the Depth derivative.
    Returns the time (ms) when the active screwing phase begins,
    or None if no active phase is detected (data should be discarded).
    """
    if "Depth" not in json_df.columns:
        return None

    depth = json_df["Depth"].values
    time = json_df["Time (ms)"].values

    # Compute smoothed absolute depth rate
    depth_diff = np.abs(np.diff(depth))
    if len(depth_diff) < IDLE_WINDOW:
        return None

    # Rolling mean of depth change rate
    kernel = np.ones(IDLE_WINDOW) / IDLE_WINDOW
    depth_rate = np.convolve(depth_diff, kernel, mode="valid")

    # Find first index where rate exceeds threshold
    active_indices = np.where(depth_rate > IDLE_DEPTH_RATE)[0]
    if len(active_indices) == 0:
        # No active screwing detected — data is unusable
        return None

    active_idx = active_indices[0]
    active_time = time[active_idx]

    # Apply margin: keep some ms before the detected start
    return max(0.0, active_time - IDLE_MARGIN_MS)


def trim_idle(df, start_time_ms):
    """Trim a DataFrame to start at start_time_ms and shift time to 0."""
    trimmed = df[df["Time (ms)"] >= start_time_ms].copy()
    trimmed["Time (ms)"] = trimmed["Time (ms)"] - start_time_ms
    trimmed = trimmed.reset_index(drop=True)
    return trimmed


# ── Step 3: Resample CSV to uniform time steps ─────────────────────────────

def resample_df(df, interval_ms=RESAMPLE_MS, smooth=False, smooth_cols=None):
    """
    Resample a DataFrame to uniform time steps using linear interpolation.
    Optionally apply Savitzky-Golay smoothing to specified columns only.
    """
    time_orig = df["Time (ms)"].values
    value_cols = [c for c in df.columns if c != "Time (ms)"]

    if len(time_orig) < 2:
        return df

    # Create uniform time grid
    t_start = time_orig[0]
    t_end = time_orig[-1]
    time_uniform = np.arange(t_start, t_end + interval_ms / 2, interval_ms)

    result = {"Time (ms)": time_uniform}

    for col in value_cols:
        values = df[col].values

        # Linear interpolation
        interp_func = interp1d(time_orig, values, kind="linear",
                               bounds_error=False, fill_value="extrapolate")
        resampled = interp_func(time_uniform)

        # Optional smoothing (only for specified columns)
        should_smooth = (smooth and len(resampled) >= SAVGOL_WINDOW
                         and (smooth_cols is None or col in smooth_cols))
        if should_smooth:
            resampled = savgol_filter(resampled, SAVGOL_WINDOW, SAVGOL_POLY)

        result[col] = resampled

    return pd.DataFrame(result)


# ── Main pipeline ───────────────────────────────────────────────────────────

def preprocess_pair(csv_path, json_path, out_csv, out_json):
    """
    Full preprocessing pipeline for one CSV/JSON pair.
    Returns list of actions taken.
    """
    actions = []

    # Load data
    csv_df = pd.read_csv(csv_path)
    json_data = load_json(json_path)
    json_df = json_to_df(json_data)

    csv_len_orig = len(csv_df)
    json_len_orig = len(json_df)

    # ── Step 1: Align CSV to JSON ──
    offset = compute_offset(csv_df, json_df)
    if offset > 0:
        csv_df = align_csv_to_json(csv_df, offset)
        actions.append(f"  Aligned: trimmed {offset:.0f} ms from CSV start "
                       f"({csv_len_orig} → {len(csv_df)} rows)")

    # ── Step 2: Remove idle phase ──
    active_start = detect_active_start_json(json_df)
    if active_start is None:
        actions.append("  SKIPPED: no active screwing detected (Depth never rises) — files not saved")
        return actions, False  # signal to caller: do not save
    if active_start > 0:
        json_df = trim_idle(json_df, active_start)
        csv_df = trim_idle(csv_df, active_start)
        actions.append(f"  Idle removed: cut first {active_start:.0f} ms "
                       f"(JSON: {json_len_orig} → {len(json_df)}, "
                       f"CSV: → {len(csv_df)} rows)")

    # ── Step 3: Resample both to uniform 2 ms ──
    csv_len_before = len(csv_df)
    json_len_before = len(json_df)
    csv_df = resample_df(csv_df, smooth=SMOOTH_CSV, smooth_cols=SMOOTH_COLS)
    json_df = resample_df(json_df, smooth=False)
    actions.append(f"  Resampled CSV: {csv_len_before} → {len(csv_df)} rows @ {RESAMPLE_MS} ms")
    actions.append(f"  Resampled JSON: {json_len_before} → {len(json_df)} rows @ {RESAMPLE_MS} ms")
    if SMOOTH_CSV:
        actions.append(f"  Smoothed CSV: Savitzky-Golay (window={SAVGOL_WINDOW}, poly={SAVGOL_POLY})")

    # ── Save ──
    csv_df.to_csv(out_csv, index=False)

    # Write updated JSON back (preserving original structure)
    json_out = df_to_json(json_df, json_data)
    save_json(json_out, out_json)

    # Summary info
    csv_dur = csv_df["Time (ms)"].iloc[-1] if len(csv_df) > 0 else 0
    json_dur = json_df["Time (ms)"].iloc[-1] if len(json_df) > 0 else 0
    actions.append(f"  Final: CSV {csv_dur:.0f} ms ({len(csv_df)} pts), "
                   f"JSON {json_dur:.0f} ms ({len(json_df)} pts)")

    return actions, True


def preprocess_subfolder(data_dir, output_dir):
    """Preprocess all paired CSV/JSON files in a subfolder."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all base names that have both CSV and JSON
    csv_files = {f.stem: f for f in sorted(data_dir.glob("*.csv"))}
    json_files = {f.stem: f for f in sorted(data_dir.glob("*.json"))}
    paired = sorted(set(csv_files.keys()) & set(json_files.keys()))
    unpaired_csv = sorted(set(csv_files.keys()) - set(json_files.keys()))
    unpaired_json = sorted(set(json_files.keys()) - set(csv_files.keys()))

    print(f"Found {len(paired)} paired files, "
          f"{len(unpaired_csv)} CSV-only, {len(unpaired_json)} JSON-only")
    print(f"  Source: {data_dir}")
    print(f"  Output: {output_dir}\n")

    if unpaired_csv:
        print(f"  WARNING: CSV without JSON: {unpaired_csv}")
    if unpaired_json:
        print(f"  WARNING: JSON without CSV: {unpaired_json}")

    total_actions = 0
    skipped = 0

    for base in paired:
        csv_path = csv_files[base]
        json_path = json_files[base]
        out_csv = output_dir / csv_path.name
        out_json = output_dir / json_path.name

        actions, saved = preprocess_pair(csv_path, json_path, out_csv, out_json)
        print(f"{base}:")
        for a in actions:
            print(a)
        total_actions += len(actions)
        if not saved:
            skipped += 1

    if skipped:
        print(f"\n  {skipped} file pair(s) skipped (no active screwing detected)")

    # Copy unpaired files as-is (just cleaned, no alignment possible)
    for base in unpaired_csv:
        src = csv_files[base]
        dst = output_dir / src.name
        pd.read_csv(src).to_csv(dst, index=False)
        print(f"{base}.csv: copied (no JSON pair)")

    for base in unpaired_json:
        src = json_files[base]
        dst = output_dir / src.name
        save_json(load_json(src), dst)
        print(f"{base}.json: copied (no CSV pair)")

    return total_actions


def main():
    if len(sys.argv) < 2:
        print("Usage: python data_preprocessing.py <subfolder>")
        print("       python data_preprocessing.py --all")
        print("Example: python data_preprocessing.py Normal")
        available = [d.name for d in DATA_ROOT.iterdir() if d.is_dir()] if DATA_ROOT.exists() else []
        print(f"\nAvailable subfolders: {available}")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--all":
        subfolders = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    else:
        subfolders = [DATA_ROOT / arg]

    print(f"Resample interval: {RESAMPLE_MS} ms (both CSV and JSON)")
    print(f"Idle detection: depth rate > {IDLE_DEPTH_RATE} mm/ms "
          f"(window={IDLE_WINDOW} ms, margin={IDLE_MARGIN_MS} ms)")
    print(f"Smoothing: {'Savitzky-Golay' if SMOOTH_CSV else 'OFF'}")

    grand_total = 0
    for data_dir in subfolders:
        if not data_dir.exists():
            print(f"Error: {data_dir} does not exist")
            sys.exit(1)
        output_dir = OUTPUT_ROOT / data_dir.name
        print(f"\n{'='*60}")
        print(f"  {data_dir.name}")
        print(f"{'='*60}")
        grand_total += preprocess_subfolder(data_dir, output_dir)

    print(f"\nDone! {grand_total} total steps applied.")
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
