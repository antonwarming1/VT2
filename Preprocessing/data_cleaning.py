"""
data_cleaning.py
================
Cleans CSV and JSON data files in C:\\github\\VT2\\data_opsamling\\<subfolder>:

1. CSV: Shifts time to start at 0 (for files where it doesn't)
2. CSV: Checks for and reports NaN values
3. JSON: Clips negative Torque values to 0
4. JSON: Clips negative Current values to 0
5. JSON: Checks for and reports NaN values
6. JSON: Fixes Angle unit encoding (Â° -> °)
7. WAV: Checks for and reports NaN values in audio, replaces with 0
8. WAV: Saves cleaned audio as CSV with Time (ms) and Amplitude columns

Cleaned files are saved to C:\\github\\VT2\\data_opsamling_cleaned\\<subfolder>.
Run with: python data_cleaning.py <subfolder>
Example:  python data_cleaning.py Normal
         python data_cleaning.py Under
         python data_cleaning.py --all   (cleans all subfolders)
"""

import pandas as pd
import json
import shutil
import sys
from pathlib import Path
import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt
import noisereduce as nr

DATA_ROOT = Path(__file__).parent.parent / "data_opsamling"
OUTPUT_ROOT = Path(__file__).parent.parent / "data_opsamling_cleaned"

# Old dataset (from earlier project)
OLD_DATA_ROOT = Path(__file__).parent.parent / "Data fra tidligere project" / "Dataset"
OLD_OUTPUT_ROOT = Path(__file__).parent.parent / "data_old_cleaned"

# ── Config ───────────────────────────────────────────────────────────────────
OLD_OR_NEW_DATA = ["old"]       # ["old"], ["new"], or ["old", "new"] Old data is from earlier project, different structure and signals than new data
PROCESS_SUBFOLDERS = ["--all"]         # For new data: ["Normal"], ["Normal","Under"], or ["--all"]
FOLDERS_OLD = ["Extrinsic data", "Intrinsic data", "Task data"]  # For old data: which subfolders to include "Intrinsic data", "Task data", "Extrinsic data"
SAMPLERATE = 2200  # Sample rate to use when loading WAV files. Set to None to keep original sample rate.

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_wav(filepath, sr=None):
    """Load a WAV file with librosa, returning the audio time series and sample rate."""
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    return y, sr


def clean_task_df(df):
    """In-memory clean for a Task CSV DataFrame: drop NaN rows, shift time to start at 0."""
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    time_col = "Time (ms)"
    if time_col in df.columns:
        t0 = df[time_col].iloc[0]
        if t0 != 0.0:
            df = df.copy()
            df[time_col] = df[time_col] - t0
    return df.reset_index(drop=True)


def clean_intrinsic_df(df):
    """In-memory clean for an Intrinsic CSV DataFrame: drop NaN, clip negative Torque/Current."""
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    df = df.copy()
    if "Torque (Nm)" in df.columns:
        df["Torque (Nm)"] = df["Torque (Nm)"].clip(lower=0)
    if "Current (V)" in df.columns:
        df["Current (V)"] = df["Current (V)"].clip(lower=0)
    return df.reset_index(drop=True)


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


def clean_intrinsic_csv(filepath, output_path):
    """Clean a single Intrinsic CSV file (old dataset). Same signals as JSON."""
    actions = []
    df = pd.read_csv(filepath)

    # Check for NaN
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        actions.append(f"  WARNING: {nan_count} NaN values — dropped rows")
        df = df.dropna()

    # Shift time to start at 0
    """Note: Some Intrinsic CSV files already start at 0, but some don't. We want to ensure all start at 0 for consistency.
    time_col = "Time (ms)"
    if time_col in df.columns:
        t0 = df[time_col].iloc[0]
        if t0 != 0.0:
            df[time_col] = df[time_col] - t0
            actions.append(f"  Time shifted by -{t0:.3f} ms")
            """

    # Clip negative Torque to 0
    if "Torque (Nm)" in df.columns:
        neg = (df["Torque (Nm)"] < 0).sum()
        if neg > 0:
            df["Torque (Nm)"] = df["Torque (Nm)"].clip(lower=0)
            actions.append(f"  Torque: clipped {neg} negative values to 0")

    # Clip negative Current to 0
    if "Current (V)" in df.columns:
        neg = (df["Current (V)"] < 0).sum()
        if neg > 0:
            df["Current (V)"] = df["Current (V)"].clip(lower=0)
            actions.append(f"  Current: clipped {neg} negative values to 0")

    df.to_csv(output_path, index=False)
    return actions


def clean_subfolder(data_dir, output_dir):
    """Clean all CSV and JSON files in a single subfolder."""
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

    return total_actions

def lowpass_filter(data, samplerate, highcut, order=6):
    nyquist = samplerate / 2
    high = highcut / nyquist

    sos = butter(order, high, btype="lowpass", output="sos")
    filtered = sosfiltfilt(sos, data, axis=0)
    return filtered

def resample_audio_df(df, target_sr=2200, original_sr = 44100):
    """Resample audio df to 2200 hz.""" 
      # Assuming original is always 44100 Hz
    if target_sr == original_sr:
        return df
    
    # Resample using librosa
    y = df["Amplitude"].values
    y_resampled = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)
    time_resampled = np.arange(len(y_resampled)) / target_sr * 1000
    
    
    return pd.DataFrame({"Time (ms)": time_resampled, "Amplitude": y_resampled})


def clean_wav(filepath, samplerate=None):
    """Clean a single WAV file. Returns list of actions taken."""
    y, sr = load_wav(filepath, sr=None)
    df = pd.DataFrame({"Time (ms)": np.arange(len(y)) / sr * 1000, "Amplitude": y})

    df['Amplitude'] = lowpass_filter(df['Amplitude'], samplerate=sr, highcut=1000, order=6)
    df = resample_audio_df(df, target_sr=samplerate, original_sr = sr)
   
    # Check for NaN
    nan_count = df["Amplitude"].isnull().sum()
    if nan_count > 0:
        df["Amplitude"] = df["Amplitude"].fillna(0)


    return df

def preprocess_audio(df, out_path, samplerate, Y_NOISE=None):
    """Read csv, apply noise reduction and lowpass filter, save cleaned csv."""
    
    samplerate = int(samplerate)
    if samplerate <= 0:
        raise ValueError(f"Invalid samplerate: {samplerate}")
    
    if "Amplitude" not in df.columns:
        raise ValueError(f"'Amplitude' column not found in dataframe")

    df["Amplitude"] = nr.reduce_noise(
        y = df["Amplitude"].to_numpy(dtype=np.float32),
        sr = samplerate,
        y_noise = Y_NOISE,
        freq_mask_smooth_hz=100,
        time_mask_smooth_ms=128,
        prop_decrease = 0.8,
        stationary = True
        )
    

    df.to_csv(out_path, index=False)
    print(f"Processed audio saved to {out_path}")

y_noise, sr_noise = librosa.load(Path(__file__).parent.parent / "soundcleaning" / "Optaget_støj.wav", sr = SAMPLERATE, mono=True)
y_noise = lowpass_filter(y_noise, samplerate=sr_noise, highcut=1000, order=6)
Y_NOISE = librosa.resample(y_noise, orig_sr=sr_noise, target_sr=SAMPLERATE)

def main():


    # Resolve subfolders from config
    if PROCESS_SUBFOLDERS == ["--all"]:
        if not DATA_ROOT.exists():
            print(f"Error: {DATA_ROOT} does not exist")
            sys.exit(1)
        subfolders = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    else:
        subfolders = [DATA_ROOT / name for name in PROCESS_SUBFOLDERS]

    grand_total = 0

    # ── Process new dataset ──
    if "new" in OLD_OR_NEW_DATA:
        for data_dir in subfolders:
            if not data_dir.exists():
                print(f"Error: {data_dir} does not exist")
                sys.exit(1)
            output_dir = OUTPUT_ROOT / data_dir.name
            print(f"\n{'='*60}")
            print(f"  {data_dir.name}")
            print(f"{'='*60}")
            grand_total += clean_subfolder(data_dir, output_dir)

    # ── Process old dataset ──
    if "old" in OLD_OR_NEW_DATA:
        print(f"\n{'#'*60}")
        print(f"  OLD DATASET")
        print(f"{'#'*60}")
        for folder_name in FOLDERS_OLD:
            src_root = OLD_DATA_ROOT / folder_name
            if not src_root.exists():
                print(f"  Skipping {folder_name} (not found)")
                continue
            for label_dir in sorted(src_root.iterdir()):
                if not label_dir.is_dir():
                    continue
                out_dir = OLD_OUTPUT_ROOT / folder_name / label_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                csv_files = sorted(label_dir.glob("*.csv"))
                wav_files = sorted(label_dir.glob("*.wav"))
                print(f"\n--- {folder_name}/{label_dir.name} ({len(csv_files)} csv, {len(wav_files)} wav) ---")
                for f in csv_files:
                    out = out_dir / f.name
                    if folder_name == "Intrinsic data":
                        actions = clean_intrinsic_csv(f, out)
                    else:
                        actions = clean_csv(f, out)
                    if actions:
                        print(f"{f.name}:")
                        for a in actions:
                            print(a)
                        grand_total += len(actions)
                    else:
                        print(f"{f.name}: OK")
                for f in wav_files:
                    out = out_dir / f.name.replace(".wav", ".csv")
                    df = clean_wav(f, samplerate=SAMPLERATE)
                    preprocess_audio(df, out, samplerate=SAMPLERATE, Y_NOISE=Y_NOISE)
                    
        Path(OLD_OUTPUT_ROOT / "Extrinsic data" / "samplerate.txt").write_text(f"{'44100' if SAMPLERATE is None else SAMPLERATE}")
                    
                

    print(f"\nDone! {grand_total} total fixes applied.")


if __name__ == "__main__":
    main()
