# data_cleaning.py

**Pipeline step 1.** Reads raw files from the dataset and outputs cleaned versions.

## What it does

| File type | Actions |
|-----------|---------|
| Task CSV | Shift time to start at 0, drop NaN rows |
| Intrinsic CSV | Drop NaN rows, clip negative Torque and Current to 0 |
| Audio WAV | Load at target samplerate, replace NaN with 0, save as CSV with `Time (ms)` and `Amplitude` columns |

## Configuration (top of file)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLD_OR_NEW_DATA` | `["old"]` | Which dataset to process: `"old"`, `"new"`, or both |
| `FOLDERS_OLD` | `["Extrinsic data", "Intrinsic data", "Task data"]` | Which subfolders of the old dataset to clean |
| `SAMPLERATE` | `2200` | Sample rate for loading WAV files. `None` keeps original |

## Paths

- Input:  `Data fra tidligere project/Dataset/`
- Output: `data_old_cleaned/`
- Also writes `data_old_cleaned/Extrinsic data/samplerate.txt` — read by `data_preprocessing.py` at import time

## In-memory helpers (used by inference pipeline)

| Function | Description |
|----------|-------------|
| `clean_task_df(df)` | Drop NaN, shift time to 0 |
| `clean_intrinsic_df(df)` | Drop NaN, clip negative Torque/Current |
| `load_wav(filepath, sr)` | Load WAV with librosa at given sample rate |
