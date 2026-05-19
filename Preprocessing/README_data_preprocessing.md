# data_preprocessing.py

**Pipeline step 2.** Removes the idle phase, resamples to uniform time steps, and smooths signals.

## What it does

For each task/intrinsic/audio triplet (matched by `base_id`):

1. **Detect plateau** — finds the time in the Intrinsic `Depth (mm)` signal where depth stops increasing (screwing is done)
2. **Trim** — removes all data before the plateau time from all three signals, shifts time to start at 0
3. **Resample** — interpolates task and intrinsic to uniform 2 ms intervals
4. **Smooth** — applies Savitzky-Golay filter to `Robot_I (A)`, `Torque (Nm)`, `Current (V)`

Audio is trimmed at the same plateau time but is NOT resampled (already uniform at 2200 Hz).

## Configuration (top of file)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLD_OR_NEW_DATA` | `["old"]` | Which dataset to process |
| `RESAMPLE_MS` | `2` | Target sample interval after resampling |
| `IDLE_DEPTH_RATE` | `0.005` | Depth rate threshold (mm/ms) for plateau detection |
| `SMOOTH_CSV` | `True` | Apply Savitzky-Golay to task signals |
| `SMOOTH_INTR` | `True` | Apply Savitzky-Golay to intrinsic signals |
| `SAVGOL_WINDOW` | `11` | Savitzky-Golay window size (must be odd) |
| `SAVGOL_POLY` | `3` | Savitzky-Golay polynomial order |

## Paths

- Input:  `data_old_cleaned/`
- Output: `data_old_preprocessed/`

## Key functions

| Function | Description |
|----------|-------------|
| `detect_plateau(df, depth_col)` | Returns time (ms) when depth stops increasing, or `None` |
| `trim_to_start(df, start_time_ms)` | Removes rows before `start_time_ms`, shifts time to 0 |
| `resample_uniform(df, smooth, smooth_cols)` | Interpolates to uniform grid, optionally smooths |
| `lowpass_filter(data, samplerate, highcut, order=6)` | Butterworth lowpass filter |
| `preprocess_old_pair_df(task_df, intr_df, sound_df)` | In-memory version — used by inference pipeline |

## Note on `samplerate.txt`

This file is read at **module import time** from `data_old_cleaned/Extrinsic data/samplerate.txt`.  
`data_cleaning.py` must have been run at least once before importing this module.
