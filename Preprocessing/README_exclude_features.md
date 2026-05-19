# exclude_features.py

**Pipeline step 3.** Drops columns that should not be used as model features.

## What it does

Reads all CSV and JSON files from `data_old_preprocessed/`, removes the columns listed in `EXCLUDE_CSV` / `EXCLUDE_JSON`, and saves the result.

## Columns removed

| Column | Reason |
|--------|--------|
| `TCP_x/y/z (mm)` | Robot position — not informative for fault detection |
| `TCP_rx/ry/rz (rad)` | Robot orientation — not informative |
| `Nset (1/min)` | Commanded speed — constant, adds no information |
| `Angle (deg)` | Redundant with Depth |
| `Depth (mm)` | Used only for plateau detection, not a fault feature |

## Configuration (top of file)

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_ROOT` | `data_old_preprocessed/` | Where to read from |
| `OUTPUT_ROOT` | `data_opsamling_final/` | Where to write to |
| `PROCESS_SUBFOLDERS` | `["--all"]` | Which label subfolders to process |

## Notes

- Audio files (`e*.csv`) pass through unchanged — they only contain `Time (ms)` and `Amplitude`, so nothing is dropped.
- The `Extrinsic data` subfolder is explicitly skipped in the folder iteration since audio does not need column filtering.

## In-memory helper (used by inference pipeline)

| Function | Description |
|----------|-------------|
| `drop_csv_columns_df(df)` | Drop `EXCLUDE_CSV` columns from a DataFrame in memory |
