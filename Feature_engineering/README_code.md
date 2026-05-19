# Feature_engineering/code.py

**Pipeline step 4.** Extracts time-series features with tsfresh and selects the most relevant ones.

## What it does

1. **Loads** all task/intrinsic/audio triplets from `data_opsamling_final/`, matched by `base_id`
2. **Converts** each signal to tsfresh long format (columns: `id`, `time`, signal values)
3. **Extracts** features using `EfficientFCParameters` (~800 features per signal)
4. **Selects** statistically relevant features using tsfresh's `select_features`
5. **Saves** four output files

## Configuration (top of file)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `"old"` | Which dataset format: `"old"` or `"new"` |
| `OLD_DATA_ROOT` | `data_opsamling_final/` | Input folder for old dataset |
| `OLD_LABEL_MAP` | `{N:0, NS:1, OT:2, P:3, UT:4}` | Label name to integer mapping |

## Output files

| File | Description |
|------|-------------|
| `features_extracted.csv` | All extracted features — task + intrinsic (before selection) |
| `features_extracted_audio.csv` | All extracted audio features (before selection) |
| `features_selected.csv` | Statistically selected features — task + intrinsic only |
| `features_selected_audio.csv` | Statistically selected features — task + intrinsic + audio |
| `labels.csv` | Integer class labels aligned to feature rows |

## Sample counts (1341 total)

| Label | Class | Count |
|-------|-------|-------|
| N | Normal | 412 |
| NS | No Screw | 159 |
| OT | Over Tightened | 310 |
| P | No Engage | 152 |
| UT | Under Tightened | 308 |

## Key functions

| Function | Description |
|----------|-------------|
| `build_old_dataset()` | Loads all triplets in label-map order, returns `(task_long, intr_long, extr_long, labels)` |
| `extract_from_long(df, name)` | Runs tsfresh extraction, drops NaN columns |
| `_csv_df_to_long(df, sample_id)` | Converts a DataFrame to tsfresh long format |

## Notes

- Only samples where **all three** files (task, intrinsic, audio) exist are included. Samples missing any one file are skipped.
- Run with `--no-select` to skip feature selection (faster, useful for testing).
- Feature extraction uses `n_jobs=10` — reduce if memory is limited.
