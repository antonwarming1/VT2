# Pipeline Overview — run_pipeline.py

Runs the full preprocessing and feature engineering pipeline end-to-end for the old dataset (Task CSV + Intrinsic CSV + Audio WAV triplets).

## Usage

```
python run_pipeline.py
```

## Steps

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `Preprocessing/data_cleaning.py` | `Data fra tidligere project/Dataset/` | `data_old_cleaned/` |
| 2 | `Preprocessing/data_preprocessing.py` | `data_old_cleaned/` | `data_old_preprocessed/` |
| 3 | `Preprocessing/exclude_features.py` | `data_old_preprocessed/` | `data_opsamling_final/` |
| 4 | `Feature_engineering/code.py` | `data_opsamling_final/` | `Feature_engineering/*.csv` |

## Output files (step 4)

| File | Description |
|------|-------------|
| `Feature_engineering/features_selected.csv` | Selected tsfresh features — task + intrinsic only |
| `Feature_engineering/features_selected_audio.csv` | Selected tsfresh features — task + intrinsic + audio |
| `Feature_engineering/labels.csv` | Class labels aligned to feature rows |

## Notes

- Steps 1–3 are fast (seconds to minutes). Step 4 (tsfresh) takes ~5–10 minutes on 1341 samples.
- Each step can also be run independently by calling its `main()` directly.
- Configure which data to process inside each script's `Config` / top-level variables section.
