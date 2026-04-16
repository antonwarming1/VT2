# data_preprocessing.py

Advanced preprocessing script for screwing cell data. Runs **after** `data_cleaning.py`.

## What it does

| Step | Problem | Solution |
|------|---------|----------|
| **1. Align CSV ↔ JSON** | CSV starts recording ~200–230 ms before JSON. The two sources are not synchronized. | Computes the duration difference and trims the beginning of CSV so both signals start at the same event. |
| **2. Remove idle phase** | The first ~1400–1600 ms of each recording is idle — no screw engagement, flat signals. | Detects active start via the JSON Depth derivative (rolling mean of change rate). Trims both CSV and JSON, shifts time to 0. |
| **3. Resample CSV** | CSV is irregularly sampled (~2–36 ms gaps). JSON is uniform at 1 ms. | Resamples CSV to uniform 2 ms intervals using linear interpolation + optional Savitzky-Golay smoothing. |

## Pipeline

```
data_opsamling_cleaned/    ──►    data_preprocessing.py    ──►    data_opsamling_preprocessed/
  Normal/                                                           Normal/
  Under/                                                            Under/
```

Input: cleaned files from `data_opsamling_cleaned/<subfolder>/` (output of `data_cleaning.py`).  
Output: preprocessed files in `data_opsamling_preprocessed/<subfolder>/`.

## Usage

```bash
python Preprocessing/data_preprocessing.py Normal
python Preprocessing/data_preprocessing.py Under
python Preprocessing/data_preprocessing.py --all
```

## Dependencies

- `pandas`
- `numpy`
- `scipy` (interpolation + Savitzky-Golay filter)

## Parameters

All configurable at the top of the script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CSV_RESAMPLE_MS` | `2` | Uniform sample interval for resampled CSV (ms) |
| `IDLE_DEPTH_RATE` | `0.005` | Depth change rate threshold (mm/ms) to detect active screwing |
| `IDLE_WINDOW` | `50` | Rolling window size (ms) for smoothing the depth derivative |
| `IDLE_MARGIN_MS` | `100` | Extra ms to keep before the detected active start |
| `SMOOTH_CSV` | `True` | Enable/disable Savitzky-Golay smoothing on resampled CSV |
| `SAVGOL_WINDOW` | `11` | Savitzky-Golay filter window length (must be odd) |
| `SAVGOL_POLY` | `3` | Savitzky-Golay polynomial order |

## Data formats

**CSV** (robot data): `Time (ms), TCP_x, TCP_y, TCP_z, TCP_rx, TCP_ry, TCP_rz, Robot_I`  
**JSON** (screwing cell): `Nset, Torque, Current, Angle, Depth` with a shared `Time` X-axis at 1 ms resolution.

## Example output

```
120320261A1:
  Idle removed: cut first 1445 ms (JSON: 3374 → 1929, CSV: → 487 rows)
  Resampled CSV: 487 → 735 rows @ 2 ms (was mean=2.0 ms, std=0.0000 ms)
  Smoothed CSV: Savitzky-Golay (window=11, poly=3)
  Final: CSV 1473 ms (735 pts), JSON 1928 ms (1929 pts)
```

## Idle detection method

The script uses the **Depth** signal derivative from the JSON data:

1. Compute `|diff(Depth)|` — absolute change per ms
2. Apply rolling mean with window of 50 ms
3. First index where rate exceeds 0.005 mm/ms = active start
4. Subtract 100 ms margin to keep a small pre-engagement buffer
5. Fallback: if Depth doesn't trigger, uses `Nset > 200` as secondary indicator
