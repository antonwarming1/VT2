"""
test_smooth.py — Visualise Savitzky-Golay smoothing with tunable parameters.

Edit the PARAMETERS block below, then run:
    python Preprocessing/test_smooth.py
Images are saved to Preprocessing/plots/smooth_test/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# ── PARAMETERS — edit these ──────────────────────────────────────────────────
SAVGOL_WINDOW  = 11     # must be odd — larger = more smoothing
SAVGOL_POLY    = 3      # polynomial order — higher = preserves peaks better
RESAMPLE_MS    = 2      # uniform grid spacing before smoothing

SMOOTH_COLS_TASK = ["Robot_I (A)"]
SMOOTH_COLS_INTR = ["Torque (Nm)", "Current (V)"]

LABEL      = "N"        # which class folder to sample from
N_SAMPLES  = 3          # how many files to test
DATA_TYPE  = "intr"     # "task" or "intr"
# ─────────────────────────────────────────────────────────────────────────────

TASK_ROOT = Path(r"C:\github\VT2\data_old_cleaned\Task data")
INTR_ROOT = Path(r"C:\github\VT2\data_old_cleaned\Intrinsic data")
OUT_DIR   = Path(__file__).parent / "plots" / "smooth_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMOOTH_COLS = SMOOTH_COLS_TASK if DATA_TYPE == "task" else SMOOTH_COLS_INTR
DATA_ROOT   = TASK_ROOT if DATA_TYPE == "task" else INTR_ROOT


def resample(df):
    time_orig    = df["Time (ms)"].values
    time_uniform = np.arange(time_orig[0], time_orig[-1] + RESAMPLE_MS / 2, RESAMPLE_MS)
    out = {"Time (ms)": time_uniform}
    for col in df.columns:
        if col == "Time (ms)":
            continue
        interp = interp1d(time_orig, df[col].values, kind="linear",
                          bounds_error=False, fill_value="extrapolate")
        out[col] = interp(time_uniform)
    return pd.DataFrame(out)


def apply_smooth(df):
    out = df.copy()
    for col in SMOOTH_COLS:
        if col in out.columns and len(out) >= SAVGOL_WINDOW:
            out[col] = savgol_filter(out[col].values, SAVGOL_WINDOW, SAVGOL_POLY)
    return out


def plot_smooth(df_before, df_after, title, save_path):
    cols = [c for c in SMOOTH_COLS if c in df_before.columns]
    if not cols:
        print("  No smoothed columns found in file, skipping.")
        return
    n = len(cols)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n), sharex="col")
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=10)
    axes[0][0].set_title("BEFORE smoothing", fontsize=9, color="#3333aa")
    axes[0][1].set_title(f"AFTER  (window={SAVGOL_WINDOW}, poly={SAVGOL_POLY})", fontsize=9, color="#226622")

    for i, col in enumerate(cols):
        t = df_before["Time (ms)"].values

        axes[i][0].plot(t, df_before[col].values, color="#4444cc", linewidth=0.7)
        axes[i][0].set_ylabel(col, fontsize=7)
        axes[i][0].grid(True, alpha=0.25)

        axes[i][1].plot(t, df_after[col].values, color="#228822", linewidth=0.7)
        axes[i][1].set_ylabel(col, fontsize=7)
        axes[i][1].grid(True, alpha=0.25)

    axes[-1][0].set_xlabel("Time (ms)")
    axes[-1][1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Run ───────────────────────────────────────────────────────────────────────
label_dir = DATA_ROOT / LABEL
if not label_dir.exists():
    raise FileNotFoundError(f"Label folder not found: {label_dir}")

files = sorted(label_dir.glob("*.csv"))[:N_SAMPLES]
print(f"Parameters: SAVGOL_WINDOW={SAVGOL_WINDOW}  SAVGOL_POLY={SAVGOL_POLY}  RESAMPLE_MS={RESAMPLE_MS}\n")

for f in files:
    df          = pd.read_csv(f)
    df_resampled = resample(df)
    df_smoothed  = apply_smooth(df_resampled)

    plot_smooth(
        df_resampled, df_smoothed,
        title=f"Smooth test — {LABEL}/{f.name}  "
              f"[window={SAVGOL_WINDOW} poly={SAVGOL_POLY}]",
        save_path=OUT_DIR / f"{f.stem}_smooth.png",
    )

print(f"\nDone. Images in {OUT_DIR}")
