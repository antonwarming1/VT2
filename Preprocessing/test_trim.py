"""
test_trim.py — Visualise depth-plateau trimming with tunable parameters.

Edit the PARAMETERS block below, then run:
    python Preprocessing/test_trim.py
Images are saved to Preprocessing/plots/trim_test/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── PARAMETERS — edit these ──────────────────────────────────────────────────
IDLE_DEPTH_RATE   = 0.01   # mm/ms — depth-rate threshold to detect active screwing
IDLE_WINDOW       = 50      # samples — rolling window for smoothing depth rate
MIN_ACTIVE_SAMPLES = 10     # consecutive samples above threshold before looking for plateau

LABEL       = "N"           # which class folder to sample from
N_SAMPLES   = 100           # how many files to test
DEPTH_ONLY  = True          # True = single depth plot with red line / False = all columns before+after
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT  = Path(r"C:\github\VT2\data_old_cleaned\Intrinsic data")
OUT_DIR    = Path(__file__).parent / "plots" / "trim_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_plateau(df):
    depth  = df["Depth (mm)"].values
    time   = df["Time (ms)"].values
    rate   = np.abs(np.diff(depth))

    if len(rate) < IDLE_WINDOW:
        return None

    kernel       = np.ones(IDLE_WINDOW) / IDLE_WINDOW
    smoothed     = np.convolve(rate, kernel, mode="valid")
    above        = np.where(smoothed > IDLE_DEPTH_RATE)[0]

    if len(above) == 0:
        return None

    # Require MIN_ACTIVE_SAMPLES consecutive samples above threshold
    run_start = above[0]
    run_end   = run_start
    for i in range(1, len(above)):
        if above[i] == above[i - 1] + 1:
            run_end = above[i]
        else:
            if run_end - run_start >= MIN_ACTIVE_SAMPLES:
                break
            run_start = above[i]
            run_end   = run_start

    if run_end - run_start < MIN_ACTIVE_SAMPLES:
        return None

    below_after = np.where(smoothed[run_end:] <= IDLE_DEPTH_RATE)[0]
    if len(below_after) == 0:
        return None

    return time[run_end + below_after[0]]


def trim_to_start(df, t):
    out = df[df["Time (ms)"] >= t].copy()
    out["Time (ms)"] -= t
    return out.reset_index(drop=True)


def plot_trim(df_raw, df_trim, plateau_ms, title, save_path):
    if DEPTH_ONLY:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_raw["Time (ms)"], df_raw["Depth (mm)"], color="#4444cc", linewidth=0.8)
        ax.axvline(plateau_ms, color="red", linestyle="--", linewidth=1.2, label=f"Plateau @ {plateau_ms:.0f} ms")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Depth (mm)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    else:
        cols = [c for c in df_raw.columns if c != "Time (ms)"]
        n    = len(cols)
        fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n), sharex="col")
        if n == 1:
            axes = [axes]
        fig.suptitle(title, fontsize=10)
        axes[0][0].set_title(f"BEFORE  (plateau @ {plateau_ms:.0f} ms)", fontsize=9, color="#3333aa")
        axes[0][1].set_title("AFTER  (from plateau onward)", fontsize=9, color="#226622")
        for i, col in enumerate(cols):
            axes[i][0].plot(df_raw["Time (ms)"], df_raw[col], color="#4444cc", linewidth=0.7)
            axes[i][0].axvline(plateau_ms, color="red", linestyle="--", linewidth=0.9)
            axes[i][0].set_ylabel(col, fontsize=7)
            axes[i][0].grid(True, alpha=0.25)
            axes[i][1].plot(df_trim["Time (ms)"], df_trim[col], color="#228822", linewidth=0.7)
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
print(f"Parameters: IDLE_DEPTH_RATE={IDLE_DEPTH_RATE}  IDLE_WINDOW={IDLE_WINDOW}  MIN_ACTIVE_SAMPLES={MIN_ACTIVE_SAMPLES}\n")

for f in files:
    df = pd.read_csv(f)
    plateau = detect_plateau(df)
    if plateau is None:
        print(f"{f.name}: NO PLATEAU DETECTED")
        continue
    print(f"{f.name}: plateau @ {plateau:.0f} ms")
    trimmed = trim_to_start(df, plateau)
    plot_trim(
        df, trimmed, plateau,
        title=f"Trim test — {LABEL}/{f.name}  "
              f"[rate={IDLE_DEPTH_RATE} win={IDLE_WINDOW} min_act={MIN_ACTIVE_SAMPLES}]",
        save_path=OUT_DIR / f"{f.stem}_trim.png",
    )

print(f"\nDone. Images in {OUT_DIR}")
