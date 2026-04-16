"""
Visualize before/after depth-based idle trimming for CSV files
in 'Data fra tidligere project/Dataset/Intrinsic data'.

Shows one file per subfolder (N, NS, OT, P, UT) with all signal channels
side by side: raw vs trimmed.

Usage: python Preprocessing/visualize_depth_trim.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.facecolor"] = "white"
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ──
DATA_ROOT = Path(r"C:\github\VT2\Data fra tidligere project\Dataset\Intrinsic data")
OUTPUT = Path(r"C:\github\VT2\visualizations")
OUTPUT.mkdir(exist_ok=True)

# Depth-based plateau detection parameters
IDLE_DEPTH_RATE = 0.005   # mm/ms threshold
IDLE_WINDOW     = 50      # rolling window size (samples)
IDLE_MARGIN_MS  = 50      # keep this many ms after detected plateau


def detect_plateau(df):
    """Find time (ms) where depth plateaus after rising (end of screwing).

    Returns (start_ms, end_ms):
      - start_ms: when depth first starts rising
      - end_ms:   when depth stops changing (plateau after rise)
    Returns (None, None) if no plateau detected.
    """
    depth = df["Depth (mm)"].values
    time  = df["Time (ms)"].values
    depth_change = np.abs(np.diff(depth))
    if len(depth_change) < IDLE_WINDOW:
        return None, None
    kernel = np.ones(IDLE_WINDOW) / IDLE_WINDOW
    smoothed_rate = np.convolve(depth_change, kernel, mode="valid")

    above = np.where(smoothed_rate > IDLE_DEPTH_RATE)[0]
    if len(above) == 0:
        return None, None

    # Start of active screwing
    start_ms = time[above[0]]

    # Find where it drops back below threshold AFTER being above (plateau)
    below_after_active = np.where(smoothed_rate[above[0]:] <= IDLE_DEPTH_RATE)[0]
    if len(below_after_active) == 0:
        return start_ms, None  # never plateaus — depth keeps changing until the end

    plateau_idx = above[0] + below_after_active[0]
    end_ms = time[plateau_idx] + IDLE_MARGIN_MS
    return start_ms, min(end_ms, time[-1])


def trim_from_plateau(df, plateau_ms):
    """Keep only rows from the plateau point onward."""
    trimmed = df[df["Time (ms)"] >= plateau_ms].copy()
    trimmed["Time (ms)"] -= plateau_ms
    return trimmed.reset_index(drop=True)


# ── Process each subfolder ──
for subfolder in sorted(DATA_ROOT.iterdir()):
    if not subfolder.is_dir():
        continue

    csv_files = sorted(subfolder.glob("*.csv"))
    if not csv_files:
        print(f"[{subfolder.name}] No CSV files found, skipping.")
        continue

    # Pick the first file as a sample
    sample_file = csv_files[0]
    df_before = pd.read_csv(sample_file)

    start_ms, end_ms = detect_plateau(df_before)
    if end_ms is None:
        print(f"[{subfolder.name}] {sample_file.name}: No plateau detected, skipping.")
        continue

    df_after = trim_from_plateau(df_before, end_ms)
    print(f"[{subfolder.name}] {sample_file.name}: plateau @ {end_ms:.1f} ms | {len(df_before)} -> {len(df_after)} rows")

    # ── Plot ──
    signal_cols = [c for c in df_before.columns if c != "Time (ms)"]
    fig, axes = plt.subplots(len(signal_cols), 2, figsize=(18, 3 * len(signal_cols)), sharex="col")
    fig.suptitle(
        f"Before vs After Depth Plateau Trim — {subfolder.name}/{sample_file.name}",
        fontsize=16, fontweight="bold",
    )

    for i, col in enumerate(signal_cols):
        # Before
        axes[i, 0].plot(df_before["Time (ms)"], df_before[col], "b-", linewidth=0.5, alpha=0.8)
        axes[i, 0].axvline(end_ms, color="red", linestyle="--", linewidth=1,
                           label=f"plateau @ {end_ms:.0f} ms")
        axes[i, 0].set_ylabel(col, fontsize=8)
        if i == 0:
            axes[i, 0].set_title("BEFORE (raw)", fontsize=12)
            axes[i, 0].legend(fontsize=8)
        # After
        axes[i, 1].plot(df_after["Time (ms)"], df_after[col], "g-", linewidth=0.5, alpha=0.8)
        if i == 0:
            axes[i, 1].set_title("AFTER (signal from depth plateau onward)", fontsize=12)

    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    plt.tight_layout()
    out_path = OUTPUT / f"depth_trim_{subfolder.name}_{sample_file.stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

print("\nDone!")
