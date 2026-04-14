"""
Visualize before/after preprocessing for one sample file.
Shows: alignment, idle removal, resampling for both CSV and JSON.

Usage: python Preprocessing/visualize_preprocessing.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.facecolor"] = "white"
import matplotlib.pyplot as plt
from pathlib import Path

CLEANED = Path(r"C:\github\VT2\data_opsamling_cleaned")
PREPROCESSED = Path(r"C:\github\VT2\data_opsamling_preprocessed")
OUTPUT = Path(r"C:\github\VT2\visualizations")
OUTPUT.mkdir(exist_ok=True)

SAMPLE = "120320261A1"
SUBFOLDER = "Normal"


def load_json_df(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    vectors = data["XML_Data"]["Wsk3Vectors"]
    x_vals = [float(v) for v in vectors["X_Axis"]["Values"]["float"]]
    axes = vectors["Y_AxesList"]["AxisData"]
    result = {"Time (ms)": x_vals}
    for axis in axes:
        name = axis["Header"]["Name"]
        values = [float(v) for v in axis["Values"]["float"]]
        result[name] = values
    return pd.DataFrame(result)


# ── Load data ──
csv_before = pd.read_csv(CLEANED / SUBFOLDER / f"{SAMPLE}.csv")
json_before = load_json_df(CLEANED / SUBFOLDER / f"{SAMPLE}.json")
csv_after = pd.read_csv(PREPROCESSED / SUBFOLDER / f"{SAMPLE}.csv")
json_after = load_json_df(PREPROCESSED / SUBFOLDER / f"{SAMPLE}.json")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Before vs After — CSV (Robot data)
# ══════════════════════════════════════════════════════════════════════════════
csv_cols = [c for c in csv_before.columns if c != "Time (ms)"]
fig, axes = plt.subplots(len(csv_cols), 2, figsize=(18, 3 * len(csv_cols)), sharex="col")
fig.suptitle(f"CSV (Robot) — Before vs After Preprocessing — {SAMPLE}", fontsize=16, fontweight="bold")

for i, col in enumerate(csv_cols):
    # Before
    axes[i, 0].plot(csv_before["Time (ms)"], csv_before[col], "b-", linewidth=0.5, alpha=0.8)
    axes[i, 0].set_ylabel(col, fontsize=8)
    if i == 0:
        axes[i, 0].set_title("BEFORE (cleaned only)", fontsize=12)
    # After
    axes[i, 1].plot(csv_after["Time (ms)"], csv_after[col], "g-", linewidth=0.5, alpha=0.8)
    if i == 0:
        axes[i, 1].set_title("AFTER (aligned + idle removed + resampled 2ms + smoothed)", fontsize=12)

axes[-1, 0].set_xlabel("Time (ms)")
axes[-1, 1].set_xlabel("Time (ms)")
plt.tight_layout()
fig.savefig(OUTPUT / f"preprocessing_csv_{SAMPLE}.png", dpi=150)
plt.close(fig)
print(f"Saved: {OUTPUT / f'preprocessing_csv_{SAMPLE}.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Before vs After — JSON (Screwing cell data)
# ══════════════════════════════════════════════════════════════════════════════
json_cols = [c for c in json_before.columns if c != "Time (ms)"]
fig, axes = plt.subplots(len(json_cols), 2, figsize=(18, 3 * len(json_cols)), sharex="col")
fig.suptitle(f"JSON (Screwing Cell) — Before vs After Preprocessing — {SAMPLE}", fontsize=16, fontweight="bold")

for i, col in enumerate(json_cols):
    axes[i, 0].plot(json_before["Time (ms)"], json_before[col], "b-", linewidth=0.5, alpha=0.8)
    axes[i, 0].set_ylabel(col, fontsize=8)
    if i == 0:
        axes[i, 0].set_title("BEFORE (cleaned only)", fontsize=12)
    axes[i, 1].plot(json_after["Time (ms)"], json_after[col], "g-", linewidth=0.5, alpha=0.8)
    if i == 0:
        axes[i, 1].set_title("AFTER (idle removed + resampled 2ms)", fontsize=12)

axes[-1, 0].set_xlabel("Time (ms)")
axes[-1, 1].set_xlabel("Time (ms)")
plt.tight_layout()
fig.savefig(OUTPUT / f"preprocessing_json_{SAMPLE}.png", dpi=150)
plt.close(fig)
print(f"Saved: {OUTPUT / f'preprocessing_json_{SAMPLE}.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Time step distribution — Before vs After (CSV)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle(f"CSV Time Step Distribution — {SAMPLE}", fontsize=14, fontweight="bold")

dt_before = csv_before["Time (ms)"].diff().dropna()
dt_after = csv_after["Time (ms)"].diff().dropna()

axes[0].hist(dt_before, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
axes[0].set_title(f"BEFORE — mean={dt_before.mean():.2f} ms, std={dt_before.std():.2f}")
axes[0].set_xlabel("Time step (ms)")
axes[0].set_ylabel("Count")
axes[0].axvline(dt_before.mean(), color="red", linestyle="--", label="mean")
axes[0].legend()

axes[1].hist(dt_after, bins=50, color="seagreen", edgecolor="black", alpha=0.8)
axes[1].set_title(f"AFTER — mean={dt_after.mean():.2f} ms, std={dt_after.std():.4f}")
axes[1].set_xlabel("Time step (ms)")
axes[1].axvline(dt_after.mean(), color="red", linestyle="--", label="mean")
axes[1].legend()

plt.tight_layout()
fig.savefig(OUTPUT / f"preprocessing_timesteps_{SAMPLE}.png", dpi=150)
plt.close(fig)
print(f"Saved: {OUTPUT / f'preprocessing_timesteps_{SAMPLE}.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Overlay — CSV Robot_I + JSON Torque aligned on same time axis
# ══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(14, 5))
fig.suptitle(f"Aligned Overlay — Robot_I (CSV) vs Torque (JSON) — {SAMPLE}", fontsize=14, fontweight="bold")

color1, color2 = "tab:blue", "tab:red"
ax1.plot(csv_after["Time (ms)"], csv_after["Robot_I (A)"], color=color1, linewidth=1, label="Robot_I (A) — CSV")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Robot_I (A)", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
ax2.plot(json_after["Time (ms)"], json_after["Torque"], color=color2, linewidth=1, alpha=0.8, label="Torque (Nm) — JSON")
ax2.set_ylabel("Torque (Nm)", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
fig.savefig(OUTPUT / f"preprocessing_overlay_{SAMPLE}.png", dpi=150)
plt.close(fig)
print(f"Saved: {OUTPUT / f'preprocessing_overlay_{SAMPLE}.png'}")

print("\nAll visualizations saved to:", OUTPUT)
