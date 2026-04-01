"""
Visualization of Intrinsic data from NS folder.
- CSV data: Screwing cell data (Nset, Torque, Current, Angle, Depth) over time
- First 5 files only
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['savefig.facecolor'] = 'white'
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_ROOT = Path(r"C:\github\VT2\Data fra tidligere project\Dataset\Intrinsic data\N")
OUTPUT_DIR = Path(r"C:\github\VT2\visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

_COLOR_CYCLE = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple",
                "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]


def load_csv(filepath):
    return pd.read_csv(filepath)


def get_file_groups(limit=5):
    """Return sorted list of (base_name, full_path) for CSV files."""
    groups = []
    for f in sorted(os.listdir(DATA_ROOT)):
        if f.endswith(".csv"):
            base = os.path.splitext(f)[0]
            groups.append((base, DATA_ROOT / f))
        if len(groups) >= limit:
            break
    return groups


def _extract_label(base):
    return base


def _find_col(df, keyword):
    """Find column name containing the given keyword (case-insensitive)."""
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None


def plot_csv_overview(file_groups):
    """Plot all CSV data in a combined overview figure."""
    csv_columns = [
        "Nset (1/min)", "Torque (Nm)", "Current (V)",
        "Angle (deg)", "Depth (mm)"
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=False)
    fig.suptitle("Intrinsic Data Overview — Screwing Cell (NS, first 5)", fontsize=16, fontweight="bold")

    for idx, (base, csv_path) in enumerate(file_groups):
        df = load_csv(csv_path)
        label = _extract_label(base)
        color = _COLOR_CYCLE[idx % len(_COLOR_CYCLE)]

        for i, col in enumerate(csv_columns):
            row, c = divmod(i, 2)
            ax = axes[row][c]
            ax.plot(df["Time (ms)"], df[col], label=label, color=color, alpha=0.6, linewidth=0.8)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

    # Last subplot empty — use for legend
    axes[2][1].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[2][1].legend(handles, labels, loc="center", ncol=2, fontsize=8, title="File")

    for row in range(3):
        for c in range(2):
            axes[row][c].set_xlabel("Time (ms)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(OUTPUT_DIR / "NS_intrinsic_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved NS_intrinsic_overview.png")


def plot_torque_vs_angle(file_groups):
    """Plot Torque vs Angle — classic screwing curve."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Torque vs Angle — Screwing Curves (NS, first 5)", fontsize=16, fontweight="bold")

    for idx, (base, csv_path) in enumerate(file_groups):
        df = load_csv(csv_path)
        angle_col = _find_col(df, "angle")
        torque_col = _find_col(df, "torque")
        if not angle_col or not torque_col:
            continue
        label = _extract_label(base)
        color = _COLOR_CYCLE[idx % len(_COLOR_CYCLE)]
        ax.plot(df[angle_col], df[torque_col], label=label, color=color, alpha=0.6, linewidth=1)

    ax.set_xlabel("Angle (deg)", fontsize=12)
    ax.set_ylabel("Torque (Nm)", fontsize=12)
    ax.legend(ncol=2, fontsize=8, title="File")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "NS_intrinsic_torque_vs_angle.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved NS_intrinsic_torque_vs_angle.png")


def plot_individual_file(base, csv_path):
    """Plot detailed view of a single CSV file."""
    df = load_csv(csv_path)
    csv_cols = [c for c in df.columns if c != "Time (ms)"]
    n_plots = len(csv_cols)

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]
    fig.suptitle(f"Detailed View — NS/{base}", fontsize=14, fontweight="bold")

    for idx, col in enumerate(csv_cols):
        axes[idx].plot(df["Time (ms)"], df[col], color="darkorange", linewidth=1)
        axes[idx].set_ylabel(col, fontsize=9)
        axes[idx].set_title(f"{col}", fontsize=10, loc="left")
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(str(OUTPUT_DIR / f"NS_{base}_detail.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved NS_{base}_detail.png")


if __name__ == "__main__":
    file_groups = get_file_groups(limit=5)
    print(f"Found {len(file_groups)} file groups:")
    for base, path in file_groups:
        print(f"  {base}")

    # 1. Intrinsic data overview
    plot_csv_overview(file_groups)

    # 2. Torque vs Angle — classic screwing curve
    plot_torque_vs_angle(file_groups)

    # 3. Detailed view for each file
    for base, csv_path in file_groups:
        plot_individual_file(base, csv_path)

    print("\nAll visualizations complete!")
