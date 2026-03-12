"""
Visualization of data from C:\\github\\VT2\\data\\020320261
- CSV data: Robot TCP positions and current over time
- JSON data: Screwing cell data (Nset, Torque, Current, Angle, Depth) over time
"""

import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['savefig.facecolor'] = 'white'
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\github\VT2\data\020320261")
OUTPUT_DIR = Path(r"C:\github\VT2\visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_csv(filepath):
    return pd.read_csv(filepath)


def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    vectors = data["XML_Data"]["Wsk3Vectors"]
    x_vals = [float(v) for v in vectors["X_Axis"]["Values"]["float"]]
    axes_data = vectors["Y_AxesList"]["AxisData"]

    result = {"Time (ms)": x_vals}
    for axis in axes_data:
        name = axis["Header"]["Name"]
        unit = axis["Header"]["Unit"]
        values = [float(v) for v in axis["Values"]["float"]]
        result[f"{name} ({unit})"] = values

    return pd.DataFrame(result)


def get_file_groups():
    """Group files by base name (e.g., 020320261A1) and return sorted list."""
    files = os.listdir(DATA_DIR)
    bases = set()
    for f in files:
        if f.endswith((".csv", ".json")):
            bases.add(os.path.splitext(f)[0])
    return sorted(bases)


def plot_csv_overview(file_groups):
    """Plot all CSV data in a combined overview figure."""
    fig, axes = plt.subplots(4, 2, figsize=(18, 14), sharex=False)
    fig.suptitle("CSV Data Overview — Robot TCP & Current (all files)", fontsize=16, fontweight="bold")

    csv_columns = [
        "TCP_x (mm)", "TCP_y (mm)", "TCP_z (mm)",
        "TCP_rx (mm)", "TCP_ry (mm)", "TCP_rz (mm)",
        "Robot_I (A)"
    ]

    # Color map for groups A, B, C
    group_colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}

    for base in file_groups:
        csv_path = DATA_DIR / f"{base}.csv"
        if not csv_path.exists():
            continue
        df = load_csv(csv_path)
        # Determine group letter (A, B, or C)
        label_part = base.replace("020320261", "")
        group = label_part[0] if label_part else "?"
        color = group_colors.get(group, "tab:gray")

        for i, col in enumerate(csv_columns):
            row, c = divmod(i, 2)
            ax = axes[row][c]
            ax.plot(df["Time (ms)"], df[col], label=label_part, color=color, alpha=0.6, linewidth=0.8)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

    # Last subplot empty — use for legend
    axes[3][1].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[3][1].legend(handles, labels, loc="center", ncol=4, fontsize=8, title="File")

    for ax in axes[3]:
        if ax.get_xlabel() == "":
            ax.set_xlabel("Time (ms)")
    axes[2][0].set_xlabel("Time (ms)")
    axes[2][1].set_xlabel("Time (ms)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(OUTPUT_DIR / "csv_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved csv_overview.png")


def plot_json_overview(file_groups):
    """Plot all JSON screwing data in a combined overview figure."""
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    fig.suptitle("JSON Data Overview — Screwing Cell (all files)", fontsize=16, fontweight="bold")

    group_colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    json_columns = None

    for base in file_groups:
        json_path = DATA_DIR / f"{base}.json"
        if not json_path.exists():
            continue
        df = load_json(json_path)
        if json_columns is None:
            json_columns = [c for c in df.columns if c != "Time (ms)"]
        label_part = base.replace("020320261", "")
        group = label_part[0] if label_part else "?"
        color = group_colors.get(group, "tab:gray")

        for i, col in enumerate(json_columns):
            row, c = divmod(i, 2)
            ax = axes[row][c]
            ax.plot(df["Time (ms)"], df[col], label=label_part, color=color, alpha=0.6, linewidth=0.8)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

    # Last subplot — legend
    axes[2][1].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[2][1].legend(handles, labels, loc="center", ncol=4, fontsize=8, title="File")

    for row in range(3):
        for c in range(2):
            axes[row][c].set_xlabel("Time (ms)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(OUTPUT_DIR / "json_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved json_overview.png")


def plot_individual_file(base):
    """Plot detailed view of a single file (both CSV and JSON)."""
    csv_path = DATA_DIR / f"{base}.csv"
    json_path = DATA_DIR / f"{base}.json"

    has_csv = csv_path.exists()
    has_json = json_path.exists()
    n_plots = (7 if has_csv else 0) + (5 if has_json else 0)

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]
    fig.suptitle(f"Detailed View — {base}", fontsize=14, fontweight="bold")

    idx = 0
    if has_csv:
        df_csv = load_csv(csv_path)
        csv_cols = ["TCP_x (mm)", "TCP_y (mm)", "TCP_z (mm)",
                    "TCP_rx (mm)", "TCP_ry (mm)", "TCP_rz (mm)", "Robot_I (A)"]
        for col in csv_cols:
            axes[idx].plot(df_csv["Time (ms)"], df_csv[col], color="steelblue", linewidth=1)
            axes[idx].set_ylabel(col, fontsize=9)
            axes[idx].set_title(f"CSV — {col}", fontsize=10, loc="left")
            axes[idx].grid(True, alpha=0.3)
            idx += 1

    if has_json:
        df_json = load_json(json_path)
        json_cols = [c for c in df_json.columns if c != "Time (ms)"]
        for col in json_cols:
            axes[idx].plot(df_json["Time (ms)"], df_json[col], color="darkorange", linewidth=1)
            axes[idx].set_ylabel(col, fontsize=9)
            axes[idx].set_title(f"JSON — {col}", fontsize=10, loc="left")
            axes[idx].grid(True, alpha=0.3)
            idx += 1

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(str(OUTPUT_DIR / f"{base}_detail.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {base}_detail.png")


def _find_col(df, keyword):
    """Find column name containing the given keyword (case-insensitive)."""
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None


def plot_torque_vs_angle(file_groups):
    """Plot Torque vs Angle from JSON data — classic screwing curve."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Torque vs Angle — Screwing Curves (all files)", fontsize=16, fontweight="bold")

    group_colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}

    for base in file_groups:
        json_path = DATA_DIR / f"{base}.json"
        if not json_path.exists():
            continue
        df = load_json(json_path)
        angle_col = _find_col(df, "angle")
        torque_col = _find_col(df, "torque")
        if not angle_col or not torque_col:
            continue
        label_part = base.replace("020320261", "")
        group = label_part[0] if label_part else "?"
        color = group_colors.get(group, "tab:gray")
        ax.plot(df[angle_col], df[torque_col], label=label_part, color=color, alpha=0.6, linewidth=1)

    ax.set_xlabel("Angle", fontsize=12)
    ax.set_ylabel("Torque (Nm)", fontsize=12)
    ax.legend(ncol=4, fontsize=8, title="File")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "torque_vs_angle.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved torque_vs_angle.png")


def plot_group_comparison(file_groups):
    """Compare groups A, B, C side by side for key metrics."""
    groups = {"A": [], "B": [], "C": []}
    for base in file_groups:
        label_part = base.replace("020320261", "")
        group = label_part[0]
        if group in groups:
            groups[group].append(base)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Group Comparison (A vs B vs C) — Torque & Angle over Time",
                 fontsize=16, fontweight="bold")

    group_colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}

    for col_idx, (group, bases) in enumerate(groups.items()):
        for base in bases:
            json_path = DATA_DIR / f"{base}.json"
            if not json_path.exists():
                continue
            df = load_json(json_path)
            torque_col = _find_col(df, "torque")
            angle_col = _find_col(df, "angle")
            label_part = base.replace("020320261", "")
            if torque_col:
                axes[0][col_idx].plot(df["Time (ms)"], df[torque_col],
                                      label=label_part, alpha=0.7, linewidth=1)
            if angle_col:
                axes[1][col_idx].plot(df["Time (ms)"], df[angle_col],
                                      label=label_part, alpha=0.7, linewidth=1)

        axes[0][col_idx].set_title(f"Group {group} — Torque", fontsize=12)
        axes[0][col_idx].set_ylabel("Torque (Nm)")
        axes[0][col_idx].grid(True, alpha=0.3)
        axes[0][col_idx].legend(fontsize=7)
        axes[1][col_idx].set_title(f"Group {group} — Angle", fontsize=12)
        axes[1][col_idx].set_ylabel("Angle")
        axes[1][col_idx].set_xlabel("Time (ms)")
        axes[1][col_idx].grid(True, alpha=0.3)
        axes[1][col_idx].legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(OUTPUT_DIR / "group_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved group_comparison.png")


if __name__ == "__main__":
    file_groups = get_file_groups()
    print(f"Found {len(file_groups)} file groups: {file_groups}")

    # 1. CSV overview — all robot TCP data
    plot_csv_overview(file_groups)

    # 2. JSON overview — all screwing cell data
    plot_json_overview(file_groups)

    # 3. Torque vs Angle — classic screwing curve
    plot_torque_vs_angle(file_groups)

    # 4. Group comparison A vs B vs C
    plot_group_comparison(file_groups)

    # 5. Detailed view for first file as example
    plot_individual_file(file_groups[4])

    print("\nAll visualizations complete!")
