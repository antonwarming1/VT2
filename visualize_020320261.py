"""
Visualization of data from C:\\github\\VT2\\data_opsamling
- CSV data: Robot TCP positions and current over time
- JSON data: Screwing cell data (Nset, Torque, Current, Angle, Depth) over time
- Subfolders: Normal, Under
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

DATA_ROOT = Path(r"C:\github\VT2\data_opsamling")
OUTPUT_DIR = Path(r"C:\github\VT2\visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# Auto-assign colors to subfolders based on folder names
_COLOR_CYCLE = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple",
                "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
SUBFOLDER_COLORS = {
    d.name: _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
    for i, d in enumerate(sorted(DATA_ROOT.iterdir())) if d.is_dir()
}


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
    """Return sorted list of (subfolder_name, base_name, full_path_dir) for all file groups."""
    groups = []
    for sub in sorted(DATA_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        bases = set()
        for f in os.listdir(sub):
            if f.endswith((".csv", ".json")):
                bases.add(os.path.splitext(f)[0])
        for b in sorted(bases):
            groups.append((sub.name, b, sub))
    return groups


def _extract_label(base):
    """Extract short label from base name by stripping the common date prefix."""
    # Remove leading digits (e.g., '120320261' -> 'A1')
    for i, ch in enumerate(base):
        if ch.isalpha():
            return base[i:]
    return base


def plot_csv_overview(file_groups):
    """Plot all CSV data in a combined overview figure."""
    fig, axes = plt.subplots(4, 2, figsize=(18, 14), sharex=False)
    fig.suptitle("CSV Data Overview — Robot TCP & Current (all files)", fontsize=16, fontweight="bold")

    csv_columns = [
        "TCP_x (mm)", "TCP_y (mm)", "TCP_z (mm)",
        "TCP_rx (mm)", "TCP_ry (mm)", "TCP_rz (mm)",
        "Robot_I (A)"
    ]

    for sub_name, base, data_dir in file_groups:
        csv_path = data_dir / f"{base}.csv"
        if not csv_path.exists():
            continue
        df = load_csv(csv_path)
        label_part = _extract_label(base)
        label = f"{sub_name}/{label_part}"
        color = SUBFOLDER_COLORS.get(sub_name, "tab:gray")

        for i, col in enumerate(csv_columns):
            row, c = divmod(i, 2)
            ax = axes[row][c]
            ax.plot(df["Time (ms)"], df[col], label=label, color=color, alpha=0.6, linewidth=0.8)
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

    json_columns = None

    for sub_name, base, data_dir in file_groups:
        json_path = data_dir / f"{base}.json"
        if not json_path.exists():
            continue
        df = load_json(json_path)
        if json_columns is None:
            json_columns = [c for c in df.columns if c != "Time (ms)"]
        label_part = _extract_label(base)
        label = f"{sub_name}/{label_part}"
        color = SUBFOLDER_COLORS.get(sub_name, "tab:gray")

        for i, col in enumerate(json_columns):
            row, c = divmod(i, 2)
            ax = axes[row][c]
            ax.plot(df["Time (ms)"], df[col], label=label, color=color, alpha=0.6, linewidth=0.8)
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


def plot_individual_file(sub_name, base, data_dir):
    """Plot detailed view of a single file (both CSV and JSON)."""
    csv_path = data_dir / f"{base}.csv"
    json_path = data_dir / f"{base}.json"

    has_csv = csv_path.exists()
    has_json = json_path.exists()
    n_plots = (7 if has_csv else 0) + (5 if has_json else 0)

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]
    fig.suptitle(f"Detailed View — {sub_name}/{base}", fontsize=14, fontweight="bold")

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
    plt.savefig(str(OUTPUT_DIR / f"{sub_name}_{base}_detail.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {sub_name}_{base}_detail.png")


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

    for sub_name, base, data_dir in file_groups:
        json_path = data_dir / f"{base}.json"
        if not json_path.exists():
            continue
        df = load_json(json_path)
        angle_col = _find_col(df, "angle")
        torque_col = _find_col(df, "torque")
        if not angle_col or not torque_col:
            continue
        label_part = _extract_label(base)
        label = f"{sub_name}/{label_part}"
        color = SUBFOLDER_COLORS.get(sub_name, "tab:gray")
        ax.plot(df[angle_col], df[torque_col], label=label, color=color, alpha=0.6, linewidth=1)

    ax.set_xlabel("Angle", fontsize=12)
    ax.set_ylabel("Torque (Nm)", fontsize=12)
    ax.legend(ncol=4, fontsize=8, title="File")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "torque_vs_angle.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved torque_vs_angle.png")


def plot_group_comparison(file_groups):
    """Compare Normal vs Under side by side for key metrics."""
    groups = {}
    for sub_name, base, data_dir in file_groups:
        groups.setdefault(sub_name, []).append((base, data_dir))

    n_groups = len(groups)
    if n_groups == 0:
        return

    fig, axes = plt.subplots(2, n_groups, figsize=(9 * n_groups, 10))
    if n_groups == 1:
        axes = axes.reshape(2, 1)
    group_names = sorted(groups.keys())
    fig.suptitle(f"Group Comparison ({' vs '.join(group_names)}) — Torque & Angle over Time",
                 fontsize=16, fontweight="bold")

    for col_idx, group_name in enumerate(group_names):
        color = SUBFOLDER_COLORS.get(group_name, "tab:gray")
        for base, data_dir in groups[group_name]:
            json_path = data_dir / f"{base}.json"
            if not json_path.exists():
                continue
            df = load_json(json_path)
            torque_col = _find_col(df, "torque")
            angle_col = _find_col(df, "angle")
            label_part = _extract_label(base)
            if torque_col:
                axes[0][col_idx].plot(df["Time (ms)"], df[torque_col],
                                      label=label_part, alpha=0.7, linewidth=1)
            if angle_col:
                axes[1][col_idx].plot(df["Time (ms)"], df[angle_col],
                                      label=label_part, alpha=0.7, linewidth=1)

        axes[0][col_idx].set_title(f"{group_name} — Torque", fontsize=12)
        axes[0][col_idx].set_ylabel("Torque (Nm)")
        axes[0][col_idx].grid(True, alpha=0.3)
        axes[0][col_idx].legend(fontsize=7)
        axes[1][col_idx].set_title(f"{group_name} — Angle", fontsize=12)
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
    print(f"Found {len(file_groups)} file groups:")
    for sub, base, _ in file_groups:
        print(f"  {sub}/{base}")

    # 1. CSV overview — all robot TCP data
    plot_csv_overview(file_groups)

    # 2. JSON overview — all screwing cell data
    plot_json_overview(file_groups)

    # 3. Torque vs Angle — classic screwing curve
    plot_torque_vs_angle(file_groups)

    # 4. Group comparison Normal vs Under
    plot_group_comparison(file_groups)

    # 5. Detailed view for first file as example
    #plot_individual_file(*file_groups[0])

    # 6. Detailed view for ALL screws (uncomment to enable)
    for sub, base, data_dir in file_groups:
        plot_individual_file(sub, base, data_dir)

    print("\nAll visualizations complete!")
