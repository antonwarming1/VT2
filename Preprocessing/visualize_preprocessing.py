"""
Visualize before/after preprocessing for one sample file.
Shows: alignment, idle removal, resampling for both CSV and JSON.

Usage:
  Standalone:  python Preprocessing/visualize_preprocessing.py
  Import:      from visualize_preprocessing import visualize_first_instance
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.facecolor"] = "white"
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
CLEANED = _ROOT / "data_opsamling_cleaned"
PREPROCESSED = _ROOT / "data_opsamling_preprocessed"
OUTPUT = _ROOT / "visualizations"


def load_json_df(filepath):
    """Load a WSK3 JSON file into a DataFrame with Time (ms) + signal columns."""
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


def plot_before_after_csv(csv_before, csv_after, sample, output_dir=None):
    """Figure 1: Before vs After — CSV (Robot data)."""
    output_dir = Path(output_dir or OUTPUT)
    csv_cols = [c for c in csv_before.columns if c != "Time (ms)"]
    fig, axes = plt.subplots(len(csv_cols), 2, figsize=(18, 3 * len(csv_cols)), sharex="col")
    fig.suptitle(f"CSV (Robot) — Before vs After Preprocessing — {sample}", fontsize=16, fontweight="bold")

    for i, col in enumerate(csv_cols):
        axes[i, 0].plot(csv_before["Time (ms)"], csv_before[col], "b-", linewidth=0.5, alpha=0.8)
        axes[i, 0].set_ylabel(col, fontsize=8)
        if i == 0:
            axes[i, 0].set_title("BEFORE (cleaned only)", fontsize=12)
        axes[i, 1].plot(csv_after["Time (ms)"], csv_after[col], "g-", linewidth=0.5, alpha=0.8)
        if i == 0:
            axes[i, 1].set_title("AFTER (aligned + idle removed + resampled 2ms + smoothed)", fontsize=12)

    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    plt.tight_layout()
    path = output_dir / f"preprocessing_csv_{sample}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_before_after_json(json_before, json_after, sample, output_dir=None):
    """Figure 2: Before vs After — JSON (Screwing cell data)."""
    output_dir = Path(output_dir or OUTPUT)
    json_cols = [c for c in json_before.columns if c != "Time (ms)"]
    fig, axes = plt.subplots(len(json_cols), 2, figsize=(18, 3 * len(json_cols)), sharex="col")
    fig.suptitle(f"JSON (Screwing Cell) — Before vs After Preprocessing — {sample}", fontsize=16, fontweight="bold")

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
    path = output_dir / f"preprocessing_json_{sample}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_timestep_distribution(csv_before, csv_after, sample, output_dir=None):
    """Figure 3: Time step distribution — Before vs After (CSV)."""
    output_dir = Path(output_dir or OUTPUT)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"CSV Time Step Distribution — {sample}", fontsize=14, fontweight="bold")

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
    path = output_dir / f"preprocessing_timesteps_{sample}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_overlay(csv_after, json_after, sample, output_dir=None):
    """Figure 4: Overlay — CSV Robot_I + JSON Torque aligned on same time axis."""
    output_dir = Path(output_dir or OUTPUT)
    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Aligned Overlay — Robot_I (CSV) vs Torque (JSON) — {sample}", fontsize=14, fontweight="bold")

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
    path = output_dir / f"preprocessing_overlay_{sample}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_first_instance(subfolder="Normal", sample="120320261A1",
                             cleaned_dir=None, preprocessed_dir=None,
                             output_dir=None):
    """
    Visualize before/after preprocessing for one sample (the first instance).

    Parameters
    ----------
    subfolder : str
        Subfolder name, e.g. "Normal" or "Under".
    sample : str
        File base name without extension, e.g. "120320261A1".
    cleaned_dir : Path or str, optional
        Path to cleaned data root (default: data_opsamling_cleaned).
    preprocessed_dir : Path or str, optional
        Path to preprocessed data root (default: data_opsamling_preprocessed).
    output_dir : Path or str, optional
        Directory to save figures (default: visualizations/).
    """
    cleaned = Path(cleaned_dir or CLEANED)
    preprocessed = Path(preprocessed_dir or PREPROCESSED)
    out = Path(output_dir or OUTPUT)
    out.mkdir(exist_ok=True)

    # Load data
    csv_before = pd.read_csv(cleaned / subfolder / f"{sample}.csv")
    json_before = load_json_df(cleaned / subfolder / f"{sample}.json")
    csv_after = pd.read_csv(preprocessed / subfolder / f"{sample}.csv")
    json_after = load_json_df(preprocessed / subfolder / f"{sample}.json")

    # Generate all four figures
    plot_before_after_csv(csv_before, csv_after, sample, out)
    plot_before_after_json(json_before, json_after, sample, out)
    plot_timestep_distribution(csv_before, csv_after, sample, out)
    plot_overlay(csv_after, json_after, sample, out)

    print(f"\nAll visualizations saved to: {out}")


def _plot_before_after_csv_pair(before_df, after_df, title_prefix, sample, output_dir):
    """Generic before/after subplot grid for any CSV DataFrame."""
    cols = [c for c in before_df.columns if c != "Time (ms)"]
    fig, axes = plt.subplots(len(cols), 2, figsize=(18, 3 * len(cols)), sharex="col")
    fig.suptitle(f"{title_prefix} — Before vs After Preprocessing — {sample}",
                 fontsize=16, fontweight="bold")
    if len(cols) == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(cols):
        axes[i, 0].plot(before_df["Time (ms)"], before_df[col], "b-", linewidth=0.5, alpha=0.8)
        axes[i, 0].set_ylabel(col, fontsize=8)
        if i == 0:
            axes[i, 0].set_title("BEFORE (raw)", fontsize=12)
        axes[i, 1].plot(after_df["Time (ms)"], after_df[col], "g-", linewidth=0.5, alpha=0.8)
        if i == 0:
            axes[i, 1].set_title("AFTER (idle removed + resampled + smoothed)", fontsize=12)

    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    plt.tight_layout()
    return fig


def visualize_old_first_instance(label="N", base_id=None,
                                 raw_intrinsic_dir=None, raw_task_dir=None,
                                 preprocessed_dir=None, output_dir=None):
    """
    Visualize before/after preprocessing for the first sample of the old dataset.

    Parameters
    ----------
    label : str
        Label subfolder, e.g. "N".
    base_id : str, optional
        File ID (without i/t prefix), e.g. "030520235006".
        If None, picks the first available pair.
    raw_intrinsic_dir : Path or str, optional
        Raw intrinsic data folder (default: Data fra tidligere project/Dataset/Intrinsic data).
    raw_task_dir : Path or str, optional
        Raw task data folder (default: Data fra tidligere project/Dataset/Task data).
    preprocessed_dir : Path or str, optional
        Preprocessed output root (default: data_old_preprocessed).
    output_dir : Path or str, optional
        Directory to save figures (default: visualizations/).
    """
    raw_intr = Path(raw_intrinsic_dir or (_ROOT / "Data fra tidligere project" / "Dataset" / "Intrinsic data"))
    raw_task = Path(raw_task_dir or (_ROOT / "Data fra tidligere project" / "Dataset" / "Task data"))
    preprocessed = Path(preprocessed_dir or (_ROOT / "data_old_preprocessed"))
    out = Path(output_dir or OUTPUT)
    out.mkdir(exist_ok=True)

    intr_dir = raw_intr / label
    task_dir = raw_task / label
    prep_dir = preprocessed / label

    if not prep_dir.exists():
        print(f"  Preprocessed folder {prep_dir} not found — run preprocessing first.")
        return

    # Pick first available pair if no base_id given
    if base_id is None:
        intr_files = {f.stem[1:]: f for f in sorted(intr_dir.glob("*.csv"))}
        task_files = {f.stem[1:]: f for f in sorted(task_dir.glob("*.csv"))}
        prep_files = {f.stem[1:]: f for f in sorted(prep_dir.glob("i*.csv"))}
        available = sorted(intr_files.keys() & task_files.keys() & prep_files.keys())
        if not available:
            print("  No matching files found.")
            return
        base_id = available[0]

    sample_label = f"old_{label}_{base_id}"

    # Load before (raw) and after (preprocessed)
    intr_before = pd.read_csv(intr_dir / f"i{base_id}.csv")
    task_before = pd.read_csv(task_dir / f"t{base_id}.csv")
    intr_after = pd.read_csv(prep_dir / f"i{base_id}.csv")
    task_after = pd.read_csv(prep_dir / f"t{base_id}.csv")

    # ── Figure 1: Task CSV before/after ──
    fig = _plot_before_after_csv_pair(task_before, task_after, "Task (Robot)", sample_label, out)
    path = out / f"preprocessing_old_task_{sample_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 2: Intrinsic CSV before/after ──
    fig = _plot_before_after_csv_pair(intr_before, intr_after, "Intrinsic (Screwing Cell)", sample_label, out)
    path = out / f"preprocessing_old_intrinsic_{sample_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 3: Time step distribution (Task) ──
    plot_timestep_distribution(task_before, task_after, f"{sample_label}_task", out)

    # ── Figure 4: Overlay — Robot_I vs Torque ──
    if "Robot_I (A)" in task_after.columns and "Torque (Nm)" in intr_after.columns:
        fig, ax1 = plt.subplots(figsize=(14, 5))
        fig.suptitle(f"Aligned Overlay — Robot_I (Task) vs Torque (Intrinsic) — {sample_label}",
                     fontsize=14, fontweight="bold")
        color1, color2 = "tab:blue", "tab:red"
        ax1.plot(task_after["Time (ms)"], task_after["Robot_I (A)"],
                 color=color1, linewidth=1, label="Robot_I (A) — Task")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Robot_I (A)", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.plot(intr_after["Time (ms)"], intr_after["Torque (Nm)"],
                 color=color2, linewidth=1, alpha=0.8, label="Torque (Nm) — Intrinsic")
        ax2.set_ylabel("Torque (Nm)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        path = out / f"preprocessing_old_overlay_{sample_label}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    print(f"\nOld dataset visualizations saved to: {out}")


if __name__ == "__main__":
    visualize_first_instance()
