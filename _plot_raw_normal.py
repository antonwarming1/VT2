"""
Plot raw data from data_opsamling/Normal — before any cleaning or preprocessing.
Shows CSV (robot) and JSON (screwing cell) side by side for all 5 file pairs.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(r"C:\github\VT2\data_opsamling\Normal")
OUT_DIR = Path(r"C:\github\VT2\visualizations")
OUT_DIR.mkdir(exist_ok=True)

def load_json_df(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
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

# Collect all paired files
bases = sorted({f.stem for f in DATA_DIR.glob("*.csv")} & {f.stem for f in DATA_DIR.glob("*.json")})

fig, axes = plt.subplots(len(bases), 2, figsize=(18, 5 * len(bases)), squeeze=False)
fig.suptitle("Raw data — data_opsamling/Normal (before cleaning/preprocessing)", fontsize=16, y=1.0)

for i, base in enumerate(bases):
    csv_df = pd.read_csv(DATA_DIR / f"{base}.csv")
    json_df = load_json_df(DATA_DIR / f"{base}.json")

    # --- Left: CSV (Robot) ---
    ax_csv = axes[i, 0]
    ax_csv.set_title(f"{base} — CSV (Robot)", fontsize=11)
    t_csv = csv_df["Time (ms)"]

    # Plot Robot_I on primary y-axis
    color_I = "tab:red"
    ax_csv.plot(t_csv, csv_df["Robot_I (A)"], color=color_I, alpha=0.8, linewidth=0.8, label="Robot_I (A)")
    ax_csv.set_ylabel("Robot_I (A)", color=color_I)
    ax_csv.tick_params(axis='y', labelcolor=color_I)

    # Plot TCP_z on secondary y-axis
    ax_csv2 = ax_csv.twinx()
    color_z = "tab:blue"
    ax_csv2.plot(t_csv, csv_df["TCP_z (mm)"], color=color_z, alpha=0.7, linewidth=0.8, label="TCP_z (mm)")
    ax_csv2.set_ylabel("TCP_z (mm)", color=color_z)
    ax_csv2.tick_params(axis='y', labelcolor=color_z)

    ax_csv.set_xlabel("Time (ms)")
    dur_csv = t_csv.iloc[-1]
    ax_csv.text(0.02, 0.95, f"Duration: {dur_csv:.0f} ms\nRows: {len(csv_df)}",
                transform=ax_csv.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Right: JSON (Screwing cell) ---
    ax_json = axes[i, 1]
    ax_json.set_title(f"{base} — JSON (WSK3)", fontsize=11)
    t_json = json_df["Time (ms)"]

    # Plot Torque and Current on primary axis
    ax_json.plot(t_json, json_df["Torque"], color="tab:orange", alpha=0.8, linewidth=0.8, label="Torque (Nm)")
    ax_json.plot(t_json, json_df["Current"], color="tab:green", alpha=0.8, linewidth=0.8, label="Current (A)")
    ax_json.set_ylabel("Torque (Nm) / Current (A)")
    ax_json.legend(loc="upper left", fontsize=8)

    # Plot Depth on secondary axis
    ax_json2 = ax_json.twinx()
    ax_json2.plot(t_json, json_df["Depth"], color="tab:purple", alpha=0.7, linewidth=0.8, label="Depth (mm)")
    ax_json2.set_ylabel("Depth (mm)", color="tab:purple")
    ax_json2.tick_params(axis='y', labelcolor="tab:purple")
    ax_json2.legend(loc="upper right", fontsize=8)

    ax_json.set_xlabel("Time (ms)")
    dur_json = t_json.iloc[-1]
    ax_json.text(0.02, 0.95, f"Duration: {dur_json:.0f} ms\nRows: {len(json_df)}",
                 transform=ax_json.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
out_path = OUT_DIR / "raw_Normal_overview.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
