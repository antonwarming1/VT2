"""
Plot Depth from old intrinsic CSV and new JSON side by side,
plus a WAV waveform from extrinsic data.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ── File paths ───────────────────────────────────────────────────────────────

OLD_CSV = r"C:\github\VT2\Data fra tidligere project\Dataset\Intrinsic data\N\i030520235006.csv"
NEW_JSON = r"C:\github\VT2\data_opsamling\Normal\120320261A1.json"
WAV_FILE = r"C:\github\VT2\Data fra tidligere project\Dataset\Extrinsic data\N\e030520236014.wav"


# ── Load data ────────────────────────────────────────────────────────────────

# Old intrinsic CSV
csv_df = pd.read_csv(OLD_CSV)

# New JSON
with open(NEW_JSON, "r", encoding="utf-8") as f:
    raw = json.load(f)
vectors = raw["XML_Data"]["Wsk3Vectors"]
json_time = [float(v) for v in vectors["X_Axis"]["Values"]["float"]]
json_depth = None
for axis in vectors["Y_AxesList"]["AxisData"]:
    if axis["Header"]["Name"] == "Depth":
        json_depth = [float(v) for v in axis["Values"]["float"]]
        break

# WAV
sample_rate, wav_data = wavfile.read(WAV_FILE)
# If stereo, take first channel
if wav_data.ndim > 1:
    wav_data = wav_data[:, 0]
wav_time_s = np.arange(len(wav_data)) / sample_rate


# ── Plot ─────────────────────────────────────────────────────────────────────

out_dir = r"C:\github\VT2\visualizations"

# 1) Old intrinsic depth
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(csv_df["Time (ms)"], csv_df["Depth (mm)"], color="tab:blue")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Depth (mm)")
ax.set_title("Old Data — Intrinsic Depth (i030520235006)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/depth_old_intrinsic.png", dpi=150)
plt.show()

# 2) New JSON depth
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(json_time, json_depth, color="tab:green")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Depth (mm)")
ax.set_title("New Data — JSON Depth (120320261A1)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/depth_new_json.png", dpi=150)
plt.show()

# 3) WAV waveform
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(wav_time_s, wav_data, color="tab:orange", linewidth=0.3)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title(f"Extrinsic WAV (e030520236014) — {sample_rate} Hz")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/wav_extrinsic.png", dpi=150)
plt.show()
