import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Plot settings
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"

# ============================================================
# Load CSV
# ============================================================

file_path = Path(
    r"C:\Users\mjuul\OneDrive - Aalborg Universitet\Dokumenter\GitHub\VT2\data_opsamling\Normal\120320261A1.csv"
)

df = pd.read_csv(file_path)

# Remove whitespace in column names
df.columns = df.columns.str.strip()

# ============================================================
# Time vector
# ============================================================

time = df["Time (ms)"] / 1000

# ============================================================
# Rotation data
# ============================================================

tcp_rx = df["TCP_rx (mm)"].astype(float)
tcp_ry = df["TCP_ry (mm)"].astype(float)
tcp_rz = -df["TCP_rz (mm)"].astype(float)

# ============================================================
# Rotation plot
# ============================================================

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

ax.plot(time, tcp_rx, linewidth=2.5, label="TCP_rx")
ax.plot(time, tcp_ry, linewidth=2.5, label="TCP_ry")
ax.plot(time, tcp_rz, linewidth=2.5, label="TCP_rz")

ax.set_xlabel("Time [s]")
ax.set_ylabel("TCP rotation")

ax.set_xlim([-1, 4])

# NO NORMALIZATION
ax.set_ylim([
    min(tcp_rx.min(), tcp_ry.min(), tcp_rz.min()) - 1,
    max(tcp_rx.max(), tcp_ry.max(), tcp_rz.max()) + 1
])

ax.grid(True, alpha=0.3)

ax.legend(loc="upper right")

plt.tight_layout()
plt.show()