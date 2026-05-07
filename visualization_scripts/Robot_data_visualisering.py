import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Use raw string r"..." so the Windows path works correctly
file_path = Path(r"C:\Users\mjuul\OneDrive - Aalborg Universitet\Dokumenter\GitHub\VT2\Data fra tidligere project\Dataset\Task data\N\t030520235006.csv")

# Read CSV
df = pd.read_csv(file_path)

# Convert time from ms to seconds
df["Time (s)"] = df["Time (ms)"] / 1000

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Column groups
position_cols = ["TCP_x (mm)", "TCP_y (mm)", "TCP_z (mm)"]
rotation_cols = ["TCP_rx (rad)", "TCP_ry (rad)", "TCP_rz (rad)"]
current_col = "Robot_I (A)"

# ---------- 1. TCP positions ----------
plt.figure()
for col in position_cols:
    plt.plot(df["Time (s)"], df[col], label=col)
plt.xlabel("Time [s]")
plt.ylabel("TCP position [mm]")
plt.title("TCP translation over time")
plt.legend()
plt.grid(True)

# ---------- 2. TCP rotations ----------
plt.figure()
for col in rotation_cols:
    if col == "TCP_rz (rad)":
        plt.plot(df["Time (s)"], -df[col], label=f"{col}")
    else:
        plt.plot(df["Time (s)"], df[col], label=col)
plt.xlabel("Time [s]")
plt.ylabel("TCP rotation [rad]")
plt.title("TCP rotation over time")
plt.legend()
plt.grid(True)

# ---------- 3. Robot current ----------
plt.figure()
plt.plot(df["Time (s)"], df[current_col])
plt.xlabel("Time [s]")
plt.ylabel("Robot current [A]")
plt.title("Robot current over time")
plt.grid(True)

# Show all plots in separate windows
plt.show()