import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TASK_ROOT = Path(r"C:\github\VT2\Data fra tidligere project\Dataset\Task data")
INTR_ROOT = Path(r"C:\github\VT2\Data fra tidligere project\Dataset\Intrinsic data")
CLASSES   = ["N", "NS", "OT", "P", "UT"]
N_EACH    = 5

fig, axes = plt.subplots(len(CLASSES), 2, figsize=(14, 4 * len(CLASSES)))
fig.suptitle("TCP_z (mm) vs Torque (Nm) — 5 instances per class", fontsize=13, y=1.01)

colors = ["#2563eb", "#ef4444", "#22c55e", "#f59e0b", "#a855f7"]

for row, cls in enumerate(CLASSES):
    task_dir = TASK_ROOT / cls
    intr_dir = INTR_ROOT / cls

    task_files = {f.stem[1:]: f for f in sorted(task_dir.glob("t*.csv"))}
    intr_files = {f.stem[1:]: f for f in sorted(intr_dir.glob("i*.csv"))}
    paired = sorted(task_files.keys() & intr_files.keys())[:N_EACH]
    print(f"{cls}: {len(paired)} paired")

    ax_tcp  = axes[row, 0]
    ax_torq = axes[row, 1]

    for i, base_id in enumerate(paired):
        task_df = pd.read_csv(task_files[base_id])
        intr_df = pd.read_csv(intr_files[base_id])

        ax_tcp.plot(task_df["Time (ms)"], task_df["TCP_z (mm)"],
                    color=colors[i], linewidth=0.9, label=base_id, alpha=0.85)
        ax_torq.plot(intr_df["Time (ms)"], intr_df["Torque (Nm)"],
                     color=colors[i], linewidth=0.9, label=base_id, alpha=0.85)

    ax_tcp.set_title(f"{cls} — TCP_z (mm)", fontsize=10)
    ax_tcp.set_xlabel("Time (ms)")
    ax_tcp.set_ylabel("TCP_z (mm)")
    ax_tcp.legend(fontsize=7, loc="upper right")
    ax_tcp.grid(True, alpha=0.3)

    ax_torq.set_title(f"{cls} — Torque (Nm)", fontsize=10)
    ax_torq.set_xlabel("Time (ms)")
    ax_torq.set_ylabel("Torque (Nm)")
    ax_torq.legend(fontsize=7, loc="upper right")
    ax_torq.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\github\VT2\plot_classes.png", dpi=120, bbox_inches="tight")
print("Saved to C:\\github\\VT2\\plot_classes.png")
