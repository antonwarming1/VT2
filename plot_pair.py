import pandas as pd
import matplotlib.pyplot as plt

task = pd.read_csv(r"C:\github\VT2\Data fra tidligere project\Dataset\Task data\N\t030520235006.csv")
intr = pd.read_csv(r"C:\github\VT2\Data fra tidligere project\Dataset\Intrinsic data\N\i030520235006.csv")

task_cols = [c for c in task.columns if c != "Time (ms)"]
intr_cols = [c for c in intr.columns if c != "Time (ms)"]

fig, axes = plt.subplots(len(task_cols) + len(intr_cols), 1, figsize=(12, 3 * (len(task_cols) + len(intr_cols))))
fig.suptitle("t030520235006 / i030520235006  (label: N)", fontsize=14, y=1.01)

for i, col in enumerate(task_cols):
    axes[i].plot(task["Time (ms)"], task[col], linewidth=0.8)
    axes[i].set_ylabel(col, fontsize=9)
    axes[i].set_xlabel("Time (ms)")
    axes[i].grid(True, alpha=0.3)

offset = len(task_cols)
for i, col in enumerate(intr_cols):
    axes[offset + i].plot(intr["Time (ms)"], intr[col], linewidth=0.8, color="tab:orange")
    axes[offset + i].set_ylabel(col, fontsize=9)
    axes[offset + i].set_xlabel("Time (ms)")
    axes[offset + i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\github\VT2\plot_pair.png", dpi=120, bbox_inches="tight")
print("Saved to C:\\github\\VT2\\plot_pair.png")
plt.show()
