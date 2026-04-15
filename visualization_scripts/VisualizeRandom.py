import os
import pandas as pd
import matplotlib.pyplot as plt
import random



base_path = r"Data fra tidligere project\Dataset\Intrinsic data"
folders = ['N', 'NS', 'OT', 'P', 'UT']
paths = []
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        raise ValueError(f"No CSV files in {folder_path}")
    random_file = random.choice(files)
    path = os.path.join(folder_path, random_file)
    paths.append(path)



def visualize_individual(path):
    df = pd.read_csv(path)
    
    # Extract folder name from path
    folder_name = os.path.basename(os.path.dirname(path))

    # Find the time column
    possible_time_columns = [c for c in df.columns if 'time' in c.lower()]
    if not possible_time_columns:
        raise ValueError("Could not find a time column in the CSV file.")
    time_col = possible_time_columns[0]

    # Columns to plot against time
    columns_to_plot = ["Torque (Nm)", "Current (V)", "Angle (deg)", "Depth (mm)"]
    selected_columns = []
    for target in columns_to_plot:
        match = next((c for c in df.columns if c.lower() == target.lower()), None)
        if match is None:
            raise ValueError(f"Could not find the required column '{target}' in the CSV file.")
        selected_columns.append(match)

    # Create one subplot per series with shared x-axis
    fig, axs = plt.subplots(len(selected_columns), 1, sharex=True, figsize=(12, 10))
    for ax, col in zip(axs, selected_columns):
        ax.plot(df[time_col], df[col], label=col)
        ax.set_ylabel(col)
        ax.grid(True)
        ax.legend(loc="upper right")

    axs[-1].set_xlabel(time_col)
    fig.suptitle(f"Torque, Current, Angle, and Depth vs Time - Folder: {folder_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show(block=False)


if __name__ == "__main__":
    for path in paths:
        visualize_individual(path)
    plt.show()



