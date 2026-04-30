import json
import matplotlib.pyplot as plt
from pathlib import Path

# --------- FONT (Times New Roman) ---------
plt.rcParams["font.family"] = "Times New Roman"

# --------- FILER ---------
file1 = Path(r"C:\Users\mjuul\OneDrive - Aalborg Universitet\Dokumenter\GitHub\VT2\data_opsamling_cleaned\Normal\120320261A1.json")
file2 = Path(r"C:\Users\mjuul\OneDrive - Aalborg Universitet\Dokumenter\GitHub\VT2\data_opsamling\Under\120320261A2.json")

# --------- STYLING ---------
style_normal = {
    "color": "blue",
    "linewidth": 1.0,
    "label": "Normal screw"
}

style_under = {
    "color": "red",
    "linewidth": 1.0,
    "label": "Under-tightened screw"
}

# --------- LOAD FUNKTION ---------
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vectors = data["XML_Data"]["Wsk3Vectors"]

    # Tid (ms → sek)
    time = [float(v) for v in vectors["X_Axis"]["Values"]["float"]]
    time = [t / 1000 for t in time]

    y_axes = vectors["Y_AxesList"]["AxisData"]

    return time, y_axes


# --------- LOAD DATA ---------
time1, y_axes1 = load_data(file1)
time2, y_axes2 = load_data(file2)

# --------- PLOT ---------
for axis1, axis2 in zip(y_axes1, y_axes2):

    name = axis1["Header"]["Name"]
    unit = axis1["Header"]["Unit"]

    values1 = [float(v) for v in axis1["Values"]["float"]]
    values2 = [float(v) for v in axis2["Values"]["float"]]

    plt.figure()

    # UNDER først (nederst)
    plt.plot(
        time2, values2,
        color=style_under["color"],
        linewidth=style_under["linewidth"],
        label=style_under["label"],
        zorder=1
    )

    # NORMAL bagefter (øverst)
    plt.plot(
        time1, values1,
        color=style_normal["color"],
        linewidth=style_normal["linewidth"],
        label=style_normal["label"],
        zorder=2
    )

    # --------- AKSER ---------
    plt.xlabel("Tid [s]")
    plt.ylabel(f"{name} [{unit}]")
    plt.title(f"{name} over Time")

    # 👉 Specifik rettelse for Nset
    if name == "Nset":
        plt.ylim(0, 400)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

plt.show()