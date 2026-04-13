import json
import pandas as pd


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

data_path = r"data_opsamling_cleaned\Under\120320261A2.json"

df = load_json(data_path)
print(df)
print(df.shape)


