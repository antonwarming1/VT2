import json
import sys
import os
import pandas as pd

sys.path.append(r"Extract_data_from_csv_and_json")

from extract_data import extract_json_data

data = extract_json_data(r"data\20022026")

df = data[0]
print(df)
print(df.shape)
