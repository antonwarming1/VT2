import pandas as pd

data = r"C:\github\VT2\data_opsamling_final\N\t0505202310001.csv"
df = pd.read_csv(data)
print(df.head())
print(df.columns)
for name in df.columns:
    print(name)
print(df.describe())
#print the collumns names and the first 5 rows of the data frame, and the description of the data frame.
print(df.info())