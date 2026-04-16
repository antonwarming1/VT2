import pandas as pd, json

csv = pd.read_csv(r'C:\github\VT2\data_opsamling\Normal\120320261A1.csv')
with open(r'C:\github\VT2\data_opsamling\Normal\120320261A1.json') as f:
    data = json.load(f)

vectors = data['XML_Data']['Wsk3Vectors']
json_times = [float(v) for v in vectors['X_Axis']['Values']['float']]

time_col = 'Time (ms)'
print('CSV:')
print(f'  Start: {csv[time_col].iloc[0]:.1f}')
print(f'  End:   {csv[time_col].iloc[-1]:.1f}')
print(f'  Duration: {csv[time_col].iloc[-1] - csv[time_col].iloc[0]:.1f}')
print(f'  Last 3 Robot_I: {csv["Robot_I (A)"].iloc[-3:].values}')

print()
print('JSON:')
print(f'  Start: {json_times[0]:.1f}')
print(f'  End:   {json_times[-1]:.1f}')
print(f'  Duration: {json_times[-1] - json_times[0]:.1f}')

axes = vectors['Y_AxesList']['AxisData']
for axis in axes:
    name = axis['Header']['Name']
    vals = [float(v) for v in axis['Values']['float']]
    print(f'  Last 3 of {name}: {vals[-3:]}')
