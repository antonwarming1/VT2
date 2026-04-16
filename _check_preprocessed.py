import pandas as pd, json

for base in ['120320261A1','120320261A3','120320261A5','120320261A7','120320261A8']:
    csv = pd.read_csv(f'data_opsamling_preprocessed/Normal/{base}.csv')
    with open(f'data_opsamling_preprocessed/Normal/{base}.json') as f:
        jd = json.load(f)
    jt = [float(v) for v in jd['XML_Data']['Wsk3Vectors']['X_Axis']['Values']['float']]
    csv_end = csv["Time (ms)"].iloc[-1]
    json_end = jt[-1]
    print(f"{base}:  CSV 0-{csv_end:.0f} ms ({len(csv)} pts)  |  JSON 0-{json_end:.0f} ms ({len(jt)} pts)  |  diff={json_end - csv_end:.0f} ms")
