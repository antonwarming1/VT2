"""Inspect what gets trimmed by the common time window step."""
import pandas as pd
import json
import numpy as np
from pathlib import Path

CLEANED = Path(r"C:\Users\Anton\OneDrive\Dokumenter\GitHub\VT2\data_opsamling_cleaned")

files = [
    ("Normal", "120320261A1"),
    ("Normal", "120320261A5"),
    ("Under", "120320261A4"),
]

for subfolder, base in files:
    csv = pd.read_csv(CLEANED / subfolder / f"{base}.csv")
    with open(CLEANED / subfolder / f"{base}.json") as f:
        data = json.load(f)

    vectors = data["XML_Data"]["Wsk3Vectors"]
    axes = vectors["Y_AxesList"]["AxisData"]

    # Get idle-trimmed times (approximate: idle ~1450 ms)
    idle_cut = 1450

    # JSON signals after CSV would end
    csv_dur = csv.iloc[-1, 0] - csv.iloc[0, 0]
    json_dur = float(vectors["X_Axis"]["Values"]["float"][-1])
    offset = max(0, csv_dur - json_dur)

    # After idle removal, CSV covers 0 to (csv_dur - offset - idle_cut)
    csv_end_after = csv_dur - offset - idle_cut
    json_end_after = json_dur - idle_cut

    print(f"\n{'='*60}")
    print(f"{base} ({subfolder})")
    print(f"  After idle removal: CSV ends ~{csv_end_after:.0f} ms, JSON ends ~{json_end_after:.0f} ms")
    print(f"  JSON data that gets CUT: {csv_end_after:.0f} to {json_end_after:.0f} ms ({json_end_after - csv_end_after:.0f} ms)")

    # What's in the trimmed part of JSON?
    for axis in axes:
        name = axis["Header"]["Name"]
        vals = np.array([float(v) for v in axis["Values"]["float"]])
        cut_start = int(csv_end_after + idle_cut)  # original JSON index
        cut_end = len(vals)

        if cut_start < cut_end:
            trimmed_vals = vals[cut_start:cut_end]
            before_vals = vals[max(0, cut_start-100):cut_start]
            print(f"  {name}:")
            print(f"    Before trim (last 100ms): mean={before_vals.mean():.3f}, min={before_vals.min():.3f}, max={before_vals.max():.3f}")
            print(f"    TRIMMED region ({len(trimmed_vals)} pts): mean={trimmed_vals.mean():.3f}, min={trimmed_vals.min():.3f}, max={trimmed_vals.max():.3f}")

            # Is the trimmed region interesting?
            if name == "Torque":
                peak_before = before_vals.max()
                peak_trimmed = trimmed_vals.max()
                print(f"    >>> Torque peak in trimmed: {peak_trimmed:.3f} vs before: {peak_before:.3f}")
            if name == "Depth":
                print(f"    >>> Depth at trim point: {before_vals[-1]:.2f}, at JSON end: {trimmed_vals[-1]:.2f}")
