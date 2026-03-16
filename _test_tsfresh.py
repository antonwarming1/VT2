import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

# Minimal test
np.random.seed(42)
df = pd.DataFrame({
    "id": [0]*100 + [1]*100,
    "time": list(range(100)) + list(range(100)),
    "value": np.random.randn(200)
})

print("Extracting on minimal data...", flush=True)
try:
    features = extract_features(df, column_id="id", column_sort="time",
                               default_fc_parameters=EfficientFCParameters(),
                               n_jobs=0, show_warnings=False,
                               disable_progressbar=True)
    print(f"Result: {features.shape}", flush=True)
    features.to_csv("_test_output.csv")
    print("Saved to _test_output.csv", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
print("OK!", flush=True)
