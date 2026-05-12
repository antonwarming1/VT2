"""
inference_pipeline.py
=====================
End-to-end per-instance inference pipeline for app.py.

Takes a raw Task CSV + Intrinsic CSV from the old dataset, runs the full
preprocessing pipeline in memory, extracts only the features the trained
model needs, and returns a single-row feature Series ready for the scaler.

Public API:
  list_raw_pairs()              -> list of (label, base_id, task_path, intr_path)
  build_kind_fc_parameters(cols)-> (task_kind_to_fc, intr_kind_to_fc)
  pipeline_one(...)             -> pd.Series | None
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.utilities.dataframe_functions import impute

# Make sibling packages importable when running from the FNN directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Preprocessing.data_cleaning import clean_task_df, clean_intrinsic_df
from Preprocessing.data_preprocessing import preprocess_old_pair_df
from Preprocessing.exclude_features import drop_csv_columns_df
from Feature_engineering.code import _csv_df_to_long

RAW_ROOT    = Path(r"C:\github\VT2\Data fra tidligere project\Dataset")
LABELS_PATH = Path(r"C:\github\VT2\Feature_engineering\labels.csv")
LABELS      = ["N", "NS", "OT", "P", "UT"]

# Each model was trained with its own test split — reconstruct the matching one
# so the app only ever evaluates a model on samples it never saw during training.
_RANDOM_STATE = 42
MODEL_TEST_SIZES = {"fnn": 0.1, "svm": 0.2, "rf": 0.2}
_DEFAULT_TEST_SIZE = MODEL_TEST_SIZES["fnn"]


def list_raw_pairs():
    """Return all (label, base_id, task_path, intr_path) tuples, in sample_id order."""
    task_root = RAW_ROOT / "Task data"
    intr_root = RAW_ROOT / "Intrinsic data"

    pairs = []
    for label in LABELS:
        task_dir = task_root / label
        intr_dir = intr_root / label
        if not task_dir.exists() or not intr_dir.exists():
            continue

        task_files = {f.stem[1:]: f for f in sorted(task_dir.glob("t*.csv"))}
        intr_files = {f.stem[1:]: f for f in sorted(intr_dir.glob("i*.csv"))}
        for base_id in sorted(task_files.keys() & intr_files.keys()):
            pairs.append((label, base_id, task_files[base_id], intr_files[base_id]))

    return pairs


def list_test_pairs(model_name="fnn"):
    """
    Return only the pairs that were in the held-out test set during training
    for the given model. Each model uses its own test_size (see MODEL_TEST_SIZES)
    so we never evaluate it on samples it saw during training.
    """
    test_size = MODEL_TEST_SIZES.get(model_name, _DEFAULT_TEST_SIZE)
    all_pairs = list_raw_pairs()
    y = pd.read_csv(LABELS_PATH, index_col=0).values.flatten()

    indices = np.arange(len(y))
    _, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=_RANDOM_STATE,
        stratify=y,
    )
    return [all_pairs[i] for i in sorted(test_idx)]


def build_kind_fc_parameters(selected_columns):
    """
    Build kind_to_fc_parameters dicts for the task and intrinsic tsfresh extraction.

    selected_columns: list of column names from features_selected.csv
                      (prefixed with "task_" or "intr_")
    Returns (task_kind_to_fc, intr_kind_to_fc).
    """
    task_cols = [c[len("task_"):] for c in selected_columns if c.startswith("task_")]
    intr_cols = [c[len("intr_"):] for c in selected_columns if c.startswith("intr_")]

    task_kind_to_fc = from_columns(task_cols) if task_cols else {}
    intr_kind_to_fc = from_columns(intr_cols) if intr_cols else {}

    return task_kind_to_fc, intr_kind_to_fc


def _extract_one_kind(long_df, kind_to_fc, prefix):
    """tsfresh extraction on one long-format frame: impute NaNs, prefix columns."""
    feats = extract_features(
        long_df,
        column_id="id", column_sort="time",
        kind_to_fc_parameters=kind_to_fc,
        n_jobs=0, show_warnings=False, disable_progressbar=True,
    )
    impute(feats)
    return feats.add_prefix(prefix)


def pipeline_one(task_csv_path, intr_csv_path, sample_id,
                 task_kind_to_fc, intr_kind_to_fc, selected_columns):
    """
    Run the full preprocessing + feature extraction pipeline for one raw instance.

    Returns a pd.Series of features aligned to selected_columns order,
    or None if no depth plateau detected (instance should be skipped).
    """
    # Step 1: Load and clean
    task_df = clean_task_df(pd.read_csv(task_csv_path))
    intr_df = clean_intrinsic_df(pd.read_csv(intr_csv_path))

    # Step 2: Plateau trim + resample + smooth
    result = preprocess_old_pair_df(task_df, intr_df)
    if result is None:
        return None
    task_df, intr_df = result

    # Step 3: Drop excluded columns (TCP positions, Nset, Angle, Depth)
    task_df = drop_csv_columns_df(task_df)
    intr_df = drop_csv_columns_df(intr_df)

    # Step 4: Convert to tsfresh long format
    task_long = _csv_df_to_long(task_df, sample_id)
    intr_long = _csv_df_to_long(intr_df, sample_id)

    # Step 5: Extract features restricted to what the model uses
    task_feats = _extract_one_kind(task_long, task_kind_to_fc, "task_")
    intr_feats = _extract_one_kind(intr_long, intr_kind_to_fc, "intr_")

    # Step 6: Combine and reindex to exact selected column order
    all_feats = pd.concat([task_feats, intr_feats], axis=1)
    all_feats = all_feats.reindex(columns=selected_columns, fill_value=0.0)
    return all_feats.iloc[0]
