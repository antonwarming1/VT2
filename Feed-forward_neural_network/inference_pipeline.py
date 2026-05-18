"""
inference_pipeline.py
=====================
End-to-end per-instance inference pipeline for app.py.

Takes a raw Task CSV + Intrinsic CSV from the old dataset, runs the full
preprocessing pipeline in memory, extracts only the features the trained
model needs, and returns a single-row feature Series ready for the scaler.

Public API (no-audio):
  list_raw_pairs()                    -> list of (label, base_id, task_path, intr_path)
  build_kind_fc_parameters(cols)      -> (task_kind_to_fc, intr_kind_to_fc)
  pipeline_one(...)                   -> pd.Series | None

Public API (audio):
  list_raw_pairs_audio()              -> list of (label, base_id, task_path, intr_path, audio_wav_path)
  list_test_pairs_audio(model_name)   -> same, test-split subset
  build_kind_fc_parameters_audio(cols)-> (task_fc, intr_fc, audio_fc)
  pipeline_one_audio(...)             -> pd.Series | None
"""

import sys
from pathlib import Path

import librosa
import noisereduce
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
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
_LABEL_REMAP = {"P": "NE"}

AUDIO_RAW_ROOT  = Path(r"C:\github\VT2\Data fra tidligere project\Dataset\Extrinsic data (clean)")
AUDIO_NOISE_REF = Path(r"C:\github\VT2\Soundcleaning\Optaget_støj.wav")
AUDIO_SAMPLERATE = 2200

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
        mapped_label = _LABEL_REMAP.get(label, label)
        for base_id in sorted(task_files.keys() & intr_files.keys()):
            pairs.append((mapped_label, base_id, task_files[base_id], intr_files[base_id]))

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
                 task_kind_to_fc, intr_kind_to_fc, selected_columns,
                 training_means=None):
    """
    Run the full preprocessing + feature extraction pipeline for one raw instance.

    Returns a pd.Series of features aligned to selected_columns order,
    or None if no depth plateau detected (instance should be skipped).

    training_means: pd.Series of per-column means from the training set.
                    Used to fill features tsfresh cannot compute for this instance.
                    Falls back to 0.0 if not provided.
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
    all_feats = all_feats.reindex(columns=selected_columns)
    if training_means is not None:
        all_feats = all_feats.fillna(training_means)
    else:
        all_feats = all_feats.fillna(0.0)
    return all_feats.iloc[0]


# ── Audio-mode extensions ─────────────────────────────────────────────────────

def list_raw_pairs_audio():
    """
    Return (label, base_id, task_path, intr_path, audio_wav_path) for every
    sample where all three raw files exist.
    """
    task_root = RAW_ROOT / "Task data"
    intr_root = RAW_ROOT / "Intrinsic data"

    pairs = []
    for label in LABELS:
        task_dir = task_root / label
        intr_dir = intr_root / label
        audio_dir = AUDIO_RAW_ROOT / label
        if not task_dir.exists() or not intr_dir.exists():
            continue

        task_files  = {f.stem[1:]: f for f in sorted(task_dir.glob("t*.csv"))}
        intr_files  = {f.stem[1:]: f for f in sorted(intr_dir.glob("i*.csv"))}
        mapped_label = _LABEL_REMAP.get(label, label)
        for base_id in sorted(task_files.keys() & intr_files.keys()):
            audio_path = audio_dir / f"e{base_id}.wav"
            if not audio_path.exists():
                continue
            pairs.append((
                mapped_label, base_id,
                task_files[base_id], intr_files[base_id], audio_path,
            ))

    return pairs


def list_test_pairs_audio(model_name="fnn"):
    """
    Return only the audio triplets that were in the held-out test set.
    Stratified split is computed over the subset of samples that have audio files.
    """
    test_size = MODEL_TEST_SIZES.get(model_name, _DEFAULT_TEST_SIZE)
    all_audio_pairs = list_raw_pairs_audio()

    # Build the full pairs list (same order as labels.csv) to get correct indices
    all_pairs = list_raw_pairs()
    pair_index = {(label, str(base_id)): i for i, (label, base_id, *_) in enumerate(all_pairs)}

    # Identify which indices in the full list correspond to audio-available samples
    audio_indices = []
    for label, base_id, *_ in all_audio_pairs:
        idx = pair_index.get((label, str(base_id)))
        if idx is not None:
            audio_indices.append(idx)

    y_full = pd.read_csv(LABELS_PATH, index_col=0).values.flatten()
    y_sub  = y_full[audio_indices]

    local_indices = np.arange(len(audio_indices))
    _, test_local = train_test_split(
        local_indices,
        test_size=test_size,
        random_state=_RANDOM_STATE,
        stratify=y_sub,
    )
    return [all_audio_pairs[i] for i in sorted(test_local)]


def build_kind_fc_parameters_audio(selected_columns):
    """
    Like build_kind_fc_parameters but also handles the 'audio_' prefix.
    Returns (task_kind_to_fc, intr_kind_to_fc, audio_kind_to_fc).
    """
    task_cols  = [c[len("task_"):]  for c in selected_columns if c.startswith("task_")]
    intr_cols  = [c[len("intr_"):]  for c in selected_columns if c.startswith("intr_")]
    audio_cols = [c[len("audio_"):] for c in selected_columns if c.startswith("audio_")]

    task_fc  = from_columns(task_cols)  if task_cols  else {}
    intr_fc  = from_columns(intr_cols)  if intr_cols  else {}
    audio_fc = from_columns(audio_cols) if audio_cols else {}

    return task_fc, intr_fc, audio_fc


def _clean_wav_df(wav_path, samplerate=AUDIO_SAMPLERATE):
    """Load a WAV file into a Time(ms)/Amplitude DataFrame (in-memory clean_wav)."""
    y, sr = librosa.load(str(wav_path), sr=samplerate, mono=True)
    y = np.where(np.isnan(y), 0.0, y)
    time_ms = np.arange(len(y)) / sr * 1000
    return pd.DataFrame({"Time (ms)": time_ms, "Amplitude": y})


def _preprocess_audio_df(df, samplerate=AUDIO_SAMPLERATE):
    """Noise-reduce and lowpass-filter an Amplitude DataFrame (in-memory preprocess_audio)."""
    noise_ref, _ = librosa.load(str(AUDIO_NOISE_REF), sr=samplerate, mono=True)
    cleaned = noisereduce.reduce_noise(
        y=df["Amplitude"].values,
        y_noise=noise_ref,
        sr=samplerate,
        prop_decrease=0.8,
        stationary=False,
        freq_mask_smooth_hz=100,
        time_mask_smooth_ms=128,
    )
    sos=butter(6, 1000, btype="low", fs=samplerate, output="sos")
    filtered=sosfiltfilt(sos, cleaned)
    out = df.copy()
    out["Amplitude"] = filtered
    return out


def pipeline_one_audio(task_csv_path, intr_csv_path, audio_wav_path, sample_id,
                        task_kind_to_fc, intr_kind_to_fc, audio_kind_to_fc,
                        selected_columns, training_means=None):
    """
    Full pipeline (task + intrinsic + audio) for one raw instance.
    Returns a pd.Series aligned to selected_columns, or None if no plateau found.
    """
    # Steps 1–3: task + intrinsic (identical to pipeline_one)
    task_df = clean_task_df(pd.read_csv(task_csv_path))
    intr_df = clean_intrinsic_df(pd.read_csv(intr_csv_path))

    result = preprocess_old_pair_df(task_df, intr_df)
    if result is None:
        return None
    task_df, intr_df = result

    task_df = drop_csv_columns_df(task_df)
    intr_df = drop_csv_columns_df(intr_df)

    # Step 4: tsfresh long format
    task_long = _csv_df_to_long(task_df, sample_id)
    intr_long = _csv_df_to_long(intr_df, sample_id)

    # Step 5: extract task + intrinsic features
    task_feats = _extract_one_kind(task_long, task_kind_to_fc, "task_")
    intr_feats = _extract_one_kind(intr_long, intr_kind_to_fc, "intr_")

    # Audio steps: clean → preprocess → long format → extract
    audio_df   = _clean_wav_df(audio_wav_path)
    audio_df   = _preprocess_audio_df(audio_df)
    audio_long = _csv_df_to_long(audio_df, sample_id)
    audio_feats = _extract_one_kind(audio_long, audio_kind_to_fc, "audio_")

    # Step 6: combine and align
    all_feats = pd.concat([task_feats, intr_feats, audio_feats], axis=1)
    all_feats = all_feats.reindex(columns=selected_columns)
    if training_means is not None:
        all_feats = all_feats.fillna(training_means)
    else:
        all_feats = all_feats.fillna(0.0)
    return all_feats.iloc[0]
