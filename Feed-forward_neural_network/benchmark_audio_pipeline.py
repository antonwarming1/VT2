"""
benchmark_audio_pipeline.py
============================
Times each stage of the full audio inference pipeline.

Run from the Feed-forward_neural_network directory in the project environment:
    python benchmark_audio_pipeline.py [--n 5] [--model fnn]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import inference_pipeline as ip
from inference_pipeline import (
    AUDIO_NOISE_REF, AUDIO_SAMPLERATE,
    _clean_wav_df, _preprocess_audio_df,
    _csv_df_to_long, _extract_one_kind,
)
from Preprocessing.data_cleaning import clean_task_df, clean_intrinsic_df
from Preprocessing.data_preprocessing import preprocess_old_pair_df
from Preprocessing.exclude_features import drop_csv_columns_df
from Feature_engineering.code import _csv_df_to_long


def _hdr(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def benchmark(n_samples: int, model_name: str):
    _hdr(f"Audio pipeline benchmark  |  model={model_name}  |  n={n_samples}")

    # ── Load shared assets ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    pairs = ip.list_test_pairs_audio(model_name)
    print(f"[setup] test pairs found  : {len(pairs)}")
    if not pairs:
        print("No audio test pairs found — check AUDIO_RAW_ROOT path.")
        return

    features_path = Path(r"C:\github\VT2\Feature_engineering\features_selected_audio.csv")
    df_feats = pd.read_csv(features_path, index_col=0)
    selected_cols = list(df_feats.columns)
    training_means = df_feats.mean()

    (task_kind_fc,
     intr_kind_fc,
     audio_kind_fc) = ip.build_kind_fc_parameters_audio(selected_cols)

    print(f"[setup] selected features : {len(selected_cols)}")
    print(f"[setup] setup time        : {time.perf_counter() - t0:.3f}s")

    # ── Per-sample timing ────────────────────────────────────────────────────
    import random
    sample = random.sample(pairs, min(n_samples, len(pairs)))

    stage_times = {
        "load_csv":      [],
        "preprocess_csv": [],
        "noise_reduce":  [],
        "lowpass":       [],
        "feat_task":     [],
        "feat_intr":     [],
        "feat_audio":    [],
        "combine":       [],
        "total":         [],
    }

    import librosa
    import noisereduce
    from scipy.signal import butter, sosfiltfilt

    noise_ref, _ = librosa.load(str(AUDIO_NOISE_REF), sr=AUDIO_SAMPLERATE, mono=True)

    for idx, (label, base_id, task_path, intr_path, audio_path) in enumerate(sample):
        print(f"\n  sample {idx+1}/{len(sample)}  id={base_id}  label={label}")
        t_total = time.perf_counter()

        # 1. Load CSVs
        t = time.perf_counter()
        task_df = clean_task_df(pd.read_csv(task_path))
        intr_df = clean_intrinsic_df(pd.read_csv(intr_path))
        stage_times["load_csv"].append(time.perf_counter() - t)

        # 2. Preprocess CSVs (plateau detection etc.)
        t = time.perf_counter()
        result = preprocess_old_pair_df(task_df, intr_df)
        if result is None:
            print("    -> no plateau found, skipping")
            continue
        task_df, intr_df = result
        task_df = drop_csv_columns_df(task_df)
        intr_df = drop_csv_columns_df(intr_df)
        stage_times["preprocess_csv"].append(time.perf_counter() - t)

        # 3. Load + noise-reduce WAV
        t = time.perf_counter()
        y, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLERATE, mono=True)
        y = np.where(np.isnan(y), 0.0, y)
        cleaned = noisereduce.reduce_noise(
            y=y, y_noise=noise_ref, sr=AUDIO_SAMPLERATE,
            prop_decrease=0.8, stationary=False,
            freq_mask_smooth_hz=100, time_mask_smooth_ms=128,
        )
        stage_times["noise_reduce"].append(time.perf_counter() - t)
        print(f"    noise_reduce   : {stage_times['noise_reduce'][-1]:.3f}s")

        # 4. Lowpass filter
        t = time.perf_counter()
        sos = butter(6, 1000, btype="low", fs=AUDIO_SAMPLERATE, output="sos")
        filtered = sosfiltfilt(sos, cleaned)
        import time as _time
        stage_times["lowpass"].append(time.perf_counter() - t)

        time_ms = np.arange(len(filtered)) / AUDIO_SAMPLERATE * 1000
        audio_df = pd.DataFrame({"Time (ms)": time_ms, "Amplitude": filtered})

        # 5. tsfresh long format
        task_long  = _csv_df_to_long(task_df,  sample_id=0)
        intr_long  = _csv_df_to_long(intr_df,  sample_id=0)
        audio_long = _csv_df_to_long(audio_df, sample_id=0)

        # 6. Feature extraction — task
        t = time.perf_counter()
        task_feats = _extract_one_kind(task_long, task_kind_fc, "task_")
        stage_times["feat_task"].append(time.perf_counter() - t)
        print(f"    feat_task      : {stage_times['feat_task'][-1]:.3f}s")

        # 7. Feature extraction — intrinsic
        t = time.perf_counter()
        intr_feats = _extract_one_kind(intr_long, intr_kind_fc, "intr_")
        stage_times["feat_intr"].append(time.perf_counter() - t)
        print(f"    feat_intr      : {stage_times['feat_intr'][-1]:.3f}s")

        # 8. Feature extraction — audio
        t = time.perf_counter()
        audio_feats = _extract_one_kind(audio_long, audio_kind_fc, "audio_")
        stage_times["feat_audio"].append(time.perf_counter() - t)
        print(f"    feat_audio     : {stage_times['feat_audio'][-1]:.3f}s")

        # 9. Combine + align
        t = time.perf_counter()
        all_feats = pd.concat([task_feats, intr_feats, audio_feats], axis=1)
        all_feats = all_feats.reindex(columns=selected_cols).fillna(training_means)
        stage_times["combine"].append(time.perf_counter() - t)

        total = time.perf_counter() - t_total
        stage_times["total"].append(total)
        print(f"    TOTAL          : {total:.3f}s")

    # ── Summary ──────────────────────────────────────────────────────────────
    _hdr("Summary (mean ± std over completed samples)")
    fmt = "{:<20} {:>8}s  ±  {:>6}s  |  min {:>6}s  max {:>6}s"
    for stage, times in stage_times.items():
        if not times:
            continue
        arr = np.array(times)
        print(fmt.format(
            stage,
            f"{arr.mean():.3f}", f"{arr.std():.3f}",
            f"{arr.min():.3f}",  f"{arr.max():.3f}",
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=5,     help="number of samples to time")
    parser.add_argument("--model", type=str, default="fnn", help="fnn | svm | rf")
    args = parser.parse_args()
    benchmark(args.n, args.model)
