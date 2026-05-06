import pandas as pd
import librosa
import numpy as np
from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt


def get_wav_files_from_subfolders(data_root: Path, subfolders=None):
    """Returner alle .wav-filer i de specificerede Soundcleaning-undermapper.

    Inkluderer kun filer i de listede undermapper og ignorerer løse .wav-filer
    direkte i Soundcleaning-roden.
    """
    if subfolders is None:
        subfolders = ["N", "NS", "OT", "P", "UT"]

    wav_files = []
    for subfolder in subfolders:
        folder = data_root / subfolder
        if not folder.exists():
            print(f"Advarsel: Undermappe findes ikke: {folder}")
            continue
        if not folder.is_dir():
            print(f"Advarsel: {folder} er ikke en mappe.")
            continue

        wav_files.extend(sorted(folder.glob("*.wav")))

    return wav_files


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "Soundcleaning"

    if not data_root.exists():
        raise FileNotFoundError(f"Soundcleaning-mappe ikke fundet: {data_root}")
    
    folders = ["N", "NS", "OT", "P", "UT"]   

    wav_files = get_wav_files_from_subfolders(data_root, subfolders=folders)
    print(f"Fandt {len(wav_files)} .wav filer i {data_root}s-undermapper.")

    y, sr = librosa.load(str(wav_files[0]), sr=None, mono=True)
    audio_df = pd.DataFrame({"Time (ms)": np.arange(len(y)) / sr * 1000, "Amplitude": y})
    # gain y by 30 dB
    
    plt.figure(figsize=(10, 4), num=1)
    plt.plot(audio_df["Time (ms)"], audio_df["Amplitude"], color="tab:blue")
    plt.title(f"Waveform for {wav_files[0].name}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    
    """for wav_file in wav_files:
        y, sr = librosa.load(str(wav_file), sr=None, mono=True)
        # ... feature extraction ..."""

    

    

















if __name__ == "__main__":
    main()
