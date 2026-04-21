import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

# ============================================================
# INDSTILLINGER - REDIGÉR KUN DET HER
# ============================================================

# Mappe med de originale lydfiler
INPUT_FOLDER = Path(
    r"C:\Users\mjuul\OneDrive - Aalborg Universitet\Dokumenter\GitHub\VT2\Data fra tidligere project\Dataset\Extrinsic data\UT"
)

# Rodmappe, hvor den nye cleaned mappe skal oprettes
OUTPUT_ROOT = Path(
    r"C:\Users\mjuul\OneDrive - Aalborg Universitet\Dokumenter\GitHub\VT2\Soundcleaning"
)

# Støjreference
NOISE_FILE = Path(
    r"C:\Users\mjuul\OneDrive - Aalborg Universitet\Dokumenter\GitHub\VT2\Soundcleaning\Optaget_støj.wav"
)

# Filnavnssuffix, så fx fil.wav bliver til fil_C.wav
OUTPUT_SUFFIX = "_C"

# Hvis True, gennemgår scriptet også undermapper
RECURSIVE = False

# Hvilke filtyper der skal behandles
FILE_EXTENSIONS = {".wav"}

# Noise reduction
PROP_DECREASE = 0.8
STATIONARY = False

# Frekvensfilter
LOWCUT = 1
HIGHCUT = 1000
FILTER_ORDER = 6

# Gain
GAIN_DB = 14.0

# Hvis True, undgår scriptet clipping ved at normalisere automatisk
AUTO_NORMALIZE_IF_CLIPPING = True

# ============================================================
# FUNKTIONER
# ============================================================

def validate_cutoffs(lowcut, highcut, samplerate):
    nyquist = samplerate / 2

    if lowcut <= 0:
        raise ValueError("LOWCUT skal være større end 0 Hz.")
    if highcut <= 0:
        raise ValueError("HIGHCUT skal være større end 0 Hz.")
    if lowcut >= highcut:
        raise ValueError("LOWCUT skal være mindre end HIGHCUT.")
    if highcut >= nyquist:
        raise ValueError(
            f"HIGHCUT skal være mindre end Nyquist-frekvensen ({nyquist:.2f} Hz)."
        )

def bandpass_filter(data, samplerate, lowcut, highcut, order=6):
    nyquist = samplerate / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, data, axis=0)
    return filtered

def apply_gain_db(data, gain_db):
    gain_linear = 10 ** (gain_db / 20.0)
    return data * gain_linear

def handle_clipping(data, auto_normalize=True):
    max_val = np.max(np.abs(data))
    if max_val > 1.0:
        if auto_normalize:
            print(f"  Advarsel: clipping fundet (max = {max_val:.4f}). Normaliserer automatisk.")
            data = data / max_val
        else:
            print(f"  Advarsel: clipping fundet (max = {max_val:.4f}). Clipper til [-1, 1].")
            data = np.clip(data, -1.0, 1.0)
    return data

def get_audio_files(folder: Path, recursive: bool, allowed_extensions: set[str]):
    if recursive:
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in allowed_extensions]
    else:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in allowed_extensions]
    return sorted(files)

def process_file(input_file: Path, output_file: Path, noise_cache: dict):
    print(f"Behandler: {input_file.name}")

    # Load original lyd
    y_old, sr_old = librosa.load(str(input_file), sr=None, mono=True)

    validate_cutoffs(LOWCUT, HIGHCUT, sr_old)

    # Hent eller lav støjreference i korrekt samplerate
    if sr_old not in noise_cache:
        print(f"  Resampler støjreference til {sr_old} Hz...")
        y_noise, sr_noise = librosa.load(str(NOISE_FILE), sr=None, mono=True)

        if sr_noise != sr_old:
            y_noise = librosa.resample(y_noise, orig_sr=sr_noise, target_sr=sr_old)

        noise_cache[sr_old] = y_noise

    y_noise = noise_cache[sr_old]

    # Noise reduction
    y_clean = nr.reduce_noise(
        y=y_old,
        sr=sr_old,
        y_noise=y_noise,
        prop_decrease=PROP_DECREASE,
        stationary=STATIONARY
    )

    # Frekvensfilter
    y_filtered = bandpass_filter(y_clean, sr_old, LOWCUT, HIGHCUT, FILTER_ORDER)

    # Gain
    y_gained = apply_gain_db(y_filtered, GAIN_DB)

    # Clipping check
    y_final = handle_clipping(y_gained, AUTO_NORMALIZE_IF_CLIPPING)

    # Gem
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_file), y_final, sr_old)

    print(f"  Gemt som: {output_file.name}")

# ============================================================
# HOVEDPROGRAM
# ============================================================

def main():
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Inputmappen findes ikke:\n{INPUT_FOLDER}")

    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"Output-roden findes ikke:\n{OUTPUT_ROOT}")

    if not NOISE_FILE.exists():
        raise FileNotFoundError(f"Støjfilen findes ikke:\n{NOISE_FILE}")

    # Opret outputmappe med samme navn som inputmappen
    output_folder = OUTPUT_ROOT / INPUT_FOLDER.name
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Inputmappe : {INPUT_FOLDER}")
    print(f"Outputmappe: {output_folder}")
    print(f"Støjfil    : {NOISE_FILE}")
    print()

    audio_files = get_audio_files(INPUT_FOLDER, RECURSIVE, FILE_EXTENSIONS)

    if not audio_files:
        print("Ingen lydfiler fundet.")
        return

    print(f"Fandt {len(audio_files)} lydfil(er).")
    print()

    noise_cache = {}
    failed_files = []

    for input_file in audio_files:
        try:
            output_name = f"{input_file.stem}{OUTPUT_SUFFIX}{input_file.suffix}"
            output_file = output_folder / output_name
            process_file(input_file, output_file, noise_cache)
        except Exception as e:
            print(f"  FEJL i {input_file.name}: {e}")
            failed_files.append((input_file.name, str(e)))

    print()
    print("Batch-kørsel færdig.")
    print(f"Output ligger i:\n{output_folder}")

    if failed_files:
        print("\nFiler med fejl:")
        for filename, error in failed_files:
            print(f"- {filename}: {error}")
    else:
        print("\nAlle filer blev behandlet uden fejl.")

if __name__ == "__main__":
    main()