import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")   # Eksterne plotvinduer
import matplotlib.pyplot as plt

# ============================================================
# INDSTILLINGER
# ============================================================

# Kun én inputfil
INPUT_AUDIO_FILE = Path(
    r"C:\Users\mjuul\OneDrive - Aalborg Universitet\Dokumenter\GitHub\VT2\Data fra tidligere project\Dataset\Extrinsic data\N\e030520236057.wav"
    
)

# Støjreference
NOISE_AUDIO_FILE = Path(
    r"C:\Users\mjuul\OneDrive - Aalborg Universitet\Dokumenter\GitHub\VT2\Soundcleaning\Optaget_støj.wav"
)

# Hvor outputfilen skal gemmes
OUTPUT_ROOT = Path(
    r"C:\Users\mjuul\OneDrive - Aalborg Universitet\Dokumenter\GitHub\VT2\Soundcleaning"
)

# Filnavnssuffix, så fx fil.wav bliver til fil_P.wav
OUTPUT_SUFFIX = "_P"

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

# Spectrogram-indstillinger
N_FFT = 2048
HOP_LENGTH = 512
WINDOW = "hann"
DB_FLOOR = -120  # nederste dB-grænse i spektrogrammet

# Y-akser:
# None = brug hele Nyquist-området automatisk
YMAX_INPUT = None
YMAX_NOISE = None
YMAX_DENOISED = None

# Disse to skal kun vise området op til HIGHCUT
YMAX_FILTERED = HIGHCUT
YMAX_GAINED = HIGHCUT


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

def plot_spectrogram(audio, sr, title, fig_num, ymax=None):
    """
    Plotter spectrogram i sit eget vindue.
    """
    D = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW)
    S_db = librosa.amplitude_to_db(np.abs(D) + 1e-12, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    fig = plt.figure(fig_num, figsize=(14, 5))
    ax = fig.add_subplot(111)

    mesh = ax.pcolormesh(
        times,
        freqs,
        S_db,
        shading="gouraud",
        vmin=DB_FLOOR,
        vmax=0
    )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Time [s]", fontsize=14)
    ax.set_ylabel("Frequency [Hz]", fontsize=14)

    if ymax is None:
        ax.set_ylim(0, sr / 2)
    else:
        ax.set_ylim(0, ymax)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Magnitude [dB]", fontsize=14)

    fig.tight_layout()
    return fig

def process_single_file():
    if not INPUT_AUDIO_FILE.exists():
        raise FileNotFoundError(f"Inputfilen findes ikke:\n{INPUT_AUDIO_FILE}")

    if not NOISE_AUDIO_FILE.exists():
        raise FileNotFoundError(f"Støjfilen findes ikke:\n{NOISE_AUDIO_FILE}")

    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"Outputmappen findes ikke:\n{OUTPUT_ROOT}")

    print(f"Inputfil : {INPUT_AUDIO_FILE}")
    print(f"Støjfil  : {NOISE_AUDIO_FILE}")
    print(f"Output   : {OUTPUT_ROOT}")
    print()

    # --------------------------------------------------------
    # 1. Load input lydfil
    # --------------------------------------------------------
    print("1) Loader input lydfil...")
    y_old, sr_old = librosa.load(str(INPUT_AUDIO_FILE), sr=None, mono=True)
    validate_cutoffs(LOWCUT, HIGHCUT, sr_old)

    # --------------------------------------------------------
    # 2. Plot spectrogram af input lydfil
    # --------------------------------------------------------
    print("2) Plotter spectrogram af input...")
    plot_spectrogram(
        y_old,
        sr_old,
        "Original sounddata",
        fig_num=1,
        ymax=YMAX_INPUT
    )

    # --------------------------------------------------------
    # 3. Load støjfil
    # --------------------------------------------------------
    print("3) Loader støjfil...")
    y_noise, sr_noise = librosa.load(str(NOISE_AUDIO_FILE), sr=None, mono=True)

    if sr_noise != sr_old:
        print(f"   Resampler støjfil fra {sr_noise} Hz til {sr_old} Hz...")
        y_noise = librosa.resample(y_noise, orig_sr=sr_noise, target_sr=sr_old)

    # --------------------------------------------------------
    # 4. Plot spectrogram af støj
    # --------------------------------------------------------
    print("4) Plotter spectrogram af støj...")
    plot_spectrogram(
        y_noise,
        sr_old,
        "Reference-noise",
        fig_num=2,
        ymax=YMAX_NOISE
    )

    # --------------------------------------------------------
    # 5. Støjreduktion og plot
    # --------------------------------------------------------
    print("5) Kører støjreduktion...")
    y_clean = nr.reduce_noise(
        y=y_old,
        sr=sr_old,
        y_noise=y_noise,
        prop_decrease=PROP_DECREASE,
        stationary=STATIONARY
    )

    print("   Plotter spectrogram efter støjreduktion...")
    plot_spectrogram(
        y_clean,
        sr_old,
        "Noise reduction",
        fig_num=3,
        ymax=YMAX_DENOISED
    )

    # --------------------------------------------------------
    # 6. Cut alt over 1000 Hz og plot
    # --------------------------------------------------------
    print("6) Filtrerer til 1-1000 Hz...")
    y_filtered = bandpass_filter(y_clean, sr_old, LOWCUT, HIGHCUT, FILTER_ORDER)

    print("   Plotter spectrogram efter frekvens-cut...")
    plot_spectrogram(
        y_filtered,
        sr_old,
        f"Frequency-isolation ({LOWCUT}-{HIGHCUT} Hz)",
        fig_num=4,
        ymax=YMAX_FILTERED
    )

    # --------------------------------------------------------
    # 7. Gain med 14 dB og plot
    # --------------------------------------------------------
    print("7) Tilføjer gain...")
    y_gained = apply_gain_db(y_filtered, GAIN_DB)
    y_final = handle_clipping(y_gained, AUTO_NORMALIZE_IF_CLIPPING)

    print("   Plotter spectrogram efter gain...")
    plot_spectrogram(
        y_final,
        sr_old,
        f"Gain (+{GAIN_DB:.1f} dB)",
        fig_num=5,
        ymax=YMAX_GAINED
    )

    # Gem outputfil
    output_name = f"{INPUT_AUDIO_FILE.stem}{OUTPUT_SUFFIX}{INPUT_AUDIO_FILE.suffix}"
    output_file = OUTPUT_ROOT / output_name
    sf.write(str(output_file), y_final, sr_old)

    print()
    print(f"Output gemt som:\n{output_file}")

    # Åbn alle vinduer
    plt.show()


# ============================================================
# HOVEDPROGRAM
# ============================================================

if __name__ == "__main__":
    process_single_file()