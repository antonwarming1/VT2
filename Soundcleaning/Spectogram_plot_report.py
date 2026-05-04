import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt, freqz_sos
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")   # Eksterne plotvinduer
import matplotlib.pyplot as plt

# ============================================================
# INDSTILLINGER
# ============================================================

BASE_PATH = Path(r"C:\Users\Nicok\OneDrive - Aalborg Universitet\8. semester\Project\Github\VT2")

# Kun én inputfil
INPUT_AUDIO_FILE = BASE_PATH / r"Data fra tidligere project\Dataset\Extrinsic data\N\e030520236057.wav"

# Støjreference
NOISE_AUDIO_FILE = BASE_PATH / r"Soundcleaning\Optaget_støj.wav"

# Hvor outputfilen skal gemmes
OUTPUT_ROOT = BASE_PATH / "Soundcleaning"

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
DB_FLOOR = -40  # nederste dB-grænse i spektrogrammet

# Y-akser:
# None = brug hele Nyquist-området automatisk
YMAX_INPUT = None
YMAX_NOISE = None
YMAX_DENOISED = None

# Disse to skal kun vise området op til HIGHCUT
YMAX_FILTERED = 2000
YMAX_GAINED = 2000

# Hvilke plots der skal laves (True/False)
#                    [Input, Noise, Denoised, Bodeplot, Filtered, Gained]
PlotInput = np.array([False, False, False,    False,    True,     False])
PlotInput = np.array([True,  True,  True,     False,    True,     False])

# Vis plot titler (slås fra for at gemme til rapporten)
PLOT_TITLES = False

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

def lowpass_filter(data, samplerate, highcut, order=6, plot_response=False, fignum=1):
    nyquist = samplerate / 2
    high = highcut / nyquist

    sos = butter(order, high, btype="lowpass", output="sos")
    filtered = sosfiltfilt(sos, data, axis=0)
    if plot_response:
        plot_frequency_response(sos, nyquist, order, fig_num=fignum)

    return filtered

def plot_frequency_response(sos, nyquist, order, fig_num, save_path=None):
    """
    Plotter frekvensresponsen for Butterworth bandpass-filteret.
    """

    w, h = freqz_sos(sos)
    freqs = w * nyquist / np.pi
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)

    fig = plt.figure(fig_num, figsize=(14, 5))
    ax = fig.add_subplot(111)

    ax.plot(freqs, magnitude_db)
    if PLOT_TITLES:
        ax.set_title(f"Lowpass Butterworth Filter (order={order})", fontsize=18, fontfamily="Times New Roman")
    ax.set_xlabel("Frequency [Hz]", fontsize=14, fontfamily="Times New Roman")
    ax.set_ylabel("Magnitude [dB]", fontsize=14, fontfamily="Times New Roman")
    ax.set_xlim(1, nyquist)
    ax.set_xscale('log')
    ax.grid(True)

    fig.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"   Filterrespons gemt: {save_path.name}")
    
    return fig

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

def plot_spectrogram(audio, sr, fig_num, fig_name, ymax=None, save_path=None, plot_ON=True):
    """
    Plotter spectrogram i sit eget vindue og gemmer det hvis save_path er angivet.
    """

    if not plot_ON:
        return None

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
        vmax=0,
        rasterized=True
    )

    if PLOT_TITLES:
        ax.set_title(fig_name, fontsize=18, fontfamily="Times New Roman")
    ax.set_xlabel("Time [s]", fontsize=14, fontfamily="Times New Roman")
    ax.set_ylabel("Frequency [Hz]", fontsize=14, fontfamily="Times New Roman")

    if ymax is None:
        ax.set_ylim(0, sr / 2)
    else:
        ax.set_ylim(0, ymax)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Magnitude [dB]", fontsize=14, fontfamily="Times New Roman")

    fig.tight_layout()
    
    # Gem figuren hvis save_path er angivet
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"   Spektrogram gemt: {save_path.name}")
    
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
    print(f"   Samplerate: {sr_old} Hz")

    # --------------------------------------------------------
    # 2. Plot spectrogram af input lydfil
    # --------------------------------------------------------
    print("2) Plotter spectrogram af input...")
    plot_spectrogram(
        y_old,
        sr_old,
        fig_num=1,
        fig_name="Input Signal",
        ymax=YMAX_INPUT,
        plot_ON=PlotInput[0]

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
        fig_num=2,
        fig_name="Noise Signal",
        ymax=YMAX_NOISE,
        plot_ON=PlotInput[1]

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
        fig_num=3,
        fig_name="Denoised Signal",
        ymax=YMAX_DENOISED,
        plot_ON=PlotInput[2]
    )

    # --------------------------------------------------------
    # 6. Cut alt over 1000 Hz og plot
    # --------------------------------------------------------
    print("6) Filtrerer til 0-1000 Hz...")
    y_filtered = lowpass_filter(y_clean, sr_old, HIGHCUT, FILTER_ORDER, plot_response=PlotInput[3], fignum=4)

    print("   Plotter spectrogram efter frekvens-cut...")
    plot_spectrogram(
        y_filtered,
        sr_old,
        fig_num=5,
        fig_name="Filtered Signal",
        ymax=YMAX_FILTERED,
        plot_ON=PlotInput[4]
    )


    # --------------------------------------------------------
    # 7. Gain med 14 dB og gem output
    # --------------------------------------------------------
    print("7) Tilføjer gain...")
    y_gained = apply_gain_db(y_filtered, GAIN_DB)
    y_final = handle_clipping(y_gained, AUTO_NORMALIZE_IF_CLIPPING)

    print("   Plotter spectrogram efter gain...")
    plot_spectrogram(
        y_final,
        sr_old,
        fig_num=6,
        fig_name="Gained Signal",
        ymax=YMAX_GAINED,
        plot_ON=PlotInput[5]
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