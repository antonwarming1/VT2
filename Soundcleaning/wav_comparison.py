import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Configuration
SOUND_FOLDER = r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2\Soundcleaning"
SUBFOLDER = "N"
WAV_FILE = "e030520235006_C.wav"

# Envelope extraction parameters (adjust these to modify envelope)
ENVELOPE_FILTER_FREQ = 1000  # Cutoff frequency for low-pass filter (Hz)
ENVELOPE_SMOOTHING = 50  # Smoothing window size (samples)

# Build the full path
wav_path = os.path.join(SOUND_FOLDER, SUBFOLDER, WAV_FILE)

# Load the WAV file
try:
    sample_rate, audio_data = wavfile.read(wav_path)
    print(f"Loaded: {WAV_FILE}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
except FileNotFoundError:
    print(f"Error: File not found at {wav_path}")
    exit()

# Create spectrogram
from scipy import signal
frequencies, times, spectrogram_data = signal.spectrogram(audio_data, sample_rate)

# Extract envelope using Hilbert transform
analytic_signal = signal.hilbert(audio_data)
envelope = np.abs(analytic_signal)

# Apply low-pass filter to envelope
nyquist = sample_rate / 2
normalized_freq = ENVELOPE_FILTER_FREQ / nyquist
if normalized_freq < 1:  # Only apply if frequency is valid
    b, a = signal.butter(4, normalized_freq, btype='low')
    envelope = signal.filtfilt(b, a, envelope)

# Apply smoothing to envelope
if ENVELOPE_SMOOTHING > 1:
    envelope = signal.savgol_filter(envelope, window_length=min(ENVELOPE_SMOOTHING, len(envelope) if len(envelope) % 2 == 1 else len(envelope) - 1), polyorder=3)

# Create spectrogram of the envelope
freq_env, times_env, spec_envelope = signal.spectrogram(envelope, sample_rate)

# Calculate maximum values for reference (0 dB)
max_spec = np.max(spectrogram_data)
max_env_spec = np.max(spec_envelope)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Plot 1: Spectrogram with linear scale
im1 = axes[0].pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data / max_spec + 1e-10), shading='gouraud', cmap='viridis')
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Frequency (Hz)")
axes[0].set_title("Spectrogram (Linear Frequency, dB Power)")
axes[0].set_ylim([0, 1000])
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label("Power (dB)")

# Plot 2: Spectrogram with logarithmic scale
im2 = axes[1].pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data / max_spec + 1e-10), shading='gouraud', cmap='viridis')
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_title("Spectrogram (Logarithmic Scale)")
axes[1].set_yscale('log')
axes[1].set_ylim([10, 1000])
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label("Power (dB)")

# Plot 3: Envelope Spectrogram
im3 = axes[2].pcolormesh(times_env, freq_env, 10 * np.log10(spec_envelope / max_env_spec + 1e-10), shading='gouraud', cmap='viridis')
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Frequency (Hz)")
axes[2].set_title("Envelope Spectrogram (Logarithmic Scale)")
axes[2].set_yscale('log')
axes[2].set_ylim([10, 1000])
cbar3 = plt.colorbar(im3, ax=axes[2])
cbar3.set_label("Power (dB)")

# Adjust layout and display
plt.tight_layout()
plt.show()
