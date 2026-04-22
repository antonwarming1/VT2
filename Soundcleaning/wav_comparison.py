import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Configuration
SOUND_FOLDER = r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2\Soundcleaning"
SUBFOLDER = "N"
WAV_FILE = "e030520235006_C.wav"

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

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: Spectrogram with linear scale
im1 = axes[0].pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data + 1e-10), shading='gouraud', cmap='viridis')
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Frequency (Hz)")
axes[0].set_title("Spectrogram (Linear Frequency, dB Power)")
axes[0].set_ylim([0, 1000])
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label("Power (dB)")

# Plot 2: Spectrogram with logarithmic scale
im2 = axes[1].pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data + 1e-10), shading='gouraud', cmap='viridis')
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_title("Spectrogram (Logarithmic Scale)")
axes[1].set_yscale('log')
axes[1].set_ylim([10, 1000])
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label("Power (dB)")

# Adjust layout and display
plt.tight_layout()
plt.show()
