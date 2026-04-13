"""
Plot one random WAV file from each class folder for comparison
Classes: N, NS, OT, P, UT
Located in: Data fra tidligere project\Dataset\Extrinsic data (clean)
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path

# Define paths
base_path = r"C:\Users\emil_\OneDrive - Aalborg Universitet\VT2\VT2\Data fra tidligere project\Dataset\Extrinsic data (clean)"
classes = ["N", "NS", "OT", "P", "UT"]

# Dictionary to store selected files and their data
wav_data = {}
selected_files = {}

# Select one random file from each class folder
for class_name in classes:
    class_path = os.path.join(base_path, class_name)
    
    # Get all WAV files in the class folder
    wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    
    if wav_files:
        # Select one random file
        random_file = random.choice(wav_files)
        file_path = os.path.join(class_path, random_file)
        selected_files[class_name] = random_file
        
        # Read the WAV file
        sample_rate, audio_data = wavfile.read(file_path)
        wav_data[class_name] = {
            'sample_rate': sample_rate,
            'audio_data': audio_data,
            'filename': random_file,
            'duration': len(audio_data) / sample_rate
        }
        
        print(f"Class {class_name}: {random_file}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {wav_data[class_name]['duration']:.2f} seconds")
        print(f"  Audio shape: {audio_data.shape}")
        print()
    else:
        print(f"No WAV files found in {class_path}")

# Plot the WAV files
fig, axes = plt.subplots(len(classes), 1, figsize=(14, 12))
fig.suptitle('Comparison of One Random WAV File from Each Class', fontsize=16, fontweight='bold')

for idx, class_name in enumerate(classes):
    if class_name in wav_data:
        data = wav_data[class_name]
        sample_rate = data['sample_rate']
        audio_data = data['audio_data']
        
        # Create time axis
        time_axis = np.arange(len(audio_data)) / sample_rate
        
        # Plot
        axes[idx].plot(time_axis, audio_data, linewidth=0.5, color='steelblue')
        axes[idx].set_title(f"Class {class_name}: {data['filename']}", fontweight='bold')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].grid(True, alpha=0.3)
        
        # Add duration info
        axes[idx].text(0.98, 0.95, f"Duration: {data['duration']:.2f}s | SR: {sample_rate} Hz", 
                      transform=axes[idx].transAxes, 
                      verticalalignment='top', 
                      horizontalalignment='right',
                      fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[-1].set_xlabel('Time (seconds)')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'wav_comparison_plot.png'), dpi=150, bbox_inches='tight')
print("Plot saved as 'wav_comparison_plot.png'")

plt.show()
