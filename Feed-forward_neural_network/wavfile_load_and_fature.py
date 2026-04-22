def load_wav(self, file_path):
        """Load WAV file, create spectrogram, and extract features"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            
            # Create spectrogram
            frequencies, times, spec_data = signal.spectrogram(y, sr)
            
            # Convert to dB scale
            spec_db = 10 * np.log10(spec_data + 1e-10)
            
            # Extract features from spectrogram
            features = []
            
            # Temporal features (statistics across time)
            features.append(np.mean(spec_db))        # Mean power
            features.append(np.std(spec_db))         # Std deviation
            features.append(np.max(spec_db))         # Max power
            features.append(np.min(spec_db))         # Min power
            
            # Spectral features (statistics across frequency)
            mean_spectrum = np.mean(spec_db, axis=1)  # Mean across time
            features.append(np.mean(mean_spectrum))   # Overall spectral mean
            features.append(np.std(mean_spectrum))    # Spectral std
            
            # Spectral centroid (weighted average of frequencies)
            spectral_mean = np.sum(frequencies[:, np.newaxis] * spec_db, axis=0) / (np.sum(spec_db, axis=0) + 1e-10)
            features.append(np.mean(spectral_mean))
            features.append(np.std(spectral_mean))
            
            # Energy in different frequency bands (up to 1kHz max)
            very_low_freq_idx = np.where(frequencies < 250)[0]
            low_mid_freq_idx = np.where((frequencies >= 250) & (frequencies < 500))[0]
            mid_freq_idx = np.where((frequencies >= 500) & (frequencies < 750))[0]
            high_freq_idx = np.where((frequencies >= 750) & (frequencies <= 1000))[0]
            
            if len(very_low_freq_idx) > 0:
                features.append(np.mean(spec_db[very_low_freq_idx]))  # Very low freq energy (0-250 Hz)
            if len(low_mid_freq_idx) > 0:
                features.append(np.mean(spec_db[low_mid_freq_idx]))   # Low-mid freq energy (250-500 Hz)
            if len(mid_freq_idx) > 0:
                features.append(np.mean(spec_db[mid_freq_idx]))       # Mid freq energy (500-750 Hz)
            if len(high_freq_idx) > 0:
                features.append(np.mean(spec_db[high_freq_idx]))      # High freq energy (750-1000 Hz)
            
            return np.array(features, dtype=float)
        
        except Exception as e:
            print(f"    WARNING: Could not extract spectrogram from {Path(file_path).name}: {str(e)}")
            return None