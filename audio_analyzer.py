"""
Stage 1: Offline Audio Analysis Module
Extracts audio characteristics from reference beatbox recordings
"""
import librosa
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import json
from pathlib import Path
import config


class AudioAnalyzer:
    """Analyzes reference audio and extracts processing parameters"""

    def __init__(self, audio_path):
        """
        Initialize analyzer with reference audio file

        Args:
            audio_path: Path to reference audio file
        """
        self.audio_path = audio_path
        self.audio = None
        self.sr = None
        self.duration = None

        # Analysis results
        self.eq_curve = None
        self.compression_params = None
        self.dynamic_range = None
        self.spectral_profile = None
        self.harmonic_content = None
        self.transient_profile = None

    def load_audio(self):
        """Load audio file and resample to target sample rate"""
        print(f"Loading audio: {self.audio_path}")
        self.audio, self.sr = librosa.load(
            self.audio_path,
            sr=config.SAMPLE_RATE,
            mono=True
        )
        self.duration = len(self.audio) / self.sr
        print(f"Loaded {self.duration:.2f}s of audio at {self.sr}Hz")

    def analyze_frequency_spectrum(self):
        """
        Analyze frequency spectrum to extract EQ curve
        Uses STFT with long windows for high frequency resolution
        """
        print("Analyzing frequency spectrum...")

        # Compute STFT
        stft = librosa.stft(
            self.audio,
            n_fft=config.FFT_SIZE,
            hop_length=config.HOP_LENGTH,
            window=config.WINDOW_TYPE
        )

        # Get magnitude spectrum
        magnitude = np.abs(stft)

        # Average over time to get long-term spectral profile
        avg_magnitude = np.mean(magnitude, axis=1)

        # Convert to dB
        avg_magnitude_db = librosa.amplitude_to_db(avg_magnitude, ref=np.max)

        # Get frequencies for the FFT bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=config.FFT_SIZE)

        # Store spectral profile
        self.spectral_profile = {
            'frequencies': freqs.tolist(),
            'magnitudes_db': avg_magnitude_db.tolist()
        }

        # Extract EQ curve at standard frequencies
        self.eq_curve = self._extract_eq_parameters(freqs, avg_magnitude_db)

        print(f"Extracted EQ curve with {len(self.eq_curve)} bands")

    def _extract_eq_parameters(self, freqs, magnitude_db):
        """
        Extract EQ parameters at standard frequency bands

        Args:
            freqs: Array of frequency values
            magnitude_db: Magnitude spectrum in dB

        Returns:
            List of EQ band parameters
        """
        eq_bands = []

        for center_freq in config.EQ_FREQUENCIES:
            # Find closest frequency bin
            idx = np.argmin(np.abs(freqs - center_freq))

            # Average over nearby bins for smoothing
            window = 10
            start_idx = max(0, idx - window)
            end_idx = min(len(magnitude_db), idx + window)

            avg_gain = np.mean(magnitude_db[start_idx:end_idx])

            # Normalize to make it relative to overall spectrum
            overall_avg = np.mean(magnitude_db)
            relative_gain = avg_gain - overall_avg

            eq_bands.append({
                'frequency': center_freq,
                'gain_db': float(relative_gain),
                'q_factor': 1.0  # Default Q factor
            })

        return eq_bands

    def analyze_dynamics(self):
        """
        Analyze dynamic range and estimate compression parameters
        """
        print("Analyzing dynamics and compression...")

        # Calculate RMS energy in windows
        window_samples = int(config.RMS_WINDOW_MS * self.sr / 1000)
        hop_samples = window_samples // 2

        # Compute RMS energy
        rms = librosa.feature.rms(
            y=self.audio,
            frame_length=window_samples,
            hop_length=hop_samples
        )[0]

        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Calculate dynamic range statistics
        dynamic_range = np.max(rms_db) - np.min(rms_db)
        rms_std = np.std(rms_db)
        rms_mean = np.mean(rms_db)

        self.dynamic_range = {
            'range_db': float(dynamic_range),
            'std_db': float(rms_std),
            'mean_db': float(rms_mean)
        }

        # Estimate compression parameters
        self.compression_params = self._estimate_compression(rms_db)

        print(f"Dynamic range: {dynamic_range:.2f} dB")
        print(f"Estimated compression ratio: {self.compression_params['ratio']:.2f}:1")

    def _estimate_compression(self, rms_db):
        """
        Estimate compressor settings from RMS envelope

        Args:
            rms_db: RMS values in dB

        Returns:
            Dictionary of compression parameters
        """
        # Analyze the distribution of levels
        rms_sorted = np.sort(rms_db)

        # Peak level (max)
        peak_db = rms_sorted[-1]

        # Estimate threshold as 75th percentile
        threshold_db = np.percentile(rms_db, 75)

        # Estimate ratio from how compressed the peaks are
        # Higher values = more compressed
        peak_range = peak_db - threshold_db
        total_range = self.dynamic_range['range_db']

        # If peaks are squashed relative to total range, higher compression
        if total_range > 0:
            compression_factor = 1.0 - (peak_range / total_range)
            ratio = 1.0 + (compression_factor * 7.0)  # Scale to 1:1 to 8:1
        else:
            ratio = 1.0

        # Clamp ratio
        ratio = np.clip(ratio, 1.0, 10.0)

        return {
            'threshold_db': float(threshold_db),
            'ratio': float(ratio),
            'attack_ms': config.ATTACK_TIME_MS,
            'release_ms': config.RELEASE_TIME_MS,
            'knee_db': 3.0,  # Soft knee
            'makeup_gain_db': 0.0
        }

    def analyze_transients(self):
        """
        Analyze transient characteristics
        Useful for preserving attack characteristics
        """
        print("Analyzing transients...")

        # Detect onsets (transients)
        onset_env = librosa.onset.onset_strength(
            y=self.audio,
            sr=self.sr
        )

        # Get onset times
        onset_frames = librosa.onset.onset_detect(
            y=self.audio,
            sr=self.sr,
            units='frames'
        )

        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)

        # Calculate onset strength statistics
        onset_strength_mean = np.mean(onset_env)
        onset_strength_std = np.std(onset_env)

        self.transient_profile = {
            'num_onsets': len(onset_times),
            'onset_times': onset_times.tolist(),
            'onset_strength_mean': float(onset_strength_mean),
            'onset_strength_std': float(onset_strength_std),
            'onset_rate_per_second': len(onset_times) / self.duration
        }

        print(f"Detected {len(onset_times)} transients")

    def analyze_harmonics(self):
        """
        Analyze harmonic content to detect saturation characteristics
        """
        print("Analyzing harmonic content...")

        # Get harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(self.audio)

        # Calculate harmonic-to-percussive ratio
        harmonic_energy = np.sum(harmonic ** 2)
        percussive_energy = np.sum(percussive ** 2)
        total_energy = harmonic_energy + percussive_energy

        hp_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.5

        # Analyze spectral characteristics
        spectral_centroid = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.audio, sr=self.sr)

        self.harmonic_content = {
            'harmonic_to_percussive_ratio': float(hp_ratio),
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff))
        }

        print(f"Harmonic/Percussive ratio: {hp_ratio:.3f}")

    def analyze_all(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 60)
        print("BEATBOX AUDIO ANALYSIS")
        print("=" * 60 + "\n")

        self.load_audio()
        self.analyze_frequency_spectrum()
        self.analyze_dynamics()
        self.analyze_transients()
        self.analyze_harmonics()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60 + "\n")

    def get_preset(self):
        """
        Get complete preset data structure

        Returns:
            Dictionary containing all extracted parameters
        """
        return {
            'metadata': {
                'source_file': str(self.audio_path),
                'duration_seconds': self.duration,
                'sample_rate': self.sr
            },
            'eq_curve': self.eq_curve,
            'compression': self.compression_params,
            'dynamic_range': self.dynamic_range,
            'spectral_profile': self.spectral_profile,
            'harmonic_content': self.harmonic_content,
            'transient_profile': self.transient_profile
        }

    def save_preset(self, preset_name):
        """
        Save extracted parameters as a preset

        Args:
            preset_name: Name for the preset file
        """
        preset = self.get_preset()
        preset_path = config.get_preset_path(preset_name)

        with open(preset_path, 'w') as f:
            json.dump(preset, f, indent=2)

        print(f"\nPreset saved: {preset_path}")
        return preset_path


def analyze_audio_file(audio_path, preset_name=None):
    """
    Convenience function to analyze audio and save preset

    Args:
        audio_path: Path to audio file
        preset_name: Name for preset (defaults to audio filename)

    Returns:
        Path to saved preset
    """
    analyzer = AudioAnalyzer(audio_path)
    analyzer.analyze_all()

    if preset_name is None:
        preset_name = Path(audio_path).stem

    return analyzer.save_preset(preset_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_analyzer.py <audio_file> [preset_name]")
        sys.exit(1)

    audio_file = sys.argv[1]
    preset_name = sys.argv[2] if len(sys.argv) > 2 else None

    analyze_audio_file(audio_file, preset_name)
