"""
Basic Audio Analyzer - Foundation for Advanced Analysis
Provides frequency spectrum, dynamics, transients, and harmonic analysis
"""
import librosa
import numpy as np
from scipy import signal
from typing import Dict, List, Optional
import config


class AudioAnalyzer:
    """
    Basic audio analyzer for extracting core audio characteristics
    """

    def __init__(self, audio_path):
        """
        Initialize analyzer

        Args:
            audio_path: Path to audio file
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
        """Load audio file"""
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
        Analyze frequency spectrum and generate EQ curve
        """
        print("Analyzing frequency spectrum...")

        # Compute STFT
        stft = librosa.stft(self.audio, n_fft=4096, hop_length=512)
        mag_spectrum = np.abs(stft)

        # Average over time
        avg_spectrum = np.mean(mag_spectrum, axis=1)

        # Convert to dB
        avg_spectrum_db = librosa.amplitude_to_db(avg_spectrum, ref=np.max)

        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)

        # Define standard EQ bands (octave-based)
        eq_bands = [
            (20, 60, "Sub Bass"),
            (60, 250, "Bass"),
            (250, 500, "Low Mids"),
            (500, 2000, "Mids"),
            (2000, 4000, "High Mids"),
            (4000, 8000, "Presence"),
            (8000, 16000, "Brilliance"),
            (16000, 20000, "Air")
        ]

        # Calculate average energy per band
        eq_curve = []

        for low_freq, high_freq, band_name in eq_bands:
            # Find indices for this band
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)

            if np.any(band_mask):
                # Average dB level in this band
                band_avg_db = np.mean(avg_spectrum_db[band_mask])

                eq_curve.append({
                    'band_name': band_name,
                    'freq_low': low_freq,
                    'freq_high': high_freq,
                    'center_freq': np.sqrt(low_freq * high_freq),  # Geometric mean
                    'level_db': float(band_avg_db)
                })

        # Normalize to mid-range (500-2000Hz)
        mid_band_level = next(b['level_db'] for b in eq_curve if b['band_name'] == "Mids")

        # Apply scaling factor to prevent extreme values (same as multiband)
        SCALING_FACTOR = 0.20

        for band in eq_curve:
            relative_db = band['level_db'] - mid_band_level
            scaled_db = relative_db * SCALING_FACTOR
            clamped_db = np.clip(scaled_db, -6.0, 6.0)
            band['gain_db'] = float(clamped_db)

        self.eq_curve = eq_curve

        # Store spectral profile
        self.spectral_profile = {
            'centroid': float(np.mean(librosa.feature.spectral_centroid(y=self.audio, sr=self.sr))),
            'rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=self.audio, sr=self.sr))),
            'bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=self.audio, sr=self.sr))),
        }

        print(f"Analyzed {len(eq_curve)} EQ bands")

    def analyze_dynamics(self):
        """
        Analyze dynamics and estimate compression parameters
        """
        print("Analyzing dynamics...")

        # Calculate RMS over time
        frame_length = 2048
        hop_length = 512

        rms = librosa.feature.rms(
            y=self.audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Calculate dynamic range
        peak_db = 20 * np.log10(np.max(np.abs(self.audio)) + 1e-10)
        rms_avg_db = 20 * np.log10(np.mean(np.abs(self.audio)) + 1e-10)

        self.dynamic_range = {
            'peak_db': float(peak_db),
            'rms_db': float(rms_avg_db),
            'crest_factor_db': float(peak_db - rms_avg_db),
            'rms_std_db': float(np.std(rms_db))
        }

        # Estimate compression parameters
        crest_factor_db = self.dynamic_range['crest_factor_db']

        # Estimate ratio (capped at 4:1 for live use)
        if crest_factor_db > 15:
            ratio = 1.0
        elif crest_factor_db > 12:
            ratio = 1.5
        elif crest_factor_db > 9:
            ratio = 2.0
        elif crest_factor_db > 7:
            ratio = 3.0
        else:
            ratio = 4.0

        # Estimate threshold based on RMS distribution
        threshold_db = float(np.percentile(rms_db, 30))  # 30th percentile

        # Estimate attack/release based on RMS variation
        rms_diff = np.diff(rms_db)
        avg_rise_time = np.mean(rms_diff[rms_diff > 0]) if np.any(rms_diff > 0) else 1.0

        # Attack: faster for more transient material
        attack_ms = np.clip(5.0 / (avg_rise_time + 0.1), 1.0, 30.0)

        # Release: based on material density
        release_ms = np.clip(50.0 + (100.0 * (1.0 - avg_rise_time / 10.0)), 50.0, 300.0)

        self.compression_params = {
            'threshold_db': float(threshold_db),
            'ratio': float(ratio),
            'attack_ms': float(attack_ms),
            'release_ms': float(release_ms),
            'knee_db': 3.0,
            'makeup_gain_db': float(-threshold_db / ratio)  # Approximate makeup gain
        }

        print(f"Estimated compression: {ratio:.1f}:1, threshold: {threshold_db:.1f} dB")

    def analyze_transients(self):
        """
        Analyze transient characteristics
        """
        print("Analyzing transients...")

        # Detect onsets
        onset_env = librosa.onset.onset_strength(y=self.audio, sr=self.sr)
        onsets = librosa.onset.onset_detect(
            y=self.audio,
            sr=self.sr,
            units='time',
            backtrack=True
        )

        if len(onsets) == 0:
            self.transient_profile = {
                'num_transients': 0,
                'avg_strength': 0.0,
                'transient_density': 0.0,
                'peak_strength': 0.0
            }
            return

        # Calculate transient strength
        onset_frames = librosa.time_to_frames(onsets, sr=self.sr)
        onset_strengths = onset_env[onset_frames]

        self.transient_profile = {
            'num_transients': int(len(onsets)),
            'avg_strength': float(np.mean(onset_strengths)),
            'peak_strength': float(np.max(onset_strengths)),
            'transient_density': float(len(onsets) / self.duration)  # Transients per second
        }

        print(f"Detected {len(onsets)} transients ({self.transient_profile['transient_density']:.1f}/sec)")

    def analyze_harmonics(self):
        """
        Analyze harmonic content
        """
        print("Analyzing harmonics...")

        # Compute harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(self.audio, margin=2.0)

        # Calculate energy ratio
        harmonic_energy = np.sum(harmonic ** 2)
        percussive_energy = np.sum(percussive ** 2)
        total_energy = harmonic_energy + percussive_energy

        harmonic_ratio = harmonic_energy / (total_energy + 1e-10)
        percussive_ratio = percussive_energy / (total_energy + 1e-10)

        # Analyze harmonic spectrum
        harmonic_stft = librosa.stft(harmonic, n_fft=4096)
        harmonic_mag = np.abs(harmonic_stft)
        harmonic_avg = np.mean(harmonic_mag, axis=1)

        # Find dominant frequencies
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)
        peaks, properties = signal.find_peaks(
            harmonic_avg,
            height=np.max(harmonic_avg) * 0.1,
            distance=20
        )

        dominant_freqs = freqs[peaks][:5]  # Top 5 harmonic frequencies

        self.harmonic_content = {
            'harmonic_ratio': float(harmonic_ratio),
            'percussive_ratio': float(percussive_ratio),
            'dominant_frequencies': dominant_freqs.tolist(),
            'harmonic_energy_db': float(20 * np.log10(np.sqrt(harmonic_energy) + 1e-10)),
            'percussive_energy_db': float(20 * np.log10(np.sqrt(percussive_energy) + 1e-10))
        }

        print(f"Harmonic ratio: {harmonic_ratio:.2%}, Percussive ratio: {percussive_ratio:.2%}")

    def analyze_all(self):
        """
        Run all analysis methods
        """
        print("\n" + "=" * 60)
        print("BASIC AUDIO ANALYSIS")
        print("=" * 60 + "\n")

        self.load_audio()
        self.analyze_frequency_spectrum()
        self.analyze_dynamics()
        self.analyze_transients()
        self.analyze_harmonics()

        print("\n" + "=" * 60)
        print("BASIC ANALYSIS COMPLETE")
        print("=" * 60 + "\n")

    def get_analysis_summary(self) -> Dict:
        """Get complete analysis summary"""
        return {
            'eq_curve': self.eq_curve,
            'compression': self.compression_params,
            'dynamic_range': self.dynamic_range,
            'spectral_profile': self.spectral_profile,
            'harmonic_content': self.harmonic_content,
            'transient_profile': self.transient_profile
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_analyzer.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    analyzer = AudioAnalyzer(audio_file)
    analyzer.analyze_all()

    # Print summary
    summary = analyzer.get_analysis_summary()
    import json
    print("\nAnalysis Summary:")
    print(json.dumps(summary, indent=2))
