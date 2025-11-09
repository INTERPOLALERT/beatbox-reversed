"""
Advanced Audio Analyzer - Professional Grade
Enhanced analysis with multiband processing, formant extraction,
and per-sound-type preset generation
"""
import librosa
import numpy as np
from scipy import signal
import json
from pathlib import Path
from typing import Dict, List, Optional
import config
from multiband_processor import create_default_crossover
from sound_classifier import OnsetBasedClassifier


class AdvancedAudioAnalyzer:
    """
    Professional-grade audio analyzer with multiband and per-sound-type analysis
    """

    def __init__(self, audio_path):
        """
        Initialize analyzer

        Args:
            audio_path: Path to reference audio file
        """
        self.audio_path = audio_path
        self.audio = None
        self.sr = None
        self.duration = None

        # Analysis results
        self.global_analysis = {}
        self.multiband_analysis = {}
        self.per_sound_analysis = {}
        self.formants = None

        # Sound classifier
        self.classifier = OnsetBasedClassifier()

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

    def analyze_multiband_eq(self, num_bands: int = 4):
        """
        Analyze EQ characteristics per frequency band

        Args:
            num_bands: Number of frequency bands (4 or 8)
        """
        print(f"\nAnalyzing {num_bands}-band EQ characteristics...")

        # Create crossover
        crossover = create_default_crossover(num_bands=num_bands, sample_rate=self.sr)

        # Split audio into bands
        bands = crossover.split_bands_offline(self.audio)

        # Analyze each band
        band_analysis = []

        for i, (band_audio, (ftype, sos)) in enumerate(zip(bands, crossover.filters)):
            # Calculate RMS energy
            rms = np.sqrt(np.mean(band_audio ** 2))
            rms_db = 20 * np.log10(rms + 1e-10)

            # Calculate peak level
            peak = np.max(np.abs(band_audio))
            peak_db = 20 * np.log10(peak + 1e-10)

            # Get frequency range
            if i == 0:
                freq_low = 20
                freq_high = crossover.crossover_freqs[0] if len(crossover.crossover_freqs) > 0 else 22000
            elif i == num_bands - 1:
                freq_low = crossover.crossover_freqs[-1]
                freq_high = 22000
            else:
                freq_low = crossover.crossover_freqs[i - 1]
                freq_high = crossover.crossover_freqs[i]

            band_info = {
                'band_index': i,
                'freq_range': [freq_low, freq_high],
                'center_freq': np.sqrt(freq_low * freq_high),  # Geometric mean
                'rms_db': float(rms_db),
                'peak_db': float(peak_db),
                'energy_ratio': float(rms ** 2),  # Relative energy
            }

            band_analysis.append(band_info)

        # Normalize energy ratios
        total_energy = sum(b['energy_ratio'] for b in band_analysis)
        for band in band_analysis:
            band['energy_ratio'] /= (total_energy + 1e-10)

        # Calculate relative gains (normalized to mid-band)
        mid_band_idx = num_bands // 2
        reference_rms = band_analysis[mid_band_idx]['rms_db']

        # CRITICAL FIX: Apply scaling + clamping to prevent mastering artifacts
        # from producing extreme multiband EQ values
        SCALING_FACTOR = 0.20  # Same as single-band EQ
        for band in band_analysis:
            # Step 1: Normalize to reference band
            relative_gain = band['rms_db'] - reference_rms

            # Step 2: Apply scaling factor
            scaled_gain = relative_gain * SCALING_FACTOR

            # Step 3: Clamp to ±6 dB for live stability
            clamped_gain = np.clip(scaled_gain, -6.0, 6.0)

            band['relative_gain_db'] = float(clamped_gain)

        self.multiband_analysis = {
            'num_bands': num_bands,
            'crossover_freqs': crossover.crossover_freqs,
            'bands': band_analysis
        }

        print(f"Analyzed {num_bands} frequency bands")

    def analyze_formants(self):
        """
        Extract formant frequencies using LPC analysis
        Formants characterize vocal tract filtering
        """
        print("\nAnalyzing formants (vocal tract characteristics)...")

        # LPC order (research recommends: 2 + sr/1000)
        lpc_order = int(2 + self.sr / 1000)

        # Frame-based analysis
        frame_length = 2048
        hop_length = 512

        # Get frames
        frames = librosa.util.frame(
            self.audio,
            frame_length=frame_length,
            hop_length=hop_length
        )

        formants_over_time = []

        for frame in frames.T:
            # Calculate LPC coefficients
            lpc_coeffs = librosa.lpc(frame, order=lpc_order)

            # Find roots of LPC polynomial
            roots = np.roots(lpc_coeffs)

            # Keep roots inside unit circle (stable poles)
            roots = roots[np.abs(roots) < 1.0]

            # Convert to frequencies
            angles = np.angle(roots)
            freqs = angles * (self.sr / (2 * np.pi))

            # Keep positive frequencies
            freqs = freqs[freqs > 0]

            # Sort by frequency
            freqs = np.sort(freqs)

            formants_over_time.append(freqs[:5] if len(freqs) >= 5 else freqs)

        # Average formants over time
        if len(formants_over_time) > 0:
            # Get most common number of formants
            formant_counts = [len(f) for f in formants_over_time if len(f) > 0]
            if len(formant_counts) > 0:
                most_common_count = max(set(formant_counts), key=formant_counts.count)

                # Filter to frames with that many formants
                filtered_formants = [f for f in formants_over_time if len(f) == most_common_count]

                if len(filtered_formants) > 0:
                    # Average each formant
                    avg_formants = np.mean(filtered_formants, axis=0)

                    self.formants = {
                        'frequencies': avg_formants.tolist(),
                        'num_formants': len(avg_formants)
                    }

                    print(f"Extracted {len(avg_formants)} formants")
                    return

        # Fallback: use spectral peaks
        print("LPC formant extraction failed, using spectral peaks...")
        self.formants = self._extract_formants_from_spectrum()

    def _extract_formants_from_spectrum(self) -> Dict:
        """Extract formant-like features from spectrum (fallback method)"""
        # Compute average spectrum
        stft = librosa.stft(self.audio, n_fft=4096)
        mag_spectrum = np.mean(np.abs(stft), axis=1)

        # Find peaks
        peaks, properties = signal.find_peaks(
            mag_spectrum,
            height=np.max(mag_spectrum) * 0.1,  # At least 10% of max
            distance=20  # Minimum separation
        )

        # Convert peak indices to frequencies
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)
        peak_freqs = freqs[peaks]

        # Keep first 5 peaks (formant-like)
        peak_freqs = peak_freqs[:5]

        return {
            'frequencies': peak_freqs.tolist(),
            'num_formants': len(peak_freqs),
            'method': 'spectral_peaks'
        }

    def analyze_per_sound_type(self):
        """
        Detect different sound types and analyze each separately
        """
        print("\nAnalyzing per sound type (kick/snare/hihat/bass)...")

        # Detect and classify onsets
        classifications = self.classifier.detect_and_classify_onsets(self.audio)

        if len(classifications) == 0:
            print("No onsets detected")
            return

        print(f"Detected {len(classifications)} sounds")

        # Group by sound type
        sound_segments = {
            'kick': [],
            'snare': [],
            'hihat': [],
            'bass': [],
            'other': []
        }

        for onset_time, sound_type, confidence in classifications:
            # Extract segment (100ms after onset)
            start_sample = int(onset_time * self.sr)
            end_sample = int((onset_time + 0.1) * self.sr)
            end_sample = min(end_sample, len(self.audio))

            segment = self.audio[start_sample:end_sample]

            sound_segments[sound_type].append(segment)

        # Analyze each sound type
        per_sound_results = {}

        for sound_type, segments in sound_segments.items():
            if len(segments) == 0:
                continue

            print(f"  Analyzing {len(segments)} {sound_type} sounds...")

            # Concatenate segments
            concatenated = np.concatenate(segments)

            # Analyze this sound type
            analysis = self._analyze_sound_segment(concatenated)
            analysis['num_occurrences'] = len(segments)

            per_sound_results[sound_type] = analysis

        self.per_sound_analysis = per_sound_results

        print(f"Analyzed {len(per_sound_results)} sound types")

    def _analyze_sound_segment(self, audio: np.ndarray) -> Dict:
        """Analyze a specific audio segment"""
        # Spectral analysis
        stft = librosa.stft(audio, n_fft=2048)
        mag_spectrum = np.mean(np.abs(stft), axis=1)
        mag_spectrum_db = librosa.amplitude_to_db(mag_spectrum, ref=np.max)

        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)

        # Find spectral centroid
        spec_centroid = np.sum(freqs * mag_spectrum) / (np.sum(mag_spectrum) + 1e-10)

        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Peak level
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak + 1e-10)

        # Compression estimate (crest factor)
        crest_factor_db = peak_db - rms_db

        return {
            'spectral_centroid': float(spec_centroid),
            'rms_db': float(rms_db),
            'peak_db': float(peak_db),
            'crest_factor_db': float(crest_factor_db),
            'estimated_compression_ratio': self._estimate_compression_from_crest(crest_factor_db)
        }

    def _estimate_compression_from_crest(self, crest_factor_db: float) -> float:
        """
        Estimate compression ratio from crest factor

        CRITICAL FIX: Cap at 4:1 for live mic stability (was 8:1)
        Mastered tracks show low crest factors but we need realistic live ratios
        """
        # Typical uncompressed: 12-20 dB crest factor
        # Moderately compressed: 6-12 dB crest factor
        # Heavily compressed: 3-6 dB crest factor

        if crest_factor_db > 15:
            return 1.0  # No compression
        elif crest_factor_db > 12:
            return 1.5  # Minimal
        elif crest_factor_db > 9:
            return 2.0  # Light
        elif crest_factor_db > 7:
            return 3.0  # Moderate
        else:
            return 4.0  # Heavy (CAPPED at 4:1 for live use)

    def analyze_all(self, num_bands: int = 4):
        """
        Run complete advanced analysis

        Args:
            num_bands: Number of frequency bands for multiband analysis
        """
        print("\n" + "=" * 60)
        print("ADVANCED BEATBOX AUDIO ANALYSIS")
        print("=" * 60 + "\n")

        self.load_audio()

        # Global analysis (from basic analyzer)
        from audio_analyzer import AudioAnalyzer
        basic_analyzer = AudioAnalyzer(self.audio_path)
        basic_analyzer.load_audio()
        basic_analyzer.analyze_frequency_spectrum()
        basic_analyzer.analyze_dynamics()
        basic_analyzer.analyze_transients()
        basic_analyzer.analyze_harmonics()

        self.global_analysis = {
            'eq_curve': basic_analyzer.eq_curve,
            'compression': basic_analyzer.compression_params,
            'dynamic_range': basic_analyzer.dynamic_range,
            'spectral_profile': basic_analyzer.spectral_profile,
            'harmonic_content': basic_analyzer.harmonic_content,
            'transient_profile': basic_analyzer.transient_profile
        }

        # Advanced analyses
        self.analyze_multiband_eq(num_bands)
        self.analyze_formants()
        self.analyze_per_sound_type()

        print("\n" + "=" * 60)
        print("ADVANCED ANALYSIS COMPLETE")
        print("=" * 60 + "\n")

    def get_advanced_preset(self) -> Dict:
        """Get complete advanced preset"""
        return {
            'metadata': {
                'source_file': str(self.audio_path),
                'duration_seconds': self.duration,
                'sample_rate': self.sr,
                'analysis_version': '2.0_advanced'
            },
            'global_analysis': self.global_analysis,
            'multiband_analysis': self.multiband_analysis,
            'formants': self.formants,
            'per_sound_analysis': self.per_sound_analysis
        }

    def save_advanced_preset(self, preset_name: str):
        """Save advanced preset"""
        preset = self.get_advanced_preset()
        preset_path = config.get_preset_path(f"{preset_name}_advanced")

        with open(preset_path, 'w') as f:
            json.dump(preset, f, indent=2)

        print(f"\nAdvanced preset saved: {preset_path}")
        return preset_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python advanced_analyzer.py <audio_file> [preset_name]")
        sys.exit(1)

    audio_file = sys.argv[1]
    preset_name = sys.argv[2] if len(sys.argv) > 2 else Path(audio_file).stem

    analyzer = AdvancedAudioAnalyzer(audio_file)
    analyzer.analyze_all(num_bands=4)
    analyzer.save_advanced_preset(preset_name)
