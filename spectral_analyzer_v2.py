"""
Spectral Analyzer V2 - Advanced EQ Curve Extraction
Extracts precise frequency response characteristics from reference audio
Uses multiple analysis methods for accuracy
"""
import librosa
import numpy as np
from scipy import signal, interpolate
from typing import Dict, List, Tuple, Optional


class SpectralAnalyzerV2:
    """
    Advanced spectral analyzer for precise EQ curve extraction
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize spectral analyzer

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

        # Analysis parameters
        self.n_fft = 8192  # Higher resolution for better frequency accuracy
        self.hop_length = 2048

        # Target EQ bands (professional mixing console style)
        self.eq_bands = self._create_eq_bands()

    def _create_eq_bands(self) -> List[Dict]:
        """
        Create professional EQ band definitions
        Matches typical parametric EQ and mixing console EQs
        """
        # Using standard ISO frequencies and octave-based bands
        bands = [
            # Low frequency
            {'name': 'Sub Bass', 'freq': 40, 'type': 'low_shelf', 'q': 0.7},
            {'name': 'Bass', 'freq': 80, 'type': 'peak', 'q': 1.0},
            {'name': 'Low Bass', 'freq': 120, 'type': 'peak', 'q': 1.0},
            {'name': 'Upper Bass', 'freq': 200, 'type': 'peak', 'q': 1.0},

            # Mid frequency
            {'name': 'Low Mid 1', 'freq': 315, 'type': 'peak', 'q': 1.0},
            {'name': 'Low Mid 2', 'freq': 500, 'type': 'peak', 'q': 1.0},
            {'name': 'Mid 1', 'freq': 800, 'type': 'peak', 'q': 1.0},
            {'name': 'Mid 2', 'freq': 1250, 'type': 'peak', 'q': 1.0},
            {'name': 'Upper Mid 1', 'freq': 2000, 'type': 'peak', 'q': 1.0},
            {'name': 'Upper Mid 2', 'freq': 3150, 'type': 'peak', 'q': 1.0},

            # High frequency
            {'name': 'Presence 1', 'freq': 5000, 'type': 'peak', 'q': 1.0},
            {'name': 'Presence 2', 'freq': 8000, 'type': 'peak', 'q': 1.0},
            {'name': 'Brilliance', 'freq': 12000, 'type': 'peak', 'q': 1.0},
            {'name': 'Air', 'freq': 16000, 'type': 'high_shelf', 'q': 0.7},
        ]

        return bands

    def analyze_spectral_response(self, audio: np.ndarray) -> Dict:
        """
        Analyze frequency response of audio

        Args:
            audio: Input audio signal

        Returns:
            Dictionary with spectral analysis results
        """
        print("\n[Spectral Analyzer V2] Analyzing frequency response...")

        # Method 1: Long-term average spectrum (LTAS)
        ltas = self._compute_ltas(audio)

        # Method 2: Percentile-based spectrum (robust to outliers)
        percentile_spectrum = self._compute_percentile_spectrum(audio)

        # Method 3: RMS spectrum (energy-based)
        rms_spectrum = self._compute_rms_spectrum(audio)

        # Combine methods for robust estimation
        combined_spectrum = self._combine_spectral_estimates(
            ltas, percentile_spectrum, rms_spectrum
        )

        # Extract EQ curve from spectrum
        eq_curve = self._extract_eq_curve(combined_spectrum)

        # Analyze spectral tilt (overall brightness)
        spectral_tilt = self._analyze_spectral_tilt(combined_spectrum)

        # Detect spectral envelope (formant-like peaks)
        spectral_peaks = self._detect_spectral_peaks(combined_spectrum)

        return {
            'eq_curve': eq_curve,
            'spectral_tilt': spectral_tilt,
            'spectral_peaks': spectral_peaks,
            'full_spectrum': combined_spectrum,
            'analysis_method': 'v2_multi_method'
        }

    def _compute_ltas(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Long-Term Average Spectrum
        """
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        mag_spectrum = np.abs(stft)

        # Average over time
        avg_spectrum = np.mean(mag_spectrum, axis=1)

        # Ensure no zeros
        avg_spectrum = np.maximum(avg_spectrum, 1e-10)

        # Convert to dB
        spectrum_db = librosa.amplitude_to_db(avg_spectrum, ref=1.0)

        return spectrum_db

    def _compute_percentile_spectrum(self, audio: np.ndarray,
                                    percentile: float = 50) -> np.ndarray:
        """
        Compute percentile-based spectrum (median by default)
        More robust to outliers than mean
        """
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        mag_spectrum = np.abs(stft)

        # Percentile over time
        percentile_spectrum = np.percentile(mag_spectrum, percentile, axis=1)
        percentile_spectrum = np.maximum(percentile_spectrum, 1e-10)

        # Convert to dB
        spectrum_db = librosa.amplitude_to_db(percentile_spectrum, ref=1.0)

        return spectrum_db

    def _compute_rms_spectrum(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute RMS-based spectrum (energy-weighted)
        """
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        mag_spectrum = np.abs(stft)

        # RMS over time
        rms_spectrum = np.sqrt(np.mean(mag_spectrum ** 2, axis=1))
        rms_spectrum = np.maximum(rms_spectrum, 1e-10)

        # Convert to dB
        spectrum_db = librosa.amplitude_to_db(rms_spectrum, ref=1.0)

        return spectrum_db

    def _combine_spectral_estimates(self, ltas: np.ndarray,
                                   percentile: np.ndarray,
                                   rms: np.ndarray) -> np.ndarray:
        """
        Combine multiple spectral estimates for robustness
        """
        # Weighted average (give more weight to median/percentile for robustness)
        combined = 0.3 * ltas + 0.4 * percentile + 0.3 * rms

        return combined

    def _extract_eq_curve(self, spectrum_db: np.ndarray) -> List[Dict]:
        """
        Extract EQ curve from spectrum
        """
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)

        eq_curve = []

        # Reference frequency for normalization (1kHz typical reference)
        ref_freq_idx = np.argmin(np.abs(freqs - 1000))
        ref_level = spectrum_db[ref_freq_idx]

        # Extract gain for each band
        for band in self.eq_bands:
            center_freq = band['freq']

            # Find frequency bin
            freq_idx = np.argmin(np.abs(freqs - center_freq))

            # Get level at this frequency
            level = spectrum_db[freq_idx]

            # Calculate gain relative to reference
            gain_db = level - ref_level

            # Apply moderate scaling to prevent extreme values
            # Professional EQ rarely exceeds ±12 dB per band
            SCALE_FACTOR = 0.25  # More conservative than before
            gain_db_scaled = gain_db * SCALE_FACTOR
            gain_db_clamped = np.clip(gain_db_scaled, -12.0, 12.0)

            eq_curve.append({
                'name': band['name'],
                'frequency': center_freq,
                'gain_db': float(gain_db_clamped),
                'q_factor': band['q'],
                'filter_type': band['type']
            })

        return eq_curve

    def _analyze_spectral_tilt(self, spectrum_db: np.ndarray) -> Dict:
        """
        Analyze overall spectral tilt (slope from low to high frequencies)
        Indicates overall brightness/darkness
        """
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)

        # Use frequencies from 100 Hz to 10 kHz for tilt calculation
        mask = (freqs >= 100) & (freqs <= 10000)
        freqs_used = freqs[mask]
        spectrum_used = spectrum_db[mask]

        # Fit linear regression to log-frequency vs magnitude
        log_freqs = np.log10(freqs_used)

        # Polyfit
        coeffs = np.polyfit(log_freqs, spectrum_used, 1)
        tilt_db_per_decade = coeffs[0]

        # Classify tilt
        if tilt_db_per_decade > 3:
            tilt_type = 'bright'
        elif tilt_db_per_decade < -3:
            tilt_type = 'dark'
        else:
            tilt_type = 'neutral'

        return {
            'tilt_db_per_decade': float(tilt_db_per_decade),
            'tilt_type': tilt_type,
            'interpretation': self._interpret_tilt(tilt_db_per_decade)
        }

    def _interpret_tilt(self, tilt: float) -> str:
        """Interpret spectral tilt value"""
        if tilt > 3:
            return "High-frequency emphasis (bright, airy sound)"
        elif tilt > 1:
            return "Slight high-frequency lift (clear, present)"
        elif tilt > -1:
            return "Balanced frequency response"
        elif tilt > -3:
            return "Slight low-frequency emphasis (warm)"
        else:
            return "Strong low-frequency emphasis (dark, warm, bass-heavy)"

    def _detect_spectral_peaks(self, spectrum_db: np.ndarray,
                               prominence_db: float = 6.0) -> List[Dict]:
        """
        Detect prominent spectral peaks (resonances)
        """
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)

        # Find peaks
        peaks, properties = signal.find_peaks(
            spectrum_db,
            prominence=prominence_db,
            distance=20,
            height=-60  # Above noise floor
        )

        # Convert to frequency and magnitude
        peak_freqs = freqs[peaks]
        peak_mags = spectrum_db[peaks]

        # Sort by magnitude (strongest first)
        sorted_indices = np.argsort(peak_mags)[::-1]

        # Take top 10 peaks
        top_peaks = []
        for i in sorted_indices[:10]:
            if i < len(peak_freqs):
                top_peaks.append({
                    'frequency': float(peak_freqs[i]),
                    'magnitude_db': float(peak_mags[i]),
                    'prominence_db': float(properties['prominences'][i])
                })

        return top_peaks

    def optimize_eq_for_live_use(self, eq_curve: List[Dict]) -> List[Dict]:
        """
        Optimize EQ curve for live microphone use
        Reduces extreme settings and ensures stability
        """
        optimized = []

        for band in eq_curve:
            # Further reduce gain for live stability
            gain = band['gain_db'] * 0.7  # 70% of analyzed gain
            gain = np.clip(gain, -9.0, 9.0)  # Max ±9 dB for live

            # Adjust Q factor for smoother response
            q = band['q_factor']
            if abs(gain) > 6:
                # Wider Q for large cuts/boosts
                q = max(0.7, q * 0.8)

            optimized.append({
                'name': band['name'],
                'frequency': band['frequency'],
                'gain_db': float(gain),
                'q_factor': float(q),
                'filter_type': band['filter_type']
            })

        return optimized


def analyze_reference_audio_spectrum(audio_path: str) -> Dict:
    """
    Convenience function to analyze reference audio

    Args:
        audio_path: Path to reference audio

    Returns:
        Spectral analysis results
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=44100, mono=True)

    # Analyze
    analyzer = SpectralAnalyzerV2(sample_rate=sr)
    results = analyzer.analyze_spectral_response(audio)

    # Optimize for live use
    results['eq_curve_optimized'] = analyzer.optimize_eq_for_live_use(results['eq_curve'])

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python spectral_analyzer_v2.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    print(f"Analyzing: {audio_file}")
    results = analyze_reference_audio_spectrum(audio_file)

    print("\n" + "="*60)
    print("SPECTRAL ANALYSIS V2 RESULTS")
    print("="*60)

    print(f"\nSpectral Tilt: {results['spectral_tilt']['tilt_db_per_decade']:.2f} dB/decade")
    print(f"Type: {results['spectral_tilt']['tilt_type']}")
    print(f"Interpretation: {results['spectral_tilt']['interpretation']}")

    print(f"\n\nEQ Curve ({len(results['eq_curve_optimized'])} bands):")
    for band in results['eq_curve_optimized']:
        print(f"  {band['name']:15s} @ {band['frequency']:6.0f} Hz: {band['gain_db']:+6.2f} dB ({band['filter_type']})")

    print(f"\n\nTop Spectral Peaks:")
    for i, peak in enumerate(results['spectral_peaks'][:5], 1):
        print(f"  {i}. {peak['frequency']:7.1f} Hz ({peak['magnitude_db']:+6.1f} dB)")
