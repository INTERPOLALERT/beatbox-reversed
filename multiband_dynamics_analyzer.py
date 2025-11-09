"""
Multiband Dynamics Analyzer - Per-Band Compression Detection
Analyzes compression characteristics for each frequency band
Critical for professional beatbox/vocal processing
"""
import librosa
import numpy as np
from scipy import signal
from typing import Dict, List
from multiband_processor import create_default_crossover


class MultibandDynamicsAnalyzer:
    """
    Analyzes dynamics processing independently for each frequency band
    """

    def __init__(self, sample_rate: int = 44100, num_bands: int = 4):
        """
        Initialize multiband dynamics analyzer

        Args:
            sample_rate: Sample rate in Hz
            num_bands: Number of frequency bands (4 or 8)
        """
        self.sample_rate = sample_rate
        self.num_bands = num_bands

        # Create crossover
        self.crossover = create_default_crossover(num_bands=num_bands,
                                                  sample_rate=sample_rate)

    def analyze_multiband_dynamics(self, audio: np.ndarray) -> Dict:
        """
        Analyze dynamics for each frequency band

        Args:
            audio: Input audio signal

        Returns:
            Per-band dynamics analysis
        """
        print(f"\n[Multiband Dynamics Analyzer] Analyzing {self.num_bands}-band dynamics...")

        # Split audio into bands
        bands = self.crossover.split_bands_offline(audio)

        # Analyze each band
        band_results = []

        for i, band_audio in enumerate(bands):
            # Get frequency range for this band
            if i == 0:
                freq_low = 20
                freq_high = self.crossover.crossover_freqs[0] if len(self.crossover.crossover_freqs) > 0 else 22000
            elif i == self.num_bands - 1:
                freq_low = self.crossover.crossover_freqs[-1]
                freq_high = 22000
            else:
                freq_low = self.crossover.crossover_freqs[i - 1]
                freq_high = self.crossover.crossover_freqs[i]

            band_name = self._get_band_name(i, freq_low, freq_high)

            print(f"  Analyzing band {i+1}/{self.num_bands}: {band_name} ({freq_low:.0f}-{freq_high:.0f} Hz)")

            # Analyze this band's dynamics
            band_dynamics = self._analyze_single_band(band_audio)

            band_results.append({
                'band_index': i,
                'band_name': band_name,
                'freq_range': [freq_low, freq_high],
                'center_freq': np.sqrt(freq_low * freq_high),  # Geometric mean
                **band_dynamics
            })

        # Detect if multiband compression is being used
        multiband_compression_detected = self._detect_multiband_compression(band_results)

        return {
            'num_bands': self.num_bands,
            'crossover_freqs': self.crossover.crossover_freqs,
            'bands': band_results,
            'multiband_compression_detected': multiband_compression_detected
        }

    def _get_band_name(self, index: int, freq_low: float, freq_high: float) -> str:
        """Get descriptive name for band"""
        if freq_high <= 250:
            return "Bass"
        elif freq_high <= 1000:
            return "Low-Mid"
        elif freq_high <= 4000:
            return "Mid"
        elif freq_high <= 8000:
            return "High-Mid"
        else:
            return "Treble"

    def _analyze_single_band(self, band_audio: np.ndarray) -> Dict:
        """
        Analyze dynamics for a single frequency band
        """
        # Skip if band is silent
        if np.max(np.abs(band_audio)) < 1e-6:
            return self._get_silent_band_analysis()

        # Calculate RMS envelope
        frame_length = 2048
        hop_length = 512

        rms = librosa.feature.rms(
            y=band_audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        rms = np.maximum(rms, 1e-10)
        rms_db = 20 * np.log10(rms)

        # 1. Estimate compression threshold
        threshold_db = float(np.percentile(rms_db[rms_db > -60], 30))
        threshold_db = np.clip(threshold_db, -40.0, -5.0)

        # 2. Estimate compression ratio
        ratio = self._estimate_band_ratio(rms_db, threshold_db)

        # 3. Analyze dynamic range in this band
        dynamic_range_db = np.percentile(rms_db, 95) - np.percentile(rms_db, 10)

        # 4. Calculate average level
        avg_rms_db = float(np.mean(rms_db))

        # 5. Calculate peak level
        peak = np.max(np.abs(band_audio))
        peak_db = 20 * np.log10(peak + 1e-10)

        # 6. Estimate attack/release for this band
        attack_ms, release_ms = self._estimate_band_attack_release(rms_db)

        # 7. Calculate crest factor
        crest_factor_db = peak_db - avg_rms_db

        # 8. Determine if this band is compressed
        is_compressed = ratio > 1.5 or dynamic_range_db < 12

        return {
            'threshold_db': float(threshold_db),
            'ratio': float(ratio),
            'attack_ms': float(attack_ms),
            'release_ms': float(release_ms),
            'dynamic_range_db': float(dynamic_range_db),
            'avg_level_db': float(avg_rms_db),
            'peak_level_db': float(peak_db),
            'crest_factor_db': float(crest_factor_db),
            'is_compressed': is_compressed
        }

    def _get_silent_band_analysis(self) -> Dict:
        """Return default analysis for silent band"""
        return {
            'threshold_db': -30.0,
            'ratio': 1.0,
            'attack_ms': 5.0,
            'release_ms': 100.0,
            'dynamic_range_db': 0.0,
            'avg_level_db': -60.0,
            'peak_level_db': -60.0,
            'crest_factor_db': 0.0,
            'is_compressed': False
        }

    def _estimate_band_ratio(self, rms_db: np.ndarray, threshold_db: float) -> float:
        """Estimate compression ratio for this band"""
        above_threshold = rms_db[rms_db >= threshold_db]
        below_threshold = rms_db[rms_db < threshold_db]

        if len(above_threshold) < 5:
            return 1.0

        # Variation above vs below threshold
        var_above = np.std(above_threshold) if len(above_threshold) > 0 else 1.0
        var_below = np.std(below_threshold) if len(below_threshold) > 0 else 1.0

        if var_above > 0:
            ratio = np.clip(var_below / (var_above + 0.1), 1.0, 8.0)
        else:
            ratio = 1.0

        return ratio

    def _estimate_band_attack_release(self, rms_db: np.ndarray) -> tuple:
        """Estimate attack and release times for this band"""
        # Analyze level changes
        diff = np.diff(rms_db)

        # Attack: how fast level rises
        rises = diff[diff > 0.5]
        if len(rises) > 5:
            avg_rise_rate = np.median(rises)
            attack_ms = np.clip(2.0 / (avg_rise_rate + 0.1), 1.0, 30.0)
        else:
            attack_ms = 5.0

        # Release: how fast level falls
        falls = -diff[diff < -0.5]
        if len(falls) > 5:
            avg_fall_rate = np.median(falls)
            release_ms = np.clip(3.0 / (avg_fall_rate + 0.1), 30.0, 300.0)
        else:
            release_ms = 100.0

        return attack_ms, release_ms

    def _detect_multiband_compression(self, band_results: List[Dict]) -> bool:
        """
        Detect if multiband compression is being used
        (different bands have different compression characteristics)
        """
        # Extract ratios from each band
        ratios = [band['ratio'] for band in band_results]

        # If ratios vary significantly between bands, multiband comp is likely
        ratio_std = np.std(ratios)

        # If standard deviation > 0.5, different bands compressed differently
        multiband_detected = ratio_std > 0.5

        return multiband_detected

    def optimize_for_live_use(self, analysis: Dict) -> Dict:
        """
        Optimize multiband dynamics parameters for live microphone use
        Reduces extreme settings for stability
        """
        optimized_bands = []

        for band in analysis['bands']:
            # Reduce ratio for live stability
            ratio = min(band['ratio'], 4.0)  # Cap at 4:1

            # Moderate threshold
            threshold = np.clip(band['threshold_db'], -35.0, -10.0)

            # Ensure reasonable attack/release
            attack = np.clip(band['attack_ms'], 1.0, 20.0)
            release = np.clip(band['release_ms'], 50.0, 250.0)

            optimized_bands.append({
                'band_index': band['band_index'],
                'band_name': band['band_name'],
                'freq_range': band['freq_range'],
                'center_freq': band['center_freq'],
                'threshold_db': float(threshold),
                'ratio': float(ratio),
                'attack_ms': float(attack),
                'release_ms': float(release),
                'enabled': band['is_compressed']
            })

        return {
            'num_bands': analysis['num_bands'],
            'crossover_freqs': analysis['crossover_freqs'],
            'bands': optimized_bands,
            'multiband_compression_detected': analysis['multiband_compression_detected']
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python multiband_dynamics_analyzer.py <audio_file> [num_bands]")
        sys.exit(1)

    audio_file = sys.argv[1]
    num_bands = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    # Load audio
    audio, sr = librosa.load(audio_file, sr=44100, mono=True)

    # Analyze
    analyzer = MultibandDynamicsAnalyzer(sample_rate=sr, num_bands=num_bands)
    results = analyzer.analyze_multiband_dynamics(audio)

    # Optimize for live use
    optimized = analyzer.optimize_for_live_use(results)

    print("\n" + "="*60)
    print(f"MULTIBAND DYNAMICS ANALYSIS ({num_bands} BANDS)")
    print("="*60)

    print(f"\nMultiband Compression Detected: {results['multiband_compression_detected']}")
    print(f"Crossover Frequencies: {[f'{f:.0f} Hz' for f in results['crossover_freqs']]}")

    print("\n" + "-"*60)
    print("PER-BAND ANALYSIS:")
    print("-"*60)

    for band in optimized['bands']:
        print(f"\n[{band['band_name']}] {band['freq_range'][0]:.0f}-{band['freq_range'][1]:.0f} Hz")
        print(f"  Threshold: {band['threshold_db']:.1f} dB")
        print(f"  Ratio: {band['ratio']:.2f}:1")
        print(f"  Attack: {band['attack_ms']:.1f} ms")
        print(f"  Release: {band['release_ms']:.0f} ms")
        print(f"  Enabled: {band['enabled']}")
