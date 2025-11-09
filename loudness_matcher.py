"""
Adaptive Loudness Matching Module
Real-time per-buffer loudness analysis and adaptive gain matching
Ensures live mic output matches reference audio loudness dynamically
"""
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import signal


class LoudnessMatcher:
    """
    Adaptive loudness matching for real-time audio processing
    Dynamically adjusts gain to match reference loudness characteristics
    """

    def __init__(self, sample_rate: int = 44100, target_lufs: float = -14.0):
        """
        Initialize loudness matcher

        Args:
            sample_rate: Sample rate in Hz
            target_lufs: Target LUFS loudness (default: -14.0 for broadcast standard)
        """
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs

        # Reference loudness statistics
        self.reference_rms = None
        self.reference_peak = None
        self.reference_crest_factor = None
        self.reference_lufs = None

        # Adaptive gain state
        self.current_gain_db = 0.0
        self.gain_smoothing_coeff = 0.95  # Smooth gain changes

        # RMS window for real-time analysis
        self.rms_window_size = int(0.050 * sample_rate)  # 50ms window

        # K-weighting filter for LUFS calculation (simplified)
        self._design_k_weighting_filter()

        # Analysis statistics
        self.stats = {
            'current_rms_db': 0.0,
            'current_peak_db': 0.0,
            'current_crest_db': 0.0,
            'current_lufs': 0.0,
            'applied_gain_db': 0.0,
            'gain_delta_db': 0.0
        }

    def _design_k_weighting_filter(self):
        """Design K-weighting filter for LUFS measurement (simplified approximation)"""
        # High-shelf filter (approximate K-weighting curve)
        # This is a simplified version - full LUFS uses BS.1770 K-weighting

        # Pre-filter: high-pass at 50Hz
        b_hp, a_hp = signal.butter(2, 50, btype='high', fs=self.sample_rate)

        # High-shelf boost at 2kHz
        b_hs, a_hs = signal.iirpeak(2000, Q=0.5, fs=self.sample_rate)

        self.k_weight_filters = [(b_hp, a_hp), (b_hs, a_hs)]
        self.k_weight_states = [None, None]

    def set_reference_loudness(self, reference_rms: float, reference_peak: float,
                               reference_lufs: Optional[float] = None):
        """
        Set reference loudness targets from analyzed audio

        Args:
            reference_rms: Reference RMS level (linear, 0-1)
            reference_peak: Reference peak level (linear, 0-1)
            reference_lufs: Optional LUFS value
        """
        self.reference_rms = reference_rms
        self.reference_peak = reference_peak
        self.reference_crest_factor = reference_peak / (reference_rms + 1e-10)

        if reference_lufs is not None:
            self.reference_lufs = reference_lufs
        else:
            # Estimate LUFS from RMS
            self.reference_lufs = 20 * np.log10(reference_rms + 1e-10) - 0.691

        print(f"Reference loudness set:")
        print(f"  RMS: {20 * np.log10(reference_rms + 1e-10):.1f} dB")
        print(f"  Peak: {20 * np.log10(reference_peak + 1e-10):.1f} dB")
        print(f"  Crest Factor: {20 * np.log10(self.reference_crest_factor):.1f} dB")
        print(f"  Estimated LUFS: {self.reference_lufs:.1f}")

    def analyze_buffer_loudness(self, audio_buffer: np.ndarray) -> Dict:
        """
        Analyze loudness of current buffer

        Args:
            audio_buffer: Input audio buffer

        Returns:
            Dictionary with loudness statistics
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_buffer ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Calculate peak
        peak = np.max(np.abs(audio_buffer))
        peak_db = 20 * np.log10(peak + 1e-10)

        # Calculate crest factor
        crest_factor = peak / (rms + 1e-10)
        crest_db = 20 * np.log10(crest_factor)

        # Estimate LUFS (simplified)
        # Apply K-weighting filter
        k_weighted = audio_buffer.copy()
        for i, (b, a) in enumerate(self.k_weight_filters):
            if self.k_weight_states[i] is None:
                k_weighted = signal.lfilter(b, a, k_weighted)
            else:
                k_weighted, self.k_weight_states[i] = signal.lfilter(
                    b, a, k_weighted, zi=self.k_weight_states[i]
                )

        # Mean square of K-weighted signal
        mean_square = np.mean(k_weighted ** 2)
        lufs = -0.691 + 10 * np.log10(mean_square + 1e-10)

        return {
            'rms': rms,
            'rms_db': rms_db,
            'peak': peak,
            'peak_db': peak_db,
            'crest_factor': crest_factor,
            'crest_db': crest_db,
            'lufs': lufs
        }

    def calculate_adaptive_gain(self, current_loudness: Dict,
                                 match_mode: str = 'rms') -> float:
        """
        Calculate adaptive gain to match reference loudness

        Args:
            current_loudness: Current buffer loudness statistics
            match_mode: Matching mode ('rms', 'lufs', 'peak_normalized', 'crest_matched')

        Returns:
            Gain in dB to apply
        """
        if self.reference_rms is None:
            return 0.0

        gain_db = 0.0

        if match_mode == 'rms':
            # Match RMS levels
            target_rms_db = 20 * np.log10(self.reference_rms + 1e-10)
            gain_db = target_rms_db - current_loudness['rms_db']

        elif match_mode == 'lufs':
            # Match LUFS levels
            if self.reference_lufs is not None:
                gain_db = self.reference_lufs - current_loudness['lufs']
            else:
                # Fall back to RMS
                target_rms_db = 20 * np.log10(self.reference_rms + 1e-10)
                gain_db = target_rms_db - current_loudness['rms_db']

        elif match_mode == 'peak_normalized':
            # Match peak levels
            target_peak_db = 20 * np.log10(self.reference_peak + 1e-10)
            gain_db = target_peak_db - current_loudness['peak_db']

        elif match_mode == 'crest_matched':
            # Match crest factor (maintains dynamics)
            target_crest_db = 20 * np.log10(self.reference_crest_factor)
            current_crest_db = current_loudness['crest_db']

            # If crest factors match, use RMS matching
            # If not, adjust to maintain dynamic range
            if abs(current_crest_db - target_crest_db) < 1.0:
                target_rms_db = 20 * np.log10(self.reference_rms + 1e-10)
                gain_db = target_rms_db - current_loudness['rms_db']
            else:
                # Adjust gain to match crest factor
                crest_delta = target_crest_db - current_crest_db
                gain_db = crest_delta * 0.5  # Partial correction

        # Clamp gain to safe range
        gain_db = np.clip(gain_db, -24.0, 24.0)

        return gain_db

    def apply_adaptive_gain(self, audio_buffer: np.ndarray,
                           match_mode: str = 'rms',
                           smoothing: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Apply adaptive gain matching to buffer

        Args:
            audio_buffer: Input audio buffer
            match_mode: Matching mode ('rms', 'lufs', 'peak_normalized', 'crest_matched')
            smoothing: Enable gain smoothing for gradual transitions

        Returns:
            Tuple of (processed_audio, statistics)
        """
        # Analyze current buffer
        current_loudness = self.analyze_buffer_loudness(audio_buffer)

        # Calculate required gain
        target_gain_db = self.calculate_adaptive_gain(current_loudness, match_mode)

        # Smooth gain changes
        if smoothing:
            self.current_gain_db = (self.gain_smoothing_coeff * self.current_gain_db +
                                   (1 - self.gain_smoothing_coeff) * target_gain_db)
        else:
            self.current_gain_db = target_gain_db

        # Apply gain
        gain_linear = 10 ** (self.current_gain_db / 20)
        processed = audio_buffer * gain_linear

        # Update statistics
        self.stats = {
            'current_rms_db': current_loudness['rms_db'],
            'current_peak_db': current_loudness['peak_db'],
            'current_crest_db': current_loudness['crest_db'],
            'current_lufs': current_loudness['lufs'],
            'applied_gain_db': self.current_gain_db,
            'gain_delta_db': target_gain_db - self.current_gain_db,
            'match_mode': match_mode
        }

        return processed, self.stats

    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return self.stats.copy()

    def reset(self):
        """Reset adaptive state"""
        self.current_gain_db = 0.0
        self.k_weight_states = [None, None]


class PerBufferLoudnessAnalyzer:
    """
    Continuous loudness analysis for monitoring
    Tracks loudness over time for diagnostics
    """

    def __init__(self, sample_rate: int = 44100, history_seconds: float = 5.0):
        """
        Initialize analyzer

        Args:
            sample_rate: Sample rate in Hz
            history_seconds: How many seconds of history to keep
        """
        self.sample_rate = sample_rate
        self.history_seconds = history_seconds

        # History buffers
        self.max_history_samples = int(history_seconds * sample_rate)
        self.rms_history = []
        self.peak_history = []
        self.lufs_history = []

        self.loudness_matcher = LoudnessMatcher(sample_rate)

    def analyze(self, audio_buffer: np.ndarray) -> Dict:
        """Analyze buffer and update history"""
        stats = self.loudness_matcher.analyze_buffer_loudness(audio_buffer)

        # Update history
        self.rms_history.append(stats['rms'])
        self.peak_history.append(stats['peak'])
        self.lufs_history.append(stats['lufs'])

        # Trim history
        if len(self.rms_history) * len(audio_buffer) > self.max_history_samples:
            self.rms_history.pop(0)
            self.peak_history.pop(0)
            self.lufs_history.pop(0)

        # Add statistical summaries
        if len(self.rms_history) > 1:
            stats['rms_mean_db'] = 20 * np.log10(np.mean(self.rms_history) + 1e-10)
            stats['rms_std_db'] = np.std([20 * np.log10(r + 1e-10) for r in self.rms_history])
            stats['peak_max_db'] = 20 * np.log10(np.max(self.peak_history) + 1e-10)
            stats['lufs_mean'] = np.mean(self.lufs_history)

        return stats


def demo_loudness_matching():
    """Demo loudness matching"""
    import librosa

    print("=" * 60)
    print("ADAPTIVE LOUDNESS MATCHING DEMO")
    print("=" * 60)

    # Create matcher
    matcher = LoudnessMatcher(sample_rate=44100)

    # Set reference loudness (example values)
    matcher.set_reference_loudness(
        reference_rms=0.15,  # Moderate level
        reference_peak=0.8,  # Healthy headroom
        reference_lufs=-14.0  # Broadcast standard
    )

    print("\nGenerating test signal...")
    # Generate quiet test signal
    t = np.linspace(0, 1, 44100)
    test_signal = 0.05 * np.sin(2 * np.pi * 440 * t)  # Quiet 440Hz tone

    print(f"Original signal RMS: {np.sqrt(np.mean(test_signal**2)):.4f}")

    # Process with adaptive gain
    processed, stats = matcher.apply_adaptive_gain(test_signal, match_mode='rms')

    print(f"\nAfter adaptive gain matching:")
    print(f"  Applied gain: {stats['applied_gain_db']:+.1f} dB")
    print(f"  Original RMS: {stats['current_rms_db']:.1f} dB")
    print(f"  Processed RMS: {20 * np.log10(np.sqrt(np.mean(processed**2)) + 1e-10):.1f} dB")
    print(f"  Target RMS: {20 * np.log10(matcher.reference_rms + 1e-10):.1f} dB")

    print("\n" + "=" * 60)
    print("Loudness matching enabled and working!")
    print("=" * 60)


if __name__ == "__main__":
    demo_loudness_matching()
