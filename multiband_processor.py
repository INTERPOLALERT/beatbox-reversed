"""
Multiband Audio Processor
Implements 4-8 band processing with Linkwitz-Riley crossovers
"""
import numpy as np
from scipy import signal
from typing import List, Tuple


class MultibandCrossover:
    """
    Linkwitz-Riley crossover filters (24dB/octave, 4th order)
    Ensures flat magnitude response when bands are summed
    """

    def __init__(self, crossover_freqs: List[float], sample_rate: int = 44100):
        """
        Initialize multiband crossover

        Args:
            crossover_freqs: List of crossover frequencies in Hz
            sample_rate: Sample rate in Hz
        """
        self.crossover_freqs = crossover_freqs
        self.sample_rate = sample_rate
        self.num_bands = len(crossover_freqs) + 1

        # Design filters
        self.filters = self._design_filters()

        # Filter states for real-time processing
        self.filter_states = {}

    def _design_filters(self):
        """Design Linkwitz-Riley crossover filters"""
        filters = []

        # Nyquist frequency
        nyquist = self.sample_rate / 2

        for i in range(self.num_bands):
            if i == 0:
                # Lowest band - lowpass only
                if len(self.crossover_freqs) > 0:
                    # 4th order Butterworth lowpass (LR is two cascaded Butterworth)
                    sos = signal.butter(4, self.crossover_freqs[0] / nyquist,
                                       btype='low', output='sos')
                else:
                    sos = None  # All-pass
                filters.append(('low', sos))

            elif i == self.num_bands - 1:
                # Highest band - highpass only
                sos = signal.butter(4, self.crossover_freqs[-1] / nyquist,
                                   btype='high', output='sos')
                filters.append(('high', sos))

            else:
                # Middle bands - bandpass
                low_freq = self.crossover_freqs[i - 1] / nyquist
                high_freq = self.crossover_freqs[i] / nyquist
                sos = signal.butter(4, [low_freq, high_freq],
                                   btype='band', output='sos')
                filters.append(('band', sos))

        return filters

    def split_bands(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Split audio into frequency bands

        Args:
            audio: Input audio signal

        Returns:
            List of band signals
        """
        bands = []

        for i, (ftype, sos) in enumerate(self.filters):
            if sos is not None:
                # Apply filter
                if i not in self.filter_states:
                    self.filter_states[i] = signal.sosfilt_zi(sos)

                band, self.filter_states[i] = signal.sosfilt(
                    sos, audio, zi=self.filter_states[i] * audio[0]
                )
            else:
                # All-pass (no filtering)
                band = audio.copy()

            bands.append(band)

        return bands

    def split_bands_offline(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Split audio into frequency bands (offline, no state)

        Args:
            audio: Input audio signal

        Returns:
            List of band signals
        """
        bands = []

        for ftype, sos in self.filters:
            if sos is not None:
                band = signal.sosfilt(sos, audio)
            else:
                band = audio.copy()

            bands.append(band)

        return bands

    def combine_bands(self, bands: List[np.ndarray]) -> np.ndarray:
        """
        Combine frequency bands back into single signal

        Args:
            bands: List of band signals

        Returns:
            Combined audio signal
        """
        if len(bands) != self.num_bands:
            raise ValueError(f"Expected {self.num_bands} bands, got {len(bands)}")

        # Ensure all bands are the same length
        min_length = min(len(b) for b in bands)
        bands = [b[:min_length] for b in bands]

        # Sum all bands
        combined = np.sum(bands, axis=0)

        return combined

    def reset_state(self):
        """Reset filter states"""
        self.filter_states = {}


class MultibandEnvelopeFollower:
    """
    Per-band envelope followers with different attack/release times
    """

    def __init__(self, num_bands: int, sample_rate: int = 44100):
        """
        Initialize envelope followers for each band

        Args:
            num_bands: Number of frequency bands
            sample_rate: Sample rate in Hz
        """
        self.num_bands = num_bands
        self.sample_rate = sample_rate

        # Attack and release times per band (in ms)
        # Faster attack for high frequencies, slower for bass
        self.attack_times = self._calculate_attack_times()
        self.release_times = self._calculate_release_times()

        # Envelope states
        self.envelope_states = [0.0] * num_bands

    def _calculate_attack_times(self) -> List[float]:
        """Calculate attack times for each band (faster for high freqs)"""
        # From 1ms (highs) to 20ms (bass)
        times = []
        for i in range(self.num_bands):
            ratio = i / (self.num_bands - 1) if self.num_bands > 1 else 0
            attack_ms = 1.0 + (20.0 - 1.0) * (1 - ratio)  # Reverse order
            times.append(attack_ms)
        return times

    def _calculate_release_times(self) -> List[float]:
        """Calculate release times for each band (slower for bass)"""
        # From 50ms (highs) to 200ms (bass)
        times = []
        for i in range(self.num_bands):
            ratio = i / (self.num_bands - 1) if self.num_bands > 1 else 0
            release_ms = 50.0 + (200.0 - 50.0) * (1 - ratio)  # Reverse order
            times.append(release_ms)
        return times

    def _time_to_coefficient(self, time_ms: float) -> float:
        """Convert time constant to filter coefficient"""
        return np.exp(-1.0 / (time_ms * self.sample_rate / 1000.0))

    def process_band(self, audio: np.ndarray, band_idx: int) -> np.ndarray:
        """
        Apply envelope following to a single band

        Args:
            audio: Input audio for this band
            band_idx: Index of the band

        Returns:
            Envelope signal
        """
        attack_coef = self._time_to_coefficient(self.attack_times[band_idx])
        release_coef = self._time_to_coefficient(self.release_times[band_idx])

        envelope = np.zeros_like(audio)
        state = self.envelope_states[band_idx]

        for i, sample in enumerate(audio):
            # Absolute value (full-wave rectification)
            rectified = abs(sample)

            # Choose attack or release
            if rectified > state:
                coef = attack_coef
            else:
                coef = release_coef

            # Update state
            state = coef * state + (1 - coef) * rectified
            envelope[i] = state

        self.envelope_states[band_idx] = state

        return envelope

    def reset(self):
        """Reset envelope states"""
        self.envelope_states = [0.0] * self.num_bands


class TransientDetector:
    """
    Dual-envelope transient detector
    Separates transient and sustained portions of audio
    """

    def __init__(self, sample_rate: int = 44100,
                 fast_attack_ms: float = 2.0,
                 slow_attack_ms: float = 50.0):
        """
        Initialize transient detector

        Args:
            sample_rate: Sample rate in Hz
            fast_attack_ms: Fast envelope attack time
            slow_attack_ms: Slow envelope attack time
        """
        self.sample_rate = sample_rate
        self.fast_attack_ms = fast_attack_ms
        self.slow_attack_ms = slow_attack_ms

        # Calculate coefficients
        self.fast_coef = np.exp(-1.0 / (fast_attack_ms * sample_rate / 1000.0))
        self.slow_coef = np.exp(-1.0 / (slow_attack_ms * sample_rate / 1000.0))

        # States
        self.fast_state = 0.0
        self.slow_state = 0.0

    def detect(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect transient vs sustained portions

        Args:
            audio: Input audio signal

        Returns:
            Tuple of (transient_mask, sustained_mask)
            Masks are 0.0 to 1.0 (can use for crossfading)
        """
        fast_env = np.zeros_like(audio)
        slow_env = np.zeros_like(audio)

        fast_state = self.fast_state
        slow_state = self.slow_state

        for i, sample in enumerate(audio):
            rectified = abs(sample)

            # Fast envelope (catches transients)
            fast_state = self.fast_coef * fast_state + (1 - self.fast_coef) * rectified
            fast_env[i] = fast_state

            # Slow envelope (overall level)
            slow_state = self.slow_coef * slow_state + (1 - self.slow_coef) * rectified
            slow_env[i] = slow_state

        # Update states
        self.fast_state = fast_state
        self.slow_state = slow_state

        # Transient strength = difference between fast and slow
        transient_signal = np.maximum(0, fast_env - slow_env)

        # Normalize to 0-1 range
        max_transient = np.max(transient_signal) if np.max(transient_signal) > 0 else 1.0
        transient_mask = transient_signal / max_transient

        # Sustained is inverse of transient
        sustained_mask = 1.0 - transient_mask

        return transient_mask, sustained_mask

    def reset(self):
        """Reset detector states"""
        self.fast_state = 0.0
        self.slow_state = 0.0


class MultibandProcessor:
    """
    Complete multiband processor with crossover, per-band processing, and recombination
    """

    def __init__(self, num_bands: int = 4, sample_rate: int = 44100):
        """
        Initialize multiband processor

        Args:
            num_bands: Number of frequency bands
            sample_rate: Sample rate in Hz
        """
        self.num_bands = num_bands
        self.sample_rate = sample_rate

        # Create crossover
        self.crossover = create_default_crossover(num_bands, sample_rate)

        # Per-band gains and parameters
        self.band_gains = np.ones(num_bands)
        self.band_enabled = [True] * num_bands

        # Per-band compression parameters
        self.band_compression_ratios = [2.0] * num_bands
        self.band_compression_thresholds = [-12.0] * num_bands

        # Simple envelope followers for compression
        self.band_envelope_states = [0.0] * num_bands

    def set_band_gain(self, band_idx: int, gain_db: float):
        """Set gain for specific band"""
        if 0 <= band_idx < self.num_bands:
            self.band_gains[band_idx] = 10 ** (gain_db / 20)

    def set_band_compression(self, band_idx: int, ratio: float, threshold_db: float):
        """Set compression parameters for specific band"""
        if 0 <= band_idx < self.num_bands:
            self.band_compression_ratios[band_idx] = ratio
            self.band_compression_thresholds[band_idx] = threshold_db

    def apply_band_compression(self, audio: np.ndarray, band_idx: int) -> np.ndarray:
        """
        Apply simple compression to a band

        Args:
            audio: Band audio
            band_idx: Band index

        Returns:
            Compressed audio
        """
        ratio = self.band_compression_ratios[band_idx]
        threshold_db = self.band_compression_thresholds[band_idx]
        threshold_linear = 10 ** (threshold_db / 20)

        # Simple RMS-based compression
        rms = np.sqrt(np.mean(audio ** 2))

        if rms > threshold_linear:
            # Calculate gain reduction
            excess_db = 20 * np.log10(rms / threshold_linear)
            reduction_db = excess_db * (1 - 1/ratio)
            gain = 10 ** (-reduction_db / 20)

            return audio * gain
        else:
            return audio

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through multiband pipeline

        Args:
            audio: Input audio

        Returns:
            Processed audio with multiband processing applied
        """
        # Split into bands
        bands = self.crossover.split_bands(audio)

        # Process each band
        processed_bands = []
        for i, band in enumerate(bands):
            if self.band_enabled[i]:
                # Apply compression
                band = self.apply_band_compression(band, i)

                # Apply gain
                band = band * self.band_gains[i]

            processed_bands.append(band)

        # Recombine
        output = self.crossover.combine_bands(processed_bands)

        return output

    def load_multiband_analysis(self, multiband_data: dict):
        """
        Load multiband analysis data from preset

        Args:
            multiband_data: Dictionary with per-band analysis
        """
        for band_idx, band_data in multiband_data.items():
            # Parse band index
            try:
                idx = int(band_idx.replace('band_', ''))
            except:
                continue

            if idx >= self.num_bands:
                continue

            # Apply band-specific parameters
            if 'gain_db' in band_data:
                self.set_band_gain(idx, band_data['gain_db'])

            if 'compression_ratio' in band_data:
                ratio = band_data['compression_ratio']
                threshold = band_data.get('compression_threshold_db', -12.0)
                self.set_band_compression(idx, ratio, threshold)

    def reset_state(self):
        """Reset processing state"""
        self.crossover.reset_state()
        self.band_envelope_states = [0.0] * self.num_bands


def create_default_crossover(num_bands: int = 4, sample_rate: int = 44100) -> MultibandCrossover:
    """
    Create a default crossover configuration

    Args:
        num_bands: Number of bands (4 or 8 recommended)
        sample_rate: Sample rate in Hz

    Returns:
        MultibandCrossover instance
    """
    if num_bands == 4:
        # 4-band: Bass, Low-mid, High-mid, Treble
        # Research recommendation: good balance of quality and efficiency
        crossover_freqs = [200, 1000, 4000]

    elif num_bands == 8:
        # 8-band: More precise control
        crossover_freqs = [80, 200, 500, 1000, 2000, 4000, 8000]

    else:
        # Generic: evenly spaced on log scale
        min_freq = 100
        max_freq = 10000
        crossover_freqs = np.logspace(
            np.log10(min_freq),
            np.log10(max_freq),
            num_bands - 1
        ).tolist()

    return MultibandCrossover(crossover_freqs, sample_rate)


if __name__ == "__main__":
    # Test multiband processing
    import matplotlib.pyplot as plt

    # Create test signal
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Multi-frequency signal
    signal_test = (
        np.sin(2 * np.pi * 100 * t) +    # Bass
        np.sin(2 * np.pi * 500 * t) +    # Low-mid
        np.sin(2 * np.pi * 2000 * t) +   # High-mid
        np.sin(2 * np.pi * 8000 * t)     # Treble
    )

    # Create crossover
    crossover = create_default_crossover(num_bands=4, sample_rate=sample_rate)

    # Split into bands
    bands = crossover.split_bands_offline(signal_test)

    # Recombine
    reconstructed = crossover.combine_bands(bands)

    # Verify reconstruction
    error = np.max(np.abs(signal_test - reconstructed))
    print(f"Reconstruction error: {error:.6f}")
    print(f"Number of bands: {len(bands)}")
    print(f"Crossover frequencies: {crossover.crossover_freqs}")

    # Test transient detector
    transient_detector = TransientDetector(sample_rate)

    # Create signal with transient
    impulse = np.zeros(int(sample_rate * 0.1))
    impulse[1000] = 1.0
    impulse = signal.lfilter([1], [1, -0.95], impulse)  # Add decay

    transient_mask, sustained_mask = transient_detector.detect(impulse)

    print(f"\nTransient detection:")
    print(f"  Peak transient: {np.max(transient_mask):.3f}")
    print(f"  Peak sustained: {np.max(sustained_mask):.3f}")
