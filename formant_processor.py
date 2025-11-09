"""
Formant-based Audio Processor
Applies formant-preserving EQ to match vocal tract characteristics
"""
import numpy as np
from scipy import signal
from typing import Dict, List, Optional


class FormantProcessor:
    """
    Processes audio to match formant characteristics from reference
    Formants define the vocal tract resonances and timbral quality
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize formant processor

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.target_formants = None
        self.formant_bandwidth = 100  # Hz, typical formant bandwidth
        self.formant_boost_db = 3.0  # dB boost at formant frequencies

    def load_formant_data(self, formant_data: Dict):
        """
        Load formant data from preset

        Args:
            formant_data: Dictionary with 'frequencies' and 'num_formants'
        """
        if 'frequencies' in formant_data:
            self.target_formants = formant_data['frequencies']
            print(f"  Loaded {len(self.target_formants)} formants: {[f'{f:.0f}Hz' for f in self.target_formants[:3]]}")

    def process(self, audio: np.ndarray, mix: float = 0.5) -> np.ndarray:
        """
        Apply formant-based processing to audio

        Args:
            audio: Input audio
            mix: Wet/dry mix (0=dry, 1=fully formant-shaped)

        Returns:
            Processed audio with formant characteristics applied
        """
        if self.target_formants is None or len(self.target_formants) == 0:
            return audio

        processed = audio.copy()

        # Apply formant boosts using parametric EQ
        for formant_freq in self.target_formants[:5]:  # Use first 5 formants
            if formant_freq > 0 and formant_freq < self.sample_rate / 2:
                processed = self._apply_formant_eq(
                    processed,
                    formant_freq,
                    self.formant_boost_db,
                    self.formant_bandwidth
                )

        # Mix wet/dry
        return audio * (1.0 - mix) + processed * mix

    def _apply_formant_eq(self, audio: np.ndarray, freq: float,
                          gain_db: float, bandwidth: float) -> np.ndarray:
        """
        Apply parametric EQ at formant frequency

        Args:
            audio: Input audio
            freq: Formant frequency in Hz
            gain_db: Boost in dB
            bandwidth: Bandwidth in Hz

        Returns:
            EQ'd audio
        """
        # Calculate Q factor from bandwidth
        # Q = f0 / bandwidth
        q = freq / bandwidth if bandwidth > 0 else 1.0
        q = max(0.5, min(q, 10.0))  # Clamp Q to reasonable range

        # Design peaking EQ filter
        nyquist = self.sample_rate / 2

        if freq >= nyquist:
            return audio

        # Convert gain to linear
        gain_linear = 10 ** (gain_db / 20)

        # Design filter using scipy
        w0 = 2 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / (2 * q)

        A = gain_linear

        # Peaking EQ coefficients
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        # Apply filter
        try:
            filtered = signal.lfilter(b, a, audio)
            return filtered
        except:
            return audio

    def set_formant_strength(self, boost_db: float):
        """
        Set formant boost strength

        Args:
            boost_db: Boost amount in dB
        """
        self.formant_boost_db = np.clip(boost_db, 0.0, 6.0)

    def set_formant_bandwidth(self, bandwidth: float):
        """
        Set formant bandwidth

        Args:
            bandwidth: Bandwidth in Hz
        """
        self.formant_bandwidth = np.clip(bandwidth, 50, 500)


def demo_formant_processor():
    """Demo formant processing"""
    import matplotlib.pyplot as plt

    # Create test signal
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create a sawtooth wave (rich harmonics)
    signal_test = signal.sawtooth(2 * np.pi * 200 * t) * 0.3

    # Create processor
    processor = FormantProcessor(sample_rate)

    # Load typical vocal formants
    formant_data = {
        'frequencies': [700, 1220, 2600, 3300, 4100],  # Typical "ah" vowel
        'num_formants': 5
    }
    processor.load_formant_data(formant_data)

    # Process
    processed = processor.process(signal_test, mix=0.8)

    print("Formant processor demo complete")
    print(f"Applied formants at: {formant_data['frequencies']} Hz")
    print(f"RMS input: {np.sqrt(np.mean(signal_test**2)):.4f}")
    print(f"RMS output: {np.sqrt(np.mean(processed**2)):.4f}")


if __name__ == "__main__":
    demo_formant_processor()
