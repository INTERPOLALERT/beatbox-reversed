"""
Harmonic Distortion and Saturation Processor
Implements tube/tape-style saturation, harmonic enhancement, and timbre shaping
"""
import numpy as np
from scipy import signal
from typing import Optional


class HarmonicSaturator:
    """
    Implements various saturation algorithms for harmonic coloration
    """

    def __init__(self):
        """Initialize saturator"""
        self.saturation_amount = 0.0  # 0-1
        self.saturation_type = 'soft'  # 'soft', 'hard', 'tube', 'tape'

    def set_saturation(self, amount: float, saturation_type: str = 'soft'):
        """
        Set saturation parameters

        Args:
            amount: Saturation amount (0-1)
            saturation_type: Type of saturation ('soft', 'hard', 'tube', 'tape')
        """
        self.saturation_amount = np.clip(amount, 0.0, 1.0)
        self.saturation_type = saturation_type

    def soft_clip(self, x: np.ndarray) -> np.ndarray:
        """
        Soft clipping (smooth saturation)

        Args:
            x: Input signal

        Returns:
            Saturated signal
        """
        # Hyperbolic tangent - smooth soft clipping
        return np.tanh(x)

    def hard_clip(self, x: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """
        Hard clipping with soft knee

        Args:
            x: Input signal
            threshold: Clipping threshold

        Returns:
            Clipped signal
        """
        return np.clip(x, -threshold, threshold)

    def tube_saturation(self, x: np.ndarray) -> np.ndarray:
        """
        Tube-style saturation (asymmetric)

        Args:
            x: Input signal

        Returns:
            Saturated signal
        """
        # Asymmetric saturation mimicking tube characteristics
        # Positive half: softer saturation
        # Negative half: harder saturation
        output = np.zeros_like(x)

        pos_mask = x >= 0
        neg_mask = x < 0

        # Positive side - soft saturation
        output[pos_mask] = np.tanh(x[pos_mask] * 1.2)

        # Negative side - harder saturation with slight bias
        output[neg_mask] = np.tanh(x[neg_mask] * 1.5) * 0.9

        return output

    def tape_saturation(self, x: np.ndarray) -> np.ndarray:
        """
        Tape-style saturation with warmth

        Args:
            x: Input signal

        Returns:
            Saturated signal
        """
        # Tape saturation - adds even harmonics
        # Use a softer curve with bias toward warmth
        output = x / (1.0 + np.abs(x) ** 1.5)

        return output

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio with saturation

        Args:
            audio: Input audio

        Returns:
            Saturated audio
        """
        if self.saturation_amount < 0.01:
            return audio

        # Apply drive (pre-gain)
        drive = 1.0 + self.saturation_amount * 4.0
        driven = audio * drive

        # Apply saturation curve
        if self.saturation_type == 'soft':
            saturated = self.soft_clip(driven)
        elif self.saturation_type == 'hard':
            saturated = self.hard_clip(driven, threshold=0.8)
        elif self.saturation_type == 'tube':
            saturated = self.tube_saturation(driven)
        elif self.saturation_type == 'tape':
            saturated = self.tape_saturation(driven)
        else:
            saturated = self.soft_clip(driven)

        # Compensate for gain
        saturated = saturated / drive

        # Mix with dry signal
        mix = self.saturation_amount
        output = audio * (1.0 - mix) + saturated * mix

        return output


class HarmonicEnhancer:
    """
    Enhances harmonics through excitation and filtering
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize harmonic enhancer

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.enhancement_amount = 0.0

    def set_enhancement(self, amount: float):
        """
        Set enhancement amount

        Args:
            amount: Enhancement amount (0-1)
        """
        self.enhancement_amount = np.clip(amount, 0.0, 1.0)

    def generate_harmonics(self, audio: np.ndarray) -> np.ndarray:
        """
        Generate harmonic content through waveshaping

        Args:
            audio: Input audio

        Returns:
            Harmonic-enhanced audio
        """
        # Generate 2nd and 3rd harmonics through simple waveshaping
        # 2nd harmonic (even) - adds warmth
        harmonic_2 = np.sign(audio) * (audio ** 2)

        # 3rd harmonic (odd) - adds edge
        harmonic_3 = audio ** 3

        # Mix harmonics
        harmonics = harmonic_2 * 0.3 + harmonic_3 * 0.2

        # High-pass filter harmonics to avoid muddiness
        sos = signal.butter(4, 200, 'hp', fs=self.sample_rate, output='sos')
        harmonics_filtered = signal.sosfilt(sos, harmonics)

        return harmonics_filtered

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio with harmonic enhancement

        Args:
            audio: Input audio

        Returns:
            Enhanced audio
        """
        if self.enhancement_amount < 0.01:
            return audio

        # Generate harmonics
        harmonics = self.generate_harmonics(audio)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(harmonics))
        if max_val > 0:
            harmonics = harmonics / max_val * 0.3

        # Mix with original
        output = audio + harmonics * self.enhancement_amount

        # Prevent clipping
        max_output = np.max(np.abs(output))
        if max_output > 1.0:
            output = output / max_output

        return output


class ExciterFilter:
    """
    Psychoacoustic exciter for adding 'air' and brightness
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize exciter

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.exciter_amount = 0.0

    def set_exciter(self, amount: float):
        """
        Set exciter amount

        Args:
            amount: Exciter amount (0-1)
        """
        self.exciter_amount = np.clip(amount, 0.0, 1.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio with exciter

        Args:
            audio: Input audio

        Returns:
            Excited audio
        """
        if self.exciter_amount < 0.01:
            return audio

        # High-pass filter (above 3kHz)
        sos_hp = signal.butter(4, 3000, 'hp', fs=self.sample_rate, output='sos')
        highs = signal.sosfilt(sos_hp, audio)

        # Apply harmonic generation to highs only
        excited = np.tanh(highs * 2.0)

        # Band-pass to keep only high frequencies
        sos_bp = signal.butter(2, [4000, 16000], 'bp', fs=self.sample_rate, output='sos')
        excited_filtered = signal.sosfilt(sos_bp, excited)

        # Mix back with original
        output = audio + excited_filtered * self.exciter_amount * 0.2

        return output


class TimbreShaper:
    """
    Complete timbre shaping processor combining saturation, harmonics, and excitation
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize timbre shaper

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

        # Components
        self.saturator = HarmonicSaturator()
        self.harmonic_enhancer = HarmonicEnhancer(sample_rate)
        self.exciter = ExciterFilter(sample_rate)

        # Settings
        self.saturation_amount = 0.0
        self.saturation_type = 'tube'
        self.harmonic_amount = 0.0
        self.exciter_amount = 0.0
        self.warmth = 0.0

    def set_saturation(self, amount: float, saturation_type: str = 'tube'):
        """
        Set saturation

        Args:
            amount: Saturation amount (0-1)
            saturation_type: Type ('soft', 'hard', 'tube', 'tape')
        """
        self.saturation_amount = amount
        self.saturation_type = saturation_type
        self.saturator.set_saturation(amount, saturation_type)

    def set_harmonics(self, amount: float):
        """
        Set harmonic enhancement

        Args:
            amount: Enhancement amount (0-1)
        """
        self.harmonic_amount = amount
        self.harmonic_enhancer.set_enhancement(amount)

    def set_exciter(self, amount: float):
        """
        Set exciter amount

        Args:
            amount: Exciter amount (0-1)
        """
        self.exciter_amount = amount
        self.exciter.set_exciter(amount)

    def set_warmth(self, amount: float):
        """
        Set overall warmth (combination of tube saturation and low harmonics)

        Args:
            amount: Warmth amount (0-1)
        """
        self.warmth = amount

        # Warmth uses tube saturation
        self.set_saturation(amount * 0.5, 'tube')
        self.set_harmonics(amount * 0.3)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio with complete timbre shaping

        Args:
            audio: Input audio

        Returns:
            Timbre-shaped audio
        """
        output = audio.copy()

        # Apply saturation first
        output = self.saturator.process(output)

        # Add harmonics
        output = self.harmonic_enhancer.process(output)

        # Add excitation/air
        output = self.exciter.process(output)

        # Final safety limiter
        max_val = np.max(np.abs(output))
        if max_val > 0.99:
            output = output / max_val * 0.99

        return output

    def analyze_and_apply(self, audio: np.ndarray, preset_params: dict) -> np.ndarray:
        """
        Apply timbre shaping based on analyzed parameters

        Args:
            audio: Input audio
            preset_params: Dictionary with 'saturation_amount' etc.

        Returns:
            Processed audio
        """
        # Extract parameters from preset
        sat_amount = preset_params.get('saturation_amount', 0.1)
        sat_type = preset_params.get('saturation_type', 'tube')

        # Apply settings
        self.set_saturation(sat_amount, sat_type)
        self.set_harmonics(sat_amount * 0.5)
        self.set_exciter(sat_amount * 0.3)

        # Process
        return self.process(audio)
