"""
Adaptive Sound-Type Processor
Real-time detection and adaptive processing based on sound type
Implements per-sound-type EQ, compression, and effects
"""
import numpy as np
from scipy import signal
import librosa
from typing import Dict, Optional, Tuple
from sound_classifier import OnsetBasedClassifier


class AdaptiveSoundProcessor:
    """
    Processes audio adaptively based on detected sound type
    Each sound type gets optimized EQ and compression
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize adaptive processor

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.classifier = OnsetBasedClassifier(sample_rate)

        # Per-sound-type processing parameters
        self.sound_profiles = self._create_sound_profiles()

        # Detection state
        self.onset_detected = False
        self.current_sound_type = 'other'
        self.onset_energy_threshold = 0.01

        # Transition smoothing
        self.transition_samples = int(0.010 * sample_rate)  # 10ms smooth transition
        self.current_eq_state = None
        self.target_eq_state = None
        self.transition_progress = 1.0

    def _create_sound_profiles(self) -> Dict:
        """
        Create optimized processing profiles for each sound type

        Returns:
            Dictionary of sound_type -> processing parameters
        """
        return {
            'kick': {
                'eq_boost_freqs': [60, 80, 100],  # Low-end punch
                'eq_boost_gains': [3.0, 4.0, 2.0],
                'eq_cut_freqs': [300, 500, 1000],  # Reduce boxiness
                'eq_cut_gains': [-2.0, -1.5, -1.0],
                'compression_ratio': 4.0,
                'compression_threshold': -12.0,
                'attack_ms': 5.0,
                'release_ms': 100.0,
                'saturation_amount': 0.2,
                'description': 'Kick - Deep punch with controlled low-end'
            },
            'snare': {
                'eq_boost_freqs': [200, 1000, 3000, 8000],  # Body and crack
                'eq_boost_gains': [1.5, 2.0, 3.0, 2.0],
                'eq_cut_freqs': [400, 600],  # Reduce muddiness
                'eq_cut_gains': [-1.5, -1.0],
                'compression_ratio': 3.0,
                'compression_threshold': -10.0,
                'attack_ms': 3.0,
                'release_ms': 80.0,
                'saturation_amount': 0.15,
                'description': 'Snare - Crisp attack with body'
            },
            'hihat': {
                'eq_boost_freqs': [6000, 8000, 12000],  # Brightness and air
                'eq_boost_gains': [2.0, 3.0, 1.5],
                'eq_cut_freqs': [200, 500, 1000],  # Remove low-end
                'eq_cut_gains': [-4.0, -3.0, -2.0],
                'compression_ratio': 2.0,
                'compression_threshold': -8.0,
                'attack_ms': 1.0,
                'release_ms': 40.0,
                'saturation_amount': 0.05,
                'description': 'Hi-hat - Crispy high-frequency detail'
            },
            'bass': {
                'eq_boost_freqs': [80, 120, 200],  # Fundamental and harmonics
                'eq_boost_gains': [3.0, 2.5, 1.5],
                'eq_cut_freqs': [400, 800],  # Clarity
                'eq_cut_gains': [-1.0, -0.5],
                'compression_ratio': 3.5,
                'compression_threshold': -14.0,
                'attack_ms': 10.0,
                'release_ms': 150.0,
                'saturation_amount': 0.25,
                'description': 'Bass - Warm low-end sustain'
            },
            'vocal': {
                'eq_boost_freqs': [1000, 2500, 5000],  # Presence and clarity
                'eq_boost_gains': [2.0, 3.0, 2.0],
                'eq_cut_freqs': [200, 10000],  # De-mud and de-harsh
                'eq_cut_gains': [-1.5, -1.0],
                'compression_ratio': 2.5,
                'compression_threshold': -15.0,
                'attack_ms': 5.0,
                'release_ms': 100.0,
                'saturation_amount': 0.10,
                'description': 'Vocal - Forward presence and clarity'
            },
            'other': {
                'eq_boost_freqs': [1000],
                'eq_boost_gains': [0.0],
                'eq_cut_freqs': [],
                'eq_cut_gains': [],
                'compression_ratio': 2.0,
                'compression_threshold': -12.0,
                'attack_ms': 5.0,
                'release_ms': 80.0,
                'saturation_amount': 0.05,
                'description': 'Other - Neutral processing'
            }
        }

    def detect_sound_type(self, audio_buffer: np.ndarray) -> Tuple[str, float]:
        """
        Detect sound type from audio buffer

        Args:
            audio_buffer: Input audio buffer

        Returns:
            Tuple of (sound_type, confidence)
        """
        # Detect onset (sharp energy increase)
        energy = np.sqrt(np.mean(audio_buffer ** 2))

        if energy > self.onset_energy_threshold:
            # Classify the sound
            sound_type, confidence = self.classifier.classify_buffer(audio_buffer)
            return sound_type, confidence
        else:
            return 'other', 0.5

    def get_adaptive_eq_params(self, sound_type: str) -> Dict:
        """
        Get EQ parameters for sound type

        Args:
            sound_type: Detected sound type

        Returns:
            Dictionary of EQ parameters
        """
        if sound_type not in self.sound_profiles:
            sound_type = 'other'

        return self.sound_profiles[sound_type]

    def apply_adaptive_eq(self, audio: np.ndarray, sound_type: str, mix: float = 1.0) -> np.ndarray:
        """
        Apply adaptive EQ based on sound type

        Args:
            audio: Input audio
            sound_type: Detected sound type
            mix: Wet/dry mix (0=dry, 1=wet)

        Returns:
            Processed audio
        """
        profile = self.get_adaptive_eq_params(sound_type)

        processed = audio.copy()

        # Apply boosts
        for freq, gain in zip(profile['eq_boost_freqs'], profile['eq_boost_gains']):
            processed = self._apply_parametric_eq(processed, freq, gain, q=1.5)

        # Apply cuts
        for freq, gain in zip(profile['eq_cut_freqs'], profile['eq_cut_gains']):
            processed = self._apply_parametric_eq(processed, freq, gain, q=1.0)

        # Mix wet/dry
        return audio * (1 - mix) + processed * mix

    def _apply_parametric_eq(self, audio: np.ndarray, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
        """
        Apply parametric EQ filter

        Args:
            audio: Input audio
            freq: Center frequency in Hz
            gain_db: Gain in dB
            q: Q factor

        Returns:
            Filtered audio
        """
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

    def process_buffer(self, audio_buffer: np.ndarray, enable_adaptive: bool = True) -> Tuple[np.ndarray, str]:
        """
        Process audio buffer with adaptive sound-type detection

        Args:
            audio_buffer: Input audio buffer
            enable_adaptive: Enable adaptive processing

        Returns:
            Tuple of (processed_audio, detected_sound_type)
        """
        if not enable_adaptive:
            return audio_buffer, 'bypass'

        # Detect sound type
        sound_type, confidence = self.detect_sound_type(audio_buffer)

        # Apply adaptive processing
        if confidence > 0.5:
            processed = self.apply_adaptive_eq(audio_buffer, sound_type, mix=0.7)
            return processed, sound_type
        else:
            return audio_buffer, 'other'


class MicroTransientProcessor:
    """
    Preserves and enhances micro-transients and articulation details
    Implements dual-envelope detection and transient shaping
    """

    def __init__(self, sample_rate: int = 44100):
        """Initialize transient processor"""
        self.sample_rate = sample_rate

        # Envelope detection
        self.attack_time_fast = 0.001  # 1ms for transients
        self.release_time_fast = 0.010  # 10ms
        self.attack_time_slow = 0.020  # 20ms for sustain
        self.release_time_slow = 0.200  # 200ms

        # Calculate coefficients
        self.fast_attack_coeff = self._time_to_coeff(self.attack_time_fast)
        self.fast_release_coeff = self._time_to_coeff(self.release_time_fast)
        self.slow_attack_coeff = self._time_to_coeff(self.attack_time_slow)
        self.slow_release_coeff = self._time_to_coeff(self.release_time_slow)

        # State
        self.fast_envelope = 0.0
        self.slow_envelope = 0.0

    def _time_to_coeff(self, time_seconds: float) -> float:
        """Convert time constant to filter coefficient"""
        return np.exp(-1.0 / (time_seconds * self.sample_rate))

    def detect_transient_envelope(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect fast (transient) and slow (sustain) envelopes

        Args:
            audio: Input audio

        Returns:
            Tuple of (fast_envelope, slow_envelope)
        """
        # Get absolute values
        rectified = np.abs(audio)

        # Fast envelope (tracks transients)
        fast_env = np.zeros_like(rectified)
        # Slow envelope (tracks sustain)
        slow_env = np.zeros_like(rectified)

        for i in range(len(rectified)):
            # Fast envelope
            if rectified[i] > self.fast_envelope:
                self.fast_envelope = self.fast_attack_coeff * self.fast_envelope + \
                                   (1 - self.fast_attack_coeff) * rectified[i]
            else:
                self.fast_envelope = self.fast_release_coeff * self.fast_envelope + \
                                   (1 - self.fast_release_coeff) * rectified[i]

            fast_env[i] = self.fast_envelope

            # Slow envelope
            if rectified[i] > self.slow_envelope:
                self.slow_envelope = self.slow_attack_coeff * self.slow_envelope + \
                                   (1 - self.slow_attack_coeff) * rectified[i]
            else:
                self.slow_envelope = self.slow_release_coeff * self.slow_envelope + \
                                   (1 - self.slow_release_coeff) * rectified[i]

            slow_env[i] = self.slow_envelope

        return fast_env, slow_env

    def extract_transients(self, audio: np.ndarray, sensitivity: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract transient and sustain components

        Args:
            audio: Input audio
            sensitivity: Transient extraction sensitivity (0-1)

        Returns:
            Tuple of (transient_component, sustain_component)
        """
        # Detect envelopes
        fast_env, slow_env = self.detect_transient_envelope(audio)

        # Transient = difference between fast and slow envelope
        transient_env = np.maximum(0, fast_env - slow_env)

        # Normalize envelopes to create masks
        max_env = np.maximum(fast_env, 1e-10)
        transient_mask = (transient_env / max_env) * sensitivity + (1 - sensitivity) * 0.5
        transient_mask = np.clip(transient_mask, 0, 1)

        sustain_mask = 1.0 - transient_mask

        # Apply masks
        transient_component = audio * transient_mask
        sustain_component = audio * sustain_mask

        return transient_component, sustain_component

    def enhance_transients(self, audio: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """
        Enhance transients while preserving sustain

        Args:
            audio: Input audio
            amount: Enhancement amount (0-1, 0.5=neutral)

        Returns:
            Enhanced audio
        """
        # Extract components
        transient, sustain = self.extract_transients(audio, sensitivity=0.7)

        # Enhance transients
        transient_gain = 1.0 + amount
        enhanced_transient = transient * transient_gain

        # Recombine
        output = enhanced_transient + sustain

        # Prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val

        return output
