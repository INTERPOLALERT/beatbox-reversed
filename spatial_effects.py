"""
Stereo/Spatial Effects Processor
Implements panning, stereo width, and reverb for spatial imaging
"""
import numpy as np
from scipy import signal
from typing import Tuple, Optional


class StereoWidthProcessor:
    """
    Adjusts stereo width of audio signal
    """

    def __init__(self):
        """Initialize stereo width processor"""
        self.width = 1.0  # 0=mono, 1=normal, 2=extra wide

    def set_width(self, width: float):
        """
        Set stereo width

        Args:
            width: Width amount (0=mono, 1=normal, 2=extra wide)
        """
        self.width = np.clip(width, 0.0, 2.0)

    def process_stereo(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process stereo channels

        Args:
            left: Left channel
            right: Right channel

        Returns:
            Tuple of (processed_left, processed_right)
        """
        # Mid-side processing
        mid = (left + right) / 2.0
        side = (left - right) / 2.0

        # Adjust side width
        side_adjusted = side * self.width

        # Convert back to left-right
        left_out = mid + side_adjusted
        right_out = mid - side_adjusted

        return left_out, right_out

    def mono_to_stereo(self, mono: np.ndarray, width: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert mono to pseudo-stereo using delay and filtering

        Args:
            mono: Mono audio
            width: Stereo width amount

        Returns:
            Tuple of (left, right)
        """
        # Create slight delay for right channel (Haas effect)
        delay_samples = int(0.0002 * 44100)  # 0.2ms delay

        # Left channel: slight high-pass
        left = mono.copy()

        # Right channel: delayed with slight low-pass
        right = np.pad(mono, (delay_samples, 0), mode='constant')[:-delay_samples]

        # Apply width
        mid = (left + right) / 2.0
        side = (left - right) / 2.0 * width

        left_out = mid + side
        right_out = mid - side

        return left_out, right_out


class SimplePanner:
    """
    Simple stereo panner
    """

    def __init__(self):
        """Initialize panner"""
        self.pan = 0.0  # -1=full left, 0=center, 1=full right

    def set_pan(self, pan: float):
        """
        Set pan position

        Args:
            pan: Pan position (-1 to 1)
        """
        self.pan = np.clip(pan, -1.0, 1.0)

    def process(self, mono: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pan mono signal to stereo

        Args:
            mono: Mono audio

        Returns:
            Tuple of (left, right)
        """
        # Constant power panning
        pan_radians = (self.pan + 1.0) * np.pi / 4.0  # Map -1..1 to 0..pi/2

        left_gain = np.cos(pan_radians)
        right_gain = np.sin(pan_radians)

        left = mono * left_gain
        right = mono * right_gain

        return left, right


class SimpleReverb:
    """
    Simple reverb using Schroeder design (4 comb filters + 2 allpass)
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize reverb

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

        # Comb filter delays (in samples) - prime numbers for diffusion
        self.comb_delays = [
            int(0.0297 * sample_rate),  # ~29.7ms
            int(0.0371 * sample_rate),  # ~37.1ms
            int(0.0411 * sample_rate),  # ~41.1ms
            int(0.0437 * sample_rate)   # ~43.7ms
        ]

        # Allpass filter delays
        self.allpass_delays = [
            int(0.005 * sample_rate),   # 5ms
            int(0.0017 * sample_rate)   # 1.7ms
        ]

        # Initialize buffers
        self.comb_buffers = [np.zeros(delay) for delay in self.comb_delays]
        self.comb_positions = [0] * len(self.comb_delays)

        self.allpass_buffers = [np.zeros(delay) for delay in self.allpass_delays]
        self.allpass_positions = [0] * len(self.allpass_delays)

        # Parameters
        self.room_size = 0.5  # 0-1
        self.damping = 0.5    # 0-1
        self.wet_mix = 0.3    # 0-1

    def set_room_size(self, size: float):
        """Set room size (0-1)"""
        self.room_size = np.clip(size, 0.0, 1.0)

    def set_damping(self, damp: float):
        """Set damping (0-1)"""
        self.damping = np.clip(damp, 0.0, 1.0)

    def set_wet_mix(self, wet: float):
        """Set wet/dry mix (0-1)"""
        self.wet_mix = np.clip(wet, 0.0, 1.0)

    def _process_comb(self, input_sample: float, buffer_idx: int) -> float:
        """Process single comb filter"""
        buffer = self.comb_buffers[buffer_idx]
        pos = self.comb_positions[buffer_idx]

        # Read delayed sample
        delayed = buffer[pos]

        # Feedback with damping
        feedback_gain = 0.7 * self.room_size
        damped = delayed * (1.0 - self.damping) + delayed * self.damping * 0.5

        # Write to buffer
        buffer[pos] = input_sample + damped * feedback_gain

        # Update position
        self.comb_positions[buffer_idx] = (pos + 1) % len(buffer)

        return delayed

    def _process_allpass(self, input_sample: float, buffer_idx: int) -> float:
        """Process single allpass filter"""
        buffer = self.allpass_buffers[buffer_idx]
        pos = self.allpass_positions[buffer_idx]

        # Read delayed sample
        delayed = buffer[pos]

        # Allpass feedback
        feedback_gain = 0.5
        output = -input_sample + delayed
        buffer[pos] = input_sample + delayed * feedback_gain

        # Update position
        self.allpass_positions[buffer_idx] = (pos + 1) % len(buffer)

        return output

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through reverb

        Args:
            audio: Input audio (mono)

        Returns:
            Reverb output
        """
        output = np.zeros_like(audio)

        for i, sample in enumerate(audio):
            # Process through parallel comb filters
            comb_sum = 0.0
            for comb_idx in range(len(self.comb_buffers)):
                comb_sum += self._process_comb(sample, comb_idx)

            comb_out = comb_sum / len(self.comb_buffers)

            # Process through series allpass filters
            allpass_out = comb_out
            for allpass_idx in range(len(self.allpass_buffers)):
                allpass_out = self._process_allpass(allpass_out, allpass_idx)

            output[i] = allpass_out

        # Mix wet/dry
        return audio * (1.0 - self.wet_mix) + output * self.wet_mix


class SpatialProcessor:
    """
    Complete spatial processing chain
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize spatial processor

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

        # Components
        self.width_processor = StereoWidthProcessor()
        self.panner = SimplePanner()
        self.reverb_left = SimpleReverb(sample_rate)
        self.reverb_right = SimpleReverb(sample_rate)

        # Settings
        self.stereo_width = 1.0
        self.pan_position = 0.0
        self.reverb_amount = 0.0
        self.reverb_size = 0.5

    def set_stereo_width(self, width: float):
        """Set stereo width (0-2)"""
        self.stereo_width = width
        self.width_processor.set_width(width)

    def set_pan(self, pan: float):
        """Set pan position (-1 to 1)"""
        self.pan_position = pan
        self.panner.set_pan(pan)

    def set_reverb(self, amount: float, size: float = 0.5):
        """
        Set reverb parameters

        Args:
            amount: Reverb wet mix (0-1)
            size: Room size (0-1)
        """
        self.reverb_amount = amount
        self.reverb_size = size

        self.reverb_left.set_wet_mix(amount)
        self.reverb_left.set_room_size(size)
        self.reverb_right.set_wet_mix(amount)
        self.reverb_right.set_room_size(size)

    def process_mono(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process mono audio to stereo with spatial effects

        Args:
            audio: Mono input audio

        Returns:
            Tuple of (left, right) stereo output
        """
        # Apply panning
        left, right = self.panner.process(audio)

        # Apply reverb if enabled
        if self.reverb_amount > 0.01:
            left = self.reverb_left.process(left)
            right = self.reverb_right.process(right)

        # Apply stereo width
        left, right = self.width_processor.process_stereo(left, right)

        return left, right

    def process_stereo(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process stereo audio with spatial effects

        Args:
            left: Left channel
            right: Right channel

        Returns:
            Tuple of (processed_left, processed_right)
        """
        # Apply reverb if enabled
        if self.reverb_amount > 0.01:
            left = self.reverb_left.process(left)
            right = self.reverb_right.process(right)

        # Apply stereo width
        left, right = self.width_processor.process_stereo(left, right)

        return left, right
