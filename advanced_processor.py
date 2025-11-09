"""
Advanced Real-Time Processor - Professional Grade
Implements multiband processing, adaptive transient preservation,
sound classification, and per-sound-type preset switching
"""
import numpy as np
import sounddevice as sd
import soundfile as sf
from pedalboard import Pedalboard, Compressor, PeakFilter, LowShelfFilter, HighShelfFilter, Limiter, Gain
from datetime import datetime
from pathlib import Path
import json
import threading
import queue
import time

import config
from multiband_processor import create_default_crossover, TransientDetector, MultibandEnvelopeFollower
from sound_classifier import OnsetBasedClassifier


class AdvancedLiveProcessor:
    """
    Professional-grade real-time audio processor with adaptive processing
    """

    def __init__(self, preset_path: str = None):
        """
        Initialize advanced processor

        Args:
            preset_path: Path to advanced preset JSON file
        """
        self.preset_path = preset_path
        self.preset = None

        # Processing components
        self.crossover = None
        self.effects_chains = {}  # Per-band or per-sound-type
        self.transient_detector = None
        self.sound_classifier = None

        # Control parameters
        self.wet_dry_mix = 1.0  # 0.0 = dry, 1.0 = wet
        self.transient_preservation = 0.8  # How much to preserve transients
        self.multiband_enabled = True
        self.classification_enabled = False  # Enable when model trained

        # Per-band mix controls
        self.band_mix_levels = [1.0, 1.0, 1.0, 1.0]  # 4-band default

        # Recording state
        self.is_recording = False
        self.recorded_audio = []
        self.recording_lock = threading.Lock()

        # Audio stream
        self.stream = None
        self.is_processing = False

        # Classification buffer (for onset detection)
        self.classification_buffer = np.array([])
        self.classification_queue = queue.Queue()
        self.current_sound_type = 'other'
        self.last_classification_time = 0

        # Load config
        self.audio_config = config.AudioConfig()

        # Safety limiter
        self.safety_limiter = Limiter(threshold_db=-1.0, release_ms=50)

        # Input/output gain
        self.input_gain_db = 0.0
        self.output_gain_db = 0.0

    def load_preset(self, preset_path: str):
        """Load advanced preset from JSON file"""
        print(f"Loading advanced preset: {preset_path}")

        with open(preset_path, 'r') as f:
            self.preset = json.load(f)

        self.preset_path = preset_path

        # Check if it's an advanced preset
        analysis_version = self.preset.get('metadata', {}).get('analysis_version', '1.0')

        if analysis_version == '2.0_advanced':
            self._build_advanced_effects_chain()
        else:
            # Fall back to basic preset
            self._build_basic_effects_chain()

        print(f"Preset loaded: {Path(preset_path).stem}")

    def _build_advanced_effects_chain(self):
        """Build advanced multiband effects chain"""
        print("Building advanced multiband effects chain...")

        # Get multiband analysis
        multiband_analysis = self.preset.get('multiband_analysis', {})

        if not multiband_analysis:
            print("No multiband analysis found, using basic chain")
            self._build_basic_effects_chain()
            return

        num_bands = multiband_analysis.get('num_bands', 4)
        bands = multiband_analysis.get('bands', [])
        crossover_freqs = multiband_analysis.get('crossover_freqs', [])

        # Create crossover
        self.crossover = create_default_crossover(
            num_bands=num_bands,
            sample_rate=self.audio_config.sample_rate
        )

        # Create effects chain for each band
        self.effects_chains = {}

        global_compression = self.preset.get('global_analysis', {}).get('compression', {})

        for band in bands:
            band_idx = band['band_index']
            relative_gain = band.get('relative_gain_db', 0.0)

            # Build effects for this band
            effects = []

            # EQ/Gain for this band
            if abs(relative_gain) > 0.5:
                effects.append(Gain(gain_db=relative_gain))

            # Compression (adapt to band characteristics)
            threshold = global_compression.get('threshold_db', -20)
            ratio = global_compression.get('ratio', 4.0)

            # Adjust compression per band
            if band_idx == 0:  # Bass band
                # Heavier compression on bass
                ratio = min(ratio * 1.2, 10.0)
            elif band_idx == num_bands - 1:  # Treble band
                # Lighter compression on highs
                ratio = max(ratio * 0.8, 1.5)

            effects.append(
                Compressor(
                    threshold_db=threshold,
                    ratio=ratio,
                    attack_ms=global_compression.get('attack_ms', 10),
                    release_ms=global_compression.get('release_ms', 100)
                )
            )

            self.effects_chains[f'band_{band_idx}'] = Pedalboard(effects)

        # Initialize transient detector
        self.transient_detector = TransientDetector(
            sample_rate=self.audio_config.sample_rate
        )

        # Initialize sound classifier
        self.sound_classifier = OnsetBasedClassifier(
            sample_rate=self.audio_config.sample_rate
        )

        print(f"Built {num_bands}-band effects chain")

    def _build_basic_effects_chain(self):
        """Build basic effects chain (fallback)"""
        print("Building basic effects chain...")

        effects = []

        # EQ from global analysis
        eq_curve = self.preset.get('global_analysis', {}).get('eq_curve', [])

        for band in eq_curve:
            freq = band['frequency']
            gain = band['gain_db']

            if abs(gain) < 0.5:
                continue

            if freq <= config.EQ_FREQUENCIES[0]:
                effects.append(LowShelfFilter(cutoff_frequency_hz=freq, gain_db=gain))
            elif freq >= config.EQ_FREQUENCIES[-1]:
                effects.append(HighShelfFilter(cutoff_frequency_hz=freq, gain_db=gain))
            else:
                effects.append(PeakFilter(cutoff_frequency_hz=freq, gain_db=gain, q=band.get('q_factor', 1.0)))

        # Compression
        comp_params = self.preset.get('global_analysis', {}).get('compression', {})

        effects.append(
            Compressor(
                threshold_db=comp_params.get('threshold_db', -20),
                ratio=comp_params.get('ratio', 4.0),
                attack_ms=comp_params.get('attack_ms', 10),
                release_ms=comp_params.get('release_ms', 100)
            )
        )

        self.effects_chains['main'] = Pedalboard(effects)
        self.multiband_enabled = False

    def process_buffer_advanced(self, audio_in: np.ndarray) -> np.ndarray:
        """
        Process audio buffer with advanced multiband + transient preservation

        Args:
            audio_in: Input audio buffer

        Returns:
            Processed audio buffer
        """
        # Apply input gain
        if self.input_gain_db != 0.0:
            audio_in = audio_in * (10 ** (self.input_gain_db / 20))

        if not self.multiband_enabled or self.crossover is None:
            # Basic processing
            audio_out = self.effects_chains['main'].process(
                audio_in,
                sample_rate=self.audio_config.sample_rate
            )
        else:
            # Advanced multiband processing
            audio_out = self._process_multiband_with_transients(audio_in)

        # Apply wet/dry mix
        if self.wet_dry_mix < 1.0:
            audio_out = (1 - self.wet_dry_mix) * audio_in + self.wet_dry_mix * audio_out

        # Apply output gain
        if self.output_gain_db != 0.0:
            audio_out = audio_out * (10 ** (self.output_gain_db / 20))

        # Safety limiter
        audio_out = self.safety_limiter.process(
            audio_out,
            sample_rate=self.audio_config.sample_rate
        )

        return audio_out

    def _process_multiband_with_transients(self, audio_in: np.ndarray) -> np.ndarray:
        """Process audio with multiband splitting and transient preservation"""

        # Detect transient vs sustained portions
        transient_mask, sustained_mask = self.transient_detector.detect(audio_in)

        # Split into frequency bands
        bands = self.crossover.split_bands(audio_in)

        # Process each band
        processed_bands = []

        for i, band_audio in enumerate(bands):
            # Get effects chain for this band
            chain_key = f'band_{i}'
            if chain_key in self.effects_chains:
                # Process sustained portion only
                band_processed = self.effects_chains[chain_key].process(
                    band_audio,
                    sample_rate=self.audio_config.sample_rate
                )

                # Blend transient and sustained
                # Preserve transients, apply effects to sustained
                blend_factor = 1.0 - (transient_mask * self.transient_preservation)
                band_out = band_audio * (transient_mask * self.transient_preservation) + \
                          band_processed * blend_factor

                # Apply per-band mix
                if i < len(self.band_mix_levels):
                    band_out = band_out * self.band_mix_levels[i]

                processed_bands.append(band_out)
            else:
                processed_bands.append(band_audio)

        # Recombine bands
        audio_out = self.crossover.combine_bands(processed_bands)

        return audio_out

    def start_processing(self, input_device=None, output_device=None, monitor=True):
        """Start real-time processing"""
        if self.effects_chains is None or len(self.effects_chains) == 0:
            raise ValueError("No effects chain loaded. Load a preset first.")

        if self.is_processing:
            print("Already processing!")
            return

        print("\nStarting advanced real-time processing...")
        print(f"Sample Rate: {self.audio_config.sample_rate} Hz")
        print(f"Buffer Size: {self.audio_config.buffer_size} samples")
        print(f"Latency: ~{self.audio_config.buffer_size / self.audio_config.sample_rate * 1000:.1f} ms")
        print(f"Multiband: {'Enabled' if self.multiband_enabled else 'Disabled'}")
        print(f"Transient Preservation: {self.transient_preservation * 100:.0f}%")

        # Set devices
        if input_device is not None:
            sd.default.device[0] = input_device
        if output_device is not None:
            sd.default.device[1] = output_device

        self.is_processing = True

        try:
            if monitor:
                self._start_callback_mode()
            else:
                self._start_processing_only()

        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop_processing()
        except Exception as e:
            print(f"\nError: {e}")
            self.stop_processing()
            raise

    def _start_callback_mode(self):
        """Start processing with live monitoring"""

        def audio_callback(indata, outdata, frames, time_info, status):
            """Audio callback for real-time processing"""
            if status:
                print(f"Status: {status}")

            try:
                # Convert to float32
                audio_in = indata[:, 0].astype(np.float32)

                # Process with advanced algorithm
                audio_out = self.process_buffer_advanced(audio_in)

                # Ensure correct shape
                if audio_out.ndim == 1:
                    audio_out = audio_out.reshape(-1, 1)

                # Copy to output buffer
                outdata[:] = audio_out

                # Record if enabled
                if self.is_recording:
                    with self.recording_lock:
                        self.recorded_audio.append(audio_out.copy())

            except Exception as e:
                print(f"Callback error: {e}")
                outdata.fill(0)

        # Create and start stream
        self.stream = sd.Stream(
            samplerate=self.audio_config.sample_rate,
            blocksize=self.audio_config.buffer_size,
            channels=1,
            callback=audio_callback
        )

        with self.stream:
            print("\n" + "=" * 60)
            print("ADVANCED LIVE PROCESSING ACTIVE")
            print("=" * 60)
            print("\nControls:")
            print("  Wet/Dry Mix: {:.0f}%".format(self.wet_dry_mix * 100))
            print("  Transient Preservation: {:.0f}%".format(self.transient_preservation * 100))
            print("\nPress Ctrl+C to stop\n")

            # Keep running
            while self.is_processing:
                time.sleep(0.1)

    def _start_processing_only(self):
        """Start processing without monitoring"""
        # Similar to live_processor.py but with advanced processing
        def audio_callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"Status: {status}")

            try:
                audio_in = indata[:, 0].astype(np.float32)
                audio_processed = self.process_buffer_advanced(audio_in)

                if self.is_recording:
                    with self.recording_lock:
                        if audio_processed.ndim == 1:
                            audio_processed = audio_processed.reshape(-1, 1)
                        self.recorded_audio.append(audio_processed.copy())

                outdata.fill(0)

            except Exception as e:
                print(f"Callback error: {e}")

        self.stream = sd.InputStream(
            samplerate=self.audio_config.sample_rate,
            blocksize=self.audio_config.buffer_size,
            channels=1,
            callback=audio_callback
        )

        with self.stream:
            print("\n" + "=" * 60)
            print("PROCESSING ACTIVE (No Monitor)")
            print("=" * 60)
            print("\nPress Ctrl+C to stop\n")

            while self.is_processing:
                time.sleep(0.1)

    def stop_processing(self):
        """Stop real-time processing"""
        self.is_processing = False

        if self.stream is not None:
            self.stream.close()
            self.stream = None

        print("\nProcessing stopped")

    def set_wet_dry_mix(self, mix: float):
        """Set wet/dry mix (0.0 = dry, 1.0 = wet)"""
        self.wet_dry_mix = np.clip(mix, 0.0, 1.0)
        print(f"Wet/Dry mix: {self.wet_dry_mix * 100:.0f}%")

    def set_transient_preservation(self, amount: float):
        """Set transient preservation amount (0.0 = none, 1.0 = full)"""
        self.transient_preservation = np.clip(amount, 0.0, 1.0)
        print(f"Transient preservation: {self.transient_preservation * 100:.0f}%")

    def set_band_mix(self, band_idx: int, level: float):
        """Set mix level for specific band"""
        if band_idx < len(self.band_mix_levels):
            self.band_mix_levels[band_idx] = np.clip(level, 0.0, 2.0)
            print(f"Band {band_idx} mix: {self.band_mix_levels[band_idx] * 100:.0f}%")

    def set_input_gain(self, gain_db: float):
        """Set input gain in dB"""
        self.input_gain_db = np.clip(gain_db, -24.0, 24.0)
        print(f"Input gain: {self.input_gain_db:+.1f} dB")

    def set_output_gain(self, gain_db: float):
        """Set output gain in dB"""
        self.output_gain_db = np.clip(gain_db, -24.0, 24.0)
        print(f"Output gain: {self.output_gain_db:+.1f} dB")

    # Recording methods (same as live_processor.py)

    def start_recording(self, filename=None):
        """Start recording processed audio"""
        if not self.is_processing:
            print("Start processing first!")
            return

        if self.is_recording:
            print("Already recording!")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            preset_name = Path(self.preset_path).stem if self.preset_path else "unknown"
            filename = f"beatbox_advanced_{preset_name}_{timestamp}.wav"

        self.recording_filename = filename

        with self.recording_lock:
            self.recorded_audio = []

        self.is_recording = True
        print(f"\n🔴 Recording started: {filename}")

    def stop_recording(self):
        """Stop recording and save file"""
        if not self.is_recording:
            print("Not recording!")
            return

        self.is_recording = False

        with self.recording_lock:
            if len(self.recorded_audio) == 0:
                print("No audio recorded!")
                return

            audio_data = np.concatenate(self.recorded_audio, axis=0)

        output_path = config.get_recording_path(self.recording_filename)

        sf.write(
            output_path,
            audio_data,
            self.audio_config.sample_rate
        )

        duration = len(audio_data) / self.audio_config.sample_rate
        print(f"\n⏹️  Recording stopped")
        print(f"   Saved: {output_path}")
        print(f"   Duration: {duration:.2f} seconds")

        return output_path


def main():
    """Main entry point"""
    import sys

    processor = AdvancedLiveProcessor()

    if len(sys.argv) < 2:
        print("Usage: python advanced_processor.py <preset_file>")
        print("\nExample: python advanced_processor.py presets/my_preset_advanced.json")
        sys.exit(1)

    preset_file = sys.argv[1]
    processor.load_preset(preset_file)

    print("\nStarting advanced live processing...")
    processor.start_processing(monitor=True)


if __name__ == "__main__":
    main()
