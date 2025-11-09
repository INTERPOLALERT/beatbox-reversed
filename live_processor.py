"""
Stage 2: Real-Time Audio Processing Module
Applies extracted preset parameters to live microphone input
"""
import numpy as np
import sounddevice as sd
import soundfile as sf
from pedalboard import Pedalboard, Compressor, PeakFilter, LowShelfFilter, HighShelfFilter
from pedalboard.io import AudioStream
import json
from datetime import datetime
from pathlib import Path
import config
import threading
import queue


class LiveProcessor:
    """Real-time audio processor with recording capability"""

    def __init__(self, preset_path=None):
        """
        Initialize live processor

        Args:
            preset_path: Path to preset JSON file
        """
        self.preset_path = preset_path
        self.preset = None
        self.effects_chain = None

        # Recording state
        self.is_recording = False
        self.recorded_audio = []
        self.recording_lock = threading.Lock()

        # Audio stream
        self.stream = None
        self.is_processing = False

        # Load config
        self.audio_config = config.AudioConfig()

    def load_preset(self, preset_path):
        """
        Load preset from JSON file

        Args:
            preset_path: Path to preset file
        """
        print(f"Loading preset: {preset_path}")

        with open(preset_path, 'r') as f:
            self.preset = json.load(f)

        self.preset_path = preset_path
        self._build_effects_chain()

        print(f"Preset loaded: {Path(preset_path).stem}")

    def _build_effects_chain(self):
        """Build Pedalboard effects chain from preset parameters"""
        if self.preset is None:
            raise ValueError("No preset loaded")

        effects = []

        # Add EQ filters
        eq_curve = self.preset.get('eq_curve', [])

        for band in eq_curve:
            freq = band['frequency']
            gain = band['gain_db']
            q = band.get('q_factor', 1.0)

            # Skip bands with negligible gain
            if abs(gain) < 0.5:
                continue

            # Use appropriate filter type based on frequency
            if freq <= config.EQ_FREQUENCIES[0]:
                # Low shelf for lowest frequency
                effects.append(
                    LowShelfFilter(cutoff_frequency_hz=freq, gain_db=gain)
                )
            elif freq >= config.EQ_FREQUENCIES[-1]:
                # High shelf for highest frequency
                effects.append(
                    HighShelfFilter(cutoff_frequency_hz=freq, gain_db=gain)
                )
            else:
                # Peak filter for mid frequencies
                effects.append(
                    PeakFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q)
                )

        # Add compressor
        comp_params = self.preset.get('compression', {})

        effects.append(
            Compressor(
                threshold_db=comp_params.get('threshold_db', -20),
                ratio=comp_params.get('ratio', 4.0),
                attack_ms=comp_params.get('attack_ms', 10),
                release_ms=comp_params.get('release_ms', 100)
            )
        )

        # Create effects chain
        self.effects_chain = Pedalboard(effects)

        print(f"Built effects chain with {len(effects)} processors")
        self._print_effects_summary()

    def _print_effects_summary(self):
        """Print summary of effects chain"""
        print("\nEffects Chain:")
        print("-" * 40)

        eq_count = 0
        comp_count = 0

        for effect in self.effects_chain:
            effect_type = type(effect).__name__

            if 'Filter' in effect_type:
                eq_count += 1
            elif 'Compressor' in effect_type:
                comp_count += 1

        print(f"  EQ Bands: {eq_count}")
        print(f"  Compressor: {comp_count}")
        print("-" * 40 + "\n")

    def list_audio_devices(self):
        """List available audio devices"""
        print("\nAvailable Audio Devices:")
        print("=" * 60)
        devices = sd.query_devices()

        for idx, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append("INPUT")
            if device['max_output_channels'] > 0:
                device_type.append("OUTPUT")

            print(f"{idx}: {device['name']}")
            print(f"   Type: {', '.join(device_type)}")
            print(f"   Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
            print(f"   Sample Rate: {device['default_samplerate']} Hz")
            print()

        print("=" * 60 + "\n")

    def start_processing(self, input_device=None, output_device=None, monitor=True):
        """
        Start real-time audio processing

        Args:
            input_device: Input device index or name (None for default)
            output_device: Output device index or name (None for default)
            monitor: Whether to monitor processed audio through speakers/headphones
        """
        if self.effects_chain is None:
            raise ValueError("No effects chain loaded. Load a preset first.")

        if self.is_processing:
            print("Already processing!")
            return

        print("\nStarting real-time processing...")
        print(f"Sample Rate: {self.audio_config.sample_rate} Hz")
        print(f"Buffer Size: {self.audio_config.buffer_size} samples")
        print(f"Latency: ~{self.audio_config.buffer_size / self.audio_config.sample_rate * 1000:.1f} ms")

        # Set devices
        if input_device is not None:
            sd.default.device[0] = input_device
        if output_device is not None:
            sd.default.device[1] = output_device

        self.is_processing = True

        try:
            if monitor:
                # Use callback mode for monitoring
                self._start_callback_mode()
            else:
                # Processing only (no monitoring)
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

        def audio_callback(indata, outdata, frames, time, status):
            """Audio callback for real-time processing"""
            if status:
                print(f"Status: {status}")

            try:
                # Convert to float32
                audio_in = indata[:, 0].astype(np.float32)

                # Apply effects chain
                audio_out = self.effects_chain.process(
                    audio_in,
                    sample_rate=self.audio_config.sample_rate
                )

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
            print("LIVE PROCESSING ACTIVE")
            print("=" * 60)
            print("\nPress Ctrl+C to stop")
            print("Commands:")
            print("  - Start recording: call start_recording()")
            print("  - Stop recording: call stop_recording()")
            print()

            # Keep running
            import time
            while self.is_processing:
                time.sleep(0.1)

    def _start_processing_only(self):
        """Start processing without monitoring (processing only mode)"""
        # This mode is for when you only want recording without live monitoring
        # to reduce latency even further

        def audio_callback(indata, outdata, frames, time, status):
            """Audio callback for processing only"""
            if status:
                print(f"Status: {status}")

            try:
                # Convert to float32
                audio_in = indata[:, 0].astype(np.float32)

                # Apply effects chain
                audio_processed = self.effects_chain.process(
                    audio_in,
                    sample_rate=self.audio_config.sample_rate
                )

                # Record if enabled
                if self.is_recording:
                    with self.recording_lock:
                        if audio_processed.ndim == 1:
                            audio_processed = audio_processed.reshape(-1, 1)
                        self.recorded_audio.append(audio_processed.copy())

                # No output (monitoring disabled)
                outdata.fill(0)

            except Exception as e:
                print(f"Callback error: {e}")

        # Create stream with input only
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

            import time
            while self.is_processing:
                time.sleep(0.1)

    def stop_processing(self):
        """Stop real-time processing"""
        self.is_processing = False

        if self.stream is not None:
            self.stream.close()
            self.stream = None

        print("\nProcessing stopped")

    def start_recording(self, filename=None):
        """
        Start recording processed audio

        Args:
            filename: Output filename (auto-generated if None)
        """
        if not self.is_processing:
            print("Start processing first!")
            return

        if self.is_recording:
            print("Already recording!")
            return

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            preset_name = Path(self.preset_path).stem if self.preset_path else "unknown"
            filename = f"beatbox_{preset_name}_{timestamp}.wav"

        self.recording_filename = filename

        # Clear recorded audio buffer
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

        # Get recorded audio
        with self.recording_lock:
            if len(self.recorded_audio) == 0:
                print("No audio recorded!")
                return

            # Concatenate all recorded chunks
            audio_data = np.concatenate(self.recorded_audio, axis=0)

        # Save to file
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
    """Main entry point for live processor"""
    import sys

    processor = LiveProcessor()

    # List devices
    processor.list_audio_devices()

    # Check if preset provided
    if len(sys.argv) < 2:
        print("Usage: python live_processor.py <preset_file>")
        print("\nExample: python live_processor.py presets/my_preset.json")
        sys.exit(1)

    # Load preset
    preset_file = sys.argv[1]
    processor.load_preset(preset_file)

    # Start processing
    print("\nStarting live processing with preset...")
    processor.start_processing(monitor=True)


if __name__ == "__main__":
    main()
