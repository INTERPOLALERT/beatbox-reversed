"""
Configuration file for Beatbox Audio Style Transfer Application
"""
import os
import json
from pathlib import Path

# Application Directories
APP_DIR = Path(__file__).parent
PRESETS_DIR = APP_DIR / "presets"
RECORDINGS_DIR = APP_DIR / "recordings"

# Create directories if they don't exist
PRESETS_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)

# Audio Settings
SAMPLE_RATE = 44100  # Hz
BUFFER_SIZE = 512  # samples (adjustable for latency vs stability trade-off)
CHANNELS = 1  # Mono for beatboxing

# Analysis Settings
FFT_SIZE = 8192  # Larger FFT for better frequency resolution
HOP_LENGTH = 2048
WINDOW_TYPE = 'hann'

# EQ Settings
NUM_EQ_BANDS = 10
EQ_FREQUENCIES = [30, 60, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]  # Hz

# Compression Settings
RMS_WINDOW_MS = 50  # milliseconds for RMS calculation
ATTACK_TIME_MS = 10
RELEASE_TIME_MS = 100

# Multiband Processing Settings (Advanced)
MULTIBAND_ENABLED = False  # Start simple, enable later
NUM_FREQUENCY_BANDS = 4
CROSSOVER_FREQUENCIES = [200, 1000, 4000]  # Hz

# Sound Classification Settings (Phase 2)
CLASSIFICATION_ENABLED = False  # Add in Phase 2
SOUND_TYPES = ['kick', 'snare', 'hihat', 'bass']

# Recording Settings
RECORDING_FORMAT = 'wav'
RECORDING_SAMPLE_WIDTH = 2  # 16-bit

# Latency Settings
TARGET_LATENCY_MS = 10  # Target latency goal

# Config File
CONFIG_FILE = APP_DIR / "audio_config.json"


class AudioConfig:
    """Manages audio device configuration"""

    def __init__(self):
        self.input_device = None
        self.output_device = None
        self.buffer_size = BUFFER_SIZE
        self.sample_rate = SAMPLE_RATE
        self.load_config()

    def load_config(self):
        """Load configuration from file"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.input_device = config.get('input_device')
                    self.output_device = config.get('output_device')
                    self.buffer_size = config.get('buffer_size', BUFFER_SIZE)
                    self.sample_rate = config.get('sample_rate', SAMPLE_RATE)
            except Exception as e:
                print(f"Error loading config: {e}")

    def save_config(self):
        """Save configuration to file"""
        try:
            config = {
                'input_device': self.input_device,
                'output_device': self.output_device,
                'buffer_size': self.buffer_size,
                'sample_rate': self.sample_rate
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, indent=2, fp=f)
        except Exception as e:
            print(f"Error saving config: {e}")


def get_preset_path(preset_name):
    """Get full path for a preset file"""
    return PRESETS_DIR / f"{preset_name}.json"


def get_recording_path(filename):
    """Get full path for a recording file"""
    if not filename.endswith('.wav'):
        filename += '.wav'
    return RECORDINGS_DIR / filename
