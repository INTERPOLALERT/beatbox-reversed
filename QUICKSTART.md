# Quick Start Guide - Universal Adaptive Audio Processing

## Get Started in 3 Steps

### 1. Enable Diagnostics & Loudness Matching

Edit `config.py`:
```python
DIAGNOSTIC_MODE_ENABLED = True  # Enable per-buffer logging
LOUDNESS_MATCHING_ENABLED = True  # Enable adaptive gain
LOUDNESS_MATCH_MODE = 'rms'  # Choose matching mode
```

### 2. Create and Load a Preset

```bash
# Analyze any reference audio
python advanced_analyzer.py path/to/reference_track.wav

# This creates: presets/reference_track_advanced.json
```

### 3. Process Live Audio

```python
from ultimate_processor import UltimateProcessor

# Create processor
processor = UltimateProcessor(sample_rate=44100, enable_diagnostics=True)

# Load preset (automatically extracts reference loudness)
processor.load_preset('presets/reference_track_advanced.json')

# Configure
processor.set_loudness_matching(enabled=True, match_mode='rms')
processor.set_transient_preservation(0.6)
processor.set_saturation(0.2, 'tube')
processor.set_reverb(0.15)

# Process buffers
for mic_buffer in audio_stream:
    output = processor.process_buffer(mic_buffer)
    # Output is automatically matched to reference loudness!

# Save diagnostic logs when done
processor.save_diagnostics()
```

## Matching Modes

- `rms`: Match RMS levels (default, most reliable)
- `lufs`: Match LUFS (broadcast standard -14.0)
- `peak_normalized`: Match peak levels (preserve headroom)
- `crest_matched`: Match crest factor (preserve dynamics)

## View Diagnostics

```bash
# View CSV logs
cat logs/diagnostic_log_*.csv

# Or load in Python
import pandas as pd
df = pd.read_csv('logs/diagnostic_log_*.csv')
print(df.head())
```

## Run Tests

```bash
python test_new_features.py
```

## Demo

```bash
# Test loudness matching
python loudness_matcher.py

# Test diagnostic logger
python diagnostic_logger.py

# Test ultimate processor
python ultimate_processor.py
```

That's it! Your live mic will now faithfully reproduce any reference audio. 🎤✨
