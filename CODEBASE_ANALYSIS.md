# Beatbox Reversed - Comprehensive Codebase Analysis

## 1. CURRENT AUDIO PROCESSING PIPELINE & ARCHITECTURE

### Two-Stage Processing Model

**Stage 1: Offline Analysis**
- `audio_analyzer.py` (378 lines): Basic analysis of reference audio
  - Frequency spectrum analysis (STFT with 8192 FFT, Hann window)
  - Dynamic range analysis using RMS envelopes
  - Transient/onset detection
  - Harmonic-to-percussive separation
  - EQ curve extraction at 10 standard frequencies (30Hz-16kHz)
  - Compression parameter estimation from crest factor
  - Normalization fix: applies 0.20 scaling factor + ±6dB clamping

- `advanced_analyzer.py` (420 lines): Professional-grade analysis
  - Multiband analysis (4 or 8 bands with Linkwitz-Riley crossovers)
  - Per-band energy and gain analysis
  - Formant extraction using LPC (Linear Prediction Coding)
  - Per-sound-type analysis (kick/snare/hihat/bass/vocal)
  - Same normalization safeguards as basic analyzer

**Stage 2: Real-Time Processing**
- `live_processor.py` (408 lines): Basic real-time processor
  - Loads JSON presets from Stage 1
  - Builds Pedalboard effects chain dynamically
  - Real-time audio callback with ~11.6ms latency @ 512 buffer
  - Thread-safe recording capability
  - Device enumeration and selection

- `advanced_processor.py` (528 lines): Advanced real-time processor
  - Multiband processing with transient preservation
  - Per-band effects chains
  - Sound classification integration
  - Wet/dry mixing
  - Input/output gain controls
  - Per-band mix level controls
  - Safety limiter

- `ultimate_processor.py` (337 lines): Complete integrated processor
  - Combines ALL advanced features
  - Adaptive per-sound-type processing
  - Micro-transient enhancement
  - Spatial effects (reverb, width, panning)
  - Harmonic processing
  - Supports both mono and stereo processing

---

## 2. EQ, COMPRESSION, TRANSIENT PROCESSING IMPLEMENTATIONS

### EQ (Equalization)

**Basic EQ System** (audio_analyzer.py, live_processor.py)
- 10-band parametric EQ at fixed frequencies: 30, 60, 125, 250, 500, 1000, 2000, 4000, 8000, 16000 Hz
- Gain range: ±6dB (safety clamped after normalization)
- Q factor: 1.0 default (peaking filters for mid-range, shelving for extremes)
- Filter types:
  - LowShelfFilter for frequencies ≤ 30Hz
  - HighShelfFilter for frequencies ≥ 16000Hz
  - PeakFilter for mid-range frequencies
- Implementation: Pedalboard library (JUCE framework)
- Extraction method: FFT magnitude averaging with window smoothing

**Multiband EQ** (multiband_processor.py, advanced_processor.py)
- 4-band or 8-band Linkwitz-Riley crossovers (24dB/octave, 4th order)
- Frequency splits:
  - 4-band: 200Hz, 1000Hz, 4000Hz
  - 8-band: 80Hz, 200Hz, 500Hz, 1000Hz, 2000Hz, 4000Hz, 8000Hz
- Per-band gains normalized to middle band reference
- Same ±6dB clamping for stability
- Flat magnitude response when bands recombined

**Per-Sound-Type EQ** (adaptive_sound_processor.py)
- Sound profiles for: kick, snare, hihat, bass, vocal, other
- **Kick**: Boosts 60-100Hz (3-4dB), cuts 300-1000Hz (1-2dB)
- **Snare**: Boosts 200-8000Hz (1-3dB), cuts 400-600Hz (1-1.5dB)
- **Hi-hat**: Boosts 6-12kHz (1-3dB), cuts 200-1000Hz (2-4dB)
- **Bass**: Boosts 80-200Hz (1-3dB), cuts 400-800Hz (0.5-1dB)
- **Vocal**: Boosts 1-5kHz (2-3dB), cuts 200Hz and 10kHz (1-1.5dB)
- Implementation: Peaking IIR filters using scipy signal design

### Compression

**Compression Parameter Estimation** (audio_analyzer.py)
- Input: RMS envelope analysis
- Crest factor calculation: Peak_dB - RMS_avg_dB
- Ratio mapping (CAPPED at 4:1 for live stability):
  - Crest > 15dB: 1.5:1 (minimal)
  - Crest > 12dB: 2.0:1 (light)
  - Crest > 9dB: 2.5:1 (moderate)
  - Crest > 7dB: 3.0:1 (noticeable)
  - Crest ≤ 7dB: 4.0:1 (heavy - CAPPED)
- Threshold: Median RMS + 3dB
- Attack time: 10ms (default)
- Release time: 100ms (default)
- Implementation: Pedalboard Compressor

**Per-Band Compression** (advanced_processor.py)
- Base ratio from global analysis
- Band-specific adjustments:
  - Bass band: Ratio × 1.2 (more compression for control)
  - Treble band: Ratio × 0.8 (less compression to preserve brightness)

### Transient Processing

**TransientDetector** (multiband_processor.py)
- Dual-envelope system separating transients from sustained content
- Fast attack envelope: 2ms attack, 100ms release
- Slow attack envelope: 50ms attack, 100ms release
- Output: transient_mask and sustained_mask (0.0 to 1.0)
- Used for crossfading between processed and unprocessed audio

**MicroTransientProcessor** (adaptive_sound_processor.py)
- Separate fast and slow envelope followers with different coefficients
- Fast attack: 1ms (catches transients)
- Slow attack: 20ms (tracks sustain)
- Extract transient component = difference between envelopes
- Enhance transients: gain boost (1.0 + amount) while preserving sustain
- Blending: transient_mask helps crossfade between original and processed

**MultibandEnvelopeFollower** (multiband_processor.py)
- Per-band envelope followers with band-dependent attack/release times
- High frequencies: 1ms attack, 50ms release (fast tracking)
- Bass frequencies: 20ms attack, 200ms release (smoother tracking)
- Output: Per-band envelope signals for monitoring

**Transient Preservation in Processing** (advanced_processor.py)
- Detects transients in each buffer
- Applies effects only to sustained portions
- Blends with original transients: `output = dry_transient + processed_sustained`
- Preserves articulation/attack characteristics

---

## 3. PRESET SYSTEM & PARAMETER APPLICATION

### Preset Structure (JSON Format)

**Basic Preset** (audio_analyzer.py output):
```json
{
  "metadata": {
    "source_file": "path/to/audio.wav",
    "duration_seconds": 5.2,
    "sample_rate": 44100
  },
  "eq_curve": [
    {"frequency": 30, "gain_db": 0.5, "q_factor": 1.0},
    ...
  ],
  "compression": {
    "threshold_db": -15.0,
    "ratio": 2.5,
    "attack_ms": 10,
    "release_ms": 100,
    "knee_db": 3.0,
    "makeup_gain_db": 1.2,
    "crest_factor_db": 8.5
  },
  "dynamic_range": {...},
  "spectral_profile": {...},
  "harmonic_content": {...},
  "transient_profile": {...}
}
```

**Advanced Preset** (advanced_analyzer.py output, analysis_version: "2.0_advanced"):
```json
{
  "metadata": {
    "analysis_version": "2.0_advanced"
  },
  "global_analysis": {
    "eq_curve": [...],
    "compression": {...},
    ...
  },
  "multiband_analysis": {
    "num_bands": 4,
    "crossover_freqs": [200, 1000, 4000],
    "bands": [
      {
        "band_index": 0,
        "freq_range": [20, 200],
        "center_freq": 63.2,
        "rms_db": -35.2,
        "peak_db": -22.1,
        "relative_gain_db": 0.5,
        "energy_ratio": 0.15
      },
      ...
    ]
  },
  "formants": {
    "frequencies": [700, 1200, 2600, ...],
    "num_formants": 4
  },
  "per_sound_analysis": {
    "kick": {
      "spectral_centroid": 85.3,
      "rms_db": -12.5,
      "peak_db": -2.1,
      "crest_factor_db": 10.4,
      "estimated_compression_ratio": 2.5,
      "num_occurrences": 8
    },
    "snare": {...},
    "hihat": {...},
    "bass": {...}
  }
}
```

### Preset Loading & Application

**In live_processor.py**:
1. Load JSON preset
2. Extract eq_curve and compression parameters
3. Build Pedalboard effects chain dynamically
4. Apply chain to live audio buffer in callback

**In advanced_processor.py**:
1. Check analysis_version to determine preset type
2. If "2.0_advanced":
   - Create multiband crossover
   - Build per-band effects chains
   - Initialize transient detector
   - Initialize sound classifier
3. If basic preset:
   - Build single effects chain
   - Disable multiband processing

**In ultimate_processor.py**:
1. Load preset
2. Configure all sub-processors (adaptive, transient, spatial, harmonic)
3. Apply per-sound-type parameters where available
4. Set global gain and mixing controls

---

## 4. ADAPTIVE PROCESSING & SOUND CLASSIFICATION

### Sound Classification System

**BeatboxSoundClassifier** (sound_classifier.py)
- Machine learning based: RandomForestClassifier (100 trees, max_depth=10)
- Training features (43 total):
  1. **MFCCs**: 13 coefficients × 2 (mean + std) = 26 features
  2. **Spectral Centroid**: mean + std = 2 features
  3. **Spectral Contrast**: 7 bands = 7 features
  4. **Zero Crossing Rate**: mean + std = 2 features
  5. **RMS Energy**: mean + std = 2 features
  6. **Spectral Rolloff**: mean + std = 2 features
  7. **Low-frequency energy ratio** (< 200Hz) = 1 feature
  8. **High-frequency energy ratio** (> 4000Hz) = 1 feature

- Fallback classification (rule-based when model not trained):
  - Kick: low_freq_ratio > 0.5 && spec_centroid < 500Hz
  - Hi-hat: high_freq_ratio > 0.3 && zcr > 0.15
  - Bass: low_freq_ratio > 0.3 && spec_centroid < 1000Hz
  - Snare: 800 < spec_centroid < 3000Hz
  - Other: default

**OnsetBasedClassifier** (sound_classifier.py)
- Wraps BeatboxSoundClassifier
- Detects onsets using librosa.onset.onset_detect()
- Extracts 100ms segments around each onset
- Classifies each segment independently
- Returns: (onset_time, sound_type, confidence)

### Adaptive Processing

**AdaptiveSoundProcessor** (adaptive_sound_processor.py)
- Detects sound type from energy threshold (0.01)
- Applies per-sound-type EQ with 0.7 wet/dry mix
- Per-sound profiles define:
  - Boost frequencies and gains
  - Cut frequencies and gains
  - Compression ratio
  - Compression threshold
  - Attack/release times
  - Saturation amount

- Processing chain:
  1. Detect sound type from onset energy
  2. Get adaptive EQ parameters
  3. Apply parametric EQ boosts
  4. Apply parametric EQ cuts
  5. Mix wet/dry based on confidence

**Transition Smoothing** (not fully implemented):
- Smooth envelope followers for transitions between sound types
- 10ms smooth transition period configured but not active

---

## 5. GAIN & LOUDNESS HANDLING

### Gain Controls

**Input Gain** (advanced_processor.py, ultimate_processor.py)
- Range: -24dB to +24dB
- Applied as first processing stage: `audio * 10^(gain_db/20)`
- Used to optimize level for effects processing

**Output Gain** (advanced_processor.py, ultimate_processor.py)
- Range: -24dB to +24dB
- Applied before safety limiting
- Compensates for effects processing gain changes

**Per-Band Mix Levels** (advanced_processor.py)
- 4 bands with independent level controls (0.0 to 2.0)
- Defaults: all 1.0
- Allows EQ-like control through band balancing

### Loudness/RMS Analysis

**RMS Window Analysis** (audio_analyzer.py)
- Window size: 50ms
- Hop size: 25ms
- Calculated using librosa.feature.rms()
- Statistics: mean, std, dynamic range (max - min)

**Peak Level Calculation**:
```
peak_db = 20 * log10(max(abs(audio)) + 1e-10)
```

**Crest Factor** (used for compression ratio estimation):
```
crest_factor_db = peak_db - rms_avg_db
```

### Safety Limiting

**Hard Limiter** (advanced_processor.py, ultimate_processor.py)
- Pedalboard Limiter: threshold_db=-1.0, release_ms=50
- Prevents clipping at output
- Applied as final processing stage

**Soft Clipping in Harmonic Processor** (harmonic_processor.py):
```
if max(abs(output)) > 0.99:
    output = output / max * 0.99
```

---

## 6. SPATIAL & STEREO PROCESSING CAPABILITIES

### Stereo Width Control

**StereoWidthProcessor** (spatial_effects.py)
- Mid-side processing
- Width parameter: 0 (mono) to 2.0 (ultra-wide)
- Processing:
  ```
  mid = (L + R) / 2
  side = (L - R) / 2
  side_adjusted = side * width
  L_out = mid + side_adjusted
  R_out = mid - side_adjusted
  ```

**Mono-to-Stereo Conversion**:
- Left channel: original mono (slight high-pass implied)
- Right channel: delayed copy (0.2ms Haas effect delay)
- Width modulation: 0.3 default
- Creates pseudo-stereo from mono input

### Panning

**SimplePanner** (spatial_effects.py)
- Constant-power panning law
- Pan parameter: -1.0 (full left) to +1.0 (full right)
- Implementation:
  ```
  pan_radians = (pan + 1.0) * π / 4.0  # Map -1..1 to 0..π/2
  left_gain = cos(pan_radians)
  right_gain = sin(pan_radians)
  ```

### Reverb

**SimpleReverb** (spatial_effects.py) - Schroeder Design
- 4 parallel comb filters (prime number delays):
  - 29.7ms (1310 samples @ 44.1kHz)
  - 37.1ms (1640 samples)
  - 41.1ms (1814 samples)
  - 43.7ms (1927 samples)
- 2 series allpass filters:
  - 5ms (220 samples)
  - 1.7ms (75 samples)

- Parameters:
  - Room size: 0.0 to 1.0 (controls feedback)
  - Damping: 0.0 to 1.0 (controls tone)
  - Wet mix: 0.0 to 1.0
  - Default: room_size=0.5, damping=0.5, wet=0.3

### Complete Spatial Processing

**SpatialProcessor** (spatial_effects.py)
- Integrates width, panning, and reverb
- Processing chain:
  1. Apply panning (mono to stereo)
  2. Apply reverb independently to each channel
  3. Apply stereo width control
  4. Return stereo output

- Methods:
  - `process_mono()`: Mono input → stereo output
  - `process_stereo()`: Stereo input → processed stereo output

---

## 7. HARMONIC PROCESSING & SATURATION

### Saturation Types

**HarmonicSaturator** (harmonic_processor.py)

1. **Soft Clipping**:
   ```
   output = tanh(input)
   ```
   Smooth saturation without harsh distortion

2. **Hard Clipping**:
   ```
   output = clip(input, -0.8, 0.8)
   ```
   Simple threshold limiting

3. **Tube Saturation** (asymmetric):
   ```
   positive: tanh(x * 1.2)        # Soft upper curve
   negative: tanh(x * 1.5) * 0.9  # Harder lower curve + bias
   ```
   Mimics vacuum tube asymmetry

4. **Tape Saturation** (warmth):
   ```
   output = x / (1.0 + abs(x)^1.5)
   ```
   Gentler saturation curve, adds even harmonics

- Processing flow:
  1. Pre-gain (drive): 1.0 + amount * 4.0
  2. Apply saturation curve
  3. Compensate: divide by drive
  4. Wet/dry mix

### Harmonic Enhancement

**HarmonicEnhancer** (harmonic_processor.py)
- Generates 2nd and 3rd harmonics through waveshaping:
  ```
  harmonic_2 = sign(x) * x^2      # Even harmonic (warmth)
  harmonic_3 = x^3                 # Odd harmonic (edge)
  harmonics = harmonic_2*0.3 + harmonic_3*0.2
  ```
- High-pass filter harmonics (200Hz cutoff) to avoid muddiness
- Normalize to prevent clipping
- Mix with original: output = input + harmonics * enhancement_amount

### Exciter (Psychoacoustic Brightness)

**ExciterFilter** (harmonic_processor.py)
- Extract high frequencies (> 3kHz with high-pass)
- Apply harmonic generation: tanh(highs * 2.0)
- Band-pass filter result (4-16kHz)
- Mix back at 20% of exciter_amount
- Creates "air" and perceived brightness without added loudness

### Complete Timbre Shaping

**TimbreShaper** (harmonic_processor.py)
- Combines saturation, harmonics, and excitation
- Processing order:
  1. Saturation
  2. Harmonic enhancement
  3. Excitation (air/brightness)
  4. Final safety limiter (clip to ±0.99)

- Control parameters:
  - Saturation amount/type
  - Harmonic enhancement amount
  - Exciter amount
  - Overall warmth (combines saturation + harmonics)

- "Warmth" parameter:
  ```
  set_saturation(warmth * 0.5, 'tube')
  set_harmonics(warmth * 0.3)
  ```

---

## 8. BUFFER MANAGEMENT & REAL-TIME PROCESSING

### Buffer Configuration

**Default Settings** (config.py):
- Sample rate: 44100 Hz
- Buffer size: 512 samples
- Latency @ 512 samples: ~11.6ms
- Channels: 1 (mono for beatboxing)

**Real-Time Audio Callback** (live_processor.py, advanced_processor.py):
```python
def audio_callback(indata, outdata, frames, time_info, status):
    # 1. Convert input to float32
    audio_in = indata[:, 0].astype(np.float32)
    
    # 2. Process through effects chain
    audio_out = effects_chain.process(
        audio_in,
        sample_rate=sample_rate
    )
    
    # 3. Handle shape and recording
    if audio_out.ndim == 1:
        audio_out = audio_out.reshape(-1, 1)
    outdata[:] = audio_out
    
    # 4. Record if enabled
    if is_recording:
        recorded_audio.append(audio_out.copy())
```

### Filter State Management

**IIR Filter State** (multiband_processor.py):
- Uses scipy.signal.sosfilt_zi() for initial conditions
- States maintained per-band for streaming processing:
  ```python
  if band_idx not in filter_states:
      filter_states[band_idx] = signal.sosfilt_zi(sos)
  
  band, filter_states[band_idx] = signal.sosfilt(
      sos, audio, zi=filter_states[band_idx] * audio[0]
  )
  ```
- Enables proper phase response and transient handling

**Envelope Follower State** (multiband_processor.py):
- Maintains fast/slow envelope states across buffer boundaries
- Stores last sample value from previous buffer
- Enables smooth envelope tracking in real-time

### Recording Management

**Thread-Safe Recording** (live_processor.py, advanced_processor.py):
- Recording lock (threading.Lock) protects audio buffer
- Recorded audio stored as list of numpy arrays
- Processing thread appends chunks, main thread reads/writes file
- On stop: concatenate all chunks and write to WAV

**File I/O** (soundfile library):
```python
sf.write(output_path, audio_data, sample_rate)
```

### Real-Time Processing Modes

**Callback Mode** (with monitoring):
- sd.Stream with duplex (input + output)
- Live audio output for monitoring
- Good for seeing real-time results
- Slight latency overhead

**Processing-Only Mode** (no monitoring):
- sd.InputStream (input only)
- Output disabled (outdata.fill(0))
- Lower latency for recording
- Can record while processing in background

---

## 9. LOGGING & DIAGNOSTIC SYSTEMS

### Console Logging

All major operations print status messages:

**Audio Analysis** (audio_analyzer.py):
```
Loading audio: path/to/file.wav
Loaded 5.20s of audio at 44100Hz
Analyzing frequency spectrum...
Extracted EQ curve with 10 bands
Analyzing dynamics and compression...
Dynamic range: 24.5 dB
Estimated compression ratio: 2.50:1
Analyzing transients...
Detected 12 transients
Analyzing harmonic content...
Harmonic/Percussive ratio: 0.450
ANALYSIS COMPLETE
```

**Real-Time Processing** (live_processor.py):
```
Loading preset: preset_name
Preset loaded: preset_name
Built effects chain with 11 processors

Effects Chain:
  EQ Bands: 8
  Compressor: 1

LIVE PROCESSING ACTIVE
Sample Rate: 44100 Hz
Buffer Size: 512 samples
Latency: ~11.6 ms

🔴 Recording started: beatbox_preset_20241109_120530.wav
⏹️  Recording stopped
   Saved: /path/to/recordings/beatbox_preset.wav
   Duration: 5.23 seconds
```

**Advanced Processing** (advanced_processor.py):
```
Building advanced multiband effects chain...
Built 4-band effects chain
Sample Rate: 44100 Hz
Buffer Size: 512 samples
Latency: ~11.6 ms
Multiband: Enabled
Transient Preservation: 80%
ADVANCED LIVE PROCESSING ACTIVE
```

### Diagnostic Data Saved in Presets

**Compression Diagnostics**:
- `crest_factor_db`: Peak - RMS difference (indicates compression need)
- `makeup_gain_db`: Compensatory gain applied
- Threshold, ratio, attack/release times

**Per-Sound Analysis**:
- `num_occurrences`: How many instances detected
- `spectral_centroid`: Brightness indicator
- `crest_factor_db`: Dynamic character of that sound type

**Transient Profile**:
- `num_onsets`: Number of detected attacks
- `onset_strength_mean`/`std`: Attack character statistics
- `onset_rate_per_second`: Rhythmic density

**Spectral Profile**:
- Full frequency response curve saved
- Allows offline visualization and analysis

### Verification Scripts

**verify_features.py** (199 lines):
- Checks all module files exist
- Verifies class structure in each module
- Lists all implemented features with checkmarks
- Summarizes capabilities per module
- Exit code indicates success/failure

---

## FILE ORGANIZATION & DEPENDENCIES

### Core Processing Modules:
1. `audio_analyzer.py` - Basic offline analysis
2. `advanced_analyzer.py` - Professional multiband/formant analysis
3. `live_processor.py` - Basic real-time processor
4. `advanced_processor.py` - Advanced real-time processor
5. `ultimate_processor.py` - Complete integration

### Processing Components:
6. `multiband_processor.py` - Crossovers, transient detection, envelope following
7. `adaptive_sound_processor.py` - Adaptive EQ, micro-transient processing
8. `sound_classifier.py` - ML-based sound classification
9. `spatial_effects.py` - Reverb, width, panning
10. `harmonic_processor.py` - Saturation, harmonics, exciter

### Support:
11. `config.py` - Configuration and paths
12. `audio_playback.py` - Playback controls
13. `visualizations.py` - Spectrum visualization
14. `gui.py` - Basic GUI (668 lines)
15. `advanced_gui.py` - Advanced GUI (857 lines)

### Data:
- `presets/` - JSON preset storage
- `recordings/` - WAV output storage
- `models/` - ML classifier models (if trained)

---

## KNOWN ISSUES / INCOMPLETE FEATURES

1. **MultibandProcessor import error** in ultimate_processor.py:
   - Imports `MultibandProcessor` which doesn't exist
   - Only instantiated, never used in processing
   - Should either create this class or remove reference

2. **Classification model not trained**:
   - OnsetBasedClassifier falls back to rule-based classification
   - ML model would improve detection accuracy
   - Training pipeline exists but no pre-trained models included

3. **Transition smoothing not active**:
   - AdaptiveSoundProcessor has transition smoothing parameters
   - Not integrated into actual processing loop

4. **Limited documentation**:
   - Code has good docstrings
   - Usage examples mostly in verification scripts
   - CLI interfaces not fully documented

---

## TECHNOLOGY STACK

- **Python 3.8+**: Core language
- **Librosa**: Audio analysis and feature extraction
- **SciPy**: Signal processing and IIR filter design
- **NumPy**: Numerical computing
- **Pedalboard**: High-performance real-time effects (Spotify's JUCE wrapper)
- **SoundDevice**: Low-latency audio I/O
- **SoundFile**: WAV file I/O
- **scikit-learn**: RandomForest classifier
- **Tkinter**: GUI (cross-platform)
- **joblib**: Model serialization

---

## SUMMARY

The beatbox-reversed project is a sophisticated audio analysis and real-time processing system with:
- **Multi-stage architecture**: Offline analysis → preset generation → real-time application
- **Comprehensive feature extraction**: EQ curves, compression, transients, harmonics, per-sound characteristics
- **Advanced real-time processing**: Multiband with transient preservation, adaptive sound-type specific processing, spatial effects, harmonic saturation
- **Safety mechanisms**: Normalization to prevent mastering artifacts (0.20 scaling + ±6dB clamping), crest factor-based compression caps (max 4:1), final safety limiting
- **Production-ready components**: Thread-safe recording, multiple processing modes, device enumeration, extensive diagnostics
- **Extensible design**: Modular processors can be combined in different ways (basic, advanced, ultimate modes)

The code quality is professional with careful attention to real-time constraints, numerical stability, and user experience.

