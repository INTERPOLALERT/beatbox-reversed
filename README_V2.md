# 🎤 Beatbox Audio Analyzer V2 - DSP Chain Extraction

## 🎯 What is V2?

**V2 is a complete rewrite focused on extracting microphone settings and post-processing effects** from professional beatbox audio. Unlike V1 which tried to match voice characteristics, **V2 extracts the actual DSP processing chain** (EQ, compression, saturation, effects) so you can apply those same settings to YOUR microphone when YOU beatbox.

### The Core Concept

```
Reference Beatbox Audio → Analyze DSP Chain → Extract Settings → Apply to Your Mic → Sound Like Pro Recording
```

You're NOT trying to sound like the beatboxer. You're trying to get the **same microphone tone and processing** they had.

## 🆕 What's New in V2?

### V2 vs V1 - Fundamental Differences

| Aspect | V1 (Old) | V2 (New) |
|--------|----------|----------|
| **Focus** | Voice characteristics (formants, pitch) | DSP processing chain (EQ, compression, effects) |
| **Goal** | Try to match vocal tract | Extract mic settings & post-processing |
| **Analysis** | Formant extraction, MFCC | Spectral EQ curve, multi-stage compression, effects detection |
| **Use Case** | Voice cloning attempt | Professional mic tone matching |
| **Accuracy** | Limited for live use | Optimized for real-time mic processing |

### V2 Modules - Complete DSP Chain Extraction

#### 1. **spectral_analyzer_v2.py** - Advanced EQ Extraction
- **14-band parametric EQ** analysis (professional console style)
- **Multiple analysis methods** (LTAS, percentile spectrum, RMS spectrum)
- **Spectral tilt detection** (overall brightness/darkness)
- **Resonance peak detection** (formant-like peaks)
- **Live-optimized** EQ curves (prevents extreme settings)

**What it detects:**
- Exact frequency boosts/cuts from reference audio
- Overall tonal balance (bright, dark, neutral)
- Key resonant frequencies

#### 2. **dynamics_analyzer_v2.py** - Compression/Limiting Detection
- **Advanced compression detection** (threshold, ratio, attack, release, knee)
- **Limiting detection** (brick-wall ceiling, limiting amount)
- **Gate/Expander detection** (noise gate threshold and settings)
- **Dynamic range analysis** (overall compression level)
- **Multi-stage compression detection** (serial compression chains)

**What it detects:**
- Compression parameters for natural dynamics
- Limiting settings for loudness
- Gating to remove background noise
- Whether multi-stage compression is used

#### 3. **effects_detector.py** - Effects Chain Detection
- **Saturation/Distortion detection** (type: tube, tape, soft, hard clipping)
- **De-essing detection** (high-frequency dynamics control for sibilance)
- **Exciter/Enhancer detection** (added brightness/air)
- **Transient shaping detection** (attack enhancement/reduction)
- **Warmth character detection** (analog tape/tube emulation)

**What it detects:**
- Type and amount of saturation
- De-esser frequency and threshold
- Harmonic enhancement
- Transient processing
- Analog warmth emulation

#### 4. **multiband_dynamics_analyzer.py** - Per-Band Compression
- **4 or 8-band analysis** (bass, low-mid, mid, high)
- **Per-band compression settings** (different compression for each frequency range)
- **Crossover frequency detection** (where bands split)
- **Multiband compression detection** (whether bands are compressed differently)

**What it detects:**
- Individual compression settings for bass, mids, highs
- Crossover frequencies
- Whether multiband processing is being used
- Per-band dynamics characteristics

#### 5. **stereo_analyzer.py** - Stereo & Spatial Effects
- **Stereo width detection** (mono, normal stereo, wide stereo)
- **Mid-side balance analysis** (M/S processing)
- **Stereo correlation** (phase relationship)
- **Panning detection** (left/right positioning)
- **Reverb detection** (spatial effects, room size)

**What it detects:**
- Stereo width enhancement
- M/S processing
- Reverb amount and character
- Panning position

#### 6. **analyzer_v2.py** - Main Integration
- **Combines all analyzers** into complete workflow
- **Generates processing chain description**
- **Saves optimized presets** for live use
- **Prints comprehensive summary**

#### 7. **processor_v2.py** - Real-time Application
- **Applies extracted settings** to live microphone
- **Builds Pedalboard chain** from analysis
- **Wet/dry mixing** for control
- **Safety limiting** to prevent clipping

## 🔧 How V2 Works

### Analysis Process

```
1. Load Reference Audio
2. Run Spectral Analysis → Extract EQ curve
3. Run Dynamics Analysis → Extract compression settings
4. Run Effects Detection → Identify saturation, de-essing, etc.
5. Run Multiband Analysis → Extract per-band processing
6. Run Stereo Analysis → Identify spatial effects
7. Optimize for Live Use → Reduce extreme settings
8. Save as Preset → Ready to apply to your mic
```

### Processing Chain Order (Typical)

```
Microphone Input
    ↓
[1] Input Gain Adjustment
    ↓
[2] EQ (14-band Parametric)
    ↓
[3] Compression (or Multiband Compression)
    ↓
[4] Effects (Saturation, De-essing, Warmth)
    ↓
[5] Limiter (Loudness/Peak Control)
    ↓
[6] Stereo Effects (Width, Reverb) [if applicable]
    ↓
[7] Output Gain
    ↓
Your Processed Sound!
```

## 🚀 Quick Start Guide

### Installation

```bash
# V2 uses the same dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Analyze Reference Audio

```bash
python analyzer_v2.py reference_audio.wav my_preset
```

This will:
- Analyze the complete DSP chain
- Print detailed summary
- Save preset as `my_preset_v2.json`

#### 2. Apply to Your Microphone

```python
from processor_v2 import BeatboxProcessorV2

# Load preset
processor = BeatboxProcessorV2(sample_rate=44100)
processor.load_preset("presets/my_preset_v2.json")

# Process audio buffer
processed_audio = processor.process(input_audio)
```

### Example Output

```
DETECTED PROCESSING CHAIN
==================================================

1. INPUT GAIN
   Adjust input to match reference level

2. EQ (14-band Parametric)
   Overall Character: Slight high-frequency lift (clear, present)
   Bass: +2.3 dB @ 80 Hz
   Low Mid 2: -1.8 dB @ 500 Hz
   Presence 1: +3.5 dB @ 5000 Hz
   Air: +2.1 dB @ 16000 Hz

3. COMPRESSION
   Threshold: -18.5 dB
   Ratio: 3.2:1
   Attack: 5.2 ms
   Release: 120 ms

4. EFFECTS
   Saturation: tube_or_tape (40%)
   De-esser: 2.5:1 @ 6000 Hz

5. LIMITER
   Ceiling: -0.3 dB
   Level: moderate

==================================================
```

## 📊 Analysis Modules Deep Dive

### Spectral Analyzer V2

**Purpose:** Extract precise EQ curve that makes the audio sound polished

**Methods:**
1. **LTAS (Long-Term Average Spectrum)** - Time-averaged frequency response
2. **Percentile Spectrum** - Robust to outliers (uses median)
3. **RMS Spectrum** - Energy-weighted response
4. **Combined Estimation** - Weighted average of all methods

**Output:**
- 14-band parametric EQ settings
- Spectral tilt (brightness measure)
- Peak resonances

**Live Optimization:**
- Scales gain by 25% to prevent extreme boosts
- Clamps to ±12 dB for stability
- Adjusts Q factors for smoother response

### Dynamics Analyzer V2

**Purpose:** Extract compression, limiting, and gating parameters

**Detection Methods:**
- **Threshold:** 35th percentile of RMS distribution
- **Ratio:** Variance ratio above/below threshold
- **Attack:** Transient rise time analysis
- **Release:** Envelope decay rate analysis
- **Knee:** Transition smoothness around threshold

**Limiting Detection:**
- Analyzes peak distribution
- Detects samples near ceiling
- Classifies limiting intensity

**Gate Detection:**
- Finds noise floor
- Detects abrupt silence cuts
- Estimates gate threshold

### Effects Detector

**Purpose:** Identify and quantify audio effects

**Saturation Detection:**
- Calculates THD (Total Harmonic Distortion)
- Analyzes harmonic content
- Detects clipping characteristics
- Classifies saturation type (tube, tape, soft, hard)

**De-essing Detection:**
- Compares high-freq vs overall dynamics
- Estimates de-esser frequency (typically 5-8 kHz)
- Calculates reduction ratio

**Warmth Detection:**
- Analyzes even harmonic content
- Measures low-mid emphasis
- Classifies analog character

### Multiband Dynamics Analyzer

**Purpose:** Extract per-band compression settings

**Process:**
1. Split audio into 4 or 8 frequency bands using Linkwitz-Riley crossovers
2. Analyze dynamics independently for each band
3. Detect if multiband compression is used
4. Optimize for live stability

**Typical Bands (4-band):**
- Bass: 20-200 Hz
- Low-Mid: 200-1000 Hz
- High-Mid: 1000-4000 Hz
- Treble: 4000-22000 Hz

## 🎛️ Using V2 Presets

### Preset Structure

```json
{
  "metadata": {
    "source_file": "reference.wav",
    "duration_seconds": 10.5,
    "sample_rate": 44100,
    "is_stereo": false,
    "analysis_version": "v2.0"
  },
  "spectral": {
    "eq_curve": [
      {
        "name": "Bass",
        "frequency": 80,
        "gain_db": 2.3,
        "q_factor": 1.0,
        "filter_type": "peak"
      }
      ...
    ],
    "spectral_tilt": {
      "tilt_db_per_decade": 1.8,
      "tilt_type": "bright"
    }
  },
  "dynamics": {
    "compression": {
      "threshold_db": -18.5,
      "ratio": 3.2,
      "attack_ms": 5.2,
      "release_ms": 120,
      "knee_db": 3.0,
      "makeup_gain_db": 4.5
    },
    "limiting": {
      "ceiling_db": -0.3,
      "is_limited": true
    }
  },
  "effects": {
    "saturation": {
      "detected": true,
      "amount": 0.4,
      "type": "tube_or_tape"
    },
    "deessing": {
      "detected": true,
      "threshold_db": -15.0,
      "frequency_hz": 6000,
      "ratio": 2.5
    }
  },
  "multiband": {
    "num_bands": 4,
    "crossover_freqs": [200, 1000, 4000],
    "bands": [...]
  }
}
```

## 🔬 Accuracy & Validation

### AUDIT #1: Analysis Accuracy

**What V2 Does Well:**
✅ EQ curve extraction (very accurate with multiple methods)
✅ Compression ratio/threshold detection (good for moderate compression)
✅ Limiting detection (accurate ceiling detection)
✅ Saturation type classification (reliable)
✅ Multiband processing detection (accurate when present)

**Limitations:**
⚠️ Cannot perfectly extract attack/release times (estimates based on envelope)
⚠️ De-esser detection is approximate (would need freq-specific dynamics analysis)
⚠️ Multi-stage compression detection is heuristic-based
⚠️ Cannot detect processing order with 100% certainty

**Live Use Optimizations:**
- All settings are scaled/clamped for stability
- EQ gains reduced by 70-75% for live use
- Compression ratios capped at 4-6:1
- Extreme settings are moderated

### AUDIT #2: Real-World Performance

**Best Results When:**
- Reference audio has clear, consistent processing
- Reference is professionally mixed/mastered
- Reference has similar source material (beatbox/vocals)

**Less Accurate When:**
- Reference has inconsistent processing
- Very dynamic/unprocessed reference
- Heavy effects that mask dynamics

**Recommendation:** Test multiple reference tracks from the same artist/album for consistent results.

## 💡 Tips for Best Results

### 1. Choose Good Reference Audio
- ✅ Professional beatbox recordings
- ✅ Consistent mixing throughout
- ✅ Similar style to what you want
- ❌ Avoid: Raw/unprocessed recordings
- ❌ Avoid: Heavily distorted/lo-fi recordings

### 2. Analyze Multiple Tracks
- Analyze 2-3 tracks from same artist/album
- Compare settings to find common patterns
- Average settings for consistency

### 3. Start with Low Mix Amount
- Set wet/dry mix to 50% initially
- Gradually increase as you get comfortable
- Full 100% may be too processed for some sources

### 4. Adjust for Your Voice
- Use detected settings as starting point
- Fine-tune EQ for your voice
- Adjust compression threshold for your mic level

### 5. Monitor Your Levels
- Watch for clipping
- Adjust input gain if too loud
- Use limiter for safety

## 🛠️ Advanced Usage

### Command-Line Analysis

```bash
# Basic analysis
python analyzer_v2.py audio.wav

# With preset name
python analyzer_v2.py audio.wav my_preset

# With 8-band multiband analysis
python analyzer_v2.py audio.wav my_preset 8
```

### Individual Module Testing

```bash
# Test spectral analyzer
python spectral_analyzer_v2.py audio.wav

# Test dynamics analyzer
python dynamics_analyzer_v2.py audio.wav

# Test effects detector
python effects_detector.py audio.wav

# Test multiband analyzer
python multiband_dynamics_analyzer.py audio.wav 4

# Test stereo analyzer
python stereo_analyzer.py audio.wav
```

### Python API

```python
from analyzer_v2 import BeatboxAnalyzerV2

# Create analyzer
analyzer = BeatboxAnalyzerV2()

# Load audio
analyzer.load_audio("reference.wav")

# Run analysis
results = analyzer.analyze_all(num_bands=4)

# Print summary
analyzer.print_summary()

# Get processing chain description
description = analyzer.create_processing_chain_description()
print(description)

# Save preset
analyzer.save_preset("my_preset")
```

## 📈 Future Improvements

### Potential V2.1 Features
- [ ] Neural network for more accurate compression detection
- [ ] Automatic processing order detection
- [ ] Multi-stage compression chain extraction
- [ ] Better de-esser parameter detection
- [ ] Automatic preset blending/averaging
- [ ] Real-time analysis from live input
- [ ] Visual waveform comparison
- [ ] A/B comparison tool

## 🤝 Integration with Existing App

V2 modules can be used alongside V1 modules:

- Use `analyzer_v2.py` for analysis
- Use `processor_v2.py` for real-time processing
- Use existing `advanced_gui.py` with modifications
- V2 presets saved with `_v2.json` suffix to avoid conflicts

## 📝 License

Same as main project.

## 🎤 Happy Beatboxing!

Transform your sound with professional-grade DSP analysis!

**Remember:** You're not copying the beatboxer's technique or voice. You're extracting the **microphone settings and post-processing effects** that make their recordings sound polished and professional!
