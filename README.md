# 🎤 Beatbox Audio Style Transfer - Professional Edition

**Reverse engineer any beatbox audio and apply those exact characteristics to your live mic in real-time!**

## 🆕 NOW WITH PROFESSIONAL-GRADE FEATURES!

**Version 2.0** adds:
- ✨ **Multiband processing** (4 or 8 bands with Linkwitz-Riley crossovers)
- ✨ **Adaptive transient preservation** (preserves your attack while adding character)
- ✨ **Sound classification** (kick/snare/hihat/bass detection)
- ✨ **Real-time controls** (wet/dry, transient blend, per-band mixing)
- ✨ **Professional visualizations** (spectrum analyzer, EQ curves)
- ✨ **Advanced analysis** (formant extraction, per-sound-type presets)

👉 **New users:** Check [`QUICKSTART.md`](QUICKSTART.md) for 5-minute setup
👉 **v2.0 features:** See [`PROFESSIONAL_UPGRADE.md`](PROFESSIONAL_UPGRADE.md) for complete details
👉 **Technical info:** Continue reading below

## 🌟 What Does This App Do?

This application allows you to:

1. **Analyze** professional beatbox recordings to extract ALL audio characteristics (EQ, compression, dynamics, etc.)
2. **Apply** those extracted settings to your microphone in real-time
3. **Beatbox live** and sound like your favorite beatboxer
4. **Record** your processed performances to save to your PC

### The Magic

Feed the app any beatbox audio → It reverse-engineers everything → Apply to your mic → Sound like that audio when you beatbox!

## 🎯 Key Features

- ✅ **Comprehensive Analysis**: Extracts EQ curves, compression settings, dynamic characteristics, harmonic content, and transient profiles
- ✅ **Real-Time Processing**: Sub-20ms latency for live performance monitoring
- ✅ **High-Quality Effects**: Uses Spotify's Pedalboard library (built on JUCE framework) for professional-grade audio processing
- ✅ **Recording Capability**: Save your processed beatboxing to WAV files
- ✅ **User-Friendly GUI**: Simple interface for analyzing, processing, and recording
- ✅ **Preset System**: Save and load multiple analyzed sounds
- ✅ **Adaptive Processing**: Intelligently applies settings while preserving your unique articulation

## 🏗️ Architecture

Built on cutting-edge research combining traditional DSP with modern audio analysis:

### Two-Stage System

**Stage 1: Offline Analysis (audio_analyzer.py)**
- Load reference beatbox audio
- Extract frequency spectrum (EQ curve) using FFT/STFT
- Analyze dynamics and estimate compression parameters
- Detect transient characteristics
- Analyze harmonic content
- Save as reusable preset

**Stage 2: Real-Time Processing (live_processor.py)**
- Load preset configuration
- Apply effects chain to live microphone input
- Monitor processed audio with minimal latency
- Record processed output to file

### Technology Stack

- **Python**: Core language for rapid development
- **Librosa**: Audio analysis and feature extraction
- **Pedalboard**: High-performance real-time audio effects (Spotify's library)
- **SoundDevice**: Low-latency audio I/O
- **NumPy/SciPy**: Signal processing and scientific computing
- **Tkinter**: Cross-platform GUI

## 📋 Requirements

- Python 3.8+
- Audio interface (USB mic or audio interface recommended for best latency)
- Headphones (to monitor processed audio)
- Windows 11, macOS, or Linux

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd beatbox-reversed
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python gui.py
```

## 📖 Usage Guide

### Quick Start: 3 Easy Steps

#### Step 1: Analyze Reference Audio

1. Launch the app: `python gui.py`
2. Go to **"1. Analyze Audio"** tab
3. Click **"Browse..."** and select a beatbox audio file (WAV, MP3, FLAC, etc.)
4. Enter a preset name (e.g., "kick_heavy", "snare_crisp")
5. Click **"🔍 Analyze Audio"**
6. Wait for analysis to complete (usually 5-30 seconds)

The app extracts:
- 10-band EQ curve
- Compression threshold, ratio, attack/release
- Dynamic range characteristics
- Transient profile
- Harmonic content

#### Step 2: Configure Audio Devices

1. Go to **"⚙️ Settings"** tab
2. Select your **microphone** from Input Device dropdown
3. Select your **headphones/speakers** from Output Device dropdown
4. Adjust buffer size if needed (lower = less latency, higher CPU usage)
5. Click **"💾 Save Settings"**

#### Step 3: Live Processing

1. Go to **"2. Live Processing"** tab
2. Select your preset from the dropdown
3. Click **"▶️ Start Processing"**
4. Start beatboxing into your mic!
5. You'll hear yourself with the applied effects in real-time

#### Bonus: Recording

While processing is active:
1. Click **"🔴 Start Recording"**
2. Perform your beatbox routine
3. Click **"⏹️ Stop Recording"**
4. Your recording is saved in the `recordings/` folder

## 🎛️ Advanced Usage

### Command Line Interface

#### Analyze Audio (Command Line)

```bash
python audio_analyzer.py reference_audio.wav my_preset_name
```

#### Live Processing (Command Line)

```bash
python live_processor.py presets/my_preset_name.json
```

### Understanding the Analysis

The analyzer extracts these parameters:

**EQ Curve**: 10 frequency bands (30Hz - 16kHz)
- Each band has: frequency, gain (dB), Q factor
- Captures the tonal balance of the reference

**Compression**:
- Threshold: Level where compression kicks in
- Ratio: Amount of compression (e.g., 4:1)
- Attack/Release: How fast compressor responds

**Dynamic Range**:
- Overall loudness variation
- Helps determine processing intensity

**Transient Profile**:
- Detects peaks and attacks
- Preserves punch and articulation

**Harmonic Content**:
- Harmonic vs percussive balance
- Spectral characteristics

### Preset Files

Presets are saved as JSON files in `presets/` directory.

Example structure:
```json
{
  "metadata": {
    "source_file": "reference.wav",
    "duration_seconds": 10.5,
    "sample_rate": 44100
  },
  "eq_curve": [
    {"frequency": 100, "gain_db": 3.5, "q_factor": 1.0},
    {"frequency": 1000, "gain_db": -2.1, "q_factor": 1.0}
  ],
  "compression": {
    "threshold_db": -18.5,
    "ratio": 4.2,
    "attack_ms": 10,
    "release_ms": 100
  }
}
```

## ⚡ Performance Optimization

### Achieving Low Latency

Target: **Sub-10ms latency** for live performance

**Tips:**

1. **Use ASIO Drivers (Windows)**
   - Install ASIO4ALL or manufacturer drivers
   - Significantly reduces latency vs standard Windows audio

2. **Reduce Buffer Size**
   - Settings tab → Buffer Size → Try 128 or 256 samples
   - Lower = less latency (but higher CPU usage)

3. **Use Quality Audio Interface**
   - USB audio interfaces have better drivers
   - Built-in sound cards often have higher latency

4. **Close Other Applications**
   - Reduce CPU load
   - Disable unnecessary background processes

5. **Wired Headphones**
   - Bluetooth adds 100-200ms latency
   - Always use wired for real-time monitoring

### Current Latency Estimates

| Buffer Size | Latency @ 44.1kHz | Latency @ 48kHz | Recommended For |
|-------------|-------------------|-----------------|-----------------|
| 64 samples  | ~1.5ms           | ~1.3ms          | Best performance systems |
| 128 samples | ~2.9ms           | ~2.7ms          | Good systems |
| 256 samples | ~5.8ms           | ~5.3ms          | Average systems |
| 512 samples | ~11.6ms          | ~10.7ms         | Older systems |

## 🔬 Technical Details

### Research-Based Implementation

This application is built on research from:
- **DDSP (Differentiable Digital Signal Processing)** - Google Magenta
- **RAVE (Real-time Audio Variational Encoder)** - IRCAM
- **iZotope Ozone Match EQ** - Industry standard spectral matching
- Academic papers on audio style transfer and compression analysis

### Key Algorithms

1. **STFT (Short-Time Fourier Transform)**
   - 8192-sample windows for high frequency resolution
   - Hann windowing to minimize spectral leakage

2. **Cepstral Analysis**
   - Separates vocal tract filtering from excitation
   - Mirrors source-filter model of human voice

3. **Envelope Detection**
   - RMS analysis for dynamic range estimation
   - Onset detection for transient characteristics

4. **IIR Filtering**
   - Low-latency implementation (0.5-1ms per filter)
   - Peak, low-shelf, and high-shelf filters

5. **Adaptive Compression**
   - Envelope followers with separate attack/release
   - Preserves natural dynamics while applying target character

### File Structure

```
beatbox-reversed/
├── gui.py                  # Main GUI application
├── audio_analyzer.py       # Stage 1: Offline analysis
├── live_processor.py       # Stage 2: Real-time processing
├── config.py              # Configuration and settings
├── requirements.txt       # Python dependencies
├── presets/              # Saved analysis presets
├── recordings/           # Saved recordings
└── bbxreverse/          # Research documents
    ├── chatgpt_core idea.txt
    ├── kimi.txt
    ├── research.txt
    ├── research 1.txt
    ├── research 2.txt
    └── Building a Beatbox Style Transfer Plugin.pdf
```

## 🎓 How It Works: Deep Dive

### The Analysis Process

When you load a reference audio file, the analyzer:

1. **Loads and resamples** audio to 44.1kHz
2. **Computes STFT** with 8192-point FFT
3. **Averages spectrum** over time for long-term profile
4. **Fits EQ curve** to standard frequency bands
5. **Calculates RMS envelope** in 50ms windows
6. **Estimates compression** from level statistics
7. **Detects onsets** for transient analysis
8. **Separates harmonics** from percussive content
9. **Saves everything** as JSON preset

### The Real-Time Process

During live processing:

1. **Audio input** captured in small buffers (512 samples)
2. **Effects chain applied**:
   - EQ filters (low-shelf, peak, high-shelf)
   - Compressor (threshold, ratio, attack, release)
3. **Processed audio** sent to output buffer
4. **Optional recording** saves to buffer
5. **Low latency** maintained through efficient algorithms

### Why This Approach Works

**Adaptive Processing**: Unlike naive spectral matching, this system:
- Preserves your unique transient articulation
- Applies tonal shaping without destroying character
- Uses compression that adapts to your dynamics
- Maintains the "feel" of your beatboxing while adding polish

**Real-Time Performance**: Achieves low latency through:
- IIR filters (not FIR) for minimal latency
- Efficient C++ backend (Pedalboard/JUCE)
- Optimized buffer management
- No neural network inference in audio thread

## 🛠️ Troubleshooting

### Common Issues

**"No sound when processing"**
- Check audio device selection in Settings
- Ensure microphone is connected and not muted
- Verify output device is selected correctly

**"High latency / delayed audio"**
- Reduce buffer size in Settings
- Install ASIO drivers (Windows)
- Close other applications
- Use wired headphones

**"Audio dropouts / glitches"**
- Increase buffer size
- Close CPU-intensive applications
- Check system performance

**"Analysis fails"**
- Check audio file format (WAV, MP3, FLAC supported)
- Ensure file is not corrupted
- Try shorter audio clips first

**"Can't find audio devices"**
- Restart application
- Check audio drivers installed
- Verify devices in system settings

### Getting Help

- Check the Issues section
- Review the research documents in `bbxreverse/` folder
- Ensure all dependencies are installed correctly

## 🚧 Future Enhancements (Phase 2)

Planned features based on research:

- [ ] **Sound Classification**: Auto-detect kick/snare/hi-hat and apply appropriate presets
- [ ] **Multiband Processing**: 4-8 frequency bands with independent processing
- [ ] **Machine Learning**: Neural network for advanced style transfer
- [ ] **Visual Feedback**: Real-time spectrogram comparison
- [ ] **Multi-Preset System**: Switch presets per sound type automatically
- [ ] **Advanced UI**: Parameter visualization and manual tweaking
- [ ] **VST Plugin Version**: Use in DAWs (requires JUCE/C++ port)
- [ ] **BaDumTss Dataset Training**: Train classifier on beatbox-specific sounds

## 📚 References

This project is built on research from:

1. **Differentiable Digital Signal Processing (DDSP)** - Google Magenta
2. **RAVE: Real-time Audio Variational Encoder** - IRCAM
3. **iZotope Ozone Match EQ** - Professional spectral matching
4. **Pedalboard** - Spotify's audio processing library
5. Academic papers on compression analysis, transient detection, and audio style transfer

See `bbxreverse/` folder for detailed research documents.

## 📝 License

[Add your license here]

## 👏 Credits

- Built by: [Your Name]
- Research compilation: Multiple AI assistants and academic papers
- Audio processing: Pedalboard (Spotify), Librosa
- Inspired by professional beatboxers worldwide

## 🎤 Happy Beatboxing!

Transform your sound, preserve your style, and record your best performances!
