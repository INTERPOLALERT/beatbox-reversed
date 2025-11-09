# 🚀 Quick Start Guide

Get up and running in 5 minutes!

## Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Launch the application
python gui.py
```

## Your First Beatbox Style Transfer (3 Steps)

### Step 1: Analyze Reference Audio (2 minutes)

1. **Get a beatbox audio file**
   - Use any WAV, MP3, or FLAC file
   - Can be a YouTube download, studio recording, or your own recording
   - Recommended: 5-30 seconds of clean beatbox audio

2. **Analyze it**
   - Click "Browse..." and select your file
   - Enter a preset name (e.g., "kick_heavy")
   - Click "🔍 Analyze Audio"
   - Wait 5-30 seconds

**Result**: You now have a preset that captures all the audio characteristics!

### Step 2: Configure Audio (1 minute)

1. Go to "⚙️ Settings" tab
2. Select your **Microphone** from Input Device
3. Select your **Headphones** from Output Device
4. Click "💾 Save Settings"

**Tip**: Use wired headphones for best latency!

### Step 3: Process Live! (30 seconds)

1. Go to "2. Live Processing" tab
2. Select your preset from dropdown
3. Click "▶️ Start Processing"
4. **Start beatboxing!**

You'll hear yourself with the applied effects in real-time! 🎉

## Recording Your Performance

While processing is active:

1. Click "🔴 Start Recording"
2. Perform your beatbox routine
3. Click "⏹️ Stop Recording"

Your recording will be saved in the `recordings/` folder!

## Tips for Best Results

### Audio Quality
- ✅ Use a decent USB microphone (even a basic one works)
- ✅ Close to mic but not touching it
- ✅ Quiet room (reduces background noise)
- ❌ Don't use laptop built-in mic (poor quality)

### Latency
- ✅ Use wired headphones (Bluetooth adds 100-200ms delay)
- ✅ Close other applications
- ✅ Reduce buffer size in Settings (try 256 or 128)
- ⚠️ Lower buffer = less latency but needs better CPU

### Reference Audio
- ✅ Use clean, well-produced beatbox recordings
- ✅ 5-30 seconds is ideal
- ✅ Single beatboxer is better than group
- ⚠️ Avoid audio with music backing tracks (analyze just the beatbox)

## Common Issues

**"I hear my voice but no effects"**
- Make sure you selected the correct preset
- Check that the preset analysis completed successfully

**"Audio sounds bad/distorted"**
- The reference audio might be heavily processed
- Try analyzing different reference audio
- Your mic might be too close or too loud

**"High latency / delay"**
- Reduce buffer size in Settings
- Install ASIO drivers (Windows)
- Use wired headphones
- Close other apps

**"No audio at all"**
- Check audio device selection in Settings
- Make sure mic is not muted
- Verify headphones are connected

## What's Being Analyzed?

When you analyze audio, the app extracts:

1. **EQ Curve** (10 frequency bands)
   - Bass, mids, highs balance
   - Tonal character

2. **Compression**
   - How punchy or smooth
   - Dynamic control

3. **Transients**
   - Attack characteristics
   - Sharpness of sounds

4. **Harmonics**
   - Harmonic vs percussive balance
   - Overall tone color

All of this gets applied to YOUR voice while preserving your unique articulation!

## Next Steps

Once you're comfortable with the basics:

- Analyze multiple beatboxers to create a preset library
- Experiment with different buffer sizes for latency
- Try the command-line interface for advanced control
- Read the full README.md for deep technical details

## Need Help?

- Check the main README.md for troubleshooting
- Review the research documents in `bbxreverse/` folder
- Make sure all Python packages are installed

## Pro Tips

🎯 **Create presets for different sounds**
- "kick_heavy" for deep kicks
- "snare_crisp" for sharp snares
- "bass_smooth" for bass sounds

🎯 **Use good reference audio**
- Studio-quality recordings work best
- Clean isolated beatbox (no music)
- Professional beatboxers as references

🎯 **Monitor your performance**
- Always use headphones (not speakers)
- Wired connection only
- Low latency is key for natural feel

Happy beatboxing! 🎤🔥
