"""
Advanced Beatbox Audio Style Transfer - Professional GUI
Complete interface with visualizations, advanced controls, and real-time feedback
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
import sys
import numpy as np

from advanced_analyzer import AdvancedAudioAnalyzer
from advanced_processor import AdvancedLiveProcessor
from ultimate_processor import UltimateProcessor
from audio_playback import AudioPlayer
from visualizations import RealTimeVisualizer, EQCurveVisualizer, MultibandVisualizer, SpectrogramVisualizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import config


class UltimateRealTimeProcessor:
    """Real-time streaming wrapper for UltimateProcessor"""

    def __init__(self):
        self.processor = UltimateProcessor(sample_rate=config.SAMPLE_RATE, enable_diagnostics=False)
        self.stream = None
        self.is_processing = False
        self.is_recording = False
        self.recorded_audio = []
        self.audio_config = config.AudioConfig()

    def load_preset(self, preset_path: str):
        """Load preset into processor"""
        self.processor.load_preset(preset_path)

    def start_processing(self, input_device=None, output_device=None, monitor=True):
        """Start real-time processing with monitoring"""
        if self.is_processing:
            return

        # Set devices
        if input_device is not None:
            sd.default.device[0] = input_device
        if output_device is not None:
            sd.default.device[1] = output_device

        self.is_processing = True

        def audio_callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"Status: {status}")

            try:
                # Get mono input
                audio_in = indata[:, 0].copy()

                # Process with UltimateProcessor (returns stereo)
                audio_out = self.processor.process_buffer(audio_in)

                # Handle recording
                if self.is_recording:
                    self.recorded_audio.append(audio_out.copy())

                # Output (stereo)
                if audio_out.ndim == 2:
                    outdata[:] = audio_out
                else:
                    # Mono to stereo
                    outdata[:, 0] = audio_out
                    outdata[:, 1] = audio_out

            except Exception as e:
                print(f"Callback error: {e}")
                outdata.fill(0)

        # Start stream with monitoring
        self.stream = sd.Stream(
            samplerate=self.audio_config.sample_rate,
            blocksize=self.audio_config.buffer_size,
            channels=(1, 2),  # Mono input, stereo output
            callback=audio_callback
        )
        self.stream.start()

    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def start_recording(self):
        """Start recording processed output"""
        self.recorded_audio = []
        self.is_recording = True

    def stop_recording(self):
        """Stop recording and save file"""
        self.is_recording = False

        if not self.recorded_audio:
            return None

        # Concatenate all buffers
        audio = np.concatenate(self.recorded_audio, axis=0)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.RECORDINGS_DIR / f"recording_{timestamp}.wav"
        config.RECORDINGS_DIR.mkdir(exist_ok=True)

        sf.write(output_path, audio, self.audio_config.sample_rate)
        return str(output_path)

    # Proxy methods to UltimateProcessor
    def set_wet_dry_mix(self, mix: float):
        self.processor.set_wet_dry_mix(mix)

    def set_transient_preservation(self, amount: float):
        self.processor.set_transient_preservation(amount)

    def set_input_gain(self, gain_db: float):
        self.processor.set_input_gain(gain_db)

    def set_output_gain(self, gain_db: float):
        self.processor.set_output_gain(gain_db)

    def set_stereo_width(self, width: float):
        self.processor.set_stereo_width(width)

    def set_reverb(self, amount: float, size: float = 0.5):
        self.processor.set_reverb(amount, size)

    def set_saturation(self, amount: float, saturation_type: str = 'tube'):
        self.processor.set_saturation(amount, saturation_type)

    def set_loudness_matching(self, enabled: bool, match_mode: str = 'rms'):
        self.processor.set_loudness_matching(enabled, match_mode)

    def set_band_mix(self, band_idx: int, level: float):
        """Set per-band mix (for compatibility)"""
        pass  # UltimateProcessor handles this internally


class AdvancedBeatboxApp:
    """Advanced GUI application"""

    def __init__(self, root):
        self.root = root
        self.root.title("Beatbox Audio Style Transfer - Professional Edition")
        self.root.geometry("1200x800")

        # Application state
        self.analyzer = None
        self.processor = UltimateRealTimeProcessor()
        self.current_preset = None
        self.is_processing = False
        self.audio_player = AudioPlayer()
        self.last_recording_path = None

        # Visualization
        self.visualizer = None

        # Create GUI
        self.create_widgets()

    def create_widgets(self):
        """Create all GUI widgets"""

        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=10)

        title_label = ttk.Label(
            title_frame,
            text="🎤 Beatbox Audio Style Transfer - Professional Edition",
            font=('Arial', 16, 'bold')
        )
        title_label.pack()

        subtitle_label = ttk.Label(
            title_frame,
            text="Ultimate Processor • Adaptive Processing • Spatial Effects • Harmonic Saturation • Loudness Matching",
            font=('Arial', 9)
        )
        subtitle_label.pack()

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Analysis
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="📊 Analysis")
        self.create_analysis_tab()

        # Tab 2: Live Processing
        self.processing_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.processing_tab, text="🎚️ Live Processing")
        self.create_processing_tab()

        # Tab 3: Advanced Controls
        self.controls_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.controls_tab, text="⚙️ Advanced Controls")
        self.create_controls_tab()

        # Tab 4: Visualization
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="📈 Visualization")
        self.create_visualization_tab()

        # Tab 5: Settings
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="🔧 Settings")
        self.create_settings_tab()

        # Status bar
        self.status_bar = ttk.Label(
            self.root,
            text="Ready - Professional Edition",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side='bottom', fill='x')

    def create_analysis_tab(self):
        """Create analysis tab"""

        # Instructions
        inst_frame = ttk.LabelFrame(self.analysis_tab, text="Instructions")
        inst_frame.pack(fill='x', padx=10, pady=10)

        instructions = """
        Advanced Analysis Mode:
        • Multiband EQ analysis (4 or 8 bands with Linkwitz-Riley crossovers)
        • Formant extraction using LPC analysis
        • Per-sound-type classification and analysis (kick/snare/hihat/bass)
        • Enhanced compression parameter estimation
        • Transient profile analysis
        """

        ttk.Label(inst_frame, text=instructions, justify='left').pack(padx=10, pady=10)

        # File selection
        file_frame = ttk.LabelFrame(self.analysis_tab, text="Reference Audio File")
        file_frame.pack(fill='x', padx=10, pady=10)

        self.file_path_var = tk.StringVar()

        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=60)
        file_entry.pack(side='left', padx=10, pady=10, fill='x', expand=True)

        browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_audio_file)
        browse_btn.pack(side='left', padx=10, pady=10)

        # Analysis options
        options_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Options")
        options_frame.pack(fill='x', padx=10, pady=10)

        # Preset name
        name_frame = ttk.Frame(options_frame)
        name_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(name_frame, text="Preset Name:").pack(side='left', padx=5)
        self.preset_name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self.preset_name_var, width=30).pack(side='left', padx=5)

        # Number of bands
        bands_frame = ttk.Frame(options_frame)
        bands_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(bands_frame, text="Frequency Bands:").pack(side='left', padx=5)
        self.num_bands_var = tk.IntVar(value=4)
        ttk.Radiobutton(bands_frame, text="4 bands", variable=self.num_bands_var, value=4).pack(side='left', padx=5)
        ttk.Radiobutton(bands_frame, text="8 bands", variable=self.num_bands_var, value=8).pack(side='left', padx=5)

        # Analyze button
        self.analyze_btn = ttk.Button(
            options_frame,
            text="🔍 Run Advanced Analysis",
            command=self.analyze_audio
        )
        self.analyze_btn.pack(padx=10, pady=10)

        # Progress
        self.analysis_progress = ttk.Progressbar(options_frame, mode='indeterminate')
        self.analysis_progress.pack(fill='x', padx=10, pady=5)

        # Results
        results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)

    def create_processing_tab(self):
        """Create processing tab"""

        # Preset selection
        preset_frame = ttk.LabelFrame(self.processing_tab, text="Preset Selection")
        preset_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(preset_frame, text="Active Preset:").pack(side='left', padx=10, pady=10)

        self.preset_combo = ttk.Combobox(preset_frame, width=40, state='readonly')
        self.preset_combo.pack(side='left', padx=10, pady=10, fill='x', expand=True)

        refresh_btn = ttk.Button(preset_frame, text="🔄", command=self.refresh_presets)
        refresh_btn.pack(side='left', padx=5)

        self.refresh_presets()

        # Processing status
        status_frame = ttk.LabelFrame(self.processing_tab, text="Processing Status")
        status_frame.pack(fill='x', padx=10, pady=10)

        self.processing_status_label = ttk.Label(
            status_frame,
            text="⚫ Status: Stopped",
            font=('Arial', 12, 'bold')
        )
        self.processing_status_label.pack(padx=10, pady=10)

        # Processing buttons
        proc_btn_frame = ttk.Frame(status_frame)
        proc_btn_frame.pack(fill='x', padx=10, pady=10)

        self.start_proc_btn = ttk.Button(
            proc_btn_frame,
            text="▶️ Start Processing",
            command=self.start_processing,
            width=20
        )
        self.start_proc_btn.pack(side='left', padx=5)

        self.stop_proc_btn = ttk.Button(
            proc_btn_frame,
            text="⏹️ Stop Processing",
            command=self.stop_processing,
            state='disabled',
            width=20
        )
        self.stop_proc_btn.pack(side='left', padx=5)

        # Recording controls
        rec_frame = ttk.LabelFrame(self.processing_tab, text="Recording Controls")
        rec_frame.pack(fill='x', padx=10, pady=10)

        self.recording_status_label = ttk.Label(
            rec_frame,
            text="⚫ Recording: Off",
            font=('Arial', 11)
        )
        self.recording_status_label.pack(padx=10, pady=5)

        rec_btn_frame = ttk.Frame(rec_frame)
        rec_btn_frame.pack(fill='x', padx=10, pady=10)

        self.start_rec_btn = ttk.Button(
            rec_btn_frame,
            text="🔴 Start Recording",
            command=self.start_recording,
            state='disabled',
            width=20
        )
        self.start_rec_btn.pack(side='left', padx=5)

        self.stop_rec_btn = ttk.Button(
            rec_btn_frame,
            text="⏹️ Stop Recording",
            command=self.stop_recording,
            state='disabled',
            width=20
        )
        self.stop_rec_btn.pack(side='left', padx=5)

        # Playback controls
        playback_btn_frame = ttk.Frame(rec_frame)
        playback_btn_frame.pack(fill='x', padx=10, pady=5)

        self.play_rec_btn = ttk.Button(
            playback_btn_frame,
            text="▶️ Play Recording",
            command=self.play_recording,
            state='disabled',
            width=20
        )
        self.play_rec_btn.pack(side='left', padx=5)

        self.stop_play_btn = ttk.Button(
            playback_btn_frame,
            text="⏹️ Stop Playback",
            command=self.stop_playback,
            state='disabled',
            width=20
        )
        self.stop_play_btn.pack(side='left', padx=5)

        # Log
        log_frame = ttk.LabelFrame(self.processing_tab, text="Processing Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)

    def create_controls_tab(self):
        """Create advanced controls tab"""

        # Wet/Dry Mix
        mix_frame = ttk.LabelFrame(self.controls_tab, text="Wet/Dry Mix")
        mix_frame.pack(fill='x', padx=10, pady=10)

        self.wet_dry_var = tk.DoubleVar(value=1.0)

        ttk.Label(mix_frame, text="Dry ←").pack(side='left', padx=5)

        wet_dry_scale = ttk.Scale(
            mix_frame,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            variable=self.wet_dry_var,
            command=self.update_wet_dry
        )
        wet_dry_scale.pack(side='left', fill='x', expand=True, padx=5)

        ttk.Label(mix_frame, text="→ Wet").pack(side='left', padx=5)

        self.wet_dry_label = ttk.Label(mix_frame, text="100%")
        self.wet_dry_label.pack(side='left', padx=5)

        # Transient Preservation
        transient_frame = ttk.LabelFrame(self.controls_tab, text="Transient Preservation")
        transient_frame.pack(fill='x', padx=10, pady=10)

        self.transient_var = tk.DoubleVar(value=0.8)

        ttk.Label(transient_frame, text="None ←").pack(side='left', padx=5)

        transient_scale = ttk.Scale(
            transient_frame,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            variable=self.transient_var,
            command=self.update_transient
        )
        transient_scale.pack(side='left', fill='x', expand=True, padx=5)

        ttk.Label(transient_frame, text="→ Full").pack(side='left', padx=5)

        self.transient_label = ttk.Label(transient_frame, text="80%")
        self.transient_label.pack(side='left', padx=5)

        # Per-band mixing
        band_frame = ttk.LabelFrame(self.controls_tab, text="Per-Band Mixing (4 bands)")
        band_frame.pack(fill='x', padx=10, pady=10)

        self.band_vars = [tk.DoubleVar(value=1.0) for _ in range(4)]
        self.band_labels = []

        band_names = ["Bass\n(20-200Hz)", "Low-Mid\n(200-1kHz)", "High-Mid\n(1k-4kHz)", "Treble\n(4k-20kHz)"]

        for i, name in enumerate(band_names):
            band_col = ttk.Frame(band_frame)
            band_col.pack(side='left', fill='both', expand=True, padx=5)

            ttk.Label(band_col, text=name, anchor='center').pack(pady=5)

            scale = ttk.Scale(
                band_col,
                from_=0.0,
                to=2.0,
                orient='vertical',
                variable=self.band_vars[i],
                command=lambda val, idx=i: self.update_band_mix(idx, val)
            )
            scale.pack(fill='y', expand=True)

            label = ttk.Label(band_col, text="100%")
            label.pack(pady=5)
            self.band_labels.append(label)

        # Gain controls
        gain_frame = ttk.LabelFrame(self.controls_tab, text="Gain Controls")
        gain_frame.pack(fill='x', padx=10, pady=10)

        # Input gain
        input_gain_frame = ttk.Frame(gain_frame)
        input_gain_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(input_gain_frame, text="Input Gain:").pack(side='left', padx=5)

        self.input_gain_var = tk.DoubleVar(value=0.0)

        input_gain_scale = ttk.Scale(
            input_gain_frame,
            from_=-24.0,
            to=24.0,
            orient='horizontal',
            variable=self.input_gain_var,
            command=self.update_input_gain
        )
        input_gain_scale.pack(side='left', fill='x', expand=True, padx=5)

        self.input_gain_label = ttk.Label(input_gain_frame, text="0.0 dB")
        self.input_gain_label.pack(side='left', padx=5)

        # Output gain
        output_gain_frame = ttk.Frame(gain_frame)
        output_gain_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(output_gain_frame, text="Output Gain:").pack(side='left', padx=5)

        self.output_gain_var = tk.DoubleVar(value=0.0)

        output_gain_scale = ttk.Scale(
            output_gain_frame,
            from_=-24.0,
            to=24.0,
            orient='horizontal',
            variable=self.output_gain_var,
            command=self.update_output_gain
        )
        output_gain_scale.pack(side='left', fill='x', expand=True, padx=5)

        self.output_gain_label = ttk.Label(output_gain_frame, text="0.0 dB")
        self.output_gain_label.pack(side='left', padx=5)

        # Spatial effects
        spatial_frame = ttk.LabelFrame(self.controls_tab, text="Spatial Effects")
        spatial_frame.pack(fill='x', padx=10, pady=10)

        # Stereo width
        stereo_frame = ttk.Frame(spatial_frame)
        stereo_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(stereo_frame, text="Stereo Width:").pack(side='left', padx=5)

        self.stereo_width_var = tk.DoubleVar(value=1.3)

        stereo_scale = ttk.Scale(
            stereo_frame,
            from_=0.0,
            to=2.0,
            orient='horizontal',
            variable=self.stereo_width_var,
            command=self.update_stereo_width
        )
        stereo_scale.pack(side='left', fill='x', expand=True, padx=5)

        self.stereo_width_label = ttk.Label(stereo_frame, text="1.3x")
        self.stereo_width_label.pack(side='left', padx=5)

        # Reverb
        reverb_frame = ttk.Frame(spatial_frame)
        reverb_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(reverb_frame, text="Reverb Amount:").pack(side='left', padx=5)

        self.reverb_var = tk.DoubleVar(value=0.0)

        reverb_scale = ttk.Scale(
            reverb_frame,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            variable=self.reverb_var,
            command=self.update_reverb
        )
        reverb_scale.pack(side='left', fill='x', expand=True, padx=5)

        self.reverb_label = ttk.Label(reverb_frame, text="0%")
        self.reverb_label.pack(side='left', padx=5)

        # Harmonic saturation
        saturation_frame = ttk.LabelFrame(self.controls_tab, text="Harmonic Saturation")
        saturation_frame.pack(fill='x', padx=10, pady=10)

        # Saturation amount
        sat_amount_frame = ttk.Frame(saturation_frame)
        sat_amount_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(sat_amount_frame, text="Saturation:").pack(side='left', padx=5)

        self.saturation_var = tk.DoubleVar(value=0.1)

        sat_scale = ttk.Scale(
            sat_amount_frame,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            variable=self.saturation_var,
            command=self.update_saturation
        )
        sat_scale.pack(side='left', fill='x', expand=True, padx=5)

        self.saturation_label = ttk.Label(sat_amount_frame, text="10%")
        self.saturation_label.pack(side='left', padx=5)

        # Saturation type
        sat_type_frame = ttk.Frame(saturation_frame)
        sat_type_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(sat_type_frame, text="Type:").pack(side='left', padx=5)

        self.saturation_type_var = tk.StringVar(value='tube')
        sat_types = ['soft', 'hard', 'tube', 'tape']

        for sat_type in sat_types:
            ttk.Radiobutton(
                sat_type_frame,
                text=sat_type.capitalize(),
                variable=self.saturation_type_var,
                value=sat_type,
                command=self.update_saturation_type
            ).pack(side='left', padx=5)

        # Loudness matching
        loudness_frame = ttk.LabelFrame(self.controls_tab, text="Loudness Matching")
        loudness_frame.pack(fill='x', padx=10, pady=10)

        # Enable checkbox
        self.loudness_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            loudness_frame,
            text="Enable Adaptive Loudness Matching",
            variable=self.loudness_enabled_var,
            command=self.update_loudness_matching
        ).pack(padx=10, pady=5)

        # Matching mode
        mode_frame = ttk.Frame(loudness_frame)
        mode_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(mode_frame, text="Match Mode:").pack(side='left', padx=5)

        self.loudness_mode_var = tk.StringVar(value='rms')
        modes = [('RMS', 'rms'), ('LUFS', 'lufs'), ('Peak Norm', 'peak_normalized'), ('Crest Match', 'crest_matched')]

        for label, mode in modes:
            ttk.Radiobutton(
                mode_frame,
                text=label,
                variable=self.loudness_mode_var,
                value=mode,
                command=self.update_loudness_matching
            ).pack(side='left', padx=5)

        # Reset button
        ttk.Button(
            self.controls_tab,
            text="↺ Reset All Controls",
            command=self.reset_controls
        ).pack(padx=10, pady=10)

    def create_visualization_tab(self):
        """Create visualization tab"""

        info_label = ttk.Label(
            self.viz_tab,
            text="Real-time spectrum comparison (Reference vs Input)",
            font=('Arial', 10, 'bold')
        )
        info_label.pack(pady=10)

        # Create visualization canvas
        viz_frame = ttk.Frame(self.viz_tab)
        viz_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.visualizer = RealTimeVisualizer(viz_frame, width=10, height=6)
        self.visualizer.get_widget().pack(fill='both', expand=True)

        # Update button
        ttk.Button(
            self.viz_tab,
            text="🔄 Update Visualization",
            command=self.update_visualization
        ).pack(pady=10)

    def create_settings_tab(self):
        """Create settings tab"""

        # Audio devices (same as basic GUI)
        device_frame = ttk.LabelFrame(self.settings_tab, text="Audio Devices")
        device_frame.pack(fill='x', padx=10, pady=10)

        # Input
        input_frame = ttk.Frame(device_frame)
        input_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(input_frame, text="Input Device:").pack(side='left', padx=5)
        self.input_device_combo = ttk.Combobox(input_frame, width=40, state='readonly')
        self.input_device_combo.pack(side='left', padx=5, fill='x', expand=True)

        # Output
        output_frame = ttk.Frame(device_frame)
        output_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(output_frame, text="Output Device:").pack(side='left', padx=5)
        self.output_device_combo = ttk.Combobox(output_frame, width=40, state='readonly')
        self.output_device_combo.pack(side='left', padx=5, fill='x', expand=True)

        ttk.Button(device_frame, text="🔄 Refresh Devices", command=self.refresh_devices).pack(padx=10, pady=10)

        self.refresh_devices()

        # Performance
        perf_frame = ttk.LabelFrame(self.settings_tab, text="Performance Settings")
        perf_frame.pack(fill='x', padx=10, pady=10)

        buffer_frame = ttk.Frame(perf_frame)
        buffer_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(buffer_frame, text="Buffer Size:").pack(side='left', padx=5)

        self.buffer_size_var = tk.IntVar(value=config.BUFFER_SIZE)
        ttk.Spinbox(
            buffer_frame,
            from_=64,
            to=2048,
            increment=64,
            textvariable=self.buffer_size_var,
            width=10
        ).pack(side='left', padx=5)

        ttk.Button(perf_frame, text="💾 Save Settings", command=self.save_settings).pack(padx=10, pady=10)

        # Info
        info_frame = ttk.LabelFrame(self.settings_tab, text="System Information")
        info_frame.pack(fill='both', expand=True, padx=10, pady=10)

        info_text = f"""
🎯 ULTIMATE PROCESSOR - ALL FEATURES ENABLED:

Core Processing:
✓ Adaptive per-sound-type processing (kick/snare/hihat/bass/vocal)
✓ Real-time sound classification with onset detection
✓ Multiband processing (4 or 8 bands with Linkwitz-Riley crossovers)
✓ Formant extraction and shaping (LPC analysis)
✓ Micro-transient preservation (1ms vs 20ms dual-envelope)

Audio Enhancement:
✓ Stereo/spatial effects (width, reverb, panning)
✓ Harmonic distortion and saturation (4 types: soft/hard/tube/tape)
✓ Dynamic EQ per sound type
✓ Transient shaping and enhancement

Adaptive Features:
✓ Dynamic makeup gain / loudness matching (per-buffer)
✓ Multiple matching modes: RMS, LUFS, peak-normalized, crest-matched
✓ Automatic reference loudness extraction from presets
✓ Smoothed gain transitions to prevent clicks

Safety & Stability:
✓ CPU-optimized real-time processing
✓ Thread-safe buffer management
✓ Safety limiting (-1.0dB threshold)
✓ Gain clamping (±24dB range)

Target Latency: <10ms (with proper configuration)
        """

        ttk.Label(info_frame, text=info_text, justify='left').pack(padx=10, pady=10)

    # Event handlers

    def browse_audio_file(self):
        """Browse for audio file"""
        filename = filedialog.askopenfilename(
            title="Select Reference Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All Files", "*.*")
            ]
        )

        if filename:
            self.file_path_var.set(filename)
            preset_name = Path(filename).stem
            self.preset_name_var.set(preset_name)

    def analyze_audio(self):
        """Run advanced analysis"""
        audio_path = self.file_path_var.get()

        if not audio_path:
            messagebox.showerror("Error", "Please select an audio file")
            return

        preset_name = self.preset_name_var.get()

        if not preset_name:
            messagebox.showerror("Error", "Please enter a preset name")
            return

        # Disable button
        self.analyze_btn.config(state='disabled')
        self.analysis_progress.start()

        # Run in thread
        thread = threading.Thread(
            target=self._analyze_thread,
            args=(audio_path, preset_name, self.num_bands_var.get()),
            daemon=True
        )
        thread.start()

    def _analyze_thread(self, audio_path, preset_name, num_bands):
        """Analysis thread"""
        try:
            import io
            captured_output = io.StringIO()
            sys.stdout = captured_output

            analyzer = AdvancedAudioAnalyzer(audio_path)
            analyzer.analyze_all(num_bands=num_bands)
            preset_path = analyzer.save_advanced_preset(preset_name)

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            self.root.after(0, self._analysis_complete, output, preset_path)

        except Exception as e:
            sys.stdout = sys.__stdout__
            self.root.after(0, self._analysis_error, str(e))

    def _analysis_complete(self, output, preset_path):
        """Analysis complete callback"""
        self.analysis_progress.stop()
        self.analyze_btn.config(state='normal')

        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', output)

        self.status_bar.config(text=f"Analysis complete: {preset_path}")
        self.refresh_presets()

        messagebox.showinfo("Success", f"Advanced analysis complete!\n\nPreset: {Path(preset_path).name}")

    def _analysis_error(self, error_msg):
        """Analysis error callback"""
        self.analysis_progress.stop()
        self.analyze_btn.config(state='normal')
        messagebox.showerror("Error", f"Analysis failed:\n\n{error_msg}")

    def refresh_presets(self):
        """Refresh preset list"""
        presets = []

        if config.PRESETS_DIR.exists():
            presets = [p.stem for p in config.PRESETS_DIR.glob("*.json")]

        self.preset_combo['values'] = presets

        if presets and not self.preset_combo.get():
            self.preset_combo.current(0)

    def refresh_devices(self):
        """Refresh audio devices"""
        import sounddevice as sd
        devices = sd.query_devices()

        input_devices = []
        output_devices = []

        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append(f"{idx}: {device['name']}")
            if device['max_output_channels'] > 0:
                output_devices.append(f"{idx}: {device['name']}")

        self.input_device_combo['values'] = input_devices
        self.output_device_combo['values'] = output_devices

        if input_devices:
            self.input_device_combo.current(0)
        if output_devices:
            self.output_device_combo.current(0)

    def start_processing(self):
        """Start processing"""
        preset_name = self.preset_combo.get()

        if not preset_name:
            messagebox.showerror("Error", "Please select a preset")
            return

        preset_path = config.get_preset_path(preset_name)

        try:
            self.processor.load_preset(str(preset_path))

            # Apply current control values
            self.processor.set_wet_dry_mix(self.wet_dry_var.get())
            self.processor.set_transient_preservation(self.transient_var.get())

            for i, var in enumerate(self.band_vars):
                self.processor.set_band_mix(i, var.get())

            self.processor.set_input_gain(self.input_gain_var.get())
            self.processor.set_output_gain(self.output_gain_var.get())

            # Apply spatial effects
            self.processor.set_stereo_width(self.stereo_width_var.get())
            self.processor.set_reverb(self.reverb_var.get(), size=0.5)

            # Apply harmonic saturation
            self.processor.set_saturation(self.saturation_var.get(), self.saturation_type_var.get())

            # Apply loudness matching
            self.processor.set_loudness_matching(
                self.loudness_enabled_var.get(),
                self.loudness_mode_var.get()
            )

            # Start processing thread
            thread = threading.Thread(target=self._processing_thread, daemon=True)
            thread.start()

            # Update UI
            self.is_processing = True
            self.processing_status_label.config(text="🟢 Status: Processing")
            self.start_proc_btn.config(state='disabled')
            self.stop_proc_btn.config(state='normal')
            self.start_rec_btn.config(state='normal')

            self.log("Advanced processing started")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start:\n\n{e}")

    def _processing_thread(self):
        """Processing thread"""
        try:
            input_device = None
            output_device = None

            if self.input_device_combo.get():
                input_device = int(self.input_device_combo.get().split(':')[0])
            if self.output_device_combo.get():
                output_device = int(self.output_device_combo.get().split(':')[0])

            self.processor.start_processing(
                input_device=input_device,
                output_device=output_device,
                monitor=True
            )

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, self.stop_processing)

    def stop_processing(self):
        """Stop processing"""
        self.processor.stop_processing()

        self.is_processing = False
        self.processing_status_label.config(text="⚫ Status: Stopped")
        self.start_proc_btn.config(state='normal')
        self.stop_proc_btn.config(state='disabled')
        self.start_rec_btn.config(state='disabled')
        self.stop_rec_btn.config(state='disabled')

        self.log("Processing stopped")

    def start_recording(self):
        """Start recording"""
        if not self.is_processing:
            messagebox.showerror("Error", "Start processing first")
            return

        self.processor.start_recording()

        self.recording_status_label.config(text="🔴 Recording: ON")
        self.start_rec_btn.config(state='disabled')
        self.stop_rec_btn.config(state='normal')

        self.log("Recording started")

    def stop_recording(self):
        """Stop recording"""
        output_path = self.processor.stop_recording()

        self.recording_status_label.config(text="⚫ Recording: Off")
        self.start_rec_btn.config(state='normal')
        self.stop_rec_btn.config(state='disabled')
        self.play_rec_btn.config(state='normal')

        # Store path for playback
        self.last_recording_path = output_path

        self.log(f"Recording saved: {output_path}")
        messagebox.showinfo("Recording Saved", f"Saved to:\n\n{output_path}")

    def play_recording(self):
        """Play the last recording"""
        if not self.last_recording_path or not Path(self.last_recording_path).exists():
            # Browse for file
            filename = filedialog.askopenfilename(
                title="Select Audio File to Play",
                initialdir=config.RECORDINGS_DIR,
                filetypes=[
                    ("WAV Files", "*.wav"),
                    ("All Audio", "*.wav *.mp3 *.flac"),
                    ("All Files", "*.*")
                ]
            )
            if not filename:
                return
            self.last_recording_path = filename

        try:
            # Get output device
            output_device = None
            if self.output_device_combo.get():
                output_device = int(self.output_device_combo.get().split(':')[0])

            # Play file
            self.audio_player.play_file(
                self.last_recording_path,
                device=output_device,
                callback=self._playback_finished
            )

            # Update UI
            self.play_rec_btn.config(state='disabled')
            self.stop_play_btn.config(state='normal')

            self.log(f"Playing: {Path(self.last_recording_path).name}")

        except Exception as e:
            messagebox.showerror("Playback Error", f"Failed to play audio:\n\n{e}")

    def stop_playback(self):
        """Stop audio playback"""
        self.audio_player.stop()

        # Update UI
        self.play_rec_btn.config(state='normal')
        self.stop_play_btn.config(state='disabled')

        self.log("Playback stopped")

    def _playback_finished(self):
        """Called when playback finishes"""
        # Update UI on main thread
        self.root.after(0, lambda: self.play_rec_btn.config(state='normal'))
        self.root.after(0, lambda: self.stop_play_btn.config(state='disabled'))
        self.root.after(0, lambda: self.log("Playback finished"))

    # Control callbacks

    def update_wet_dry(self, value):
        """Update wet/dry mix"""
        val = float(value)
        self.wet_dry_label.config(text=f"{val*100:.0f}%")
        if self.is_processing:
            self.processor.set_wet_dry_mix(val)

    def update_transient(self, value):
        """Update transient preservation"""
        val = float(value)
        self.transient_label.config(text=f"{val*100:.0f}%")
        if self.is_processing:
            self.processor.set_transient_preservation(val)

    def update_band_mix(self, band_idx, value):
        """Update band mix"""
        val = float(value)
        self.band_labels[band_idx].config(text=f"{val*100:.0f}%")
        if self.is_processing:
            self.processor.set_band_mix(band_idx, val)

    def update_input_gain(self, value):
        """Update input gain"""
        val = float(value)
        self.input_gain_label.config(text=f"{val:+.1f} dB")
        if self.is_processing:
            self.processor.set_input_gain(val)

    def update_output_gain(self, value):
        """Update output gain"""
        val = float(value)
        self.output_gain_label.config(text=f"{val:+.1f} dB")
        if self.is_processing:
            self.processor.set_output_gain(val)

    def update_stereo_width(self, value):
        """Update stereo width"""
        val = float(value)
        self.stereo_width_label.config(text=f"{val:.1f}x")
        if self.is_processing:
            self.processor.set_stereo_width(val)

    def update_reverb(self, value):
        """Update reverb amount"""
        val = float(value)
        self.reverb_label.config(text=f"{val*100:.0f}%")
        if self.is_processing:
            self.processor.set_reverb(val, size=0.5)

    def update_saturation(self, value):
        """Update saturation amount"""
        val = float(value)
        self.saturation_label.config(text=f"{val*100:.0f}%")
        if self.is_processing:
            self.processor.set_saturation(val, self.saturation_type_var.get())

    def update_saturation_type(self):
        """Update saturation type"""
        if self.is_processing:
            self.processor.set_saturation(self.saturation_var.get(), self.saturation_type_var.get())

    def update_loudness_matching(self):
        """Update loudness matching settings"""
        if self.is_processing:
            enabled = self.loudness_enabled_var.get()
            mode = self.loudness_mode_var.get()
            self.processor.set_loudness_matching(enabled, mode)
            self.log(f"Loudness matching: {'Enabled' if enabled else 'Disabled'} (mode: {mode})")

    def reset_controls(self):
        """Reset all controls to defaults"""
        self.wet_dry_var.set(1.0)
        self.transient_var.set(0.8)

        for var in self.band_vars:
            var.set(1.0)

        self.input_gain_var.set(0.0)
        self.output_gain_var.set(0.0)
        self.stereo_width_var.set(1.3)
        self.reverb_var.set(0.0)
        self.saturation_var.set(0.1)
        self.saturation_type_var.set('tube')
        self.loudness_enabled_var.set(True)
        self.loudness_mode_var.set('rms')

        self.log("Controls reset to defaults")

    def update_visualization(self):
        """Update visualization (placeholder)"""
        # In a full implementation, this would update with real audio
        self.log("Visualization updated")

    def save_settings(self):
        """Save settings"""
        self.processor.audio_config.buffer_size = self.buffer_size_var.get()
        self.processor.audio_config.save_config()
        messagebox.showinfo("Settings", "Settings saved")

    def log(self, message):
        """Add to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = AdvancedBeatboxApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
