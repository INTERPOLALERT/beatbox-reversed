"""
Beatbox Audio Style Transfer - Graphical User Interface
Simple GUI for analyzing reference audio and processing live input
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
import sys

from audio_analyzer import AudioAnalyzer
from live_processor import LiveProcessor
import config


class BeatboxApp:
    """Main application GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("Beatbox Audio Style Transfer")
        self.root.geometry("900x700")

        # Application state
        self.analyzer = None
        self.processor = LiveProcessor()
        self.current_preset = None
        self.is_processing = False

        # Create GUI
        self.create_widgets()

    def create_widgets(self):
        """Create all GUI widgets"""

        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=10)

        title_label = ttk.Label(
            title_frame,
            text="🎤 Beatbox Audio Style Transfer",
            font=('Arial', 16, 'bold')
        )
        title_label.pack()

        subtitle_label = ttk.Label(
            title_frame,
            text="Reverse engineer audio characteristics and apply to your live beatboxing",
            font=('Arial', 10)
        )
        subtitle_label.pack()

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Analysis
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="1. Analyze Audio")
        self.create_analysis_tab()

        # Tab 2: Live Processing
        self.processing_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.processing_tab, text="2. Live Processing")
        self.create_processing_tab()

        # Tab 3: Settings
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="⚙️ Settings")
        self.create_settings_tab()

        # Status bar
        self.status_bar = ttk.Label(
            self.root,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side='bottom', fill='x')

    def create_analysis_tab(self):
        """Create the analysis tab"""

        # Instructions
        inst_frame = ttk.LabelFrame(self.analysis_tab, text="Instructions")
        inst_frame.pack(fill='x', padx=10, pady=10)

        instructions = """
        1. Load a reference beatbox audio file (the sound you want to emulate)
        2. Click 'Analyze Audio' to extract all characteristics (EQ, compression, dynamics)
        3. Save the preset with a descriptive name
        4. Go to 'Live Processing' tab to apply the preset to your mic
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

        # Analysis controls
        control_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis")
        control_frame.pack(fill='x', padx=10, pady=10)

        # Preset name
        preset_frame = ttk.Frame(control_frame)
        preset_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(preset_frame, text="Preset Name:").pack(side='left', padx=5)

        self.preset_name_var = tk.StringVar()
        preset_entry = ttk.Entry(preset_frame, textvariable=self.preset_name_var, width=40)
        preset_entry.pack(side='left', padx=5, fill='x', expand=True)

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill='x', padx=10, pady=10)

        self.analyze_btn = ttk.Button(
            button_frame,
            text="🔍 Analyze Audio",
            command=self.analyze_audio
        )
        self.analyze_btn.pack(side='left', padx=5)

        # Progress
        self.analysis_progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.analysis_progress.pack(fill='x', padx=10, pady=10)

        # Results
        results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)

    def create_processing_tab(self):
        """Create the live processing tab"""

        # Instructions
        inst_frame = ttk.LabelFrame(self.processing_tab, text="Instructions")
        inst_frame.pack(fill='x', padx=10, pady=10)

        instructions = """
        1. Select a preset from the dropdown (or analyze audio first in the previous tab)
        2. Configure your audio devices in Settings tab
        3. Click 'Start Processing' to hear your voice with the applied effects
        4. Click 'Start Recording' to save your beatboxing to a file
        """

        ttk.Label(inst_frame, text=instructions, justify='left').pack(padx=10, pady=10)

        # Preset selection
        preset_frame = ttk.LabelFrame(self.processing_tab, text="Preset Selection")
        preset_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(preset_frame, text="Active Preset:").pack(side='left', padx=10, pady=10)

        self.preset_combo = ttk.Combobox(preset_frame, width=40, state='readonly')
        self.preset_combo.pack(side='left', padx=10, pady=10, fill='x', expand=True)

        refresh_btn = ttk.Button(preset_frame, text="🔄 Refresh", command=self.refresh_presets)
        refresh_btn.pack(side='left', padx=10, pady=10)

        self.refresh_presets()

        # Processing controls
        control_frame = ttk.LabelFrame(self.processing_tab, text="Live Processing Controls")
        control_frame.pack(fill='x', padx=10, pady=10)

        # Processing status
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill='x', padx=10, pady=10)

        self.processing_status_label = ttk.Label(
            status_frame,
            text="⚫ Status: Stopped",
            font=('Arial', 12, 'bold')
        )
        self.processing_status_label.pack(side='left', padx=10)

        # Processing buttons
        proc_btn_frame = ttk.Frame(control_frame)
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

        # Recording status
        rec_status_frame = ttk.Frame(rec_frame)
        rec_status_frame.pack(fill='x', padx=10, pady=10)

        self.recording_status_label = ttk.Label(
            rec_status_frame,
            text="⚫ Recording: Off",
            font=('Arial', 12)
        )
        self.recording_status_label.pack(side='left', padx=10)

        # Recording buttons
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

        # Log
        log_frame = ttk.LabelFrame(self.processing_tab, text="Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80)
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)

    def create_settings_tab(self):
        """Create the settings tab"""

        # Audio devices
        device_frame = ttk.LabelFrame(self.settings_tab, text="Audio Devices")
        device_frame.pack(fill='x', padx=10, pady=10)

        # Input device
        input_frame = ttk.Frame(device_frame)
        input_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(input_frame, text="Input Device (Microphone):").pack(side='left', padx=5)

        self.input_device_combo = ttk.Combobox(input_frame, width=40, state='readonly')
        self.input_device_combo.pack(side='left', padx=5, fill='x', expand=True)

        # Output device
        output_frame = ttk.Frame(device_frame)
        output_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(output_frame, text="Output Device (Headphones):").pack(side='left', padx=5)

        self.output_device_combo = ttk.Combobox(output_frame, width=40, state='readonly')
        self.output_device_combo.pack(side='left', padx=5, fill='x', expand=True)

        # Refresh devices button
        refresh_dev_btn = ttk.Button(device_frame, text="🔄 Refresh Devices", command=self.refresh_devices)
        refresh_dev_btn.pack(padx=10, pady=10)

        self.refresh_devices()

        # Performance settings
        perf_frame = ttk.LabelFrame(self.settings_tab, text="Performance Settings")
        perf_frame.pack(fill='x', padx=10, pady=10)

        # Buffer size
        buffer_frame = ttk.Frame(perf_frame)
        buffer_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(buffer_frame, text="Buffer Size (samples):").pack(side='left', padx=5)

        self.buffer_size_var = tk.IntVar(value=config.BUFFER_SIZE)
        buffer_spinbox = ttk.Spinbox(
            buffer_frame,
            from_=64,
            to=2048,
            increment=64,
            textvariable=self.buffer_size_var,
            width=10
        )
        buffer_spinbox.pack(side='left', padx=5)

        ttk.Label(buffer_frame, text="(Lower = less latency, higher CPU)").pack(side='left', padx=5)

        # Save settings button
        save_btn = ttk.Button(perf_frame, text="💾 Save Settings", command=self.save_settings)
        save_btn.pack(padx=10, pady=10)

        # Info
        info_frame = ttk.LabelFrame(self.settings_tab, text="Information")
        info_frame.pack(fill='both', expand=True, padx=10, pady=10)

        info_text = f"""
Sample Rate: {config.SAMPLE_RATE} Hz
Default Buffer: {config.BUFFER_SIZE} samples
Estimated Latency: ~{config.BUFFER_SIZE / config.SAMPLE_RATE * 1000:.1f} ms

Directories:
  Presets: {config.PRESETS_DIR}
  Recordings: {config.RECORDINGS_DIR}

Tips for Low Latency:
  - Use ASIO drivers if available (Windows)
  - Use smaller buffer sizes (64-256 samples)
  - Close other applications
  - Use dedicated audio interface if possible
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

            # Auto-generate preset name from filename
            preset_name = Path(filename).stem
            self.preset_name_var.set(preset_name)

    def analyze_audio(self):
        """Analyze audio file in background thread"""
        audio_path = self.file_path_var.get()

        if not audio_path:
            messagebox.showerror("Error", "Please select an audio file first")
            return

        preset_name = self.preset_name_var.get()

        if not preset_name:
            messagebox.showerror("Error", "Please enter a preset name")
            return

        # Disable button and start progress
        self.analyze_btn.config(state='disabled')
        self.analysis_progress.start()

        # Run analysis in thread
        thread = threading.Thread(
            target=self._analyze_thread,
            args=(audio_path, preset_name),
            daemon=True
        )
        thread.start()

    def _analyze_thread(self, audio_path, preset_name):
        """Background thread for analysis"""
        try:
            # Redirect stdout to capture prints
            import io
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Run analysis
            analyzer = AudioAnalyzer(audio_path)
            analyzer.analyze_all()
            preset_path = analyzer.save_preset(preset_name)

            # Get output
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            # Update GUI
            self.root.after(0, self._analysis_complete, output, preset_path)

        except Exception as e:
            sys.stdout = sys.__stdout__
            self.root.after(0, self._analysis_error, str(e))

    def _analysis_complete(self, output, preset_path):
        """Called when analysis completes"""
        self.analysis_progress.stop()
        self.analyze_btn.config(state='normal')

        # Show results
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', output)

        # Update status
        self.status_bar.config(text=f"Analysis complete: {preset_path}")

        # Refresh presets
        self.refresh_presets()

        messagebox.showinfo("Success", f"Analysis complete!\n\nPreset saved: {Path(preset_path).name}")

    def _analysis_error(self, error_msg):
        """Called when analysis fails"""
        self.analysis_progress.stop()
        self.analyze_btn.config(state='normal')

        messagebox.showerror("Analysis Error", f"Error during analysis:\n\n{error_msg}")

    def refresh_presets(self):
        """Refresh list of available presets"""
        presets = []

        if config.PRESETS_DIR.exists():
            presets = [p.stem for p in config.PRESETS_DIR.glob("*.json")]

        self.preset_combo['values'] = presets

        if presets and not self.preset_combo.get():
            self.preset_combo.current(0)

    def refresh_devices(self):
        """Refresh audio device lists"""
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

        # Select defaults
        if input_devices:
            self.input_device_combo.current(0)
        if output_devices:
            self.output_device_combo.current(0)

    def save_settings(self):
        """Save application settings"""
        self.processor.audio_config.buffer_size = self.buffer_size_var.get()
        self.processor.audio_config.save_config()

        messagebox.showinfo("Settings", "Settings saved successfully")

    def start_processing(self):
        """Start live processing"""
        preset_name = self.preset_combo.get()

        if not preset_name:
            messagebox.showerror("Error", "Please select a preset first")
            return

        # Load preset
        preset_path = config.get_preset_path(preset_name)

        try:
            self.processor.load_preset(preset_path)

            # Start processing in thread
            thread = threading.Thread(
                target=self._processing_thread,
                daemon=True
            )
            thread.start()

            # Update UI
            self.is_processing = True
            self.processing_status_label.config(text="🟢 Status: Processing")
            self.start_proc_btn.config(state='disabled')
            self.stop_proc_btn.config(state='normal')
            self.start_rec_btn.config(state='normal')

            self.log("Processing started")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start processing:\n\n{e}")

    def _processing_thread(self):
        """Background thread for processing"""
        try:
            # Get device indices
            input_device = None
            output_device = None

            if self.input_device_combo.get():
                input_device = int(self.input_device_combo.get().split(':')[0])

            if self.output_device_combo.get():
                output_device = int(self.output_device_combo.get().split(':')[0])

            # Start processing
            self.processor.start_processing(
                input_device=input_device,
                output_device=output_device,
                monitor=True
            )

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Processing Error", str(e)))
            self.root.after(0, self.stop_processing)

    def stop_processing(self):
        """Stop live processing"""
        self.processor.stop_processing()

        # Update UI
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

        # Update UI
        self.recording_status_label.config(text="🔴 Recording: ON")
        self.start_rec_btn.config(state='disabled')
        self.stop_rec_btn.config(state='normal')

        self.log("Recording started")

    def stop_recording(self):
        """Stop recording"""
        output_path = self.processor.stop_recording()

        # Update UI
        self.recording_status_label.config(text="⚫ Recording: Off")
        self.start_rec_btn.config(state='normal')
        self.stop_rec_btn.config(state='disabled')

        self.log(f"Recording saved: {output_path}")

        messagebox.showinfo("Recording Saved", f"Recording saved to:\n\n{output_path}")

    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = BeatboxApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
