"""
Visualization System
Real-time spectrograms, EQ curves, and audio analysis displays
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for embedding
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import librosa
import config


class SpectrumAnalyzer:
    """Real-time spectrum analyzer visualization"""

    def __init__(self, sample_rate=44100, fft_size=2048):
        """
        Initialize spectrum analyzer

        Args:
            sample_rate: Sample rate in Hz
            fft_size: FFT size
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=fft_size)

    def compute_spectrum(self, audio: np.ndarray) -> tuple:
        """
        Compute frequency spectrum

        Args:
            audio: Audio signal

        Returns:
            Tuple of (frequencies, magnitudes_db)
        """
        # Zero-pad if necessary
        if len(audio) < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - len(audio)))

        # Compute FFT
        fft_data = np.fft.rfft(audio, n=self.fft_size)
        magnitudes = np.abs(fft_data)

        # Convert to dB
        magnitudes_db = 20 * np.log10(magnitudes + 1e-10)

        return self.freqs, magnitudes_db


class EQCurveVisualizer:
    """Visualize EQ curves"""

    @staticmethod
    def plot_eq_curve(eq_bands: list, ax=None):
        """
        Plot EQ curve

        Args:
            eq_bands: List of EQ band dicts with frequency, gain_db
            ax: Matplotlib axis (creates new if None)

        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        # Extract frequencies and gains
        freqs = [band['frequency'] for band in eq_bands]
        gains = [band['gain_db'] for band in eq_bands]

        # Plot bars
        colors = ['red' if g < 0 else 'green' for g in gains]
        ax.bar(range(len(freqs)), gains, color=colors, alpha=0.7)

        # Formatting
        ax.set_xticks(range(len(freqs)))
        ax.set_xticklabels([f"{f}Hz" if f < 1000 else f"{f//1000}kHz" for f in freqs], rotation=45)
        ax.set_ylabel('Gain (dB)')
        ax.set_title('EQ Curve')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        return ax


class SpectrogramVisualizer:
    """Spectrogram visualization"""

    @staticmethod
    def compute_spectrogram(audio: np.ndarray, sr: int = 44100, n_fft: int = 2048) -> tuple:
        """
        Compute spectrogram

        Args:
            audio: Audio signal
            sr: Sample rate
            n_fft: FFT size

        Returns:
            Tuple of (times, frequencies, spectrogram_db)
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=n_fft // 4)

        # Convert to dB
        spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        # Time and frequency axes
        times = librosa.frames_to_time(np.arange(spec_db.shape[1]), sr=sr, hop_length=n_fft // 4)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        return times, freqs, spec_db

    @staticmethod
    def plot_spectrogram(audio: np.ndarray, sr: int = 44100, ax=None):
        """
        Plot spectrogram

        Args:
            audio: Audio signal
            sr: Sample rate
            ax: Matplotlib axis

        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        times, freqs, spec_db = SpectrogramVisualizer.compute_spectrogram(audio, sr)

        # Plot
        im = ax.pcolormesh(times, freqs, spec_db, shading='gouraud', cmap='viridis')

        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Spectrogram')
        ax.set_ylim([0, 8000])  # Focus on relevant frequencies

        # Colorbar
        plt.colorbar(im, ax=ax, format='%+2.0f dB')

        return ax


class ComparisonVisualizer:
    """Compare reference vs input audio"""

    @staticmethod
    def plot_comparison(reference_audio: np.ndarray,
                       input_audio: np.ndarray,
                       sr: int = 44100,
                       fig=None):
        """
        Plot comparison of reference and input

        Args:
            reference_audio: Reference audio
            input_audio: Input audio
            sr: Sample rate
            fig: Matplotlib figure

        Returns:
            Matplotlib figure
        """
        if fig is None:
            fig = plt.figure(figsize=(12, 8))

        # Spectrogram comparison
        ax1 = fig.add_subplot(2, 2, 1)
        SpectrogramVisualizer.plot_spectrogram(reference_audio, sr, ax1)
        ax1.set_title('Reference Spectrogram')

        ax2 = fig.add_subplot(2, 2, 2)
        SpectrogramVisualizer.plot_spectrogram(input_audio, sr, ax2)
        ax2.set_title('Input Spectrogram')

        # Spectrum comparison
        ax3 = fig.add_subplot(2, 2, 3)

        spectrum_analyzer = SpectrumAnalyzer(sample_rate=sr)

        ref_freqs, ref_mags = spectrum_analyzer.compute_spectrum(reference_audio)
        input_freqs, input_mags = spectrum_analyzer.compute_spectrum(input_audio)

        ax3.plot(ref_freqs, ref_mags, label='Reference', alpha=0.7)
        ax3.plot(input_freqs, input_mags, label='Input', alpha=0.7)
        ax3.set_xlim([20, 8000])
        ax3.set_xscale('log')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Magnitude (dB)')
        ax3.set_title('Spectrum Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Waveform comparison
        ax4 = fig.add_subplot(2, 2, 4)

        ref_time = np.linspace(0, len(reference_audio) / sr, len(reference_audio))
        input_time = np.linspace(0, len(input_audio) / sr, len(input_audio))

        ax4.plot(ref_time, reference_audio, label='Reference', alpha=0.7)
        ax4.plot(input_time, input_audio, label='Input', alpha=0.7)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Amplitude')
        ax4.set_title('Waveform Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()

        return fig


class MultibandVisualizer:
    """Visualize multiband analysis"""

    @staticmethod
    def plot_multiband_analysis(multiband_data: dict, ax=None):
        """
        Plot multiband analysis results

        Args:
            multiband_data: Multiband analysis dict
            ax: Matplotlib axis

        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        bands = multiband_data.get('bands', [])

        if len(bands) == 0:
            return ax

        # Extract data
        band_labels = []
        energy_ratios = []
        relative_gains = []

        for band in bands:
            freq_range = band['freq_range']
            band_labels.append(f"{freq_range[0]:.0f}-\n{freq_range[1]:.0f}Hz")
            energy_ratios.append(band['energy_ratio'] * 100)  # Convert to percentage
            relative_gains.append(band['relative_gain_db'])

        x = np.arange(len(bands))
        width = 0.35

        # Plot energy distribution
        ax2 = ax.twinx()

        bars1 = ax.bar(x - width/2, relative_gains, width, label='Relative Gain (dB)', alpha=0.7)
        bars2 = ax2.bar(x + width/2, energy_ratios, width, label='Energy %', alpha=0.7, color='orange')

        # Formatting
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Relative Gain (dB)', color='blue')
        ax2.set_ylabel('Energy Distribution (%)', color='orange')
        ax.set_title('Multiband Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels)
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3)

        # Legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        return ax


class RealTimeVisualizer:
    """Real-time visualization for GUI embedding"""

    def __init__(self, master, width=6, height=4):
        """
        Initialize real-time visualizer

        Args:
            master: Tkinter parent widget
            width: Figure width in inches
            height: Figure height in inches
        """
        self.master = master

        # Create figure
        self.figure = Figure(figsize=(width, height), dpi=100)
        self.ax = self.figure.add_subplot(111)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()

        # Spectrum analyzer
        self.spectrum_analyzer = SpectrumAnalyzer()

        # Initialize empty plot
        self.line_ref, = self.ax.plot([], [], label='Reference', alpha=0.7, linewidth=2)
        self.line_input, = self.ax.plot([], [], label='Input', alpha=0.7, linewidth=2)

        self.ax.set_xlim([20, 8000])
        self.ax.set_ylim([-80, 0])
        self.ax.set_xscale('log')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Magnitude (dB)')
        self.ax.set_title('Real-Time Spectrum Comparison')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        self.figure.tight_layout()

    def update(self, reference_audio: np.ndarray = None, input_audio: np.ndarray = None):
        """
        Update visualization

        Args:
            reference_audio: Reference audio buffer
            input_audio: Input audio buffer
        """
        if reference_audio is not None:
            freqs, mags = self.spectrum_analyzer.compute_spectrum(reference_audio)
            self.line_ref.set_data(freqs, mags)

        if input_audio is not None:
            freqs, mags = self.spectrum_analyzer.compute_spectrum(input_audio)
            self.line_input.set_data(freqs, mags)

        self.canvas.draw()

    def get_widget(self):
        """Get the canvas widget for packing"""
        return self.canvas_widget


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization modules...")

    # Create test audio
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Multi-frequency signal
    test_audio = (
        np.sin(2 * np.pi * 100 * t) +
        np.sin(2 * np.pi * 500 * t) +
        np.sin(2 * np.pi * 2000 * t) +
        np.sin(2 * np.pi * 8000 * t)
    )

    # Test EQ curve plot
    eq_bands = [
        {'frequency': 100, 'gain_db': 3.0},
        {'frequency': 500, 'gain_db': -2.0},
        {'frequency': 2000, 'gain_db': 1.5},
        {'frequency': 8000, 'gain_db': 4.0}
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # EQ Curve
    EQCurveVisualizer.plot_eq_curve(eq_bands, axes[0, 0])

    # Spectrogram
    SpectrogramVisualizer.plot_spectrogram(test_audio, sr, axes[0, 1])

    # Spectrum
    spectrum_analyzer = SpectrumAnalyzer(sample_rate=sr)
    freqs, mags = spectrum_analyzer.compute_spectrum(test_audio)
    axes[1, 0].plot(freqs, mags)
    axes[1, 0].set_xlim([20, 10000])
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude (dB)')
    axes[1, 0].set_title('Spectrum')
    axes[1, 0].grid(True, alpha=0.3)

    # Multiband
    multiband_data = {
        'num_bands': 4,
        'bands': [
            {'band_index': 0, 'freq_range': [20, 200], 'energy_ratio': 0.3, 'relative_gain_db': 2.0},
            {'band_index': 1, 'freq_range': [200, 1000], 'energy_ratio': 0.25, 'relative_gain_db': -1.0},
            {'band_index': 2, 'freq_range': [1000, 4000], 'energy_ratio': 0.25, 'relative_gain_db': 1.5},
            {'band_index': 3, 'freq_range': [4000, 20000], 'energy_ratio': 0.2, 'relative_gain_db': 3.0}
        ]
    }

    MultibandVisualizer.plot_multiband_analysis(multiband_data, axes[1, 1])

    plt.tight_layout()
    plt.savefig('test_visualizations.png', dpi=150)
    print("Saved test_visualizations.png")
