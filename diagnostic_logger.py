"""
Diagnostic Logger Module
Per-buffer real-time diagnostics and logging system
Tracks applied EQ, compression, transient shaping, and frequency content
"""
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import csv


class DiagnosticLogger:
    """
    Real-time diagnostic logger for audio processing
    Logs per-buffer processing parameters and audio characteristics
    """

    def __init__(self, enabled: bool = False, log_dir: str = "logs"):
        """
        Initialize diagnostic logger

        Args:
            enabled: Enable logging
            log_dir: Directory for log files
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Log buffers
        self.buffer_logs = []
        self.max_buffer_logs = 1000  # Keep last 1000 buffers in memory

        # Statistics
        self.total_buffers_processed = 0
        self.session_start_time = datetime.now()

        # Current session log file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_log_path = None
        self.json_log_path = None

        if self.enabled:
            self._initialize_log_files()

    def _initialize_log_files(self):
        """Initialize log files for current session"""
        # CSV for time-series data
        self.csv_log_path = self.log_dir / f"diagnostic_log_{self.session_id}.csv"

        # JSON for summary data
        self.json_log_path = self.log_dir / f"diagnostic_summary_{self.session_id}.json"

        # Write CSV header
        with open(self.csv_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'buffer_index',
                'timestamp',
                'rms_db',
                'peak_db',
                'crest_db',
                'lufs',
                'spectral_centroid',
                'spectral_rolloff',
                'zero_crossing_rate',
                'applied_gain_db',
                'detected_sound_type',
                'eq_applied',
                'compression_applied',
                'transient_amount',
                'saturation_amount',
                'reverb_amount',
                'stereo_width'
            ])

        print(f"Diagnostic logging enabled: {self.csv_log_path}")

    def log_buffer(self, buffer_data: Dict):
        """
        Log data for a single buffer

        Args:
            buffer_data: Dictionary with buffer processing data
        """
        if not self.enabled:
            return

        self.total_buffers_processed += 1
        buffer_data['buffer_index'] = self.total_buffers_processed
        buffer_data['timestamp'] = (datetime.now() - self.session_start_time).total_seconds()

        # Add to memory buffer
        self.buffer_logs.append(buffer_data.copy())

        # Trim if too large
        if len(self.buffer_logs) > self.max_buffer_logs:
            self.buffer_logs.pop(0)

        # Write to CSV
        if self.csv_log_path:
            self._append_to_csv(buffer_data)

    def _append_to_csv(self, buffer_data: Dict):
        """Append buffer data to CSV log"""
        with open(self.csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                buffer_data.get('buffer_index', 0),
                f"{buffer_data.get('timestamp', 0.0):.3f}",
                f"{buffer_data.get('rms_db', 0.0):.2f}",
                f"{buffer_data.get('peak_db', 0.0):.2f}",
                f"{buffer_data.get('crest_db', 0.0):.2f}",
                f"{buffer_data.get('lufs', 0.0):.2f}",
                f"{buffer_data.get('spectral_centroid', 0.0):.1f}",
                f"{buffer_data.get('spectral_rolloff', 0.0):.1f}",
                f"{buffer_data.get('zero_crossing_rate', 0.0):.4f}",
                f"{buffer_data.get('applied_gain_db', 0.0):.2f}",
                buffer_data.get('detected_sound_type', 'unknown'),
                buffer_data.get('eq_applied', 'none'),
                buffer_data.get('compression_applied', 'none'),
                f"{buffer_data.get('transient_amount', 0.0):.2f}",
                f"{buffer_data.get('saturation_amount', 0.0):.2f}",
                f"{buffer_data.get('reverb_amount', 0.0):.2f}",
                f"{buffer_data.get('stereo_width', 1.0):.2f}"
            ])

    def get_buffer_statistics(self, last_n: int = 100) -> Dict:
        """
        Get statistics for last N buffers

        Args:
            last_n: Number of recent buffers to analyze

        Returns:
            Dictionary with statistical summary
        """
        if len(self.buffer_logs) == 0:
            return {}

        recent_logs = self.buffer_logs[-last_n:]

        # Extract metrics
        rms_values = [log.get('rms_db', 0) for log in recent_logs]
        peak_values = [log.get('peak_db', 0) for log in recent_logs]
        crest_values = [log.get('crest_db', 0) for log in recent_logs]
        gain_values = [log.get('applied_gain_db', 0) for log in recent_logs]

        # Sound type distribution
        sound_types = [log.get('detected_sound_type', 'unknown') for log in recent_logs]
        sound_type_counts = {}
        for st in sound_types:
            sound_type_counts[st] = sound_type_counts.get(st, 0) + 1

        stats = {
            'num_buffers': len(recent_logs),
            'rms_mean_db': np.mean(rms_values),
            'rms_std_db': np.std(rms_values),
            'rms_min_db': np.min(rms_values),
            'rms_max_db': np.max(rms_values),
            'peak_mean_db': np.mean(peak_values),
            'peak_max_db': np.max(peak_values),
            'crest_mean_db': np.mean(crest_values),
            'gain_mean_db': np.mean(gain_values),
            'gain_std_db': np.std(gain_values),
            'sound_type_distribution': sound_type_counts,
            'total_buffers_processed': self.total_buffers_processed
        }

        return stats

    def save_summary(self):
        """Save summary JSON file"""
        if not self.enabled or not self.json_log_path:
            return

        stats = self.get_buffer_statistics(last_n=len(self.buffer_logs))

        summary = {
            'session_id': self.session_id,
            'session_start_time': self.session_start_time.isoformat(),
            'total_buffers_processed': self.total_buffers_processed,
            'duration_seconds': (datetime.now() - self.session_start_time).total_seconds(),
            'statistics': stats
        }

        with open(self.json_log_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nDiagnostic summary saved: {self.json_log_path}")

    def print_live_stats(self, last_n: int = 10):
        """Print live statistics to console"""
        if len(self.buffer_logs) < last_n:
            return

        stats = self.get_buffer_statistics(last_n=last_n)

        print(f"\n--- Live Stats (last {last_n} buffers) ---")
        print(f"RMS: {stats['rms_mean_db']:+.1f} ± {stats['rms_std_db']:.1f} dB")
        print(f"Peak: {stats['peak_max_db']:+.1f} dB")
        print(f"Crest: {stats['crest_mean_db']:.1f} dB")
        print(f"Gain: {stats['gain_mean_db']:+.1f} ± {stats['gain_std_db']:.1f} dB")

        if stats.get('sound_type_distribution'):
            print("Sound types:", stats['sound_type_distribution'])

    def enable(self):
        """Enable logging"""
        if not self.enabled:
            self.enabled = True
            self._initialize_log_files()

    def disable(self):
        """Disable logging"""
        if self.enabled:
            self.save_summary()
            self.enabled = False


class PerBufferAnalyzer:
    """
    Analyzes audio characteristics per buffer for diagnostics
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize analyzer

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

    def analyze_buffer(self, audio_buffer: np.ndarray) -> Dict:
        """
        Analyze audio buffer characteristics

        Args:
            audio_buffer: Input audio buffer

        Returns:
            Dictionary with analysis results
        """
        # Basic loudness metrics
        rms = np.sqrt(np.mean(audio_buffer ** 2))
        peak = np.max(np.abs(audio_buffer))
        crest_factor = peak / (rms + 1e-10)

        rms_db = 20 * np.log10(rms + 1e-10)
        peak_db = 20 * np.log10(peak + 1e-10)
        crest_db = 20 * np.log10(crest_factor)

        # Spectral features (fast approximations)
        fft = np.fft.rfft(audio_buffer)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio_buffer), 1/self.sample_rate)

        # Spectral centroid (brightness)
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0.0

        # Spectral rolloff (where 85% of energy is below)
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
        else:
            spectral_rolloff = 0.0

        # Zero crossing rate (roughness/noisiness)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_buffer)))) / 2
        zcr = zero_crossings / len(audio_buffer)

        return {
            'rms': rms,
            'rms_db': rms_db,
            'peak': peak,
            'peak_db': peak_db,
            'crest_factor': crest_factor,
            'crest_db': crest_db,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zcr,
            'dominant_freq_hz': freqs[np.argmax(magnitude)] if len(magnitude) > 0 else 0.0
        }

    def analyze_frequency_content(self, audio_buffer: np.ndarray,
                                  bands: List[Tuple[float, float]] = None) -> Dict:
        """
        Analyze energy distribution across frequency bands

        Args:
            audio_buffer: Input audio buffer
            bands: List of (low_freq, high_freq) tuples

        Returns:
            Dictionary with per-band energy
        """
        if bands is None:
            # Default bands: sub, bass, low-mid, mid, high-mid, high
            bands = [
                (20, 60),      # Sub bass
                (60, 250),     # Bass
                (250, 500),    # Low-mid
                (500, 2000),   # Mid
                (2000, 6000),  # High-mid
                (6000, 20000)  # High
            ]

        fft = np.fft.rfft(audio_buffer)
        magnitude = np.abs(fft) ** 2  # Power
        freqs = np.fft.rfftfreq(len(audio_buffer), 1/self.sample_rate)

        band_energy = {}
        total_energy = np.sum(magnitude)

        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            energy = np.sum(magnitude[mask])
            energy_ratio = energy / (total_energy + 1e-10)
            energy_db = 10 * np.log10(energy + 1e-10)

            band_energy[f"{low}-{high}Hz"] = {
                'energy': float(energy),
                'energy_db': float(energy_db),
                'energy_ratio': float(energy_ratio)
            }

        return band_energy


def demo_diagnostic_logger():
    """Demo diagnostic logging"""
    print("=" * 60)
    print("DIAGNOSTIC LOGGER DEMO")
    print("=" * 60)

    # Create logger
    logger = DiagnosticLogger(enabled=True)
    analyzer = PerBufferAnalyzer(sample_rate=44100)

    print("\nSimulating buffer processing...")

    # Simulate processing 20 buffers
    for i in range(20):
        # Generate test buffer
        t = np.linspace(0, 0.01, 441)  # 10ms buffer
        freq = 440 * (1 + i * 0.1)  # Varying frequency
        amplitude = 0.1 * (1 + np.random.randn() * 0.2)  # Varying amplitude
        test_buffer = amplitude * np.sin(2 * np.pi * freq * t)

        # Analyze buffer
        analysis = analyzer.analyze_buffer(test_buffer)

        # Add processing info
        analysis.update({
            'detected_sound_type': ['kick', 'snare', 'hihat', 'other'][i % 4],
            'applied_gain_db': np.random.randn() * 2,
            'eq_applied': 'adaptive',
            'compression_applied': 'multiband',
            'transient_amount': 0.5,
            'saturation_amount': 0.1,
            'reverb_amount': 0.2,
            'stereo_width': 1.3
        })

        # Log buffer
        logger.log_buffer(analysis)

        if (i + 1) % 10 == 0:
            logger.print_live_stats(last_n=10)

    # Get final statistics
    stats = logger.get_buffer_statistics()

    print(f"\n\nFinal Statistics:")
    print(f"  Total buffers: {stats['total_buffers_processed']}")
    print(f"  RMS: {stats['rms_mean_db']:.1f} ± {stats['rms_std_db']:.1f} dB")
    print(f"  Peak: {stats['peak_max_db']:.1f} dB")
    print(f"  Sound types: {stats['sound_type_distribution']}")

    # Save summary
    logger.save_summary()

    print("\n" + "=" * 60)
    print("Diagnostic logging enabled and working!")
    print("=" * 60)


if __name__ == "__main__":
    demo_diagnostic_logger()
