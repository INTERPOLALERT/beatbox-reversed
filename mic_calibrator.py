"""
Microphone Calibration Module
Analyzes user's microphone characteristics to create a mic profile
This profile is used to adapt presets for accurate results
"""
import librosa
import numpy as np
import pyloudnorm as pyln
from scipy import signal
from typing import Dict, Optional
import json
from pathlib import Path
import config


class MicCalibrator:
    """
    Analyzes user's microphone input to create a calibration profile
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize mic calibrator

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(sample_rate)  # For LUFS measurement

    def calibrate(self, calibration_audio: np.ndarray) -> Dict:
        """
        Analyze calibration audio (user's mic recording) and create profile

        Args:
            calibration_audio: 5-10 second recording of user's normal beatboxing/speaking

        Returns:
            Microphone profile dictionary
        """
        print("\n" + "="*60)
        print("MICROPHONE CALIBRATION")
        print("="*60)
        print("\nAnalyzing your microphone characteristics...")

        # 1. Loudness Analysis
        loudness_profile = self._analyze_loudness(calibration_audio)
        print(f"\n[1/5] Loudness Analysis")
        print(f"  RMS Level: {loudness_profile['rms_db']:.1f} dB")
        print(f"  LUFS: {loudness_profile['lufs']:.1f} LUFS")
        print(f"  Peak Level: {loudness_profile['peak_db']:.1f} dB")

        # 2. Spectral Tilt Analysis (tonal bias)
        spectral_profile = self._analyze_spectral_tilt(calibration_audio)
        print(f"\n[2/5] Spectral Balance")
        print(f"  Overall Tilt: {spectral_profile['tilt_db_per_octave']:+.2f} dB/oct ({spectral_profile['character']})")
        print(f"  Low Freq Bias: {spectral_profile['low_freq_bias_db']:+.1f} dB")
        print(f"  Mid Freq Bias: {spectral_profile['mid_freq_bias_db']:+.1f} dB")
        print(f"  High Freq Bias: {spectral_profile['high_freq_bias_db']:+.1f} dB")

        # 3. Dynamic Range Analysis
        dynamics_profile = self._analyze_dynamics(calibration_audio)
        print(f"\n[3/5] Dynamic Characteristics")
        print(f"  Crest Factor: {dynamics_profile['crest_factor_db']:.1f} dB")
        print(f"  Dynamic Range: {dynamics_profile['dynamic_range_db']:.1f} dB")
        print(f"  Transient Character: {dynamics_profile['transient_strength']}")

        # 4. Noise Floor Analysis
        noise_profile = self._analyze_noise_floor(calibration_audio)
        print(f"\n[4/5] Noise Floor")
        print(f"  Noise Floor: {noise_profile['noise_floor_db']:.1f} dB")
        print(f"  SNR: {noise_profile['snr_db']:.1f} dB")
        print(f"  Noise Type: {noise_profile['noise_type']}")

        # 5. Frequency-Specific Loudness (per-band analysis)
        band_profile = self._analyze_frequency_bands(calibration_audio)
        print(f"\n[5/5] Per-Band Analysis")
        for band_name, band_data in band_profile.items():
            print(f"  {band_name:12s}: {band_data['level_db']:+5.1f} dB")

        # Compile complete profile
        mic_profile = {
            'loudness': loudness_profile,
            'spectral': spectral_profile,
            'dynamics': dynamics_profile,
            'noise': noise_profile,
            'frequency_bands': band_profile,
            'sample_rate': self.sample_rate,
            'calibration_duration': len(calibration_audio) / self.sample_rate
        }

        print("\n" + "="*60)
        print("✅ CALIBRATION COMPLETE")
        print("="*60)

        # Print recommendations
        self._print_recommendations(mic_profile)

        return mic_profile

    def _analyze_loudness(self, audio: np.ndarray) -> Dict:
        """Analyze loudness characteristics"""
        # RMS level
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Peak level
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak + 1e-10)

        # LUFS (integrated loudness)
        try:
            lufs = self.meter.integrated_loudness(audio)
        except:
            # Fallback if audio is too quiet
            lufs = rms_db

        # Loudness range (LRA)
        percentile_95 = 20 * np.log10(np.percentile(np.abs(audio), 95) + 1e-10)
        percentile_10 = 20 * np.log10(np.percentile(np.abs(audio), 10) + 1e-10)
        lra_db = percentile_95 - percentile_10

        return {
            'rms_db': float(rms_db),
            'peak_db': float(peak_db),
            'lufs': float(lufs),
            'lra_db': float(lra_db)
        }

    def _analyze_spectral_tilt(self, audio: np.ndarray) -> Dict:
        """Analyze spectral tilt and frequency balance"""
        # Compute average spectrum
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag_spectrum = np.mean(np.abs(stft), axis=1)
        mag_spectrum_db = 20 * np.log10(mag_spectrum + 1e-10)

        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)

        # Define frequency bands
        low_mask = (freqs >= 80) & (freqs <= 250)
        mid_mask = (freqs >= 250) & (freqs <= 2000)
        high_mask = (freqs >= 2000) & (freqs <= 8000)

        # Average level in each band
        low_level = np.mean(mag_spectrum_db[low_mask])
        mid_level = np.mean(mag_spectrum_db[mid_mask])
        high_level = np.mean(mag_spectrum_db[high_mask])

        # Calculate tilt (dB per octave)
        # Use linear regression on log-frequency scale
        log_freqs = np.log2(freqs[freqs > 0])
        spectrum_for_fit = mag_spectrum_db[freqs > 0]

        # Fit line to spectrum
        valid_idx = np.isfinite(spectrum_for_fit)
        if np.sum(valid_idx) > 10:
            slope, intercept = np.polyfit(log_freqs[valid_idx], spectrum_for_fit[valid_idx], 1)
            tilt_db_per_octave = float(slope)
        else:
            tilt_db_per_octave = 0.0

        # Characterize tilt
        if tilt_db_per_octave > 1.0:
            character = "bright"
        elif tilt_db_per_octave < -1.0:
            character = "dark"
        else:
            character = "balanced"

        # Calculate frequency bias relative to flat response
        # (positive = louder than average, negative = quieter)
        avg_level = np.mean([low_level, mid_level, high_level])
        low_bias = low_level - avg_level
        mid_bias = mid_level - avg_level
        high_bias = high_level - avg_level

        return {
            'tilt_db_per_octave': float(tilt_db_per_octave),
            'character': character,
            'low_freq_bias_db': float(low_bias),
            'mid_freq_bias_db': float(mid_bias),
            'high_freq_bias_db': float(high_bias),
            'low_freq_level_db': float(low_level),
            'mid_freq_level_db': float(mid_level),
            'high_freq_level_db': float(high_level)
        }

    def _analyze_dynamics(self, audio: np.ndarray) -> Dict:
        """Analyze dynamic characteristics"""
        # Crest factor (peak to RMS ratio)
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        crest_factor = peak / (rms + 1e-10)
        crest_factor_db = 20 * np.log10(crest_factor)

        # Dynamic range (95th percentile - 10th percentile)
        p95 = np.percentile(np.abs(audio), 95)
        p10 = np.percentile(np.abs(audio), 10)
        dynamic_range = p95 / (p10 + 1e-10)
        dynamic_range_db = 20 * np.log10(dynamic_range)

        # Transient strength analysis
        # Use envelope follower with fast/slow attack
        fast_env = self._envelope_follower(audio, attack_ms=2.0, release_ms=50.0)
        slow_env = self._envelope_follower(audio, attack_ms=30.0, release_ms=200.0)

        transient_signal = np.maximum(0, fast_env - slow_env)
        transient_energy = np.sum(transient_signal ** 2)
        total_energy = np.sum(audio ** 2) + 1e-10
        transient_ratio = transient_energy / total_energy

        # Characterize transient strength
        if transient_ratio > 0.15:
            transient_strength = "strong"
        elif transient_ratio > 0.08:
            transient_strength = "moderate"
        else:
            transient_strength = "soft"

        return {
            'crest_factor_db': float(crest_factor_db),
            'dynamic_range_db': float(dynamic_range_db),
            'transient_ratio': float(transient_ratio),
            'transient_strength': transient_strength
        }

    def _envelope_follower(self, audio: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
        """Simple envelope follower"""
        attack_coef = np.exp(-1.0 / (attack_ms * self.sample_rate / 1000.0))
        release_coef = np.exp(-1.0 / (release_ms * self.sample_rate / 1000.0))

        envelope = np.zeros_like(audio)
        state = 0.0

        for i, sample in enumerate(audio):
            rectified = abs(sample)
            if rectified > state:
                coef = attack_coef
            else:
                coef = release_coef
            state = coef * state + (1 - coef) * rectified
            envelope[i] = state

        return envelope

    def _analyze_noise_floor(self, audio: np.ndarray) -> Dict:
        """Analyze noise floor characteristics"""
        # Find quiet segments (bottom 10% of RMS values)
        frame_length = 2048
        hop_length = 512

        rms_frames = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_frames = np.maximum(rms_frames, 1e-10)

        # Noise floor is estimated from quietest 5% of frames
        noise_rms = np.percentile(rms_frames, 5)
        noise_floor_db = 20 * np.log10(noise_rms)

        # Signal level (95th percentile)
        signal_rms = np.percentile(rms_frames, 95)
        signal_db = 20 * np.log10(signal_rms)

        # SNR
        snr_db = signal_db - noise_floor_db

        # Characterize noise type by analyzing noise spectrum
        # Extract quietest frames
        quiet_threshold = np.percentile(rms_frames, 10)
        quiet_frames_mask = rms_frames < quiet_threshold

        # Reconstruct quiet segments
        stft = librosa.stft(audio, n_fft=2048, hop_length=hop_length)
        noise_stft = stft[:, quiet_frames_mask]

        if noise_stft.shape[1] > 0:
            noise_spectrum = np.mean(np.abs(noise_stft), axis=1)
            noise_spectrum_db = 20 * np.log10(noise_spectrum + 1e-10)

            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)

            # Analyze noise spectral shape
            low_noise = np.mean(noise_spectrum_db[(freqs >= 50) & (freqs <= 200)])
            mid_noise = np.mean(noise_spectrum_db[(freqs >= 500) & (freqs <= 2000)])
            high_noise = np.mean(noise_spectrum_db[(freqs >= 4000) & (freqs <= 8000)])

            # Classify noise type
            if low_noise > mid_noise + 3:
                noise_type = "low_frequency_hum"  # AC hum, room rumble
            elif high_noise > mid_noise + 3:
                noise_type = "high_frequency_hiss"  # Electronic noise
            else:
                noise_type = "broadband"  # White/pink noise
        else:
            noise_type = "minimal"

        return {
            'noise_floor_db': float(noise_floor_db),
            'snr_db': float(snr_db),
            'noise_type': noise_type
        }

    def _analyze_frequency_bands(self, audio: np.ndarray) -> Dict:
        """Analyze level in each frequency band"""
        # Define standard bands
        bands = {
            'Sub': (20, 80),
            'Bass': (80, 250),
            'Low-Mid': (250, 500),
            'Mid': (500, 2000),
            'High-Mid': (2000, 4000),
            'Presence': (4000, 8000),
            'Brilliance': (8000, 16000)
        }

        band_profile = {}

        for band_name, (low_freq, high_freq) in bands.items():
            # Design bandpass filter
            sos = signal.butter(4, [low_freq, high_freq], 'bp', fs=self.sample_rate, output='sos')
            band_signal = signal.sosfilt(sos, audio)

            # Measure RMS level
            band_rms = np.sqrt(np.mean(band_signal ** 2))
            band_level_db = 20 * np.log10(band_rms + 1e-10)

            band_profile[band_name] = {
                'level_db': float(band_level_db),
                'freq_range': (low_freq, high_freq)
            }

        return band_profile

    def _print_recommendations(self, profile: Dict):
        """Print recommendations based on profile"""
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)

        # Check input level
        rms_db = profile['loudness']['rms_db']
        if rms_db < -30:
            print("⚠ Input level is very low")
            print("  → Increase your mic gain or speak/beatbox louder")
        elif rms_db < -20:
            print("ℹ Input level is moderate")
            print("  → Consider increasing gain slightly for better SNR")
        else:
            print("✓ Input level is good")

        # Check SNR
        snr_db = profile['noise']['snr_db']
        if snr_db < 30:
            print(f"\n⚠ Low signal-to-noise ratio ({snr_db:.1f} dB)")
            print("  → Try to reduce background noise")
            if profile['noise']['noise_type'] == 'low_frequency_hum':
                print("  → AC hum detected - check grounding and use high-pass filter")
        else:
            print(f"\n✓ Good signal-to-noise ratio ({snr_db:.1f} dB)")

        # Check spectral balance
        character = profile['spectral']['character']
        if character == "dark":
            print(f"\n⚠ Microphone has a dark character")
            print("  → Presets will be adjusted to boost high frequencies")
        elif character == "bright":
            print(f"\n⚠ Microphone has a bright character")
            print("  → Presets will be adjusted to reduce high frequencies")
        else:
            print(f"\n✓ Microphone has balanced frequency response")

        print("\n" + "="*60)

    def save_profile(self, profile: Dict, profile_name: str, output_dir: Optional[Path] = None) -> Path:
        """
        Save mic profile to JSON

        Args:
            profile: Mic profile dictionary
            profile_name: Name for profile
            output_dir: Output directory (default: config.PRESETS_DIR)

        Returns:
            Path to saved profile
        """
        if output_dir is None:
            output_dir = config.PRESETS_DIR

        profile_path = output_dir / f"mic_profile_{profile_name}.json"

        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)

        print(f"\n✅ Mic profile saved: {profile_path}")

        return profile_path

    @staticmethod
    def load_profile(profile_path: str) -> Dict:
        """
        Load mic profile from JSON

        Args:
            profile_path: Path to profile JSON

        Returns:
            Mic profile dictionary
        """
        with open(profile_path, 'r') as f:
            profile = json.load(f)

        print(f"✅ Mic profile loaded: {profile_path}")

        return profile


def calibrate_microphone(audio_path: str, profile_name: str = "default") -> Dict:
    """
    Convenience function to calibrate microphone from audio file

    Args:
        audio_path: Path to calibration audio (5-10 sec of beatboxing/speaking)
        profile_name: Name for the profile

    Returns:
        Microphone profile dictionary
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=44100, mono=True)

    # Calibrate
    calibrator = MicCalibrator(sample_rate=sr)
    profile = calibrator.calibrate(audio)

    # Save profile
    calibrator.save_profile(profile, profile_name)

    return profile


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mic_calibrator.py <calibration_audio.wav> [profile_name]")
        print("\nRecord 5-10 seconds of your normal beatboxing or speaking")
        print("This will analyze your microphone characteristics")
        sys.exit(1)

    audio_file = sys.argv[1]
    profile_name = sys.argv[2] if len(sys.argv) > 2 else "default"

    # Run calibration
    profile = calibrate_microphone(audio_file, profile_name)

    print("\n✅ Calibration complete!")
    print(f"📁 Profile saved as: mic_profile_{profile_name}.json")
    print("\nYou can now use this profile to adapt presets to your microphone!")
