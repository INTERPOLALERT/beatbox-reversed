"""
Effects Detector - Identify Saturation, De-essing, and Other Effects
Detects and characterizes audio effects applied to reference audio
"""
import librosa
import numpy as np
from scipy import signal, stats
from typing import Dict, List, Optional


class EffectsDetector:
    """
    Detects various audio effects in reference audio
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize effects detector

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

    def analyze_effects(self, audio: np.ndarray) -> Dict:
        """
        Comprehensive effects analysis

        Args:
            audio: Input audio signal

        Returns:
            Dictionary with detected effects
        """
        print("\n[Effects Detector] Analyzing audio effects...")

        # Detect saturation/distortion
        saturation = self._detect_saturation(audio)

        # Detect de-essing
        deessing = self._detect_deessing(audio)

        # Detect exciter/enhancer
        exciter = self._detect_exciter(audio)

        # Detect stereo enhancement (if stereo input)
        # For now assume mono, will handle in stereo_analyzer

        # Detect transient shaping
        transient_shaping = self._detect_transient_shaping(audio)

        # Detect tape/analog warmth
        warmth = self._detect_warmth_character(audio)

        return {
            'saturation': saturation,
            'deessing': deessing,
            'exciter': exciter,
            'transient_shaping': transient_shaping,
            'warmth': warmth
        }

    def _detect_saturation(self, audio: np.ndarray) -> Dict:
        """
        Detect saturation/harmonic distortion
        """
        # Analyze harmonic content
        # Saturation adds harmonics

        # Method 1: Total Harmonic Distortion (THD)
        thd = self._calculate_thd(audio)

        # Method 2: Harmonic to noise ratio
        harmonic, percussive = librosa.effects.hpss(audio, margin=2.0)
        harmonic_energy = np.sum(harmonic ** 2)
        total_energy = np.sum(audio ** 2) + 1e-10
        harmonic_ratio = harmonic_energy / total_energy

        # Method 3: Waveshaping detection (look for clipping characteristics)
        clipping_detected, clipping_type = self._detect_clipping_type(audio)

        # Estimate saturation amount (0-1 scale)
        saturation_amount = np.clip(thd / 0.1, 0.0, 1.0)  # Normalize THD

        # Determine saturation type
        if clipping_detected:
            if clipping_type == 'hard':
                sat_type = 'hard_clipping'
            else:
                sat_type = 'soft_clipping'
        elif harmonic_ratio > 0.6:
            sat_type = 'tube_or_tape'
        elif thd > 0.05:
            sat_type = 'soft_saturation'
        else:
            sat_type = 'clean'

        return {
            'detected': saturation_amount > 0.02,
            'amount': float(saturation_amount),
            'type': sat_type,
            'thd': float(thd),
            'harmonic_ratio': float(harmonic_ratio)
        }

    def _calculate_thd(self, audio: np.ndarray, fundamental_freq: float = 200) -> float:
        """
        Calculate Total Harmonic Distortion
        """
        # Compute spectrum
        spectrum = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)

        # Find fundamental and harmonics (approximate)
        # Use energy in harmonic bands vs fundamental

        # Simplified THD: ratio of high-frequency energy to total energy
        # (true THD requires tracking specific harmonics)

        low_freq_mask = freqs < 500
        high_freq_mask = (freqs >= 500) & (freqs < 8000)

        low_energy = np.sum(np.abs(spectrum[low_freq_mask]) ** 2)
        high_energy = np.sum(np.abs(spectrum[high_freq_mask]) ** 2)
        total_energy = low_energy + high_energy + 1e-10

        # Harmonic distortion shows as excess high-frequency energy
        thd_estimate = (high_energy / total_energy) * 0.3  # Scale factor

        return float(np.clip(thd_estimate, 0.0, 1.0))

    def _detect_clipping_type(self, audio: np.ndarray) -> tuple:
        """
        Detect if audio is clipped and classify clipping type
        """
        # Hard clipping: samples at exactly ±1.0 (or close)
        max_val = np.max(np.abs(audio))

        if max_val > 0.99:
            # Check how many samples are at the limit
            at_limit = np.sum(np.abs(audio) > 0.98)
            total_samples = len(audio)

            if (at_limit / total_samples) > 0.001:  # More than 0.1% clipped
                # Analyze clipping characteristics
                # Hard clipping: many consecutive samples at limit
                # Soft clipping: smoother transitions

                # Simple heuristic: check derivative at peaks
                peaks = signal.find_peaks(np.abs(audio), height=0.95)[0]

                if len(peaks) > 5:
                    # Check if peaks are flat (hard clip) or rounded (soft clip)
                    flatness_scores = []

                    for peak in peaks[:20]:
                        if peak > 5 and peak < len(audio) - 5:
                            region = audio[peak-5:peak+6]
                            # Flat region has low variance
                            flatness = np.std(region)
                            flatness_scores.append(flatness)

                    if len(flatness_scores) > 0:
                        avg_flatness = np.mean(flatness_scores)

                        if avg_flatness < 0.01:
                            return (True, 'hard')
                        else:
                            return (True, 'soft')

                return (True, 'unknown')

        return (False, 'none')

    def _detect_deessing(self, audio: np.ndarray) -> Dict:
        """
        Enhanced de-essing detection with multi-band high-frequency analysis
        """
        # De-esser reduces harsh sibilants (5-10 kHz range)
        # Enhanced: Analyze multiple HF bands to find optimal de-esser frequency

        frame_length = 2048
        hop_length = 512

        # Define high-frequency sub-bands for analysis
        hf_bands = {
            '4-6 kHz': (4000, 6000),
            '6-8 kHz': (6000, 8000),
            '8-10 kHz': (8000, 10000)
        }

        band_dynamics = {}
        most_controlled_band = None
        max_control = 0.0

        # Analyze each HF band
        for band_name, (low_freq, high_freq) in hf_bands.items():
            # Extract band
            sos = signal.butter(4, [low_freq, high_freq], 'bp', fs=self.sample_rate, output='sos')
            band_signal = signal.sosfilt(sos, audio)

            # Calculate RMS over time
            band_rms = librosa.feature.rms(y=band_signal, frame_length=frame_length,
                                          hop_length=hop_length)[0]
            band_rms = np.maximum(band_rms, 1e-10)
            band_rms_db = 20 * np.log10(band_rms)

            # Calculate dynamic range
            band_dr = np.percentile(band_rms_db, 95) - np.percentile(band_rms_db, 10)

            # Calculate peak control (how much peaks are reduced)
            band_peaks = band_rms_db[band_rms_db > np.percentile(band_rms_db, 80)]
            if len(band_peaks) > 0:
                peak_control = np.std(band_peaks)  # Lower std = more control
            else:
                peak_control = 10.0

            band_dynamics[band_name] = {
                'dynamic_range_db': float(band_dr),
                'peak_control': float(peak_control),
                'center_freq': int((low_freq + high_freq) / 2)
            }

            # Track most controlled band
            control_score = 1.0 / (peak_control + 1e-10)
            if control_score > max_control:
                max_control = control_score
                most_controlled_band = band_name

        # Overall HF analysis
        sos_hp = signal.butter(4, 4000, 'hp', fs=self.sample_rate, output='sos')
        highs = signal.sosfilt(sos_hp, audio)

        hf_rms = librosa.feature.rms(y=highs, frame_length=frame_length,
                                      hop_length=hop_length)[0]
        hf_rms = np.maximum(hf_rms, 1e-10)
        hf_rms_db = 20 * np.log10(hf_rms)

        hf_dynamic_range = np.percentile(hf_rms_db, 95) - np.percentile(hf_rms_db, 10)

        # Compare to overall dynamic range
        overall_rms = librosa.feature.rms(y=audio, frame_length=frame_length,
                                          hop_length=hop_length)[0]
        overall_rms = np.maximum(overall_rms, 1e-10)
        overall_rms_db = 20 * np.log10(overall_rms)
        overall_dynamic_range = np.percentile(overall_rms_db, 95) - np.percentile(overall_rms_db, 10)

        # Detection logic
        if overall_dynamic_range > 0:
            hf_control_ratio = hf_dynamic_range / overall_dynamic_range

            deessing_detected = hf_control_ratio < 0.7
            deessing_amount = np.clip(1.0 - hf_control_ratio, 0.0, 1.0)
        else:
            deessing_detected = False
            deessing_amount = 0.0

        # Estimate de-esser parameters
        if deessing_detected and most_controlled_band:
            # Threshold: level where HF starts getting controlled
            threshold_db = np.percentile(hf_rms_db, 60)

            # Frequency: use center of most controlled band
            freq_hz = band_dynamics[most_controlled_band]['center_freq']

            # Ratio: based on amount of control
            # More control = higher ratio
            ratio = np.clip(2.0 + (deessing_amount * 4.0), 1.0, 8.0)

            # Attack/Release for de-esser (typically fast)
            attack_ms = 1.0  # Very fast attack to catch sibilants
            release_ms = 50.0  # Medium release

            # Q factor (bandwidth)
            q_factor = 1.5  # Moderate Q for de-esser
        else:
            threshold_db = -20.0
            freq_hz = 6000
            ratio = 1.0
            attack_ms = 1.0
            release_ms = 50.0
            q_factor = 1.5

        return {
            'detected': deessing_detected,
            'amount': float(deessing_amount),
            'threshold_db': float(threshold_db),
            'frequency_hz': float(freq_hz),
            'ratio': float(ratio),
            'attack_ms': float(attack_ms),
            'release_ms': float(release_ms),
            'q_factor': float(q_factor),
            'hf_control_ratio': float(hf_control_ratio) if overall_dynamic_range > 0 else 1.0,
            'band_analysis': band_dynamics,
            'most_controlled_band': most_controlled_band
        }

    def _detect_exciter(self, audio: np.ndarray) -> Dict:
        """
        Detect exciter/harmonic enhancer (adds high-frequency harmonics)
        """
        # Exciter adds generated harmonics above ~3kHz

        # Analyze very high frequency content (8-16 kHz)
        sos_vhf = signal.butter(4, [8000, 16000], 'bp', fs=self.sample_rate, output='sos')
        vhf = signal.sosfilt(sos_vhf, audio)

        vhf_energy = np.sum(vhf ** 2)
        total_energy = np.sum(audio ** 2) + 1e-10

        vhf_ratio = vhf_energy / total_energy

        # High VHF ratio suggests exciter/enhancer
        exciter_detected = vhf_ratio > 0.03

        exciter_amount = np.clip((vhf_ratio - 0.02) / 0.08, 0.0, 1.0)

        return {
            'detected': exciter_detected,
            'amount': float(exciter_amount),
            'vhf_energy_ratio': float(vhf_ratio)
        }

    def _detect_transient_shaping(self, audio: np.ndarray) -> Dict:
        """
        Detect transient enhancement or reduction
        """
        # Separate transient and sustained components
        # Use dual-envelope detection

        # Fast envelope (tracks transients)
        fast_env = self._envelope_follower(audio, attack_ms=1.0, release_ms=20.0)

        # Slow envelope (tracks sustain)
        slow_env = self._envelope_follower(audio, attack_ms=20.0, release_ms=100.0)

        # Transient strength = difference between fast and slow
        transient_signal = np.maximum(0, fast_env - slow_env)

        # Analyze transient energy
        transient_energy = np.sum(transient_signal ** 2)
        total_energy = np.sum(audio ** 2) + 1e-10

        transient_ratio = transient_energy / total_energy

        # Classify
        if transient_ratio > 0.15:
            shaping = 'enhanced'  # Transients boosted
            amount = np.clip((transient_ratio - 0.1) / 0.2, 0.0, 1.0)
        elif transient_ratio < 0.05:
            shaping = 'reduced'   # Transients suppressed
            amount = np.clip((0.1 - transient_ratio) / 0.1, 0.0, 1.0)
        else:
            shaping = 'natural'
            amount = 0.0

        return {
            'shaping_type': shaping,
            'amount': float(amount),
            'transient_ratio': float(transient_ratio)
        }

    def _envelope_follower(self, audio: np.ndarray,
                          attack_ms: float,
                          release_ms: float) -> np.ndarray:
        """
        Simple envelope follower
        """
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

    def _detect_warmth_character(self, audio: np.ndarray) -> Dict:
        """
        Detect analog warmth characteristics (tape/tube emulation)
        """
        # Analog warmth characteristics:
        # 1. Even harmonic content (2nd, 4th harmonics)
        # 2. Slight low-frequency boost
        # 3. Gentle high-frequency roll-off
        # 4. Subtle saturation

        # Analyze harmonic content
        harmonic, percussive = librosa.effects.hpss(audio, margin=2.0)

        # Compute spectrum
        stft = librosa.stft(harmonic, n_fft=4096)
        mag_spectrum = np.mean(np.abs(stft), axis=1)
        mag_spectrum = np.maximum(mag_spectrum, 1e-10)

        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=4096)

        # Check for even harmonic emphasis
        # (simplified: check if low-mids are boosted relative to highs)

        low_mid_mask = (freqs >= 200) & (freqs <= 800)
        high_mask = (freqs >= 4000) & (freqs <= 8000)

        low_mid_energy = np.mean(mag_spectrum[low_mid_mask])
        high_energy = np.mean(mag_spectrum[high_mask]) + 1e-10

        warmth_ratio = low_mid_energy / high_energy

        # Warmth detected if low-mids are emphasized
        warmth_detected = warmth_ratio > 1.5

        warmth_amount = np.clip((warmth_ratio - 1.0) / 2.0, 0.0, 1.0)

        # Determine warmth type
        if warmth_amount > 0.5:
            warmth_type = 'tube_or_tape'
        elif warmth_amount > 0.2:
            warmth_type = 'subtle_warmth'
        else:
            warmth_type = 'clean'

        return {
            'detected': warmth_detected,
            'amount': float(warmth_amount),
            'type': warmth_type,
            'warmth_ratio': float(warmth_ratio)
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python effects_detector.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    # Load audio
    audio, sr = librosa.load(audio_file, sr=44100, mono=True)

    # Analyze
    detector = EffectsDetector(sample_rate=sr)
    results = detector.analyze_effects(audio)

    print("\n" + "="*60)
    print("EFFECTS DETECTION RESULTS")
    print("="*60)

    print("\n[SATURATION/DISTORTION]")
    sat = results['saturation']
    print(f"  Detected: {sat['detected']}")
    print(f"  Type: {sat['type']}")
    print(f"  Amount: {sat['amount']:.3f}")
    print(f"  THD: {sat['thd']:.3f}")

    print("\n[DE-ESSING]")
    de = results['deessing']
    print(f"  Detected: {de['detected']}")
    print(f"  Amount: {de['amount']:.3f}")
    print(f"  Threshold: {de['threshold_db']:.1f} dB")
    print(f"  Frequency: {de['frequency_hz']:.0f} Hz")
    print(f"  Ratio: {de['ratio']:.1f}:1")

    print("\n[EXCITER/ENHANCER]")
    exc = results['exciter']
    print(f"  Detected: {exc['detected']}")
    print(f"  Amount: {exc['amount']:.3f}")

    print("\n[TRANSIENT SHAPING]")
    trans = results['transient_shaping']
    print(f"  Type: {trans['shaping_type']}")
    print(f"  Amount: {trans['amount']:.3f}")

    print("\n[WARMTH CHARACTER]")
    warmth = results['warmth']
    print(f"  Detected: {warmth['detected']}")
    print(f"  Type: {warmth['type']}")
    print(f"  Amount: {warmth['amount']:.3f}")
