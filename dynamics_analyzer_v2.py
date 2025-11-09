"""
Dynamics Analyzer V2 - Advanced Compression, Limiting, and Gate Detection
Extracts precise dynamics processing parameters from reference audio
"""
import librosa
import numpy as np
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional


class DynamicsAnalyzerV2:
    """
    Advanced dynamics analyzer for compression, limiting, expansion, and gating
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize dynamics analyzer

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

    def analyze_dynamics(self, audio: np.ndarray) -> Dict:
        """
        Comprehensive dynamics analysis

        Args:
            audio: Input audio signal

        Returns:
            Complete dynamics analysis
        """
        print("\n[Dynamics Analyzer V2] Analyzing dynamics processing...")

        # Analyze compression
        compression_params = self._analyze_compression(audio)

        # Analyze limiting
        limiting_params = self._analyze_limiting(audio)

        # Analyze gate/expander
        gate_params = self._analyze_gate(audio)

        # Analyze overall dynamics
        dynamic_range = self._analyze_dynamic_range(audio)

        # Detect multi-stage compression
        multi_stage = self._detect_multi_stage_compression(audio)

        return {
            'compression': compression_params,
            'limiting': limiting_params,
            'gate': gate_params,
            'dynamic_range': dynamic_range,
            'multi_stage': multi_stage,
            'analysis_method': 'v2_advanced'
        }

    def _analyze_compression(self, audio: np.ndarray) -> Dict:
        """
        Advanced compression detection using level analysis
        """
        # Calculate RMS envelope
        frame_length = 2048
        hop_length = 512

        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Ensure no zeros
        rms = np.maximum(rms, 1e-10)
        rms_db = 20 * np.log10(rms)

        # Calculate peak envelope
        peak_env = self._calculate_peak_envelope(audio, frame_length, hop_length)
        peak_env = np.maximum(peak_env, 1e-10)
        peak_db = 20 * np.log10(peak_env)

        # Analyze compression characteristics
        # 1. Estimate threshold (where compression starts to act)
        threshold_db = self._estimate_threshold(rms_db)

        # 2. Estimate ratio (from level distribution)
        ratio = self._estimate_ratio(rms_db, threshold_db)

        # 3. Estimate attack time (from transient response)
        attack_ms = self._estimate_attack_time(audio, rms)

        # 4. Estimate release time (from envelope decay)
        release_ms = self._estimate_release_time(rms_db)

        # 5. Estimate knee width
        knee_db = self._estimate_knee_width(rms_db, threshold_db)

        # 6. Calculate makeup gain
        makeup_gain_db = self._estimate_makeup_gain(rms_db, threshold_db, ratio)

        # Classify compression amount
        compression_level = self._classify_compression_level(ratio, threshold_db)

        return {
            'threshold_db': float(threshold_db),
            'ratio': float(ratio),
            'attack_ms': float(attack_ms),
            'release_ms': float(release_ms),
            'knee_db': float(knee_db),
            'makeup_gain_db': float(makeup_gain_db),
            'compression_level': compression_level,
            'crest_factor_db': float(np.mean(peak_db - rms_db))
        }

    def _calculate_peak_envelope(self, audio: np.ndarray,
                                frame_length: int,
                                hop_length: int) -> np.ndarray:
        """Calculate peak envelope"""
        frames = librosa.util.frame(audio, frame_length=frame_length,
                                   hop_length=hop_length)
        peak_env = np.max(np.abs(frames), axis=0)
        return peak_env

    def _estimate_threshold(self, rms_db: np.ndarray) -> float:
        """
        Estimate compression threshold
        Uses histogram analysis to find where compression starts
        """
        # Find the level where the distribution starts to flatten
        # (indicative of compression acting)

        # Use 30th percentile as threshold estimate
        # (below this, compression typically doesn't act much)
        threshold = np.percentile(rms_db[rms_db > -60], 35)

        # Clamp to reasonable range
        threshold = np.clip(threshold, -40.0, -5.0)

        return threshold

    def _estimate_ratio(self, rms_db: np.ndarray, threshold_db: float) -> float:
        """
        Estimate compression ratio from level distribution
        """
        # Separate into below-threshold and above-threshold samples
        below_threshold = rms_db[rms_db < threshold_db]
        above_threshold = rms_db[rms_db >= threshold_db]

        if len(above_threshold) < 10:
            return 1.0  # No compression detected

        # Calculate dynamic range above and below threshold
        range_above = np.std(above_threshold) if len(above_threshold) > 0 else 1.0
        range_below = np.std(below_threshold) if len(below_threshold) > 0 else 1.0

        # Ratio estimate: reduced variation above threshold indicates compression
        if range_below > 0:
            ratio_estimate = range_below / (range_above + 0.1)
        else:
            ratio_estimate = 1.0

        # Clamp to realistic range (1:1 to 6:1 for live use)
        ratio = np.clip(ratio_estimate, 1.0, 6.0)

        return ratio

    def _estimate_attack_time(self, audio: np.ndarray, rms: np.ndarray) -> float:
        """
        Estimate attack time from transient response
        """
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=self.sample_rate,
            units='frames',
            hop_length=512,
            backtrack=False
        )

        if len(onset_frames) < 3:
            return 5.0  # Default

        # Analyze how quickly level changes after onsets
        attack_times = []

        for onset_frame in onset_frames[:20]:  # Analyze first 20 onsets
            if onset_frame + 10 < len(rms):
                # Look at level rise after onset
                segment = rms[onset_frame:onset_frame+10]
                if len(segment) > 0 and np.max(segment) > np.min(segment):
                    # Time to reach 90% of peak
                    peak_val = np.max(segment)
                    target = 0.9 * peak_val
                    rise_frames = np.where(segment >= target)[0]
                    if len(rise_frames) > 0:
                        rise_time_frames = rise_frames[0]
                        rise_time_ms = (rise_time_frames * 512 / self.sample_rate) * 1000
                        attack_times.append(rise_time_ms)

        if len(attack_times) > 0:
            avg_attack = np.median(attack_times)
            # Fast attack is typical for compressed material
            # Clamp to reasonable range
            attack = np.clip(avg_attack, 0.5, 30.0)
        else:
            attack = 5.0  # Default medium-fast attack

        return attack

    def _estimate_release_time(self, rms_db: np.ndarray) -> float:
        """
        Estimate release time from envelope decay
        """
        # Find decay slopes in the envelope
        # Calculate first derivative
        diff = np.diff(rms_db)

        # Find negative slopes (decay regions)
        decay_regions = diff < -0.1

        if np.sum(decay_regions) < 10:
            return 100.0  # Default

        # Analyze decay rates
        decay_rates = []

        # Find continuous decay regions
        i = 0
        while i < len(decay_regions):
            if decay_regions[i]:
                # Start of decay region
                start = i
                while i < len(decay_regions) and decay_regions[i]:
                    i += 1
                end = i

                if end - start > 5:  # At least 5 frames
                    # Calculate decay rate (dB/second)
                    decay_db = rms_db[start] - rms_db[end]
                    decay_time_sec = (end - start) * 512 / self.sample_rate

                    if decay_time_sec > 0 and decay_db > 0:
                        decay_rate = decay_db / decay_time_sec
                        decay_rates.append(decay_rate)
            i += 1

        if len(decay_rates) > 0:
            avg_decay_rate = np.median(decay_rates)

            # Convert decay rate to release time
            # Faster decay = shorter release
            # Typical: 6 dB decay corresponds to release time
            release_ms = (6.0 / (avg_decay_rate + 0.1)) * 1000

            # Clamp to reasonable range
            release = np.clip(release_ms, 30.0, 300.0)
        else:
            release = 100.0  # Default medium release

        return release

    def _estimate_knee_width(self, rms_db: np.ndarray, threshold_db: float) -> float:
        """
        Estimate compressor knee width
        """
        # Analyze transition region around threshold
        near_threshold = rms_db[(rms_db > threshold_db - 6) & (rms_db < threshold_db + 6)]

        if len(near_threshold) < 10:
            return 3.0  # Default soft knee

        # Soft knee = more variation around threshold
        # Hard knee = sharp transition
        transition_variation = np.std(near_threshold)

        # Map variation to knee width (0-6 dB)
        knee = np.clip(transition_variation * 0.5, 0.0, 6.0)

        return knee

    def _estimate_makeup_gain(self, rms_db: np.ndarray,
                             threshold_db: float,
                             ratio: float) -> float:
        """
        Estimate makeup gain applied after compression
        """
        # Calculate expected gain reduction
        above_threshold = rms_db[rms_db >= threshold_db]

        if len(above_threshold) < 10:
            return 0.0

        avg_above = np.mean(above_threshold)
        excess_db = avg_above - threshold_db

        # Expected gain reduction
        reduction_db = excess_db * (1 - 1/ratio)

        # Makeup gain typically compensates for this
        makeup = np.clip(reduction_db * 0.7, 0.0, 12.0)

        return makeup

    def _classify_compression_level(self, ratio: float, threshold_db: float) -> str:
        """Classify compression intensity"""
        if ratio < 1.5:
            return "none_or_minimal"
        elif ratio < 2.5:
            return "light"
        elif ratio < 4.0:
            return "moderate"
        elif ratio < 6.0:
            return "heavy"
        else:
            return "extreme"

    def _analyze_limiting(self, audio: np.ndarray) -> Dict:
        """
        Analyze brick-wall limiting
        """
        # Calculate peak level
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak + 1e-10)

        # Check for clipping or limiting
        # Limiter typically keeps peaks below -0.1 to -0.3 dB
        ceiling_db = peak_db

        # Detect limiting by looking at peak distribution
        # Limited audio has many samples near the ceiling
        threshold_for_limiting = peak * 0.95
        samples_at_ceiling = np.sum(np.abs(audio) >= threshold_for_limiting)
        total_samples = len(audio)

        limiting_percentage = (samples_at_ceiling / total_samples) * 100

        # Classify limiting
        if limiting_percentage > 0.1:
            is_limited = True
            limiting_level = "heavy" if limiting_percentage > 1.0 else "moderate"
        else:
            is_limited = False
            limiting_level = "none"

        return {
            'ceiling_db': float(ceiling_db),
            'is_limited': is_limited,
            'limiting_level': limiting_level,
            'limiting_percentage': float(limiting_percentage)
        }

    def _analyze_gate(self, audio: np.ndarray) -> Dict:
        """
        Analyze noise gate / expander
        """
        # Calculate RMS in small windows
        frame_length = 1024
        hop_length = 256

        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        rms = np.maximum(rms, 1e-10)
        rms_db = 20 * np.log10(rms)

        # Estimate gate threshold (where signal is cut off)
        # Look for a floor level
        noise_floor = np.percentile(rms_db, 5)

        # Check if there's gating (abrupt cuts to silence)
        very_quiet_frames = np.sum(rms_db < noise_floor + 3)
        gating_percentage = (very_quiet_frames / len(rms_db)) * 100

        is_gated = gating_percentage > 5  # More than 5% of frames gated

        return {
            'gate_threshold_db': float(noise_floor),
            'is_gated': is_gated,
            'gating_percentage': float(gating_percentage),
            'noise_floor_db': float(noise_floor)
        }

    def _analyze_dynamic_range(self, audio: np.ndarray) -> Dict:
        """
        Analyze overall dynamic range
        """
        # Peak level
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak + 1e-10)

        # RMS level
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Crest factor
        crest_factor_db = peak_db - rms_db

        # Dynamic range (using percentiles for robustness)
        # Calculate short-term loudness
        frame_length = 2048
        hop_length = 512

        frame_rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        frame_rms = np.maximum(frame_rms, 1e-10)
        frame_rms_db = 20 * np.log10(frame_rms)

        # Dynamic range = difference between loud and quiet parts
        loud_level = np.percentile(frame_rms_db, 95)
        quiet_level = np.percentile(frame_rms_db, 10)
        dynamic_range_db = loud_level - quiet_level

        # Classify dynamic range
        if dynamic_range_db > 20:
            dr_classification = "very_dynamic"
        elif dynamic_range_db > 12:
            dr_classification = "dynamic"
        elif dynamic_range_db > 6:
            dr_classification = "moderate"
        else:
            dr_classification = "heavily_compressed"

        return {
            'peak_db': float(peak_db),
            'rms_db': float(rms_db),
            'crest_factor_db': float(crest_factor_db),
            'dynamic_range_db': float(dynamic_range_db),
            'classification': dr_classification
        }

    def _detect_multi_stage_compression(self, audio: np.ndarray) -> Dict:
        """
        Detect if multiple compression stages are being used
        """
        # Multi-stage compression shows:
        # 1. Non-linear compression curve
        # 2. Multiple "knees" in level distribution

        frame_rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        frame_rms = np.maximum(frame_rms, 1e-10)
        frame_rms_db = 20 * np.log10(frame_rms)

        # Analyze histogram of levels
        hist, bin_edges = np.histogram(frame_rms_db, bins=50, range=(-60, 0))

        # Look for multiple peaks in histogram (indicates multiple stages)
        peaks, _ = signal.find_peaks(hist, prominence=len(frame_rms_db) * 0.02)

        multi_stage_detected = len(peaks) > 1

        return {
            'multi_stage_detected': multi_stage_detected,
            'num_stages_estimate': len(peaks) if multi_stage_detected else 1
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dynamics_analyzer_v2.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    # Load audio
    audio, sr = librosa.load(audio_file, sr=44100, mono=True)

    # Analyze
    analyzer = DynamicsAnalyzerV2(sample_rate=sr)
    results = analyzer.analyze_dynamics(audio)

    print("\n" + "="*60)
    print("DYNAMICS ANALYSIS V2 RESULTS")
    print("="*60)

    print("\n[COMPRESSION]")
    comp = results['compression']
    print(f"  Threshold: {comp['threshold_db']:.1f} dB")
    print(f"  Ratio: {comp['ratio']:.2f}:1")
    print(f"  Attack: {comp['attack_ms']:.1f} ms")
    print(f"  Release: {comp['release_ms']:.0f} ms")
    print(f"  Knee: {comp['knee_db']:.1f} dB")
    print(f"  Makeup Gain: {comp['makeup_gain_db']:.1f} dB")
    print(f"  Level: {comp['compression_level']}")

    print("\n[LIMITING]")
    lim = results['limiting']
    print(f"  Ceiling: {lim['ceiling_db']:.2f} dB")
    print(f"  Limited: {lim['is_limited']}")
    print(f"  Level: {lim['limiting_level']}")

    print("\n[GATE/EXPANDER]")
    gate = results['gate']
    print(f"  Gate Threshold: {gate['gate_threshold_db']:.1f} dB")
    print(f"  Gated: {gate['is_gated']}")
    print(f"  Noise Floor: {gate['noise_floor_db']:.1f} dB")

    print("\n[DYNAMIC RANGE]")
    dr = results['dynamic_range']
    print(f"  Peak: {dr['peak_db']:.1f} dB")
    print(f"  RMS: {dr['rms_db']:.1f} dB")
    print(f"  Crest Factor: {dr['crest_factor_db']:.1f} dB")
    print(f"  Dynamic Range: {dr['dynamic_range_db']:.1f} dB")
    print(f"  Classification: {dr['classification']}")

    print("\n[MULTI-STAGE]")
    multi = results['multi_stage']
    print(f"  Multi-stage: {multi['multi_stage_detected']}")
    print(f"  Estimated stages: {multi['num_stages_estimate']}")
