"""
Stereo Analyzer - Stereo Width, Imaging, and Spatial Effects Detection
Analyzes stereo processing applied to reference audio
"""
import librosa
import numpy as np
from scipy import signal
from typing import Dict, Optional


class StereoAnalyzer:
    """
    Analyzes stereo characteristics and spatial processing
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize stereo analyzer

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

    def analyze_stereo(self, audio: np.ndarray, is_stereo: bool = None) -> Dict:
        """
        Analyze stereo characteristics

        Args:
            audio: Input audio (mono or stereo)
            is_stereo: Whether input is stereo (auto-detect if None)

        Returns:
            Stereo analysis results
        """
        print("\n[Stereo Analyzer] Analyzing stereo characteristics...")

        # Determine if audio is stereo
        if is_stereo is None:
            is_stereo = len(audio.shape) > 1 and audio.shape[0] == 2

        if not is_stereo:
            # Mono audio - analyze mono characteristics
            return self._analyze_mono(audio)
        else:
            # Stereo audio - full stereo analysis
            return self._analyze_stereo_full(audio)

    def _analyze_mono(self, audio: np.ndarray) -> Dict:
        """Analyze mono audio"""
        return {
            'is_stereo': False,
            'stereo_width': 0.0,
            'mid_side_balance': 0.5,  # Equal M/S
            'correlation': 1.0,  # Perfect correlation (mono)
            'recommendation': 'mono_source'
        }

    def _analyze_stereo_full(self, audio: np.ndarray) -> Dict:
        """Analyze stereo audio"""
        # Separate left and right channels
        left = audio[0, :]
        right = audio[1, :]

        # 1. Calculate stereo width
        stereo_width = self._calculate_stereo_width(left, right)

        # 2. Analyze mid-side balance
        mid, side = self._convert_to_mid_side(left, right)
        mid_side_balance = self._analyze_mid_side_balance(mid, side)

        # 3. Calculate stereo correlation
        correlation = self._calculate_correlation(left, right)

        # 4. Detect panning/imaging
        panning = self._detect_panning(left, right)

        # 5. Detect reverb/space
        reverb = self._detect_stereo_reverb(left, right)

        # 6. Classify stereo type
        stereo_type = self._classify_stereo_type(stereo_width, correlation)

        return {
            'is_stereo': True,
            'stereo_width': stereo_width,
            'mid_side_balance': mid_side_balance,
            'correlation': correlation,
            'panning': panning,
            'reverb': reverb,
            'stereo_type': stereo_type,
            'recommendation': self._get_recommendation(stereo_width, correlation)
        }

    def _calculate_stereo_width(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Calculate stereo width (0=mono, 1=normal stereo, 2=wide stereo)
        """
        # Convert to mid-side
        mid, side = self._convert_to_mid_side(left, right)

        # Calculate energy ratio
        mid_energy = np.sum(mid ** 2)
        side_energy = np.sum(side ** 2)
        total_energy = mid_energy + side_energy + 1e-10

        # Width is related to side energy
        side_ratio = side_energy / total_energy

        # Map to width scale (0-2)
        # Normal stereo: ~0.5 side ratio = width 1.0
        # Mono: 0 side ratio = width 0.0
        # Wide: high side ratio = width 1.5-2.0

        width = np.clip(side_ratio * 2.0, 0.0, 2.0)

        return float(width)

    def _convert_to_mid_side(self, left: np.ndarray,
                            right: np.ndarray) -> tuple:
        """Convert L/R to Mid/Side"""
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        return mid, side

    def _analyze_mid_side_balance(self, mid: np.ndarray, side: np.ndarray) -> float:
        """
        Analyze mid-side balance (0=all mid, 1=all side)
        """
        mid_energy = np.sum(mid ** 2)
        side_energy = np.sum(side ** 2)
        total_energy = mid_energy + side_energy + 1e-10

        side_ratio = side_energy / total_energy

        return float(side_ratio)

    def _calculate_correlation(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Calculate stereo correlation (-1 to +1)
        +1 = identical (mono), 0 = uncorrelated, -1 = inverted
        """
        # Pearson correlation
        if len(left) != len(right):
            return 1.0

        left_norm = left - np.mean(left)
        right_norm = right - np.mean(right)

        numerator = np.sum(left_norm * right_norm)
        denominator = np.sqrt(np.sum(left_norm ** 2) * np.sum(right_norm ** 2)) + 1e-10

        correlation = numerator / denominator

        return float(np.clip(correlation, -1.0, 1.0))

    def _detect_panning(self, left: np.ndarray, right: np.ndarray) -> Dict:
        """
        Detect panning position
        """
        # Calculate RMS for left and right
        left_rms = np.sqrt(np.mean(left ** 2))
        right_rms = np.sqrt(np.mean(right ** 2))

        total_rms = left_rms + right_rms + 1e-10

        # Pan position: -1 (left) to +1 (right), 0 = center
        pan_position = (right_rms - left_rms) / total_rms

        # Classify panning
        if abs(pan_position) < 0.1:
            pan_type = 'center'
        elif pan_position < -0.5:
            pan_type = 'hard_left'
        elif pan_position > 0.5:
            pan_type = 'hard_right'
        elif pan_position < 0:
            pan_type = 'left'
        else:
            pan_type = 'right'

        return {
            'pan_position': float(pan_position),
            'type': pan_type
        }

    def _detect_stereo_reverb(self, left: np.ndarray, right: np.ndarray) -> Dict:
        """
        Detect stereo reverb presence
        """
        # Reverb creates decorrelated late reflections

        # Analyze correlation over time
        frame_length = 4096
        hop_length = 2048

        left_frames = librosa.util.frame(left, frame_length=frame_length,
                                         hop_length=hop_length)
        right_frames = librosa.util.frame(right, frame_length=frame_length,
                                          hop_length=hop_length)

        correlations = []

        for i in range(min(left_frames.shape[1], right_frames.shape[1])):
            l_frame = left_frames[:, i]
            r_frame = right_frames[:, i]

            # Calculate correlation for this frame
            l_norm = l_frame - np.mean(l_frame)
            r_norm = r_frame - np.mean(r_frame)

            num = np.sum(l_norm * r_norm)
            denom = np.sqrt(np.sum(l_norm ** 2) * np.sum(r_norm ** 2)) + 1e-10

            corr = num / denom
            correlations.append(corr)

        correlations = np.array(correlations)

        # Reverb shows decreasing correlation over time
        # (early reflections are correlated, late are not)

        avg_correlation = np.mean(correlations)
        correlation_std = np.std(correlations)

        # If correlation varies and average is moderate, likely reverb
        reverb_detected = (correlation_std > 0.2) and (avg_correlation < 0.8)

        # Estimate reverb amount
        if reverb_detected:
            reverb_amount = np.clip((0.8 - avg_correlation) / 0.5, 0.0, 1.0)
        else:
            reverb_amount = 0.0

        return {
            'detected': reverb_detected,
            'amount': float(reverb_amount),
            'avg_correlation': float(avg_correlation)
        }

    def _classify_stereo_type(self, width: float, correlation: float) -> str:
        """Classify type of stereo processing"""
        if width < 0.2:
            return 'mono'
        elif width > 1.5:
            if correlation > 0.8:
                return 'artificially_widened'
            else:
                return 'naturally_wide'
        elif correlation > 0.9:
            return 'narrow_stereo'
        elif correlation > 0.5:
            return 'normal_stereo'
        else:
            return 'wide_decorrelated'

    def _get_recommendation(self, width: float, correlation: float) -> str:
        """Get recommendation for live use"""
        if width < 0.3:
            return "Source is mono or near-mono. Consider mono processing for beatbox."
        elif width > 1.5:
            return "Wide stereo detected. May want to reduce width for live use."
        else:
            return "Normal stereo width. Suitable for processing."

    def create_stereo_enhancement_preset(self, analysis: Dict) -> Dict:
        """
        Create preset for stereo enhancement based on analysis
        """
        if not analysis['is_stereo']:
            return {
                'enabled': False,
                'width': 1.0,
                'reverb_amount': 0.0,
                'reverb_size': 0.5,
                'pan': 0.0
            }

        # Extract parameters
        width = analysis['stereo_width']
        reverb_amount = analysis['reverb']['amount'] if analysis['reverb']['detected'] else 0.0
        pan = analysis['panning']['pan_position']

        # Moderate for live use
        width = np.clip(width, 0.8, 1.3)  # Don't go too wide
        reverb_amount = np.clip(reverb_amount * 0.5, 0.0, 0.3)  # Reduce reverb for clarity

        return {
            'enabled': True,
            'width': float(width),
            'reverb_amount': float(reverb_amount),
            'reverb_size': 0.5,  # Medium room
            'pan': float(pan)
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stereo_analyzer.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    # Load audio (preserve stereo if present)
    audio, sr = librosa.load(audio_file, sr=44100, mono=False)

    # Analyze
    analyzer = StereoAnalyzer(sample_rate=sr)
    results = analyzer.analyze_stereo(audio)

    print("\n" + "="*60)
    print("STEREO ANALYSIS RESULTS")
    print("="*60)

    print(f"\nIs Stereo: {results['is_stereo']}")

    if results['is_stereo']:
        print(f"Stereo Width: {results['stereo_width']:.2f}")
        print(f"Correlation: {results['correlation']:.3f}")
        print(f"Stereo Type: {results['stereo_type']}")

        print(f"\nPanning:")
        print(f"  Position: {results['panning']['pan_position']:.2f}")
        print(f"  Type: {results['panning']['type']}")

        print(f"\nReverb:")
        print(f"  Detected: {results['reverb']['detected']}")
        print(f"  Amount: {results['reverb']['amount']:.2f}")

        # Create preset
        preset = analyzer.create_stereo_enhancement_preset(results)
        print(f"\nStereo Enhancement Preset:")
        print(f"  Width: {preset['width']:.2f}")
        print(f"  Reverb: {preset['reverb_amount']:.2f}")
        print(f"  Pan: {preset['pan']:.2f}")

    print(f"\nRecommendation: {results['recommendation']}")
