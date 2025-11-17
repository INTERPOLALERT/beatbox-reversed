"""
Auto Gain Normalization Module
Ensures consistent input levels so effects chain behaves predictably
"""
import numpy as np
import pyloudnorm as pyln
from typing import Dict, Optional, Tuple


class GainNormalizer:
    """
    Automatically normalizes audio input levels to a target level
    """

    def __init__(self, sample_rate: int = 44100, target_lufs: float = -18.0):
        """
        Initialize gain normalizer

        Args:
            sample_rate: Sample rate in Hz
            target_lufs: Target loudness in LUFS (default: -18 LUFS for speech/beatbox)
        """
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.meter = pyln.Meter(sample_rate)

        # State for real-time operation
        self.current_gain_db = 0.0
        self.smoothing_factor = 0.95  # For smooth gain changes

    def calculate_normalization_gain(self, audio: np.ndarray, method: str = 'lufs') -> Dict:
        """
        Calculate gain needed to normalize audio to target level

        Args:
            audio: Input audio signal
            method: Normalization method ('lufs', 'rms', or 'peak')

        Returns:
            Dictionary with gain information
        """
        if method == 'lufs':
            return self._calculate_lufs_gain(audio)
        elif method == 'rms':
            return self._calculate_rms_gain(audio)
        elif method == 'peak':
            return self._calculate_peak_gain(audio)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _calculate_lufs_gain(self, audio: np.ndarray) -> Dict:
        """Calculate gain using LUFS (recommended for speech/beatbox)"""
        try:
            # Measure integrated loudness
            current_lufs = self.meter.integrated_loudness(audio)

            # Calculate gain needed
            gain_db = self.target_lufs - current_lufs

            # Safety limits
            gain_db = np.clip(gain_db, -24.0, 24.0)

            return {
                'gain_db': float(gain_db),
                'current_lufs': float(current_lufs),
                'target_lufs': float(self.target_lufs),
                'method': 'lufs'
            }
        except Exception as e:
            # Fallback to RMS if audio is too quiet or has issues
            return self._calculate_rms_gain(audio)

    def _calculate_rms_gain(self, audio: np.ndarray, target_rms_db: float = -12.0) -> Dict:
        """Calculate gain using RMS level"""
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))
        current_rms_db = 20 * np.log10(rms + 1e-10)

        # Calculate gain needed
        gain_db = target_rms_db - current_rms_db

        # Safety limits
        gain_db = np.clip(gain_db, -24.0, 24.0)

        return {
            'gain_db': float(gain_db),
            'current_rms_db': float(current_rms_db),
            'target_rms_db': float(target_rms_db),
            'method': 'rms'
        }

    def _calculate_peak_gain(self, audio: np.ndarray, target_peak: float = 0.8) -> Dict:
        """Calculate gain using peak normalization"""
        # Find peak level
        peak = np.max(np.abs(audio))

        if peak < 1e-10:
            # Audio is silent
            return {
                'gain_db': 0.0,
                'current_peak': 0.0,
                'target_peak': target_peak,
                'method': 'peak'
            }

        # Calculate gain needed
        gain_linear = target_peak / peak
        gain_db = 20 * np.log10(gain_linear)

        # Safety limits
        gain_db = np.clip(gain_db, -24.0, 24.0)

        return {
            'gain_db': float(gain_db),
            'current_peak': float(peak),
            'target_peak': float(target_peak),
            'method': 'peak'
        }

    def normalize(self, audio: np.ndarray, method: str = 'lufs') -> Tuple[np.ndarray, float]:
        """
        Normalize audio to target level

        Args:
            audio: Input audio signal
            method: Normalization method

        Returns:
            Tuple of (normalized_audio, applied_gain_db)
        """
        # Calculate gain
        gain_info = self.calculate_normalization_gain(audio, method=method)
        gain_db = gain_info['gain_db']

        # Apply gain
        gain_linear = 10 ** (gain_db / 20.0)
        normalized = audio * gain_linear

        # Safety clipping
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized, gain_db

    def normalize_realtime(self, audio: np.ndarray, method: str = 'rms',
                          adapt_speed: float = 0.95) -> Tuple[np.ndarray, float]:
        """
        Normalize audio with smooth gain changes for real-time processing

        Args:
            audio: Input audio buffer
            method: Normalization method
            adapt_speed: Smoothing factor (0-1, higher = slower adaptation)

        Returns:
            Tuple of (normalized_audio, applied_gain_db)
        """
        # Calculate target gain
        gain_info = self.calculate_normalization_gain(audio, method=method)
        target_gain_db = gain_info['gain_db']

        # Smooth gain changes
        self.current_gain_db = (adapt_speed * self.current_gain_db +
                                (1 - adapt_speed) * target_gain_db)

        # Apply smoothed gain
        gain_linear = 10 ** (self.current_gain_db / 20.0)
        normalized = audio * gain_linear

        # Safety clipping
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized, self.current_gain_db

    def create_normalizer_preset(self, audio: np.ndarray, method: str = 'lufs') -> Dict:
        """
        Analyze audio and create a normalization preset

        Args:
            audio: Input audio signal
            method: Normalization method

        Returns:
            Normalization preset dictionary
        """
        # Calculate normalization parameters
        gain_info = self.calculate_normalization_gain(audio, method=method)

        # Additional analysis
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        crest_factor = peak / (rms + 1e-10)
        crest_factor_db = 20 * np.log10(crest_factor)

        try:
            lufs = self.meter.integrated_loudness(audio)
        except Exception as e:
            lufs = 20 * np.log10(rms + 1e-10)

        preset = {
            'normalization': {
                'method': method,
                'gain_db': gain_info['gain_db'],
                'target_lufs': self.target_lufs
            },
            'input_analysis': {
                'rms_db': float(20 * np.log10(rms + 1e-10)),
                'peak_db': float(20 * np.log10(peak + 1e-10)),
                'lufs': float(lufs),
                'crest_factor_db': float(crest_factor_db)
            },
            'recommendations': self._generate_recommendations(gain_info, rms, peak)
        }

        return preset

    def _generate_recommendations(self, gain_info: Dict, rms: float, peak: float) -> list:
        """Generate recommendations based on analysis"""
        recommendations = []

        gain_db = gain_info['gain_db']

        if gain_db > 12:
            recommendations.append({
                'level': 'warning',
                'message': 'Input level is very low - increase mic gain or speak louder'
            })
        elif gain_db > 6:
            recommendations.append({
                'level': 'info',
                'message': 'Input level is low - consider increasing mic gain'
            })
        elif gain_db < -12:
            recommendations.append({
                'level': 'warning',
                'message': 'Input level is too high - reduce mic gain to avoid distortion'
            })
        elif gain_db < -6:
            recommendations.append({
                'level': 'info',
                'message': 'Input level is high - consider reducing mic gain slightly'
            })
        else:
            recommendations.append({
                'level': 'success',
                'message': 'Input level is optimal'
            })

        # Check for clipping
        if peak > 0.99:
            recommendations.append({
                'level': 'error',
                'message': 'Input is clipping! Reduce mic gain immediately'
            })

        # Check dynamic range
        crest_factor = peak / (rms + 1e-10)
        crest_factor_db = 20 * np.log10(crest_factor)

        if crest_factor_db < 6:
            recommendations.append({
                'level': 'warning',
                'message': 'Low dynamic range - signal may be over-compressed'
            })
        elif crest_factor_db > 20:
            recommendations.append({
                'level': 'info',
                'message': 'Very high dynamic range - compression may be beneficial'
            })

        return recommendations

    def print_analysis(self, preset: Dict):
        """Print normalization analysis"""
        print("\n" + "="*60)
        print("GAIN NORMALIZATION ANALYSIS")
        print("="*60)

        norm = preset['normalization']
        analysis = preset['input_analysis']

        print(f"\n[INPUT LEVELS]")
        print(f"  RMS: {analysis['rms_db']:.1f} dB")
        print(f"  Peak: {analysis['peak_db']:.1f} dB")
        print(f"  LUFS: {analysis['lufs']:.1f} LUFS")
        print(f"  Crest Factor: {analysis['crest_factor_db']:.1f} dB")

        print(f"\n[NORMALIZATION]")
        print(f"  Method: {norm['method'].upper()}")
        print(f"  Required Gain: {norm['gain_db']:+.1f} dB")
        print(f"  Target LUFS: {norm['target_lufs']:.1f} LUFS")

        print(f"\n[RECOMMENDATIONS]")
        for rec in preset['recommendations']:
            icon = {
                'error': '❌',
                'warning': '⚠',
                'info': 'ℹ',
                'success': '✓'
            }.get(rec['level'], '•')
            print(f"  {icon} {rec['message']}")

        print("\n" + "="*60)


def normalize_audio_file(input_path: str, output_path: str,
                         target_lufs: float = -18.0,
                         method: str = 'lufs') -> Dict:
    """
    Normalize an audio file and save result

    Args:
        input_path: Input audio file
        output_path: Output audio file
        target_lufs: Target loudness in LUFS
        method: Normalization method

    Returns:
        Normalization report
    """
    import librosa
    import soundfile as sf

    # Load audio
    audio, sr = librosa.load(input_path, sr=44100, mono=True)

    # Normalize
    normalizer = GainNormalizer(sample_rate=sr, target_lufs=target_lufs)
    normalized, gain_db = normalizer.normalize(audio, method=method)

    # Create analysis
    preset = normalizer.create_normalizer_preset(audio, method=method)
    normalizer.print_analysis(preset)

    # Save
    sf.write(output_path, normalized, sr)

    print(f"\n✅ Normalized audio saved: {output_path}")
    print(f"Applied gain: {gain_db:+.1f} dB")

    return preset


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python gain_normalizer.py <input.wav> <output.wav> [target_lufs] [method]")
        print("\nMethods: lufs (default), rms, peak")
        print("Target LUFS: -18 (default, good for speech/beatbox)")
        print("\nExample: python gain_normalizer.py input.wav normalized.wav -18 lufs")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_lufs = float(sys.argv[3]) if len(sys.argv) > 3 else -18.0
    method = sys.argv[4] if len(sys.argv) > 4 else 'lufs'

    # Normalize
    result = normalize_audio_file(input_file, output_file, target_lufs, method)

    print("\n✅ Normalization complete!")
