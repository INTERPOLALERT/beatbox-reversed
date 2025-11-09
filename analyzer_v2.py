"""
Beatbox Audio Analyzer V2 - Complete DSP Chain Extraction
Integrates all analysis modules to extract comprehensive processing parameters
Main entry point for analyzing reference audio
"""
import librosa
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
import config

# Import v2 analyzers
from spectral_analyzer_v2 import SpectralAnalyzerV2
from dynamics_analyzer_v2 import DynamicsAnalyzerV2
from effects_detector import EffectsDetector
from multiband_dynamics_analyzer import MultibandDynamicsAnalyzer
from stereo_analyzer import StereoAnalyzer


class BeatboxAnalyzerV2:
    """
    Complete beatbox audio analyzer - extracts full DSP processing chain
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize analyzer V2

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

        # Initialize all analyzers
        self.spectral_analyzer = SpectralAnalyzerV2(sample_rate)
        self.dynamics_analyzer = DynamicsAnalyzerV2(sample_rate)
        self.effects_detector = EffectsDetector(sample_rate)
        self.multiband_analyzer = MultibandDynamicsAnalyzer(sample_rate, num_bands=4)
        self.stereo_analyzer = StereoAnalyzer(sample_rate)

        # Analysis results
        self.audio_path = None
        self.audio = None
        self.sr = None
        self.duration = None
        self.is_stereo = False

        self.results = {}

    def load_audio(self, audio_path: str):
        """
        Load audio file

        Args:
            audio_path: Path to audio file
        """
        print(f"\n{'='*60}")
        print(f"BEATBOX ANALYZER V2 - DSP CHAIN EXTRACTION")
        print(f"{'='*60}\n")
        print(f"Loading: {audio_path}")

        self.audio_path = audio_path

        # Load as mono for most analysis
        audio_mono, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        self.audio = audio_mono
        self.sr = sr
        self.duration = len(audio_mono) / sr

        # Also try to load stereo for stereo analysis
        try:
            audio_stereo, _ = librosa.load(audio_path, sr=self.sample_rate, mono=False)
            if len(audio_stereo.shape) > 1 and audio_stereo.shape[0] == 2:
                self.is_stereo = True
                self.audio_stereo = audio_stereo
            else:
                self.is_stereo = False
        except:
            self.is_stereo = False

        print(f"Duration: {self.duration:.2f}s")
        print(f"Sample Rate: {sr} Hz")
        print(f"Channels: {'Stereo' if self.is_stereo else 'Mono'}")

    def analyze_all(self, num_bands: int = 4) -> Dict:
        """
        Run complete analysis

        Args:
            num_bands: Number of bands for multiband analysis

        Returns:
            Complete analysis results
        """
        if self.audio is None:
            raise ValueError("No audio loaded. Call load_audio() first.")

        print(f"\n{'='*60}")
        print("RUNNING COMPLETE ANALYSIS")
        print(f"{'='*60}")

        # 1. Spectral Analysis (EQ Curve)
        print("\n[1/5] Spectral Analysis...")
        spectral_results = self.spectral_analyzer.analyze_spectral_response(self.audio)
        spectral_results_optimized = {
            'eq_curve': self.spectral_analyzer.optimize_eq_for_live_use(spectral_results['eq_curve']),
            'spectral_tilt': spectral_results['spectral_tilt'],
            'spectral_peaks': spectral_results['spectral_peaks']
        }

        # 2. Dynamics Analysis (Compression, Limiting, Gate)
        print("\n[2/5] Dynamics Analysis...")
        dynamics_results = self.dynamics_analyzer.analyze_dynamics(self.audio)

        # 3. Effects Detection (Saturation, De-essing, etc.)
        print("\n[3/5] Effects Detection...")
        effects_results = self.effects_detector.analyze_effects(self.audio)

        # 4. Multiband Dynamics Analysis
        print(f"\n[4/5] Multiband Dynamics Analysis ({num_bands} bands)...")
        multiband_results = self.multiband_analyzer.analyze_multiband_dynamics(self.audio)
        multiband_optimized = self.multiband_analyzer.optimize_for_live_use(multiband_results)

        # 5. Stereo Analysis
        print("\n[5/5] Stereo Analysis...")
        if self.is_stereo:
            stereo_results = self.stereo_analyzer.analyze_stereo(self.audio_stereo, is_stereo=True)
            stereo_preset = self.stereo_analyzer.create_stereo_enhancement_preset(stereo_results)
        else:
            stereo_results = self.stereo_analyzer.analyze_stereo(self.audio, is_stereo=False)
            stereo_preset = None

        # Compile all results
        self.results = {
            'metadata': {
                'source_file': str(self.audio_path),
                'duration_seconds': self.duration,
                'sample_rate': self.sr,
                'is_stereo': self.is_stereo,
                'analysis_version': 'v2.0'
            },
            'spectral': spectral_results_optimized,
            'dynamics': dynamics_results,
            'effects': effects_results,
            'multiband': multiband_optimized,
            'stereo': stereo_results,
            'stereo_preset': stereo_preset
        }

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}\n")

        return self.results

    def print_summary(self):
        """Print analysis summary"""
        if not self.results:
            print("No analysis results available")
            return

        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}\n")

        # Spectral
        print("[EQ CURVE]")
        for band in self.results['spectral']['eq_curve'][:8]:  # Show first 8
            print(f"  {band['name']:15s} {band['frequency']:6.0f} Hz: {band['gain_db']:+6.2f} dB")

        # Spectral Tilt
        tilt = self.results['spectral']['spectral_tilt']
        print(f"\nSpectral Tilt: {tilt['tilt_db_per_decade']:+.2f} dB/decade ({tilt['tilt_type']})")

        # Dynamics
        print("\n[COMPRESSION]")
        comp = self.results['dynamics']['compression']
        print(f"  Threshold: {comp['threshold_db']:.1f} dB")
        print(f"  Ratio: {comp['ratio']:.2f}:1")
        print(f"  Attack: {comp['attack_ms']:.1f} ms")
        print(f"  Release: {comp['release_ms']:.0f} ms")
        print(f"  Level: {comp['compression_level']}")

        # Limiting
        lim = self.results['dynamics']['limiting']
        print(f"\n[LIMITING]")
        print(f"  Ceiling: {lim['ceiling_db']:.2f} dB")
        print(f"  Limited: {lim['is_limited']} ({lim['limiting_level']})")

        # Effects
        print("\n[EFFECTS DETECTED]")
        sat = self.results['effects']['saturation']
        print(f"  Saturation: {sat['type']} (amount: {sat['amount']:.2f})")

        de = self.results['effects']['deessing']
        if de['detected']:
            print(f"  De-essing: {de['amount']:.2f} @ {de['frequency_hz']:.0f} Hz")

        warmth = self.results['effects']['warmth']
        if warmth['detected']:
            print(f"  Warmth: {warmth['type']} (amount: {warmth['amount']:.2f})")

        # Multiband
        print(f"\n[MULTIBAND COMPRESSION]")
        print(f"  Detected: {self.results['multiband']['multiband_compression_detected']}")
        print(f"  Bands: {self.results['multiband']['num_bands']}")

        for band in self.results['multiband']['bands']:
            if band['enabled']:
                print(f"  {band['band_name']:10s} ({band['freq_range'][0]:.0f}-{band['freq_range'][1]:.0f} Hz): "
                      f"Ratio {band['ratio']:.1f}:1, Threshold {band['threshold_db']:.1f} dB")

        # Stereo
        print(f"\n[STEREO]")
        print(f"  Is Stereo: {self.results['stereo']['is_stereo']}")
        if self.results['stereo']['is_stereo']:
            print(f"  Width: {self.results['stereo']['stereo_width']:.2f}")
            print(f"  Type: {self.results['stereo']['stereo_type']}")

        print(f"\n{'='*60}\n")

    def save_preset(self, preset_name: str, output_dir: Optional[Path] = None) -> Path:
        """
        Save analysis as preset

        Args:
            preset_name: Name for preset
            output_dir: Output directory (default: config.PRESETS_DIR)

        Returns:
            Path to saved preset
        """
        if not self.results:
            raise ValueError("No analysis results to save")

        if output_dir is None:
            output_dir = config.PRESETS_DIR

        preset_path = output_dir / f"{preset_name}_v2.json"

        with open(preset_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Preset saved: {preset_path}")

        return preset_path

    def create_processing_chain_description(self) -> str:
        """
        Create human-readable description of the detected processing chain
        """
        if not self.results:
            return "No analysis available"

        lines = []
        lines.append("DETECTED PROCESSING CHAIN")
        lines.append("="*50)

        # Determine processing order (typical: EQ → Compression → Effects → Limiting)

        lines.append("\n1. INPUT GAIN")
        lines.append("   Adjust input to match reference level")

        lines.append("\n2. EQ (14-band Parametric)")
        tilt = self.results['spectral']['spectral_tilt']
        lines.append(f"   Overall Character: {tilt['interpretation']}")
        for band in self.results['spectral']['eq_curve']:
            if abs(band['gain_db']) > 1.0:  # Only show significant adjustments
                lines.append(f"   {band['name']}: {band['gain_db']:+.1f} dB @ {band['frequency']:.0f} Hz")

        if self.results['multiband']['multiband_compression_detected']:
            lines.append("\n3. MULTIBAND COMPRESSION")
            for band in self.results['multiband']['bands']:
                if band['enabled']:
                    lines.append(f"   {band['band_name']}: {band['ratio']:.1f}:1, "
                               f"Threshold {band['threshold_db']:.1f} dB")
        else:
            lines.append("\n3. COMPRESSION")
            comp = self.results['dynamics']['compression']
            lines.append(f"   Threshold: {comp['threshold_db']:.1f} dB")
            lines.append(f"   Ratio: {comp['ratio']:.1f}:1")
            lines.append(f"   Attack: {comp['attack_ms']:.1f} ms")
            lines.append(f"   Release: {comp['release_ms']:.0f} ms")

        lines.append("\n4. EFFECTS")
        sat = self.results['effects']['saturation']
        if sat['detected']:
            lines.append(f"   Saturation: {sat['type']} ({sat['amount']:.0%})")

        de = self.results['effects']['deessing']
        if de['detected']:
            lines.append(f"   De-esser: {de['ratio']:.1f}:1 @ {de['frequency_hz']:.0f} Hz")

        warmth = self.results['effects']['warmth']
        if warmth['detected']:
            lines.append(f"   Warmth: {warmth['type']}")

        if self.results['dynamics']['limiting']['is_limited']:
            lines.append("\n5. LIMITER")
            lim = self.results['dynamics']['limiting']
            lines.append(f"   Ceiling: {lim['ceiling_db']:.1f} dB")
            lines.append(f"   Level: {lim['limiting_level']}")

        if self.results['stereo']['is_stereo']:
            lines.append("\n6. STEREO PROCESSING")
            stereo = self.results['stereo']
            lines.append(f"   Width: {stereo['stereo_width']:.2f}")
            if stereo['reverb']['detected']:
                lines.append(f"   Reverb: {stereo['reverb']['amount']:.0%}")

        lines.append("\n" + "="*50)

        return "\n".join(lines)


def analyze_audio_v2(audio_path: str, preset_name: str = None,
                    num_bands: int = 4) -> Dict:
    """
    Convenience function to analyze audio

    Args:
        audio_path: Path to audio file
        preset_name: Optional preset name to save
        num_bands: Number of bands for multiband analysis

    Returns:
        Analysis results
    """
    analyzer = BeatboxAnalyzerV2()
    analyzer.load_audio(audio_path)
    results = analyzer.analyze_all(num_bands=num_bands)
    analyzer.print_summary()

    print("\n" + analyzer.create_processing_chain_description())

    if preset_name:
        analyzer.save_preset(preset_name)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyzer_v2.py <audio_file> [preset_name] [num_bands]")
        print("\nExample: python analyzer_v2.py beatbox_sample.wav my_preset 4")
        sys.exit(1)

    audio_file = sys.argv[1]
    preset_name = sys.argv[2] if len(sys.argv) > 2 else Path(audio_file).stem
    num_bands = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    # Run analysis
    results = analyze_audio_v2(audio_file, preset_name, num_bands)

    print("\n✅ Analysis complete!")
    print(f"📁 Preset saved as: {preset_name}_v2.json")
    print(f"\nYou can now apply these settings to your microphone for live beatboxing!")
