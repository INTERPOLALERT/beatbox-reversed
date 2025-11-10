"""
Calibration Workflow - Unified UX for Complete Preset Adaptation
Integrates: Analysis → Calibration → Adaptation → Validation → Ready-to-Use Preset
"""
import librosa
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Callable
import config

# Import all calibration modules
from analyzer_v2 import BeatboxAnalyzerV2
from mic_calibrator import MicCalibrator
from gain_normalizer import GainNormalizer
from preset_adapter import PresetAdapter
from validation_engine import ValidationEngine
from processor_v2 import BeatboxProcessorV2


class CalibrationWorkflow:
    """
    Complete end-to-end calibration workflow
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize calibration workflow

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

        # Initialize components
        self.analyzer = BeatboxAnalyzerV2(sample_rate)
        self.mic_calibrator = MicCalibrator(sample_rate)
        self.gain_normalizer = GainNormalizer(sample_rate)
        self.preset_adapter = PresetAdapter()
        self.validation_engine = ValidationEngine(sample_rate)
        self.processor = BeatboxProcessorV2(sample_rate)

        # State
        self.reference_audio = None
        self.reference_preset = None
        self.calibration_audio = None
        self.mic_profile = None
        self.adapted_preset = None
        self.final_preset = None

        self.workflow_complete = False

    def run_complete_workflow(self,
                            reference_audio_path: str,
                            calibration_audio_path: str,
                            test_audio_path: Optional[str] = None,
                            preset_name: str = "calibrated_preset",
                            auto_validate: bool = True) -> Dict:
        """
        Run complete calibration workflow from start to finish

        Args:
            reference_audio_path: Path to reference beatbox audio (target sound)
            calibration_audio_path: Path to user's mic calibration recording (5-10 sec)
            test_audio_path: Optional path to test audio for validation
            preset_name: Name for final preset
            auto_validate: Whether to run auto-validation loop

        Returns:
            Complete workflow results
        """
        print("\n" + "="*70)
        print("  BEATBOX PRESET CALIBRATION WORKFLOW")
        print("  Complete Mic-Adaptive Preset System")
        print("="*70)

        results = {
            'steps_completed': [],
            'warnings': [],
            'success': False
        }

        # STEP 1: Analyze Reference Audio
        print("\n" + "="*70)
        print("STEP 1: ANALYZE REFERENCE AUDIO")
        print("="*70)
        print(f"Analyzing: {reference_audio_path}")

        try:
            self.reference_preset = self._step1_analyze_reference(reference_audio_path, preset_name)
            results['steps_completed'].append('reference_analysis')
            print("\n✅ Step 1 Complete: Reference analysis successful")
        except Exception as e:
            print(f"\n❌ Step 1 Failed: {e}")
            results['error'] = str(e)
            return results

        # STEP 2: Calibrate Microphone
        print("\n" + "="*70)
        print("STEP 2: CALIBRATE YOUR MICROPHONE")
        print("="*70)
        print(f"Analyzing: {calibration_audio_path}")

        try:
            self.mic_profile = self._step2_calibrate_mic(calibration_audio_path, preset_name)
            results['steps_completed'].append('mic_calibration')
            print("\n✅ Step 2 Complete: Mic calibration successful")
        except Exception as e:
            print(f"\n❌ Step 2 Failed: {e}")
            results['error'] = str(e)
            return results

        # STEP 3: Adapt Preset to Mic
        print("\n" + "="*70)
        print("STEP 3: ADAPT PRESET TO YOUR MICROPHONE")
        print("="*70)

        try:
            self.adapted_preset = self._step3_adapt_preset()
            results['steps_completed'].append('preset_adaptation')
            print("\n✅ Step 3 Complete: Preset adapted to your microphone")
        except Exception as e:
            print(f"\n❌ Step 3 Failed: {e}")
            results['error'] = str(e)
            return results

        # STEP 4: Validate & Auto-Tune (Optional)
        if auto_validate and test_audio_path:
            print("\n" + "="*70)
            print("STEP 4: VALIDATE & AUTO-TUNE PRESET")
            print("="*70)
            print(f"Testing with: {test_audio_path}")

            try:
                self.final_preset = self._step4_validate_preset(
                    test_audio_path,
                    reference_audio_path
                )
                results['steps_completed'].append('validation')
                print("\n✅ Step 4 Complete: Preset validated and tuned")
            except Exception as e:
                print(f"\n⚠ Step 4 Warning: {e}")
                results['warnings'].append(f"Validation failed: {e}")
                # Use adapted preset as final
                self.final_preset = self.adapted_preset
        else:
            # Skip validation
            self.final_preset = self.adapted_preset
            print("\n⚠ Skipping validation (no test audio provided)")

        # STEP 5: Save Final Preset
        print("\n" + "="*70)
        print("STEP 5: SAVE CALIBRATED PRESET")
        print("="*70)

        try:
            preset_path = self._step5_save_preset(preset_name)
            results['steps_completed'].append('save_preset')
            results['preset_path'] = str(preset_path)
            print(f"\n✅ Step 5 Complete: Preset saved to {preset_path}")
        except Exception as e:
            print(f"\n❌ Step 5 Failed: {e}")
            results['error'] = str(e)
            return results

        # Workflow Complete
        self.workflow_complete = True
        results['success'] = True

        self._print_final_summary(results)

        return results

    def _step1_analyze_reference(self, audio_path: str, preset_name: str) -> Dict:
        """Step 1: Analyze reference audio and extract preset"""
        # Load reference audio
        self.reference_audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Run analysis
        self.analyzer.load_audio(audio_path)
        preset = self.analyzer.analyze_all()

        # Save reference preset
        preset_path = self.analyzer.save_preset(f"{preset_name}_reference")

        return preset

    def _step2_calibrate_mic(self, audio_path: str, profile_name: str) -> Dict:
        """Step 2: Calibrate microphone characteristics"""
        # Load calibration audio
        self.calibration_audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Run calibration
        profile = self.mic_calibrator.calibrate(self.calibration_audio)

        # Save mic profile
        profile_path = self.mic_calibrator.save_profile(profile, profile_name)

        return profile

    def _step3_adapt_preset(self) -> Dict:
        """Step 3: Adapt preset to microphone profile"""
        if self.reference_preset is None:
            raise ValueError("No reference preset available")
        if self.mic_profile is None:
            raise ValueError("No mic profile available")

        # Adapt preset
        adapted = self.preset_adapter.adapt_preset(
            reference_preset=self.reference_preset,
            mic_profile=self.mic_profile
        )

        return adapted

    def _step4_validate_preset(self, test_audio_path: str,
                               reference_audio_path: str) -> Dict:
        """Step 4: Validate and auto-tune preset"""
        # Load test audio
        test_audio, sr = librosa.load(test_audio_path, sr=self.sample_rate, mono=True)

        # Load reference audio if not already loaded
        if self.reference_audio is None:
            self.reference_audio, sr = librosa.load(reference_audio_path,
                                                   sr=self.sample_rate, mono=True)

        # Create processing function for validation loop
        def process_func(audio: np.ndarray, preset: Dict) -> np.ndarray:
            """Process audio with given preset"""
            # Build processor with preset
            temp_processor = BeatboxProcessorV2(self.sample_rate)

            # Manually set preset data
            temp_processor.preset_data = preset
            temp_processor._build_processing_chain()

            # Process audio
            processed = temp_processor.process(audio)

            return processed

        # Run validation loop
        final_preset, validation_history = self.validation_engine.validation_loop(
            reference_audio=self.reference_audio,
            process_func=process_func,
            initial_preset=self.adapted_preset,
            test_audio=test_audio,
            max_iterations=3
        )

        return final_preset

    def _step5_save_preset(self, preset_name: str) -> Path:
        """Step 5: Save final calibrated preset"""
        if self.final_preset is None:
            raise ValueError("No final preset available")

        output_dir = config.PRESETS_DIR
        preset_path = output_dir / f"{preset_name}_calibrated.json"

        with open(preset_path, 'w') as f:
            json.dump(self.final_preset, f, indent=2)

        return preset_path

    def _print_final_summary(self, results: Dict):
        """Print final workflow summary"""
        print("\n" + "="*70)
        print("  CALIBRATION WORKFLOW COMPLETE!")
        print("="*70)

        print("\n📋 Steps Completed:")
        for i, step in enumerate(results['steps_completed'], 1):
            step_name = step.replace('_', ' ').title()
            print(f"  {i}. {step_name}")

        if results.get('warnings'):
            print("\n⚠ Warnings:")
            for warning in results['warnings']:
                print(f"  • {warning}")

        if results.get('preset_path'):
            print(f"\n✅ Final Preset: {results['preset_path']}")

        print("\n" + "="*70)
        print("🎤 YOUR MICROPHONE IS NOW CALIBRATED!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Load the calibrated preset in processor_v2.py or advanced_gui.py")
        print("  2. Start beatboxing with your SM7B!")
        print("  3. The preset will sound like the reference audio")
        print("\n" + "="*70)

    def quick_calibrate(self, reference_path: str, mic_calibration_path: str,
                       preset_name: str = "quick_preset") -> Path:
        """
        Quick calibration workflow (skip validation)

        Args:
            reference_path: Path to reference audio
            mic_calibration_path: Path to mic calibration recording
            preset_name: Name for preset

        Returns:
            Path to calibrated preset
        """
        print("\n🚀 QUICK CALIBRATION MODE (Skip Validation)")

        results = self.run_complete_workflow(
            reference_audio_path=reference_path,
            calibration_audio_path=mic_calibration_path,
            preset_name=preset_name,
            auto_validate=False
        )

        if results['success']:
            return Path(results['preset_path'])
        else:
            raise RuntimeError(f"Calibration failed: {results.get('error', 'Unknown error')}")

    def full_calibrate(self, reference_path: str, mic_calibration_path: str,
                      test_audio_path: str, preset_name: str = "full_preset") -> Path:
        """
        Full calibration workflow (with validation)

        Args:
            reference_path: Path to reference audio
            mic_calibration_path: Path to mic calibration recording
            test_audio_path: Path to test audio for validation
            preset_name: Name for preset

        Returns:
            Path to calibrated preset
        """
        print("\n🎯 FULL CALIBRATION MODE (With Validation)")

        results = self.run_complete_workflow(
            reference_audio_path=reference_path,
            calibration_audio_path=mic_calibration_path,
            test_audio_path=test_audio_path,
            preset_name=preset_name,
            auto_validate=True
        )

        if results['success']:
            return Path(results['preset_path'])
        else:
            raise RuntimeError(f"Calibration failed: {results.get('error', 'Unknown error')}")


def quick_calibrate(reference_audio: str, mic_calibration: str, preset_name: str = "my_preset") -> Path:
    """
    Convenience function for quick calibration

    Args:
        reference_audio: Path to reference beatbox audio
        mic_calibration: Path to user's mic calibration recording
        preset_name: Name for preset

    Returns:
        Path to calibrated preset
    """
    workflow = CalibrationWorkflow()
    return workflow.quick_calibrate(reference_audio, mic_calibration, preset_name)


def full_calibrate(reference_audio: str, mic_calibration: str,
                  test_audio: str, preset_name: str = "my_preset") -> Path:
    """
    Convenience function for full calibration with validation

    Args:
        reference_audio: Path to reference beatbox audio
        mic_calibration: Path to user's mic calibration recording
        test_audio: Path to test audio for validation
        preset_name: Name for preset

    Returns:
        Path to calibrated preset
    """
    workflow = CalibrationWorkflow()
    return workflow.full_calibrate(reference_audio, mic_calibration, test_audio, preset_name)


if __name__ == "__main__":
    import sys

    print("="*70)
    print("  BEATBOX CALIBRATION WORKFLOW")
    print("="*70)
    print("\nUsage:")
    print("  Quick Mode (no validation):")
    print("    python calibration_workflow.py quick <reference.wav> <mic_calibration.wav> [preset_name]")
    print("\n  Full Mode (with validation):")
    print("    python calibration_workflow.py full <reference.wav> <mic_calibration.wav> <test.wav> [preset_name]")
    print("\nExample:")
    print("  python calibration_workflow.py quick reference_beatbox.wav my_mic_test.wav my_preset")
    print("="*70)

    if len(sys.argv) < 4:
        sys.exit(0)

    mode = sys.argv[1].lower()

    if mode == "quick":
        if len(sys.argv) < 4:
            print("\n❌ Error: Quick mode requires <reference.wav> <mic_calibration.wav>")
            sys.exit(1)

        reference = sys.argv[2]
        mic_cal = sys.argv[3]
        preset_name = sys.argv[4] if len(sys.argv) > 4 else "quick_preset"

        print(f"\n🚀 Running Quick Calibration...")
        preset_path = quick_calibrate(reference, mic_cal, preset_name)
        print(f"\n✅ Success! Preset saved: {preset_path}")

    elif mode == "full":
        if len(sys.argv) < 5:
            print("\n❌ Error: Full mode requires <reference.wav> <mic_calibration.wav> <test.wav>")
            sys.exit(1)

        reference = sys.argv[2]
        mic_cal = sys.argv[3]
        test = sys.argv[4]
        preset_name = sys.argv[5] if len(sys.argv) > 5 else "full_preset"

        print(f"\n🎯 Running Full Calibration with Validation...")
        preset_path = full_calibrate(reference, mic_cal, test, preset_name)
        print(f"\n✅ Success! Preset saved: {preset_path}")

    else:
        print(f"\n❌ Error: Unknown mode '{mode}'. Use 'quick' or 'full'")
        sys.exit(1)
