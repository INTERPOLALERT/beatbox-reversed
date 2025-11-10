#!/usr/bin/env python3
"""
Beatbox Calibration CLI Tool
User-friendly command-line interface for complete microphone calibration
"""
import sys
import argparse
from pathlib import Path
from calibration_workflow import CalibrationWorkflow, quick_calibrate, full_calibrate


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("  🎤 BEATBOX MICROPHONE CALIBRATION SYSTEM")
    print("  Adaptive Preset Matching for Perfect Sound")
    print("="*70 + "\n")


def print_help():
    """Print detailed help"""
    print_banner()
    print("USAGE:")
    print("  Quick Mode (Skip Validation):")
    print("    python calibrate.py quick <reference.wav> <mic_test.wav> [preset_name]\n")
    print("  Full Mode (With Validation):")
    print("    python calibrate.py full <reference.wav> <mic_test.wav> <validation.wav> [preset_name]\n")

    print("\nARGUMENTS:")
    print("  reference.wav    - Reference beatbox audio (the sound you want)")
    print("  mic_test.wav     - 5-10 second recording of YOUR normal beatboxing")
    print("  validation.wav   - (Full mode only) Another test recording for validation")
    print("  preset_name      - Optional name for your preset (default: 'my_preset')\n")

    print("EXAMPLES:")
    print("  # Quick calibration (fastest, good for most users)")
    print("  python calibrate.py quick pro_beatbox.wav my_beatbox_test.wav my_preset\n")

    print("  # Full calibration with validation (most accurate)")
    print("  python calibrate.py full pro_beatbox.wav my_test1.wav my_test2.wav pro_preset\n")

    print("WORKFLOW:")
    print("  1. Analyze reference audio → Extract DSP chain")
    print("  2. Calibrate your mic → Profile characteristics")
    print("  3. Adapt preset → Compensate for mic differences")
    print("  4. Validate (full mode) → Auto-tune for perfect match")
    print("  5. Save calibrated preset → Ready to use!\n")

    print("TIPS:")
    print("  • Use high-quality reference audio (professional beatbox recording)")
    print("  • Record your mic test in the same room/setup you'll use")
    print("  • Beatbox/speak normally - don't be too quiet or too loud")
    print("  • Quick mode is fine for most users (saves time)")
    print("  • Full mode is recommended if you want perfect accuracy\n")

    print("="*70)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Beatbox Microphone Calibration System',
        add_help=False
    )

    parser.add_argument('mode', nargs='?', choices=['quick', 'full', 'help'],
                       help='Calibration mode')
    parser.add_argument('reference', nargs='?', help='Reference audio file')
    parser.add_argument('mic_test', nargs='?', help='Microphone test recording')
    parser.add_argument('validation', nargs='?', help='Validation audio (full mode only)')
    parser.add_argument('preset_name', nargs='?', default=None, help='Preset name')
    parser.add_argument('-h', '--help', action='store_true', help='Show help')

    args = parser.parse_args()

    # Show help
    if args.help or args.mode == 'help' or args.mode is None:
        print_help()
        return 0

    # Validate mode
    if args.mode not in ['quick', 'full']:
        print(f"❌ Error: Invalid mode '{args.mode}'. Use 'quick' or 'full'")
        print("Run 'python calibrate.py help' for usage information")
        return 1

    # Quick mode
    if args.mode == 'quick':
        if not args.reference or not args.mic_test:
            print("❌ Error: Quick mode requires <reference.wav> <mic_test.wav>")
            print("Run 'python calibrate.py help' for usage information")
            return 1

        # Check if files exist
        if not Path(args.reference).exists():
            print(f"❌ Error: Reference file not found: {args.reference}")
            return 1
        if not Path(args.mic_test).exists():
            print(f"❌ Error: Mic test file not found: {args.mic_test}")
            return 1

        preset_name = args.preset_name if args.preset_name else "my_preset"

        print_banner()
        print("🚀 QUICK CALIBRATION MODE")
        print(f"Reference: {args.reference}")
        print(f"Mic Test:  {args.mic_test}")
        print(f"Preset:    {preset_name}")
        print("\n" + "="*70)

        try:
            preset_path = quick_calibrate(args.reference, args.mic_test, preset_name)
            print("\n" + "="*70)
            print("✅ CALIBRATION SUCCESSFUL!")
            print("="*70)
            print(f"\n📁 Preset saved: {preset_path}")
            print("\n🎤 Next steps:")
            print("  1. Load this preset in processor_v2.py or advanced_gui.py")
            print("  2. Start beatboxing!")
            print("  3. Your mic will sound like the reference audio")
            print("\n" + "="*70)
            return 0
        except Exception as e:
            print("\n" + "="*70)
            print("❌ CALIBRATION FAILED")
            print("="*70)
            print(f"\nError: {e}")
            print("\nTroubleshooting:")
            print("  • Check that audio files are valid WAV files")
            print("  • Ensure mic test recording is at least 5 seconds")
            print("  • Make sure you have write permissions in the presets directory")
            return 1

    # Full mode
    elif args.mode == 'full':
        if not args.reference or not args.mic_test or not args.validation:
            print("❌ Error: Full mode requires <reference.wav> <mic_test.wav> <validation.wav>")
            print("Run 'python calibrate.py help' for usage information")
            return 1

        # Check if files exist
        if not Path(args.reference).exists():
            print(f"❌ Error: Reference file not found: {args.reference}")
            return 1
        if not Path(args.mic_test).exists():
            print(f"❌ Error: Mic test file not found: {args.mic_test}")
            return 1
        if not Path(args.validation).exists():
            print(f"❌ Error: Validation file not found: {args.validation}")
            return 1

        # Determine preset name
        if args.preset_name:
            preset_name = args.preset_name
        else:
            # If validation arg looks like a preset name (no extension), use it
            if not args.validation.endswith(('.wav', '.mp3', '.flac')):
                preset_name = args.validation
                print("⚠ Warning: Validation file looks like a name, not an audio file")
                print("   If you meant to skip validation, use 'quick' mode instead")
                return 1
            else:
                preset_name = "my_preset"

        print_banner()
        print("🎯 FULL CALIBRATION MODE (With Validation)")
        print(f"Reference:   {args.reference}")
        print(f"Mic Test:    {args.mic_test}")
        print(f"Validation:  {args.validation}")
        print(f"Preset:      {preset_name}")
        print("\n" + "="*70)

        try:
            preset_path = full_calibrate(args.reference, args.mic_test, args.validation, preset_name)
            print("\n" + "="*70)
            print("✅ CALIBRATION & VALIDATION SUCCESSFUL!")
            print("="*70)
            print(f"\n📁 Preset saved: {preset_path}")
            print("\n🎤 Next steps:")
            print("  1. Load this preset in processor_v2.py or advanced_gui.py")
            print("  2. Start beatboxing!")
            print("  3. Your mic will sound like the reference audio (validated)")
            print("\n" + "="*70)
            return 0
        except Exception as e:
            print("\n" + "="*70)
            print("❌ CALIBRATION FAILED")
            print("="*70)
            print(f"\nError: {e}")
            print("\nTroubleshooting:")
            print("  • Check that audio files are valid WAV files")
            print("  • Ensure recordings are at least 5 seconds each")
            print("  • Make sure you have write permissions in the presets directory")
            print("  • Try 'quick' mode if validation is causing issues")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
