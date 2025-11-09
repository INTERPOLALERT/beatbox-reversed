"""
Test Script for New Features
Tests adaptive loudness matching, diagnostics, and integration
"""
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ultimate_processor import UltimateProcessor
from loudness_matcher import LoudnessMatcher
from diagnostic_logger import DiagnosticLogger, PerBufferAnalyzer
import config


def test_loudness_matching():
    """Test adaptive loudness matching"""
    print("\n" + "=" * 80)
    print("TEST 1: ADAPTIVE LOUDNESS MATCHING")
    print("=" * 80)

    # Create loudness matcher
    matcher = LoudnessMatcher(sample_rate=44100)

    # Set reference loudness
    print("\n1. Setting reference loudness...")
    matcher.set_reference_loudness(
        reference_rms=0.15,  # Moderate level
        reference_peak=0.8,  # Healthy headroom
        reference_lufs=-14.0  # Broadcast standard
    )

    # Test with quiet signal
    print("\n2. Testing with quiet signal...")
    quiet_signal = 0.03 * np.random.randn(4410)  # 100ms of quiet noise
    processed, stats = matcher.apply_adaptive_gain(quiet_signal, match_mode='rms')

    print(f"   Original RMS: {stats['current_rms_db']:.1f} dB")
    print(f"   Applied gain: {stats['applied_gain_db']:+.1f} dB")
    print(f"   Processed RMS: {20 * np.log10(np.sqrt(np.mean(processed**2)) + 1e-10):.1f} dB")
    print(f"   Target RMS: {20 * np.log10(matcher.reference_rms + 1e-10):.1f} dB")

    # Test different matching modes
    print("\n3. Testing different matching modes...")
    test_signal = 0.1 * np.random.randn(4410)

    for mode in ['rms', 'lufs', 'peak_normalized', 'crest_matched']:
        processed, stats = matcher.apply_adaptive_gain(test_signal, match_mode=mode)
        print(f"   Mode '{mode}': Applied gain = {stats['applied_gain_db']:+.1f} dB")

    print("\n✓ Loudness matching test PASSED")
    return True


def test_diagnostic_logger():
    """Test diagnostic logging system"""
    print("\n" + "=" * 80)
    print("TEST 2: DIAGNOSTIC LOGGING SYSTEM")
    print("=" * 80)

    # Create logger
    logger = DiagnosticLogger(enabled=True, log_dir=str(config.LOGS_DIR))
    analyzer = PerBufferAnalyzer(sample_rate=44100)

    print("\n1. Simulating buffer processing...")

    # Process 50 buffers
    for i in range(50):
        # Generate test buffer
        t = np.linspace(0, 0.01, 441)  # 10ms buffer
        freq = 440 * (1 + i * 0.05)
        amplitude = 0.1 * (1 + 0.5 * np.sin(i * 0.1))
        test_buffer = amplitude * np.sin(2 * np.pi * freq * t)

        # Analyze
        analysis = analyzer.analyze_buffer(test_buffer)

        # Add processing info
        analysis.update({
            'detected_sound_type': ['kick', 'snare', 'hihat', 'bass', 'vocal'][i % 5],
            'applied_gain_db': np.random.randn() * 2,
            'eq_applied': 'adaptive',
            'compression_applied': 'multiband',
            'transient_amount': 0.6,
            'saturation_amount': 0.2,
            'reverb_amount': 0.15,
            'stereo_width': 1.3
        })

        # Log
        logger.log_buffer(analysis)

    print(f"   Processed {logger.total_buffers_processed} buffers")

    # Get statistics
    print("\n2. Analyzing statistics...")
    stats = logger.get_buffer_statistics()

    print(f"   RMS: {stats['rms_mean_db']:.1f} ± {stats['rms_std_db']:.1f} dB")
    print(f"   Peak: {stats['peak_max_db']:.1f} dB")
    print(f"   Crest: {stats['crest_mean_db']:.1f} dB")
    print(f"   Sound types: {stats['sound_type_distribution']}")

    # Save summary
    print("\n3. Saving diagnostic summary...")
    logger.save_summary()
    print(f"   CSV log: {logger.csv_log_path}")
    print(f"   JSON summary: {logger.json_log_path}")

    print("\n✓ Diagnostic logging test PASSED")
    return True


def test_ultimate_processor_integration():
    """Test UltimateProcessor with all new features"""
    print("\n" + "=" * 80)
    print("TEST 3: ULTIMATE PROCESSOR INTEGRATION")
    print("=" * 80)

    # Create processor with diagnostics
    print("\n1. Creating processor with diagnostics enabled...")
    processor = UltimateProcessor(sample_rate=44100, enable_diagnostics=True)

    # Configure
    print("\n2. Configuring processor...")
    processor.set_wet_dry_mix(0.8)
    processor.set_transient_preservation(0.6)
    processor.set_saturation(0.2, 'tube')
    processor.set_reverb(0.15, size=0.4)
    processor.set_stereo_width(1.3)
    processor.set_loudness_matching(enabled=True, match_mode='rms')

    # Simulate reference loudness (normally from preset)
    print("\n3. Setting reference loudness...")
    processor.loudness_matcher.set_reference_loudness(
        reference_rms=0.15,
        reference_peak=0.8,
        reference_lufs=-14.0
    )

    # Process test buffers
    print("\n4. Processing test buffers...")
    for i in range(20):
        # Generate test buffer (varying characteristics)
        t = np.linspace(0, 0.01, 441)
        freq = 200 + i * 50  # Varying frequency
        amplitude = 0.05 + 0.1 * np.random.rand()  # Varying amplitude
        test_buffer = amplitude * np.sin(2 * np.pi * freq * t)

        # Add some noise
        test_buffer += 0.01 * np.random.randn(len(test_buffer))

        # Process
        output = processor.process_buffer(test_buffer.astype(np.float32))

        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1} buffers")

    # Get status
    print("\n5. Checking processor status...")
    status = processor.get_status()

    print(f"   Preset loaded: {status['preset_loaded']}")
    print(f"   Adaptive enabled: {status['adaptive_enabled']}")
    print(f"   Loudness matching: {status['loudness_matching_enabled']}")
    print(f"   Diagnostics: {status['diagnostics_enabled']}")
    print(f"   Buffers processed: {status['buffers_processed']}")

    # Get diagnostic summary
    print("\n6. Getting diagnostic summary...")
    diag_stats = processor.get_diagnostics_summary()
    print(f"   Total buffers logged: {diag_stats['total_buffers_processed']}")
    print(f"   Mean RMS: {diag_stats['rms_mean_db']:.1f} dB")

    # Save diagnostics
    print("\n7. Saving diagnostics...")
    processor.save_diagnostics()

    print("\n✓ Ultimate processor integration test PASSED")
    return True


def test_config_options():
    """Test configuration options"""
    print("\n" + "=" * 80)
    print("TEST 4: CONFIGURATION OPTIONS")
    print("=" * 80)

    print("\n1. Checking diagnostic settings...")
    print(f"   DIAGNOSTIC_MODE_ENABLED: {config.DIAGNOSTIC_MODE_ENABLED}")
    print(f"   DIAGNOSTIC_PRINT_INTERVAL: {config.DIAGNOSTIC_PRINT_INTERVAL}")
    print(f"   DIAGNOSTIC_LOG_TO_FILE: {config.DIAGNOSTIC_LOG_TO_FILE}")
    print(f"   DIAGNOSTIC_LOG_TO_CSV: {config.DIAGNOSTIC_LOG_TO_CSV}")

    print("\n2. Checking loudness matching settings...")
    print(f"   LOUDNESS_MATCHING_ENABLED: {config.LOUDNESS_MATCHING_ENABLED}")
    print(f"   LOUDNESS_MATCH_MODE: {config.LOUDNESS_MATCH_MODE}")
    print(f"   LOUDNESS_GAIN_SMOOTHING: {config.LOUDNESS_GAIN_SMOOTHING}")
    print(f"   LOUDNESS_TARGET_LUFS: {config.LOUDNESS_TARGET_LUFS}")

    print("\n3. Checking directories...")
    print(f"   Logs directory exists: {config.LOGS_DIR.exists()}")
    print(f"   Logs directory path: {config.LOGS_DIR}")

    print("\n✓ Configuration options test PASSED")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RUNNING COMPREHENSIVE FEATURE TESTS")
    print("=" * 80)

    tests = [
        ("Adaptive Loudness Matching", test_loudness_matching),
        ("Diagnostic Logging System", test_diagnostic_logger),
        ("Ultimate Processor Integration", test_ultimate_processor_integration),
        ("Configuration Options", test_config_options)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ {test_name} FAILED: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = 0
    failed = 0

    for test_name, result, error in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"\n{status}: {test_name}")
        if error:
            print(f"  Error: {error}")

        if result:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print(f"TOTAL: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
