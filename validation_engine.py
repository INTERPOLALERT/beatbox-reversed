"""
Live Validation & Auto-Adjustment Engine
Tests processed mic output against reference audio and auto-tunes preset
"""
import numpy as np
import librosa
import pyloudnorm as pyln
from scipy import signal
from typing import Dict, List, Tuple, Optional
import copy


class ValidationEngine:
    """
    Validates processed audio against reference and auto-adjusts preset
    """

    def __init__(self, sample_rate: int = 44100, tolerance_db: float = 2.0):
        """
        Initialize validation engine

        Args:
            sample_rate: Sample rate in Hz
            tolerance_db: Acceptable deviation in dB for validation
        """
        self.sample_rate = sample_rate
        self.tolerance_db = tolerance_db
        self.meter = pyln.Meter(sample_rate)

        # Validation thresholds
        self.tonal_tolerance_db = tolerance_db
        self.loudness_tolerance_lufs = 3.0
        self.dynamic_tolerance_db = 3.0

        # Auto-adjustment parameters
        self.max_iterations = 3
        self.adjustment_step_size = 0.5  # Conservative adjustment step

    def validate(self, reference_audio: np.ndarray,
                processed_audio: np.ndarray,
                preset: Dict) -> Dict:
        """
        Validate processed audio against reference

        Args:
            reference_audio: Reference audio (target sound)
            processed_audio: Processed mic output
            preset: Current preset being used

        Returns:
            Validation results dictionary
        """
        print("\n" + "="*60)
        print("VALIDATION ENGINE - COMPARING OUTPUT TO REFERENCE")
        print("="*60)

        # 1. Tonal Difference (Spectral Similarity)
        print("\n[1/4] Tonal Analysis...")
        tonal_diff = self._compare_tonal_balance(reference_audio, processed_audio)

        # 2. Loudness Difference
        print("\n[2/4] Loudness Analysis...")
        loudness_diff = self._compare_loudness(reference_audio, processed_audio)

        # 3. Dynamic Difference (Crest Factor, Dynamic Range)
        print("\n[3/4] Dynamics Analysis...")
        dynamic_diff = self._compare_dynamics(reference_audio, processed_audio)

        # 4. High-Frequency Dynamics (De-esser effectiveness)
        print("\n[4/4] High-Frequency Dynamics...")
        hf_diff = self._compare_hf_dynamics(reference_audio, processed_audio)

        # Compile results
        validation_results = {
            'tonal_difference': tonal_diff,
            'loudness_difference': loudness_diff,
            'dynamic_difference': dynamic_diff,
            'hf_difference': hf_diff,
            'passes_validation': self._check_validation_pass(tonal_diff, loudness_diff, dynamic_diff, hf_diff)
        }

        # Print summary
        self._print_validation_summary(validation_results)

        return validation_results

    def _compare_tonal_balance(self, reference: np.ndarray, processed: np.ndarray) -> Dict:
        """Compare tonal balance across frequency bands"""
        # Define frequency bands
        bands = {
            'Low': (80, 250),
            'Low-Mid': (250, 500),
            'Mid': (500, 2000),
            'High-Mid': (2000, 4000),
            'High': (4000, 8000)
        }

        band_differences = {}
        total_deviation = 0.0

        for band_name, (low_freq, high_freq) in bands.items():
            # Extract band for reference
            sos = signal.butter(4, [low_freq, high_freq], 'bp', fs=self.sample_rate, output='sos')

            ref_band = signal.sosfilt(sos, reference)
            proc_band = signal.sosfilt(sos, processed)

            # Measure RMS level
            ref_rms = np.sqrt(np.mean(ref_band ** 2))
            proc_rms = np.sqrt(np.mean(proc_band ** 2))

            # Calculate difference in dB
            if ref_rms > 1e-10 and proc_rms > 1e-10:
                diff_db = 20 * np.log10(proc_rms / ref_rms)
            else:
                diff_db = 0.0

            band_differences[band_name] = {
                'difference_db': float(diff_db),
                'freq_range': (low_freq, high_freq),
                'within_tolerance': abs(diff_db) <= self.tonal_tolerance_db
            }

            total_deviation += abs(diff_db)

            # Print result
            status = "✓" if abs(diff_db) <= self.tonal_tolerance_db else "✗"
            print(f"  {status} {band_name:12s} ({low_freq:5d}-{high_freq:5d} Hz): {diff_db:+5.1f} dB")

        avg_deviation = total_deviation / len(bands)

        return {
            'band_differences': band_differences,
            'average_deviation_db': float(avg_deviation),
            'max_deviation_db': float(max([abs(b['difference_db']) for b in band_differences.values()])),
            'within_tolerance': avg_deviation <= self.tonal_tolerance_db
        }

    def _compare_loudness(self, reference: np.ndarray, processed: np.ndarray) -> Dict:
        """Compare overall loudness"""
        # RMS comparison
        ref_rms = np.sqrt(np.mean(reference ** 2))
        proc_rms = np.sqrt(np.mean(processed ** 2))

        rms_diff_db = 20 * np.log10((proc_rms + 1e-10) / (ref_rms + 1e-10))

        # LUFS comparison
        try:
            ref_lufs = self.meter.integrated_loudness(reference)
            proc_lufs = self.meter.integrated_loudness(processed)
            lufs_diff = proc_lufs - ref_lufs
        except Exception as e:
            lufs_diff = rms_diff_db

        # Peak comparison
        ref_peak = np.max(np.abs(reference))
        proc_peak = np.max(np.abs(processed))
        peak_diff_db = 20 * np.log10((proc_peak + 1e-10) / (ref_peak + 1e-10))

        within_tolerance = abs(lufs_diff) <= self.loudness_tolerance_lufs

        status = "✓" if within_tolerance else "✗"
        print(f"  {status} RMS Difference: {rms_diff_db:+.1f} dB")
        print(f"  {status} LUFS Difference: {lufs_diff:+.1f} LUFS")
        print(f"  {status} Peak Difference: {peak_diff_db:+.1f} dB")

        return {
            'rms_difference_db': float(rms_diff_db),
            'lufs_difference': float(lufs_diff),
            'peak_difference_db': float(peak_diff_db),
            'within_tolerance': within_tolerance
        }

    def _compare_dynamics(self, reference: np.ndarray, processed: np.ndarray) -> Dict:
        """Compare dynamic characteristics"""
        # Crest factor comparison
        ref_rms = np.sqrt(np.mean(reference ** 2))
        ref_peak = np.max(np.abs(reference))
        ref_crest = 20 * np.log10((ref_peak + 1e-10) / (ref_rms + 1e-10))

        proc_rms = np.sqrt(np.mean(processed ** 2))
        proc_peak = np.max(np.abs(processed))
        proc_crest = 20 * np.log10((proc_peak + 1e-10) / (proc_rms + 1e-10))

        crest_diff_db = proc_crest - ref_crest

        # Dynamic range comparison (95th - 10th percentile)
        ref_p95 = np.percentile(np.abs(reference), 95)
        ref_p10 = np.percentile(np.abs(reference), 10)
        ref_dr = 20 * np.log10((ref_p95 + 1e-10) / (ref_p10 + 1e-10))

        proc_p95 = np.percentile(np.abs(processed), 95)
        proc_p10 = np.percentile(np.abs(processed), 10)
        proc_dr = 20 * np.log10((proc_p95 + 1e-10) / (proc_p10 + 1e-10))

        dr_diff_db = proc_dr - ref_dr

        within_tolerance = abs(crest_diff_db) <= self.dynamic_tolerance_db

        status = "✓" if within_tolerance else "✗"
        print(f"  {status} Crest Factor Difference: {crest_diff_db:+.1f} dB")
        print(f"  {status} Dynamic Range Difference: {dr_diff_db:+.1f} dB")

        return {
            'crest_factor_difference_db': float(crest_diff_db),
            'dynamic_range_difference_db': float(dr_diff_db),
            'within_tolerance': within_tolerance
        }

    def _compare_hf_dynamics(self, reference: np.ndarray, processed: np.ndarray) -> Dict:
        """Compare high-frequency dynamics (for de-esser validation)"""
        # Extract high frequencies (4-8 kHz)
        sos = signal.butter(4, [4000, 8000], 'bp', fs=self.sample_rate, output='sos')

        ref_hf = signal.sosfilt(sos, reference)
        proc_hf = signal.sosfilt(sos, processed)

        # Measure HF RMS
        ref_hf_rms = np.sqrt(np.mean(ref_hf ** 2))
        proc_hf_rms = np.sqrt(np.mean(proc_hf ** 2))

        hf_level_diff_db = 20 * np.log10((proc_hf_rms + 1e-10) / (ref_hf_rms + 1e-10))

        # Measure HF dynamic range
        ref_hf_p95 = np.percentile(np.abs(ref_hf), 95)
        ref_hf_p10 = np.percentile(np.abs(ref_hf), 10)
        ref_hf_dr = 20 * np.log10((ref_hf_p95 + 1e-10) / (ref_hf_p10 + 1e-10))

        proc_hf_p95 = np.percentile(np.abs(proc_hf), 95)
        proc_hf_p10 = np.percentile(np.abs(proc_hf), 10)
        proc_hf_dr = 20 * np.log10((proc_hf_p95 + 1e-10) / (proc_hf_p10 + 1e-10))

        hf_dr_diff_db = proc_hf_dr - ref_hf_dr

        within_tolerance = abs(hf_level_diff_db) <= (self.tonal_tolerance_db * 1.5)  # Slightly more lenient for HF

        status = "✓" if within_tolerance else "✗"
        print(f"  {status} HF Level Difference: {hf_level_diff_db:+.1f} dB")
        print(f"  {status} HF Dynamic Range Difference: {hf_dr_diff_db:+.1f} dB")

        return {
            'hf_level_difference_db': float(hf_level_diff_db),
            'hf_dr_difference_db': float(hf_dr_diff_db),
            'within_tolerance': within_tolerance
        }

    def _check_validation_pass(self, tonal_diff: Dict, loudness_diff: Dict,
                               dynamic_diff: Dict, hf_diff: Dict) -> bool:
        """Check if validation passes all criteria"""
        return (tonal_diff['within_tolerance'] and
                loudness_diff['within_tolerance'] and
                dynamic_diff['within_tolerance'] and
                hf_diff['within_tolerance'])

    def _print_validation_summary(self, results: Dict):
        """Print validation summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        passes = results['passes_validation']

        if passes:
            print("\n✅ VALIDATION PASSED")
            print("Processed audio matches reference within tolerance")
        else:
            print("\n⚠ VALIDATION FAILED")
            print("Processed audio deviates from reference")

        print(f"\nTonal Deviation: {results['tonal_difference']['average_deviation_db']:.1f} dB "
              f"(tolerance: {self.tonal_tolerance_db:.1f} dB)")
        print(f"Loudness Deviation: {results['loudness_difference']['lufs_difference']:.1f} LUFS "
              f"(tolerance: {self.loudness_tolerance_lufs:.1f} LUFS)")
        print(f"Dynamic Deviation: {results['dynamic_difference']['crest_factor_difference_db']:.1f} dB "
              f"(tolerance: {self.dynamic_tolerance_db:.1f} dB)")

        print("\n" + "="*60)

    def auto_adjust_preset(self, reference_audio: np.ndarray,
                          processed_audio: np.ndarray,
                          preset: Dict,
                          validation_results: Dict) -> Dict:
        """
        Automatically adjust preset based on validation results

        Args:
            reference_audio: Reference audio
            processed_audio: Processed audio
            preset: Current preset
            validation_results: Results from validation

        Returns:
            Adjusted preset
        """
        print("\n" + "="*60)
        print("AUTO-ADJUSTING PRESET")
        print("="*60)

        adjusted_preset = copy.deepcopy(preset)

        # 1. Adjust Input Gain (based on loudness difference)
        adjusted_preset = self._adjust_input_gain(adjusted_preset, validation_results['loudness_difference'])

        # 2. Adjust EQ (based on tonal difference)
        adjusted_preset = self._adjust_eq(adjusted_preset, validation_results['tonal_difference'])

        # 3. Adjust Compression (based on dynamic difference)
        adjusted_preset = self._adjust_compression(adjusted_preset, validation_results['dynamic_difference'])

        # 4. Adjust High-Frequency Processing (based on HF difference)
        adjusted_preset = self._adjust_hf_processing(adjusted_preset, validation_results['hf_difference'])

        print("\n✅ Auto-adjustment complete")

        return adjusted_preset

    def _adjust_input_gain(self, preset: Dict, loudness_diff: Dict) -> Dict:
        """Adjust input gain based on loudness difference"""
        lufs_diff = loudness_diff['lufs_difference']

        if abs(lufs_diff) <= self.loudness_tolerance_lufs:
            print("\n[1/4] Input Gain: Within tolerance, no adjustment needed")
            return preset

        # Calculate adjustment
        # If processed is louder than reference (positive diff), reduce input gain
        gain_adjustment = -lufs_diff * self.adjustment_step_size

        # Apply adjustment
        if 'metadata' not in preset:
            preset['metadata'] = {}

        current_gain = preset['metadata'].get('input_gain_offset_db', 0.0)
        new_gain = current_gain + gain_adjustment

        preset['metadata']['input_gain_offset_db'] = float(np.clip(new_gain, -24.0, 24.0))

        print(f"\n[1/4] Input Gain: Adjusted by {gain_adjustment:+.1f} dB ({current_gain:+.1f} → {new_gain:+.1f} dB)")

        return preset

    def _adjust_eq(self, preset: Dict, tonal_diff: Dict) -> Dict:
        """Adjust EQ based on tonal differences"""
        if tonal_diff['within_tolerance']:
            print("\n[2/4] EQ: Within tolerance, no adjustment needed")
            return preset

        if 'spectral' not in preset:
            print("\n[2/4] EQ: No spectral data, skipping")
            return preset

        eq_curve = preset['spectral']['eq_curve']
        band_diffs = tonal_diff['band_differences']

        adjustments_made = 0

        # Map band names to frequency ranges
        band_freq_map = {
            'Low': (80, 250),
            'Low-Mid': (250, 500),
            'Mid': (500, 2000),
            'High-Mid': (2000, 4000),
            'High': (4000, 8000)
        }

        for band_name, band_data in band_diffs.items():
            if not band_data['within_tolerance']:
                diff_db = band_data['difference_db']

                # If processed is louder than reference in this band, reduce EQ gain
                eq_adjustment = -diff_db * self.adjustment_step_size

                # Find EQ bands in this frequency range
                freq_range = band_freq_map[band_name]

                for eq_band in eq_curve:
                    if freq_range[0] <= eq_band['frequency'] <= freq_range[1]:
                        eq_band['gain_db'] += eq_adjustment
                        adjustments_made += 1

        print(f"\n[2/4] EQ: Adjusted {adjustments_made} bands")

        return preset

    def _adjust_compression(self, preset: Dict, dynamic_diff: Dict) -> Dict:
        """Adjust compression based on dynamic differences"""
        if dynamic_diff['within_tolerance']:
            print("\n[3/4] Compression: Within tolerance, no adjustment needed")
            return preset

        if 'dynamics' not in preset:
            print("\n[3/4] Compression: No dynamics data, skipping")
            return preset

        compression = preset['dynamics']['compression']

        crest_diff = dynamic_diff['crest_factor_difference_db']

        # If processed has lower crest factor than reference, reduce compression
        # (lower crest factor = more compressed)
        if crest_diff < -self.dynamic_tolerance_db:
            # Too much compression - increase threshold or reduce ratio
            threshold_adjustment = 2.0 * self.adjustment_step_size
            ratio_adjustment = -0.2 * self.adjustment_step_size

            print(f"\n[3/4] Compression: Reducing compression (crest factor too low)")
        elif crest_diff > self.dynamic_tolerance_db:
            # Too little compression - decrease threshold or increase ratio
            threshold_adjustment = -2.0 * self.adjustment_step_size
            ratio_adjustment = 0.2 * self.adjustment_step_size

            print(f"\n[3/4] Compression: Increasing compression (crest factor too high)")
        else:
            print(f"\n[3/4] Compression: Within tolerance")
            return preset

        # Apply adjustments
        compression['threshold_db'] = float(np.clip(compression['threshold_db'] + threshold_adjustment, -40.0, -5.0))
        compression['ratio'] = float(np.clip(compression['ratio'] + ratio_adjustment, 1.0, 20.0))

        print(f"  Threshold adjusted by {threshold_adjustment:+.1f} dB")
        print(f"  Ratio adjusted by {ratio_adjustment:+.2f}")

        return preset

    def _adjust_hf_processing(self, preset: Dict, hf_diff: Dict) -> Dict:
        """Adjust high-frequency processing"""
        if hf_diff['within_tolerance']:
            print("\n[4/4] High-Frequency: Within tolerance, no adjustment needed")
            return preset

        if 'effects' not in preset:
            print("\n[4/4] High-Frequency: No effects data, skipping")
            return preset

        hf_level_diff = hf_diff['hf_level_difference_db']

        # Adjust de-esser if present
        if 'deessing' in preset['effects'] and preset['effects']['deessing'].get('detected', False):
            deessing = preset['effects']['deessing']

            # If processed HF is quieter than reference, reduce de-essing
            if hf_level_diff < -self.tonal_tolerance_db:
                # Too much de-essing
                ratio_adjustment = -0.3 * self.adjustment_step_size
                threshold_adjustment = 2.0 * self.adjustment_step_size

                print(f"\n[4/4] High-Frequency: Reducing de-essing")
            else:
                # Too little de-essing
                ratio_adjustment = 0.3 * self.adjustment_step_size
                threshold_adjustment = -2.0 * self.adjustment_step_size

                print(f"\n[4/4] High-Frequency: Increasing de-essing")

            deessing['ratio'] = float(np.clip(deessing['ratio'] + ratio_adjustment, 1.0, 8.0))
            deessing['threshold_db'] = float(deessing['threshold_db'] + threshold_adjustment)

            print(f"  De-esser ratio adjusted by {ratio_adjustment:+.2f}")
            print(f"  De-esser threshold adjusted by {threshold_adjustment:+.1f} dB")

        return preset

    def validation_loop(self, reference_audio: np.ndarray,
                       process_func,
                       initial_preset: Dict,
                       test_audio: np.ndarray,
                       max_iterations: Optional[int] = None) -> Tuple[Dict, List[Dict]]:
        """
        Run validation loop with auto-adjustment

        Args:
            reference_audio: Reference audio (target sound)
            process_func: Function that processes audio: process_func(audio, preset) -> processed_audio
            initial_preset: Starting preset
            test_audio: Test audio to process (user's mic input)
            max_iterations: Maximum adjustment iterations (default: self.max_iterations)

        Returns:
            Tuple of (final_preset, validation_history)
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        print("\n" + "="*60)
        print("VALIDATION LOOP WITH AUTO-ADJUSTMENT")
        print("="*60)

        current_preset = copy.deepcopy(initial_preset)
        validation_history = []

        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1} / {max_iterations}")
            print(f"{'='*60}")

            # Process test audio with current preset
            processed = process_func(test_audio, current_preset)

            # Validate
            validation_results = self.validate(reference_audio, processed, current_preset)

            validation_history.append({
                'iteration': iteration + 1,
                'preset': copy.deepcopy(current_preset),
                'validation_results': validation_results
            })

            # Check if validation passes
            if validation_results['passes_validation']:
                print(f"\n✅ Validation passed on iteration {iteration + 1}!")
                print("Preset is optimized for your microphone.")
                break

            # If not last iteration, auto-adjust
            if iteration < max_iterations - 1:
                print(f"\n⚙ Auto-adjusting preset for iteration {iteration + 2}...")
                current_preset = self.auto_adjust_preset(reference_audio, processed,
                                                        current_preset, validation_results)
            else:
                print(f"\n⚠ Maximum iterations reached ({max_iterations})")
                print("Preset is close but may not perfectly match reference.")

        return current_preset, validation_history


if __name__ == "__main__":
    import sys

    print("Validation Engine Module")
    print("This module is used by the calibration workflow")
    print("Run calibration_workflow.py for complete validation testing")
