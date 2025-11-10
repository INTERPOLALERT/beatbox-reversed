"""
Adaptive Preset Matching Module
Modifies extracted presets to match user's microphone characteristics
Ensures presets behave identically regardless of mic setup
"""
import numpy as np
import json
from typing import Dict, Optional
from pathlib import Path
import copy


class PresetAdapter:
    """
    Adapts reference presets to match user's microphone profile
    """

    def __init__(self):
        """Initialize preset adapter"""
        self.reference_preset = None
        self.mic_profile = None
        self.adapted_preset = None

    def load_reference_preset(self, preset_path: str):
        """
        Load reference preset (from analysis)

        Args:
            preset_path: Path to reference preset JSON
        """
        with open(preset_path, 'r') as f:
            self.reference_preset = json.load(f)

        print(f"✅ Reference preset loaded: {preset_path}")

    def load_mic_profile(self, profile_path: str):
        """
        Load microphone profile

        Args:
            profile_path: Path to mic profile JSON
        """
        with open(profile_path, 'r') as f:
            self.mic_profile = json.load(f)

        print(f"✅ Mic profile loaded: {profile_path}")

    def adapt_preset(self, reference_preset: Optional[Dict] = None,
                    mic_profile: Optional[Dict] = None) -> Dict:
        """
        Adapt preset to microphone characteristics

        Args:
            reference_preset: Reference preset (or use loaded)
            mic_profile: Mic profile (or use loaded)

        Returns:
            Adapted preset dictionary
        """
        if reference_preset is not None:
            self.reference_preset = reference_preset
        if mic_profile is not None:
            self.mic_profile = mic_profile

        if self.reference_preset is None:
            raise ValueError("No reference preset loaded")
        if self.mic_profile is None:
            raise ValueError("No mic profile loaded")

        print("\n" + "="*60)
        print("ADAPTIVE PRESET MATCHING")
        print("="*60)

        # Deep copy to avoid modifying original
        adapted = copy.deepcopy(self.reference_preset)

        # 1. Adapt Input Gain
        print("\n[1/5] Adapting Input Gain...")
        adapted = self._adapt_input_gain(adapted)

        # 2. Adapt EQ Curve
        print("\n[2/5] Adapting EQ Curve...")
        adapted = self._adapt_eq_curve(adapted)

        # 3. Adapt Compression Parameters
        print("\n[3/5] Adapting Compression...")
        adapted = self._adapt_compression(adapted)

        # 4. Adapt High-Frequency Processing
        print("\n[4/5] Adapting High-Frequency Processing...")
        adapted = self._adapt_high_frequency_processing(adapted)

        # 5. Adapt Limiting
        print("\n[5/5] Adapting Limiting...")
        adapted = self._adapt_limiting(adapted)

        # Store adapted preset
        self.adapted_preset = adapted

        # Add adaptation metadata
        adapted['adaptation_metadata'] = {
            'adapted': True,
            'mic_profile_used': True,
            'adaptation_version': '1.0'
        }

        print("\n" + "="*60)
        print("✅ PRESET ADAPTATION COMPLETE")
        print("="*60)

        self._print_adaptation_summary()

        return adapted

    def _adapt_input_gain(self, preset: Dict) -> Dict:
        """Adapt input gain based on mic loudness profile"""
        # Get mic loudness characteristics
        mic_loudness = self.mic_profile['loudness']
        mic_rms_db = mic_loudness['rms_db']
        mic_lufs = mic_loudness.get('lufs', mic_rms_db)

        # Target level for processing chain (standard: -18 LUFS)
        target_lufs = -18.0

        # Calculate gain offset needed
        gain_offset_db = target_lufs - mic_lufs

        # Safety limits
        gain_offset_db = np.clip(gain_offset_db, -24.0, 24.0)

        # Add to metadata
        if 'metadata' not in preset:
            preset['metadata'] = {}

        preset['metadata']['input_gain_offset_db'] = float(gain_offset_db)

        print(f"  Input gain offset: {gain_offset_db:+.1f} dB")
        print(f"  (Mic LUFS: {mic_lufs:.1f} → Target: {target_lufs:.1f})")

        return preset

    def _adapt_eq_curve(self, preset: Dict) -> Dict:
        """Adapt EQ curve to compensate for mic frequency response"""
        if 'spectral' not in preset or 'spectral' not in self.mic_profile:
            print("  No spectral data - skipping EQ adaptation")
            return preset

        # Get mic spectral characteristics
        mic_spectral = self.mic_profile['spectral']
        mic_low_bias = mic_spectral['low_freq_bias_db']
        mic_mid_bias = mic_spectral['mid_freq_bias_db']
        mic_high_bias = mic_spectral['high_freq_bias_db']

        # Get reference EQ curve
        eq_curve = preset['spectral']['eq_curve']

        # Adapt each band
        adaptations_made = 0

        for band in eq_curve:
            freq = band['frequency']
            original_gain = band['gain_db']

            # Determine which frequency range this band is in
            if freq < 250:
                # Low frequencies - compensate for mic low bias
                compensation = -mic_low_bias
            elif freq < 2000:
                # Mid frequencies - compensate for mic mid bias
                compensation = -mic_mid_bias
            else:
                # High frequencies - compensate for mic high bias
                compensation = -mic_high_bias

            # Apply compensation
            # Use a scaling factor to avoid over-correction
            compensation_scaled = compensation * 0.7

            new_gain = original_gain + compensation_scaled
            band['gain_db'] = float(new_gain)

            # Track significant adaptations
            if abs(compensation_scaled) > 0.5:
                adaptations_made += 1

        print(f"  Adapted {adaptations_made} EQ bands")
        print(f"  Low freq compensation: {-mic_low_bias*0.7:+.1f} dB")
        print(f"  Mid freq compensation: {-mic_mid_bias*0.7:+.1f} dB")
        print(f"  High freq compensation: {-mic_high_bias*0.7:+.1f} dB")

        return preset

    def _adapt_compression(self, preset: Dict) -> Dict:
        """Adapt compression parameters for mic dynamics"""
        if 'dynamics' not in preset or 'dynamics' not in self.mic_profile:
            print("  No dynamics data - skipping compression adaptation")
            return preset

        # Get mic dynamics characteristics
        mic_dynamics = self.mic_profile['dynamics']
        mic_crest_factor = mic_dynamics['crest_factor_db']
        mic_transient_strength = mic_dynamics['transient_strength']

        # Get reference compression
        compression = preset['dynamics']['compression']

        # Adapt threshold based on mic input level
        mic_rms_db = self.mic_profile['loudness']['rms_db']
        target_rms_db = -12.0  # Standard operating level

        # Adjust threshold
        threshold_offset = target_rms_db - mic_rms_db
        original_threshold = compression['threshold_db']
        new_threshold = original_threshold + threshold_offset

        # Limit threshold adjustment
        new_threshold = np.clip(new_threshold, -40.0, -5.0)
        compression['threshold_db'] = float(new_threshold)

        # Adapt attack/release for mic transient character
        original_attack = compression['attack_ms']
        original_release = compression['release_ms']

        if mic_transient_strength == 'strong':
            # Strong transients - slightly faster attack to catch them
            attack_scale = 0.9
        elif mic_transient_strength == 'soft':
            # Soft transients - slightly slower attack to avoid over-compression
            attack_scale = 1.1
        else:
            attack_scale = 1.0

        compression['attack_ms'] = float(original_attack * attack_scale)

        print(f"  Threshold: {original_threshold:.1f} → {new_threshold:.1f} dB ({threshold_offset:+.1f} dB)")
        print(f"  Attack: {original_attack:.1f} → {compression['attack_ms']:.1f} ms")
        print(f"  (Transient character: {mic_transient_strength})")

        return preset

    def _adapt_high_frequency_processing(self, preset: Dict) -> Dict:
        """Adapt high-frequency processing (de-esser, exciter)"""
        if 'effects' not in preset:
            print("  No effects data - skipping HF adaptation")
            return preset

        # Get mic high-frequency characteristics
        mic_spectral = self.mic_profile.get('spectral', {})
        mic_high_bias = mic_spectral.get('high_freq_bias_db', 0.0)

        effects = preset['effects']

        # Adapt de-esser if present
        if 'deessing' in effects and effects['deessing'].get('detected', False):
            deessing = effects['deessing']

            # If mic is naturally bright, de-essing may need to be more aggressive
            # If mic is dark, de-essing can be gentler
            if mic_high_bias > 2.0:
                # Bright mic - increase de-essing
                ratio_increase = 0.3
                threshold_decrease = 2.0
            elif mic_high_bias < -2.0:
                # Dark mic - decrease de-essing
                ratio_increase = -0.2
                threshold_decrease = -2.0
            else:
                ratio_increase = 0.0
                threshold_decrease = 0.0

            original_ratio = deessing.get('ratio', 2.0)
            original_threshold = deessing.get('threshold_db', -20.0)

            deessing['ratio'] = float(np.clip(original_ratio + ratio_increase, 1.0, 8.0))
            deessing['threshold_db'] = float(original_threshold - threshold_decrease)

            print(f"  De-esser ratio: {original_ratio:.1f} → {deessing['ratio']:.1f}")
            print(f"  De-esser threshold: {original_threshold:.1f} → {deessing['threshold_db']:.1f} dB")

        # Adapt exciter if present
        if 'exciter' in effects and effects['exciter'].get('detected', False):
            exciter = effects['exciter']
            original_amount = exciter.get('amount', 0.0)

            # If mic is dark, may want more exciter
            # If mic is bright, may want less exciter
            if mic_high_bias < -2.0:
                amount_scale = 1.2
            elif mic_high_bias > 2.0:
                amount_scale = 0.8
            else:
                amount_scale = 1.0

            exciter['amount'] = float(np.clip(original_amount * amount_scale, 0.0, 1.0))

            print(f"  Exciter amount: {original_amount:.2f} → {exciter['amount']:.2f}")

        return preset

    def _adapt_limiting(self, preset: Dict) -> Dict:
        """Adapt limiter ceiling based on expected levels"""
        if 'dynamics' not in preset:
            print("  No dynamics data - skipping limiting adaptation")
            return preset

        limiting = preset['dynamics']['limiting']

        if not limiting.get('is_limited', False):
            print("  No limiting detected - skipping")
            return limiting

        # Get mic dynamics
        mic_dynamics = self.mic_profile.get('dynamics', {})
        mic_crest_factor = mic_dynamics.get('crest_factor_db', 12.0)

        # Get original ceiling
        original_ceiling = limiting['ceiling_db']

        # If mic has very high crest factor (strong peaks), may need lower ceiling
        # If mic has low crest factor (already compressed), can have higher ceiling
        if mic_crest_factor > 15:
            ceiling_adjust = -1.0  # More headroom needed
        elif mic_crest_factor < 8:
            ceiling_adjust = 0.5   # Can push closer to 0 dB
        else:
            ceiling_adjust = 0.0

        new_ceiling = np.clip(original_ceiling + ceiling_adjust, -6.0, -0.3)
        limiting['ceiling_db'] = float(new_ceiling)

        print(f"  Limiter ceiling: {original_ceiling:.1f} → {new_ceiling:.1f} dB")
        print(f"  (Mic crest factor: {mic_crest_factor:.1f} dB)")

        return preset

    def _print_adaptation_summary(self):
        """Print summary of adaptations made"""
        print("\n" + "="*60)
        print("ADAPTATION SUMMARY")
        print("="*60)

        if not self.adapted_preset:
            print("No adaptations made")
            return

        # Input gain
        if 'metadata' in self.adapted_preset:
            gain_offset = self.adapted_preset['metadata'].get('input_gain_offset_db', 0.0)
            print(f"\nInput Gain Offset: {gain_offset:+.1f} dB")

        # EQ compensation
        mic_spectral = self.mic_profile['spectral']
        print(f"\nMic Character: {mic_spectral['character']}")
        print(f"  Low Bias: {mic_spectral['low_freq_bias_db']:+.1f} dB")
        print(f"  High Bias: {mic_spectral['high_freq_bias_db']:+.1f} dB")

        # Dynamics adaptation
        mic_dynamics = self.mic_profile['dynamics']
        print(f"\nMic Transient Character: {mic_dynamics['transient_strength']}")

        print("\n" + "="*60)

    def save_adapted_preset(self, preset_name: str, output_dir: Optional[Path] = None) -> Path:
        """
        Save adapted preset to JSON

        Args:
            preset_name: Name for adapted preset
            output_dir: Output directory

        Returns:
            Path to saved preset
        """
        if self.adapted_preset is None:
            raise ValueError("No adapted preset available. Run adapt_preset() first.")

        if output_dir is None:
            from config import PRESETS_DIR
            output_dir = PRESETS_DIR

        preset_path = output_dir / f"{preset_name}_adapted.json"

        with open(preset_path, 'w') as f:
            json.dump(self.adapted_preset, f, indent=2)

        print(f"\n✅ Adapted preset saved: {preset_path}")

        return preset_path

    def compare_presets(self) -> Dict:
        """
        Compare reference and adapted presets

        Returns:
            Dictionary with comparison data
        """
        if not self.reference_preset or not self.adapted_preset:
            raise ValueError("Both reference and adapted presets must be loaded")

        comparison = {
            'input_gain_change': 0.0,
            'eq_changes': [],
            'compression_changes': {},
            'effects_changes': {}
        }

        # Input gain
        if 'metadata' in self.adapted_preset:
            comparison['input_gain_change'] = self.adapted_preset['metadata'].get('input_gain_offset_db', 0.0)

        # EQ changes
        if 'spectral' in self.reference_preset and 'spectral' in self.adapted_preset:
            ref_eq = self.reference_preset['spectral']['eq_curve']
            adapted_eq = self.adapted_preset['spectral']['eq_curve']

            for ref_band, adapted_band in zip(ref_eq, adapted_eq):
                change = adapted_band['gain_db'] - ref_band['gain_db']
                if abs(change) > 0.1:
                    comparison['eq_changes'].append({
                        'frequency': ref_band['frequency'],
                        'name': ref_band['name'],
                        'original': ref_band['gain_db'],
                        'adapted': adapted_band['gain_db'],
                        'change': change
                    })

        # Compression changes
        if 'dynamics' in self.reference_preset and 'dynamics' in self.adapted_preset:
            ref_comp = self.reference_preset['dynamics']['compression']
            adapted_comp = self.adapted_preset['dynamics']['compression']

            comparison['compression_changes'] = {
                'threshold_change': adapted_comp['threshold_db'] - ref_comp['threshold_db'],
                'attack_change': adapted_comp['attack_ms'] - ref_comp['attack_ms']
            }

        return comparison


def adapt_preset_to_mic(preset_path: str, mic_profile_path: str, output_name: str) -> Dict:
    """
    Convenience function to adapt a preset to a microphone profile

    Args:
        preset_path: Path to reference preset
        mic_profile_path: Path to mic profile
        output_name: Name for adapted preset

    Returns:
        Adapted preset dictionary
    """
    adapter = PresetAdapter()
    adapter.load_reference_preset(preset_path)
    adapter.load_mic_profile(mic_profile_path)

    adapted = adapter.adapt_preset()
    adapter.save_adapted_preset(output_name)

    return adapted


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python preset_adapter.py <preset.json> <mic_profile.json> <output_name>")
        print("\nExample:")
        print("  python preset_adapter.py reference_preset_v2.json mic_profile_default.json my_adapted_preset")
        sys.exit(1)

    preset_file = sys.argv[1]
    profile_file = sys.argv[2]
    output_name = sys.argv[3]

    # Adapt preset
    adapted = adapt_preset_to_mic(preset_file, profile_file, output_name)

    print("\n✅ Preset adaptation complete!")
    print(f"📁 Adapted preset saved as: {output_name}_adapted.json")
    print("\nYou can now use this preset with your microphone!")
