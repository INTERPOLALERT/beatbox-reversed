"""
Beatbox Processor V2 - Real-time DSP Chain Application
Applies extracted processing parameters to live microphone input
"""
import numpy as np
from pedalboard import (
    Pedalboard, Compressor, Limiter, Gain,
    HighShelfFilter, LowShelfFilter, PeakFilter,
    Reverb, Distortion
)
import json
from typing import Dict, Optional
import config


class BeatboxProcessorV2:
    """
    Real-time processor that applies V2 analysis results to live audio
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize processor

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

        # Processing chain
        self.pedalboard = None

        # Preset data
        self.preset_data = None
        self.preset_loaded = False

        # Control parameters
        self.wet_dry_mix = 1.0  # 0=dry, 1=fully processed
        self.input_gain_db = 0.0
        self.output_gain_db = 0.0

    def load_preset(self, preset_path: str):
        """
        Load V2 preset and build processing chain

        Args:
            preset_path: Path to preset JSON
        """
        print(f"\nLoading preset: {preset_path}")

        with open(preset_path, 'r') as f:
            self.preset_data = json.load(f)

        # Build processing chain
        self._build_processing_chain()

        self.preset_loaded = True

        print("✅ Preset loaded and processing chain built")

    def _build_processing_chain(self):
        """
        Build Pedalboard processing chain from preset data
        """
        if not self.preset_data:
            raise ValueError("No preset data loaded")

        chain = []

        print("\nBuilding processing chain...")

        # 1. INPUT GAIN
        if self.input_gain_db != 0:
            chain.append(Gain(gain_db=self.input_gain_db))
            print(f"  [1] Input Gain: {self.input_gain_db:+.1f} dB")

        # 2. EQ (from spectral analysis)
        eq_filters = self._build_eq_chain()
        if eq_filters:
            chain.extend(eq_filters)
            print(f"  [2] EQ: {len(eq_filters)} bands")

        # 3. COMPRESSION (from dynamics analysis)
        if 'dynamics' in self.preset_data:
            compressor = self._build_compressor()
            if compressor:
                chain.append(compressor)
                comp = self.preset_data['dynamics']['compression']
                print(f"  [3] Compressor: {comp['ratio']:.1f}:1, Threshold {comp['threshold_db']:.1f} dB")

        # 4. EFFECTS
        if 'effects' in self.preset_data:
            effects = self._build_effects()
            chain.extend(effects)
            if effects:
                print(f"  [4] Effects: {len(effects)} units")

        # 5. LIMITER (from dynamics analysis)
        if 'dynamics' in self.preset_data:
            limiter = self._build_limiter()
            if limiter:
                chain.append(limiter)
                lim = self.preset_data['dynamics']['limiting']
                print(f"  [5] Limiter: Ceiling {lim['ceiling_db']:.1f} dB")

        # 6. STEREO EFFECTS (if applicable)
        if self.preset_data.get('stereo_preset'):
            stereo_effects = self._build_stereo_effects()
            chain.extend(stereo_effects)
            if stereo_effects:
                print(f"  [6] Stereo: {len(stereo_effects)} effects")

        # 7. OUTPUT GAIN
        if self.output_gain_db != 0:
            chain.append(Gain(gain_db=self.output_gain_db))
            print(f"  [7] Output Gain: {self.output_gain_db:+.1f} dB")

        # Create pedalboard
        self.pedalboard = Pedalboard(chain)

        print(f"\nTotal processing units: {len(chain)}")

    def _build_eq_chain(self) -> list:
        """Build EQ filter chain"""
        if 'spectral' not in self.preset_data:
            return []

        eq_curve = self.preset_data['spectral']['eq_curve']
        filters = []

        for band in eq_curve:
            freq = band['frequency']
            gain = band['gain_db']
            q = band['q_factor']
            filter_type = band['filter_type']

            # Skip if gain is negligible
            if abs(gain) < 0.5:
                continue

            # Create appropriate filter type
            if filter_type == 'low_shelf':
                filters.append(LowShelfFilter(cutoff_frequency_hz=freq, gain_db=gain))
            elif filter_type == 'high_shelf':
                filters.append(HighShelfFilter(cutoff_frequency_hz=freq, gain_db=gain))
            elif filter_type == 'peak':
                filters.append(PeakFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))

        return filters

    def _build_compressor(self) -> Optional[Compressor]:
        """Build compressor from dynamics analysis"""
        comp_params = self.preset_data['dynamics']['compression']

        # Only enable if significant compression detected
        if comp_params['ratio'] < 1.3:
            return None

        return Compressor(
            threshold_db=comp_params['threshold_db'],
            ratio=comp_params['ratio'],
            attack_ms=comp_params['attack_ms'],
            release_ms=comp_params['release_ms']
        )

    def _build_limiter(self) -> Optional[Limiter]:
        """Build limiter from dynamics analysis"""
        lim_params = self.preset_data['dynamics']['limiting']

        if not lim_params['is_limited']:
            return None

        # Use ceiling from analysis, but ensure it's safe for live use
        ceiling_db = max(lim_params['ceiling_db'], -0.5)  # At least -0.5 dB headroom

        return Limiter(
            threshold_db=ceiling_db,
            release_ms=50.0  # Fast limiter
        )

    def _build_effects(self) -> list:
        """Build effects from effects detection"""
        effects = []

        effects_params = self.preset_data['effects']

        # Saturation/Distortion
        if effects_params['saturation']['detected']:
            sat_amount = effects_params['saturation']['amount']

            if sat_amount > 0.1:
                # Use distortion with low drive for saturation
                drive_db = sat_amount * 12.0  # Map to dB range
                effects.append(Distortion(drive_db=drive_db))

        # De-esser (implement as high-freq compressor - approximate)
        # Note: Pedalboard doesn't have de-esser, would need custom implementation

        # Warmth (subtle low-frequency boost)
        if effects_params['warmth']['detected']:
            warmth_amount = effects_params['warmth']['amount']

            if warmth_amount > 0.2:
                # Boost low-mids slightly
                effects.append(LowShelfFilter(cutoff_frequency_hz=300,
                                             gain_db=warmth_amount * 3.0))

        return effects

    def _build_stereo_effects(self) -> list:
        """Build stereo effects"""
        stereo_preset = self.preset_data.get('stereo_preset')

        if not stereo_preset or not stereo_preset['enabled']:
            return []

        effects = []

        # Reverb
        if stereo_preset['reverb_amount'] > 0.05:
            effects.append(Reverb(
                room_size=stereo_preset['reverb_size'],
                wet_level=stereo_preset['reverb_amount'],
                dry_level=1.0 - stereo_preset['reverb_amount']
            ))

        return effects

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio buffer

        Args:
            audio: Input audio buffer

        Returns:
            Processed audio
        """
        if not self.preset_loaded:
            return audio

        if self.pedalboard is None:
            return audio

        # Process with pedalboard
        processed = self.pedalboard(audio, sample_rate=self.sample_rate)

        # Apply wet/dry mix
        if self.wet_dry_mix < 1.0:
            processed = audio * (1.0 - self.wet_dry_mix) + processed * self.wet_dry_mix

        # Safety limiting
        max_val = np.max(np.abs(processed))
        if max_val > 0.99:
            processed = processed / max_val * 0.99

        return processed

    def set_wet_dry_mix(self, mix: float):
        """Set wet/dry mix (0-1)"""
        self.wet_dry_mix = np.clip(mix, 0.0, 1.0)

    def set_input_gain(self, gain_db: float):
        """Set input gain in dB"""
        self.input_gain_db = np.clip(gain_db, -24.0, 24.0)
        # Rebuild chain if preset is loaded
        if self.preset_loaded:
            self._build_processing_chain()

    def set_output_gain(self, gain_db: float):
        """Set output gain in dB"""
        self.output_gain_db = np.clip(gain_db, -24.0, 24.0)
        # Rebuild chain if preset is loaded
        if self.preset_loaded:
            self._build_processing_chain()

    def get_chain_description(self) -> str:
        """Get description of current processing chain"""
        if not self.preset_loaded:
            return "No preset loaded"

        if not self.preset_data:
            return "No processing chain"

        description = []
        description.append("ACTIVE PROCESSING CHAIN")
        description.append("=" * 50)

        if self.input_gain_db != 0:
            description.append(f"Input Gain: {self.input_gain_db:+.1f} dB")

        if 'spectral' in self.preset_data:
            eq_count = len([b for b in self.preset_data['spectral']['eq_curve']
                          if abs(b['gain_db']) >= 0.5])
            description.append(f"EQ: {eq_count} active bands")

        if 'dynamics' in self.preset_data:
            comp = self.preset_data['dynamics']['compression']
            if comp['ratio'] >= 1.3:
                description.append(f"Compressor: {comp['ratio']:.1f}:1")

        if 'effects' in self.preset_data:
            sat = self.preset_data['effects']['saturation']
            if sat['detected']:
                description.append(f"Saturation: {sat['type']}")

        if 'dynamics' in self.preset_data:
            if self.preset_data['dynamics']['limiting']['is_limited']:
                description.append("Limiter: Active")

        if self.output_gain_db != 0:
            description.append(f"Output Gain: {self.output_gain_db:+.1f} dB")

        description.append(f"\nWet/Dry Mix: {self.wet_dry_mix:.0%}")
        description.append("=" * 50)

        return "\n".join(description)


if __name__ == "__main__":
    import sys
    import sounddevice as sd

    if len(sys.argv) < 2:
        print("Usage: python processor_v2.py <preset_file>")
        print("\nThis will apply the preset to your microphone in real-time.")
        sys.exit(1)

    preset_file = sys.argv[1]

    # Initialize processor
    processor = BeatboxProcessorV2(sample_rate=44100)
    processor.load_preset(preset_file)

    print("\n" + processor.get_chain_description())

    print("\n✅ Processor ready!")
    print("Use this processor with live_processor.py or advanced_gui.py")
