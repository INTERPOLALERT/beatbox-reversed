"""
Audio Playback Module
Simple audio playback for recorded files
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional
import threading


class AudioPlayer:
    """
    Simple audio file player
    """

    def __init__(self):
        """Initialize audio player"""
        self.is_playing = False
        self.playback_thread = None
        self.stop_requested = False
        self.current_file = None

    def play_file(self, file_path: str, device: Optional[int] = None, callback: Optional[callable] = None):
        """
        Play audio file

        Args:
            file_path: Path to audio file
            device: Output device index (None for default)
            callback: Optional callback when playback completes
        """
        if self.is_playing:
            self.stop()

        self.current_file = file_path
        self.stop_requested = False

        # Start playback in thread
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            args=(file_path, device, callback),
            daemon=True
        )
        self.playback_thread.start()

    def _playback_worker(self, file_path: str, device: Optional[int], callback: Optional[callable]):
        """
        Worker thread for playback

        Args:
            file_path: Path to audio file
            device: Output device index
            callback: Completion callback
        """
        try:
            # Load audio file
            audio, sample_rate = sf.read(file_path, always_2d=True)

            self.is_playing = True

            # Play audio
            sd.play(audio, samplerate=sample_rate, device=device)

            # Wait for playback to finish or stop requested
            while sd.get_stream().active and not self.stop_requested:
                sd.sleep(100)

            # Stop if requested
            if self.stop_requested:
                sd.stop()

            self.is_playing = False

            # Call callback if provided
            if callback and not self.stop_requested:
                callback()

        except Exception as e:
            print(f"Playback error: {e}")
            self.is_playing = False

    def stop(self):
        """Stop playback"""
        if self.is_playing:
            self.stop_requested = True
            sd.stop()
            self.is_playing = False

    def is_playing(self) -> bool:
        """
        Check if currently playing

        Returns:
            True if playing, False otherwise
        """
        return self.is_playing


def play_audio_file(file_path: str, blocking: bool = False):
    """
    Convenience function to play audio file

    Args:
        file_path: Path to audio file
        blocking: If True, wait for playback to finish
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Load and play
    audio, sample_rate = sf.read(file_path)
    sd.play(audio, samplerate=sample_rate)

    if blocking:
        sd.wait()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_playback.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    print(f"Playing: {audio_file}")

    play_audio_file(audio_file, blocking=True)

    print("Playback complete")
