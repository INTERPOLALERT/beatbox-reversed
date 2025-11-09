"""
Sound Classification System
Classifies beatbox sounds into categories: kick, snare, hihat, bass
Uses MFCC features + spectral features with machine learning
"""
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
import config


class BeatboxSoundClassifier:
    """
    Classifies beatbox sounds using audio features
    """

    # Sound categories
    SOUND_TYPES = ['kick', 'snare', 'hihat', 'bass', 'other']

    def __init__(self):
        """Initialize classifier"""
        self.classifier = None
        self.scaler = None
        self.is_trained = False

    def extract_features(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """
        Extract features from audio segment

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Feature vector
        """
        features = []

        # 1. MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)

        # 2. Spectral Centroid (brightness)
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(spec_centroid))
        features.append(np.std(spec_centroid))

        # 3. Spectral Contrast (texture)
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.extend(np.mean(spec_contrast, axis=1))

        # 4. Zero Crossing Rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # 5. RMS Energy (loudness)
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        # 6. Spectral Rolloff (frequency distribution)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))

        # 7. Low-frequency energy ratio (for bass/kick detection)
        # Calculate energy below 200 Hz vs total
        stft = librosa.stft(audio)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 2)
        low_freq_mask = freqs < 200
        low_energy = np.sum(np.abs(stft[low_freq_mask, :]) ** 2)
        total_energy = np.sum(np.abs(stft) ** 2) + 1e-10
        features.append(low_energy / total_energy)

        # 8. High-frequency energy ratio (for hihat detection)
        # Energy above 4000 Hz
        high_freq_mask = freqs > 4000
        high_energy = np.sum(np.abs(stft[high_freq_mask, :]) ** 2)
        features.append(high_energy / total_energy)

        return np.array(features)

    def train(self, training_data: Dict[str, list], save_path: Optional[Path] = None):
        """
        Train classifier on labeled data

        Args:
            training_data: Dict mapping sound_type -> list of audio arrays
            save_path: Optional path to save trained model
        """
        X = []
        y = []

        print("Extracting features from training data...")

        for sound_type, audio_samples in training_data.items():
            if sound_type not in self.SOUND_TYPES:
                print(f"Warning: Unknown sound type '{sound_type}', skipping")
                continue

            for audio in audio_samples:
                features = self.extract_features(audio)
                X.append(features)
                y.append(sound_type)

        X = np.array(X)
        y = np.array(y)

        print(f"Training on {len(X)} samples across {len(set(y))} classes...")

        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(X_scaled, y)

        self.is_trained = True

        # Calculate training accuracy
        train_accuracy = self.classifier.score(X_scaled, y)
        print(f"Training accuracy: {train_accuracy:.2%}")

        # Save model if path provided
        if save_path:
            self.save(save_path)

    def predict(self, audio: np.ndarray, sr: int = 44100) -> Tuple[str, float]:
        """
        Predict sound type from audio

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Tuple of (predicted_type, confidence)
        """
        if not self.is_trained:
            # Use rule-based classification as fallback
            return self._rule_based_classification(audio, sr)

        # Extract features
        features = self.extract_features(audio, sr)
        features = features.reshape(1, -1)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)

        return prediction, confidence

    def _rule_based_classification(self, audio: np.ndarray, sr: int = 44100) -> Tuple[str, float]:
        """
        Simple rule-based classification (fallback when model not trained)

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Tuple of (predicted_type, confidence)
        """
        # Extract key features
        features = self.extract_features(audio, sr)

        # Simple rules based on feature inspection
        # Feature indices (from extract_features):
        # [0-12]: MFCC means
        # [13-25]: MFCC stds
        # [26-27]: Spectral centroid mean/std
        # [28-34]: Spectral contrast
        # [35-36]: ZCR mean/std
        # [37-38]: RMS mean/std
        # [39-40]: Rolloff mean/std
        # [41]: Low-freq energy ratio
        # [42]: High-freq energy ratio

        low_freq_ratio = features[41]
        high_freq_ratio = features[42]
        spec_centroid = features[26]
        zcr_mean = features[35]

        # Decision rules
        if low_freq_ratio > 0.5 and spec_centroid < 500:
            # Strong low-frequency energy, low brightness = KICK
            return 'kick', 0.7

        elif high_freq_ratio > 0.3 and zcr_mean > 0.15:
            # High-frequency energy, high noisiness = HIHAT
            return 'hihat', 0.7

        elif low_freq_ratio > 0.3 and spec_centroid < 1000:
            # Moderate low-freq, low-mid brightness = BASS
            return 'bass', 0.6

        elif spec_centroid > 800 and spec_centroid < 3000:
            # Mid-range brightness = SNARE
            return 'snare', 0.6

        else:
            # Unknown
            return 'other', 0.5

    def save(self, path: Path):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'sound_types': self.SOUND_TYPES
        }

        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load(self, path: Path):
        """Load trained model"""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.SOUND_TYPES = model_data.get('sound_types', self.SOUND_TYPES)
        self.is_trained = True

        print(f"Model loaded from {path}")


class OnsetBasedClassifier:
    """
    Classifies sounds at detected onsets in real-time
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize onset-based classifier

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.classifier = BeatboxSoundClassifier()

        # Try to load pre-trained model
        model_path = config.APP_DIR / "models" / "beatbox_classifier.pkl"
        if model_path.exists():
            try:
                self.classifier.load(model_path)
                print("Loaded pre-trained classifier")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Using rule-based classification")

        # Onset detection state
        self.onset_buffer_ms = 100  # ms of audio to analyze per onset
        self.min_onset_interval_ms = 50  # minimum time between onsets

    def detect_and_classify_onsets(self, audio: np.ndarray) -> list:
        """
        Detect onsets and classify each sound

        Args:
            audio: Audio signal

        Returns:
            List of (onset_time, sound_type, confidence)
        """
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=self.sample_rate,
            units='frames',
            backtrack=False
        )

        # Convert to time
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)

        results = []

        # Classify each onset
        for onset_time in onset_times:
            # Extract audio segment around onset
            start_sample = int(onset_time * self.sample_rate)
            end_sample = int((onset_time + self.onset_buffer_ms / 1000) * self.sample_rate)
            end_sample = min(end_sample, len(audio))

            if end_sample - start_sample < self.sample_rate * 0.01:
                # Too short, skip
                continue

            segment = audio[start_sample:end_sample]

            # Classify
            sound_type, confidence = self.classifier.predict(segment, self.sample_rate)

            results.append((onset_time, sound_type, confidence))

        return results

    def classify_buffer(self, audio_buffer: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single audio buffer (for real-time use)

        Args:
            audio_buffer: Audio buffer to classify

        Returns:
            Tuple of (sound_type, confidence)
        """
        return self.classifier.predict(audio_buffer, self.sample_rate)


def create_training_dataset_from_files(audio_files: Dict[str, list], sr: int = 44100) -> Dict[str, list]:
    """
    Create training dataset from audio files

    Args:
        audio_files: Dict mapping sound_type -> list of file paths
        sr: Sample rate

    Returns:
        Dict mapping sound_type -> list of audio arrays
    """
    training_data = {}

    for sound_type, file_paths in audio_files.items():
        audio_samples = []

        for file_path in file_paths:
            # Load audio
            audio, _ = librosa.load(file_path, sr=sr, mono=True)

            # Detect onsets and split
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=sr,
                units='frames'
            )

            if len(onset_frames) == 0:
                # No onsets, use whole file
                audio_samples.append(audio)
            else:
                # Extract segments around onsets
                for onset_frame in onset_frames:
                    start_sample = librosa.frames_to_samples(onset_frame)
                    end_sample = start_sample + int(sr * 0.1)  # 100ms
                    end_sample = min(end_sample, len(audio))

                    segment = audio[start_sample:end_sample]

                    if len(segment) >= sr * 0.01:  # At least 10ms
                        audio_samples.append(segment)

        training_data[sound_type] = audio_samples

    return training_data


if __name__ == "__main__":
    # Test classifier
    print("Testing BeatboxSoundClassifier...")

    # Create synthetic test signals
    sr = 44100
    duration = 0.1

    # Kick: Low frequency sine
    t = np.linspace(0, duration, int(sr * duration))
    kick = np.sin(2 * np.pi * 80 * t) * np.exp(-t * 30)

    # Snare: Noise with mid-range emphasis
    snare = np.random.randn(int(sr * duration)) * np.exp(-t * 20)
    snare = librosa.effects.preemphasis(snare)

    # Hihat: High-frequency noise
    hihat = np.random.randn(int(sr * duration)) * np.exp(-t * 50)
    hihat = librosa.effects.preemphasis(hihat, coef=0.97)

    # Bass: Low-mid frequency sustained
    bass = np.sin(2 * np.pi * 150 * t) + np.sin(2 * np.pi * 75 * t)

    # Test with rule-based classifier
    classifier = BeatboxSoundClassifier()

    test_sounds = {
        'kick': kick,
        'snare': snare,
        'hihat': hihat,
        'bass': bass
    }

    print("\nRule-based classification results:")
    for name, sound in test_sounds.items():
        predicted, confidence = classifier.predict(sound, sr)
        print(f"  {name}: predicted={predicted}, confidence={confidence:.2f}")

    # Test feature extraction
    print("\nFeature vector size:")
    features = classifier.extract_features(kick, sr)
    print(f"  {len(features)} features")
