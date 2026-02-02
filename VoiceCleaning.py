import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, filtfilt, medfilt
import os


class VoiceCleaner:
    def __init__(self, target_sr=24000):
        self.target_sr = target_sr

    def load_audio(self, path):
        """Load and convert to mono"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return librosa.load(path, sr=self.target_sr, mono=True)

    def remove_silence(self, y, top_db=30):
        """Remove silence from start/end"""
        return librosa.effects.trim(y, top_db=top_db, frame_length=2048, hop_length=512)[0]

    def highpass_filter(self, y, cutoff=80):
        """Remove low-frequency rumble"""
        nyquist = self.target_sr / 2
        normal_cutoff = cutoff / nyquist
        b, a = butter(5, normal_cutoff, btype='high')
        return filtfilt(b, a, y)

    def reduce_noise(self, y, reduction_strength=0.5):
        """
        Intelligent noise reduction
        reduction_strength: 0.3 (gentle) to 0.8 (aggressive)
        """
        # Use first 0.5s as noise profile, or quietest 0.5s
        noise_duration = min(0.5, len(y) / self.target_sr / 2)
        noise_samples = int(noise_duration * self.target_sr)

        # Find quietest section for noise profile
        window_size = noise_samples
        min_rms = float('inf')
        noise_start = 0

        for i in range(0, len(y) - window_size, window_size // 2):
            window_rms = np.sqrt(np.mean(y[i:i + window_size] ** 2))
            if window_rms < min_rms:
                min_rms = window_rms
                noise_start = i

        noise_sample = y[noise_start:noise_start + window_size]

        return nr.reduce_noise(
            y=y,
            sr=self.target_sr,
            y_noise=noise_sample,
            prop_decrease=reduction_strength,
            stationary=False,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50
        )

    def select_best_segment(self, y, target_duration=15, min_duration=6, max_duration=28):
        """Select best segment for voice cloning"""
        duration = len(y) / self.target_sr

        if duration <= max_duration:
            return y

        # Calculate speech activity in windows
        segment_samples = int(target_duration * self.target_sr)
        hop = int(0.5 * self.target_sr)

        best_start = 0
        best_score = 0

        for start in range(0, len(y) - segment_samples, hop):
            segment = y[start:start + segment_samples]

            # Multi-factor scoring
            rms = np.sqrt(np.mean(segment ** 2))
            zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=self.target_sr))

            # Prefer segments with speech-like characteristics
            score = rms * (1 - zcr) * (spectral_centroid / 2000)

            if score > best_score:
                best_score = score
                best_start = start

        print(f"Selected best {target_duration}s segment from {duration:.1f}s")
        return y[best_start:best_start + segment_samples]

    def normalize_audio(self, y, target_db=-20):
        """Normalize to target RMS level"""
        rms = np.sqrt(np.mean(y ** 2))
        if rms < 1e-10:
            return y
        target_rms = 10 ** (target_db / 20)
        return y * (target_rms / rms)

    def soft_limit(self, y, threshold=0.95, ratio=0.5):
        """Soft limiter to prevent clipping"""
        peak = np.max(np.abs(y))
        if peak <= threshold:
            return y

        # Soft knee compression above threshold
        mask = np.abs(y) > threshold
        excess = np.abs(y[mask]) - threshold
        y[mask] = np.sign(y[mask]) * (threshold + excess * ratio)

        return y

    def add_fades(self, y, fade_ms=20):
        """Add fade in/out to prevent clicks"""
        fade_samples = int(fade_ms * self.target_sr / 1000)
        fade_samples = min(fade_samples, len(y) // 10)  # Max 10% of audio

        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        y[:fade_samples] *= fade_in
        y[-fade_samples:] *= fade_out

        return y

    def process(self, input_path, output_path,
                noise_reduction=0.5,
                target_duration=15,
                target_db=-20):
        """
        Complete cleaning pipeline

        Args:
            input_path: Input audio file
            output_path: Output cleaned file
            noise_reduction: 0.3 (gentle) to 0.8 (aggressive)
            target_duration: Target segment length if audio > 30s
            target_db: Target RMS level (-20 to -18 recommended)
        """
        print(f"Processing: {input_path}")

        # Load
        y, sr = self.load_audio(input_path)
        original_duration = len(y) / sr
        print(f"Loaded: {original_duration:.1f}s @ {sr}Hz")

        # 1. Remove silence
        y = self.remove_silence(y)

        # 2. High-pass filter
        y = self.highpass_filter(y, cutoff=80)

        # 3. Noise reduction
        if noise_reduction > 0:
            y = self.reduce_noise(y, reduction_strength=noise_reduction)

        # 4. Select optimal segment
        y = self.select_best_segment(y, target_duration=target_duration)

        # 5. Normalize
        y = self.normalize_audio(y, target_db=target_db)

        # 6. Soft limiting
        y = self.soft_limit(y, threshold=0.95, ratio=0.5)

        # 7. Fade in/out
        y = self.add_fades(y, fade_ms=20)

        # Save
        sf.write(output_path, y, self.target_sr, subtype='PCM_16')

        # Report
        final_duration = len(y) / self.target_sr
        final_rms = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-10)
        final_peak = np.max(np.abs(y))

        print(f"\n✓ Saved: {output_path}")
        print(f"  Duration: {final_duration:.1f}s")
        print(f"  RMS: {final_rms:.1f} dBFS")
        print(f"  Peak: {final_peak:.2f}")

        return output_path


# Usage
if __name__ == "__main__":
    cleaner = VoiceCleaner(target_sr=24000)

    input_file = r"C:\Users\romme\PycharmProjects\VoicesINOut\CelebrityVoices\donald-trump.wav"
    output_file = "Trump_clean.wav"

    # Standard cleaning
    cleaner.process(
        input_file,
        output_file,
        noise_reduction=0.5,  # Adjust 0.3-0.8
        target_duration=15,
        target_db=-20
    )

    # For very noisy audio, use more aggressive settings:
    # cleaner.process(input_file, output_file, noise_reduction=0.7, target_db=-18)