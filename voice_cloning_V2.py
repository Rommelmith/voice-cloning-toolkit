import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs])

from TTS.api import TTS
import sounddevice as sd
import soundfile as sf
import numpy as np


class VoiceCloner:
    def __init__(self, speaker_wav):
        self.speaker_wav = speaker_wav
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing TTS on {self.device}...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        print("✓ TTS model loaded")

    def clone_voice(self, text, language="en", temperature=0.65,
                    repetition_penalty=5.0, speed=1.0):
        print(f"Generating speech for: '{text[:50]}...'")

        audio = self.tts.tts(
            text=text,
            speaker_wav=self.speaker_wav,
            language=language,
            # Advanced parameters for better quality
            temperature=temperature,
            length_penalty=1.9,
            repetition_penalty=repetition_penalty,
            top_k=50,
            top_p=0.85,
            speed=speed
        )

        return np.array(audio)

    def play(self, audio, sample_rate=24000):
        """Play audio"""
        sd.play(audio, sample_rate)
        sd.wait()

    def save(self, audio, output_path, sample_rate=24000):
        """Save audio to file"""
        sf.write(output_path, audio, sample_rate)
        print(f"✓ Saved to {output_path}")


# Usage
if __name__ == "__main__":
    cloner = VoiceCloner("Trump_clean.wav")

    # Test different texts
    texts = [
        "Hello Rommel, this is a cloned voice, and I am your personal AI assistant.",
        "The weather today is absolutely fantastic, don't you think?",
        "I've been thinking about the future of artificial intelligence.",
    ]

    for i, text in enumerate(texts):
        print(f"\n--- Test {i + 1} ---")

        # Generate with optimal settings
        audio = cloner.clone_voice(
            text=text,
            language="en",
            temperature=0.65,  # Balanced
            repetition_penalty=5.0,
            speed=1.0
        )

        # Save
        cloner.save(audio, f"output_{i + 1}.wav")

        # Play
        cloner.play(audio)