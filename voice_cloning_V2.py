import torch
import soundfile as sf_lib
import numpy as np

# Patch torchaudio.load to use soundfile instead of torchcodec (Windows compatibility)
import torchaudio
_original_load = torchaudio.load

def _patched_load(filepath, *args, **kwargs):
    """Use soundfile to load audio instead of torchcodec"""
    audio, sample_rate = sf_lib.read(filepath, dtype='float32')
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    else:
        audio = audio.T
    return torch.from_numpy(audio), sample_rate

torchaudio.load = _patched_load

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([BaseDatasetConfig])

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs])

from TTS.api import TTS
import sounddevice as sd
sf = sf_lib  # Reuse the already imported soundfile


class VoiceCloner:
    def __init__(self, speaker_wav):
        self.speaker_wav = speaker_wav
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing TTS on {self.device}...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        print("[OK] TTS model loaded")

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
        print(f"[OK] Saved to {output_path}")


# Usage
if __name__ == "__main__":
    cloner = VoiceCloner("CelebrityVoices/EmmaWatsonSample2_clean.wav")

    # Test different texts
    texts = [
        "Hello Rommel, this is a cloned voice, and I am your personal AI assistant. The weather today is absolutely fantastic, don't you think? I've been thinking about the future of artificial intelligence.",
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