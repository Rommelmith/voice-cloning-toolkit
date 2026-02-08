import torch
import numpy as np
import soundfile as sf
import sounddevice as sd

# Optional: Patch torchaudio.load to use soundfile (helps on some Windows installs)
try:
    import torchaudio

    def _patched_load(filepath, *args, **kwargs):
        audio, sample_rate = sf.read(filepath, dtype="float32")
        # soundfile returns shape: (samples,) or (samples, channels)
        if audio.ndim == 1:
            audio = audio[None, :]          # (1, samples)
        else:
            audio = audio.T                 # (channels, samples)
        return torch.from_numpy(audio), sample_rate

    torchaudio.load = _patched_load
except Exception:
    # If torchaudio isn't installed or fails to import, we just continue.
    pass


# Fix for some PyTorch safe deserialization cases (XTTS config objects)
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([BaseDatasetConfig, XttsConfig, XttsAudioConfig, XttsArgs])

from TTS.api import TTS


class VoiceCloner:
    def __init__(self, speaker_wav: str, use_gpu: bool = True):
        self.speaker_wav = speaker_wav

        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        print(f"Initializing XTTS v2 on: {self.device}")

        # Let Coqui decide device; gpu=True is fine if CUDA is available
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(self.device == "cuda"))
        print("[OK] Model loaded")

    def clone_voice(
        self,
        text: str,
        language: str = "en",
        temperature: float = 0.75,
        repetition_penalty: float = 6.0,
        speed: float = 1.0,
    ) -> np.ndarray:
        audio = self.tts.tts(
            text=text,
            speaker_wav=self.speaker_wav,
            language=language,
            temperature=temperature,
            length_penalty=1.9,
            repetition_penalty=repetition_penalty,
            top_k=50,
            top_p=0.85,
            speed=speed,
        )
        return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def play(audio: np.ndarray, sample_rate: int = 24000):
        sd.play(audio, sample_rate)
        sd.wait()

    @staticmethod
    def save(audio: np.ndarray, output_path: str, sample_rate: int = 24000):
        sf.write(output_path, audio, sample_rate)
        print(f"[OK] Saved: {output_path}")


if __name__ == "__main__":
    cloner = VoiceCloner(
        speaker_wav=r"C:\Users\romme\PycharmProjects\VoicesINOut\CelebrityVoices\EmmaWatsonSample2_clean.wav",
        use_gpu=True,
    )

    while True:
        text = input("\nEnter text (or 'q' to quit): ").strip()
        if not text or text.lower() == "q":
            break

        audio = cloner.clone_voice(text=text, language="en")
        cloner.play(audio)

        # Uncomment if you want to save output:
        # cloner.save(audio, "output.wav")
