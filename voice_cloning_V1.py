import torch

# Allowlist XTTS classes for PyTorch 2.6+ "weights_only" loading
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs])

from TTS.api import TTS
import sounddevice as sd

SPEAKER_WAV = "Trump_clean.wav"
text = "Hello Rommel, this is a cloned voice, and i am your personal AI assistant"
TEXT = text

def main():
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
    audio = tts.tts(text=TEXT, speaker_wav=SPEAKER_WAV, language="en")
    sd.play(audio, 24000)
    sd.wait()

if __name__ == "__main__":
    main()

