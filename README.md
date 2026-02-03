# Voice Cloning Toolkit

A simple voice cloning project using Coqui TTS's XTTS v2 model. Feed it a voice sample, give it some text, and it generates speech in that voice.

## What's Inside

- `VoiceCleaning.py` - Cleans up your audio samples (removes noise, normalizes volume, etc.)
- `voice_cloning_V1.py` - Basic version, just plays the output
- `voice_cloning_V2.py` - Full version with save/play and tweakable settings

## Installation

Create a virtual environment first:
```bash
python -m venv .venv_tts
.venv_tts\Scripts\activate  # Windows
source .venv_tts/bin/activate  # Linux/Mac
```

Then install dependencies based on your setup:

| GPU (CUDA 12.8) | CPU Only |
|-----------------|----------|
| `pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128` | `pip install torch torchaudio torchvision` |
| `pip install TTS sounddevice soundfile librosa noisereduce` | `pip install TTS sounddevice soundfile librosa noisereduce` |

For GPU, you need an NVIDIA card with CUDA 12.8 support. RTX 30/40/50 series work great.

## Quick Start

1. Drop your voice sample in `CelebrityVoices/` folder (WAV format, 30-90 seconds of clean speech works best)

2. Clean it up:
```bash
python VoiceCleaning.py
```

3. Run the cloner:
```bash
python voice_cloning_V2.py
```

That's it. Output saves to `output_1.wav`.

## Tweaking the Output

Edit `voice_cloning_V2.py` to change these:

| Setting | What it does | Default |
|---------|--------------|---------|
| `temperature` | Higher = more expressive, lower = more robotic | 0.65 |
| `repetition_penalty` | Stops words from repeating | 5.0 |
| `speed` | Faster or slower speech | 1.0 |

## Common Issues

**CUDA out of memory** - Close other GPU apps or it'll fall back to CPU automatically.

**Sounds robotic** - Bump up temperature to 0.75.

**Words repeating** - Increase repetition_penalty to 7.0.

## Heads Up

Don't clone someone's voice without their permission. Be responsible with this stuff.
