# Voice Cloning Toolkit

A research-oriented voice synthesis pipeline built from first principles to understand the complete workflow from raw audio to voice conversion. This project prioritizes transparency, reproducibility, and systematic experimentation over black-box solutions.

## Overview

Voice cloning sits at the intersection of signal processing, deep learning, and audio engineering. Rather than using pre-packaged APIs, this toolkit implements a modular pipeline that exposes each transformation step—from audio preprocessing to feature extraction to model inference—allowing for detailed inspection, component swapping, and controlled experimentation.

**Core focus areas:**
- Signal processing and audio normalization pipelines
- Multi-stage data preprocessing with quality validation
- Comparative analysis of generation parameters and their impact on output quality
- Reproducible experiment design with consistent sample management

## Motivation

This project emerged from studying recent advances in voice conversion and neural audio synthesis. The goal was to move beyond surface-level implementations and build genuine understanding of:

- How preprocessing choices (sample rate, normalization, silence removal) cascade through the pipeline
- Why certain samples produce better results than others
- Trade-offs between different voice conversion parameters
- The relationship between model settings and audio fidelity

## Technical Architecture
```
CelebrityVoices/          # Sample voices for testing
VoiceCleaning.py          # Audio preprocessing pipeline
voice_cloning_V1.py       # Minimal proof-of-concept implementation
voice_cloning_V2.py       # Feature-rich implementation with parameter tuning
```

### Key Components

**Audio Preprocessing Pipeline (`VoiceCleaning.py`)**

The `VoiceCleaner` class implements a sophisticated 7-stage processing pipeline:

1. **Silence Removal** - Energy-based trimming using librosa's top_db threshold
2. **High-pass Filtering** - 5th-order Butterworth filter to remove low-frequency rumble (80Hz cutoff)
3. **Intelligent Noise Reduction** - Adaptive noise profiling using the quietest audio segment
4. **Smart Segment Selection** - Multi-factor scoring (RMS, zero-crossing rate, spectral centroid) to identify optimal speech segments
5. **RMS Normalization** - Target-based amplitude normalization (default: -20 dBFS)
6. **Soft Limiting** - Prevents clipping while preserving dynamics using soft-knee compression
7. **Fade In/Out** - Removes click artifacts at audio boundaries

Configurable parameters:
- `noise_reduction`: 0.3 (gentle) to 0.8 (aggressive)
- `target_duration`: Optimal segment length extraction
- `target_db`: RMS normalization level

**Voice Cloning Implementations**

Both implementations use Coqui TTS's XTTS v2 model (`tts_models/multilingual/multi-dataset/xtts_v2`), a state-of-the-art multilingual voice cloning architecture.

*V1 - Minimal Implementation (`voice_cloning_V1.py`)*
- Proof-of-concept with minimal code
- Direct TTS API usage without wrappers
- Real-time playback only (no save functionality)
- Default parameters for quick testing
- Perfect for understanding the basic workflow

*V2 - Production-Ready Implementation (`voice_cloning_V2.py`)*
- Object-oriented `VoiceCloner` class for reusability
- Automatic device detection (CUDA/CPU)
- Advanced parameter tuning:
  - `temperature` (0.65): Controls randomness/creativity
  - `repetition_penalty` (5.0): Prevents word repetition
  - `length_penalty` (1.9): Influences output duration
  - `top_k` (50): Limits vocabulary sampling
  - `top_p` (0.85): Nucleus sampling threshold
  - `speed` (1.0): Playback speed adjustment
- Save and playback functionality
- Batch processing support for comparing multiple texts

## Getting Started

### Prerequisites
Set up a Python environment with audio processing and deep learning dependencies:
```bash
# Create isolated environment
python -m venv .venv_tts
source .venv_tts/bin/activate  # On Windows: .venv_tts\Scripts\activate

# Install dependencies
pip install numpy librosa soundfile noisereduce scipy
pip install TTS torch sounddevice
```

### Workflow

**1. Prepare Source Audio**

Place high-quality voice samples in `CelebrityVoices/`:
- **Duration:** 30-90 seconds optimal (quality > quantity)
- **Content:** Clean speech without music or overlapping audio
- **Format:** WAV preferred, consistent sample rate
- **Quality:** Minimal background noise, clear articulation

**2. Preprocessing**
```bash
python VoiceCleaning.py
```

The script automatically:
- Loads audio and resamples to 24kHz
- Applies the 7-stage cleaning pipeline
- Selects the best 15-second segment if audio is longer
- Outputs cleaned WAV with quality metrics

**Custom preprocessing:**
```python
from VoiceCleaning import VoiceCleaner

cleaner = VoiceCleaner(target_sr=24000)
cleaner.process(
    "input.wav",
    "output.wav",
    noise_reduction=0.5,    # Adjust 0.3-0.8 based on noise level
    target_duration=15,     # Extract optimal 15s segment
    target_db=-20           # RMS normalization target
)
```

**3. Run Cloning Experiments**

*Quick test with V1:*
```bash
python voice_cloning_V1.py
```
Simple proof-of-concept that generates and plays a single sample.

*Production workflow with V2:*
```bash
python voice_cloning_V2.py
```
Generates multiple samples with optimized parameters and saves outputs.

**Custom voice cloning with V2:**
```python
from voice_cloning_V2 import VoiceCloner

cloner = VoiceCloner("path/to/cleaned_voice.wav")

# Generate with custom parameters
audio = cloner.clone_voice(
    text="Your text here",
    language="en",
    temperature=0.65,        # Lower = more consistent, higher = more varied
    repetition_penalty=5.0,  # Higher = less repetition
    speed=1.0               # Adjust playback speed
)

# Save output
cloner.save(audio, "output.wav")

# Or play directly
cloner.play(audio)
```

## Engineering Decisions

**Why a 7-stage preprocessing pipeline?**  
Each stage addresses a specific audio quality issue. The order matters: silence removal before noise reduction, normalization after segment selection, limiting before fades. This sequence minimizes artifacts and preserves voice characteristics.

**Why intelligent segment selection?**  
Not all audio is equal. The multi-factor scoring system (RMS × low-ZCR × spectral centroid) identifies segments with strong, stable speech characteristics—exactly what voice cloning models need.

**Why adaptive noise reduction?**  
Static noise profiles fail on variable background noise. The algorithm automatically finds the quietest segment to profile noise, then applies targeted reduction while preserving voice timbre.

**Why two cloning implementations?**  
V1 serves as a minimal reference implementation for understanding the basic workflow. V2 provides production-ready functionality with parameter tuning, batch processing, and proper error handling. This separation makes it easy to understand fundamentals before exploring advanced features.

**Why XTTS v2?**  
XTTS v2 offers excellent multilingual support, requires minimal reference audio (as little as 6 seconds), and produces high-quality results. It's also open-source and well-documented.

**Why expose generation parameters?**  
Voice cloning isn't one-size-fits-all. Different texts, voices, and use cases benefit from different parameter settings. V2 exposes these knobs for experimentation and optimization.

## Technical Challenges & Solutions

| Challenge | Approach |
|-----------|----------|
| Variable input quality | Multi-stage adaptive preprocessing pipeline |
| Selecting optimal segments | Speech activity scoring with multiple acoustic features |
| Preserving voice characteristics | Gentle noise reduction + soft limiting instead of hard clipping |
| Preventing artifacts | Strategic fade application and soft-knee compression |
| Sample rate consistency | Mandatory resampling to 24kHz matching XTTS requirements |
| Repetitive generation | High repetition penalty (5.0) in V2 |
| Inconsistent quality | Tuned temperature (0.65) and nucleus sampling (top_p=0.85) |
| Cross-platform compatibility | Device auto-detection (CUDA/CPU) |

## Parameter Tuning Guide (V2)

| Parameter | Range | Effect | Recommended |
|-----------|-------|--------|-------------|
| `temperature` | 0.1-1.0 | Creativity vs consistency | 0.65 (balanced) |
| `repetition_penalty` | 1.0-10.0 | Prevents word repetition | 5.0 (strong prevention) |
| `length_penalty` | 0.5-2.0 | Affects output duration | 1.9 (slightly longer) |
| `top_k` | 10-100 | Vocabulary sampling size | 50 (moderate diversity) |
| `top_p` | 0.5-1.0 | Nucleus sampling threshold | 0.85 (focused sampling) |
| `speed` | 0.5-2.0 | Playback speed | 1.0 (natural) |

**Tuning tips:**
- Lower temperature (0.4-0.5) for robotic/monotone voices
- Higher temperature (0.7-0.85) for expressive/varied voices
- Increase repetition_penalty if words are repeated excessively
- Adjust speed for different speaking rates without affecting pitch

## Limitations & Future Work

**Current limitations:**
- Single-speaker focus (no multi-speaker disentanglement)
- Requires GPU for real-time generation (CPU is slow)
- Quality heavily dependent on source audio characteristics
- XTTS v2 occasionally produces artifacts on very long texts

**Potential enhancements:**
- Automated hyperparameter tuning based on input analysis
- Quantitative evaluation metrics (MOS prediction, speaker embedding distance)
- Real-time streaming generation
- Web interface for interactive experimentation
- Fine-tuning XTTS on custom datasets
- Support for additional voice conversion architectures

## Ethics & Responsible Use

This is an educational research project. Voice synthesis technology requires careful consideration:

- **Consent:** Never clone voices without explicit permission
- **Disclosure:** Synthetic audio should be clearly marked
- **Compliance:** Respect local laws regarding deepfakes and impersonation
- **Harm prevention:** Do not use for fraud, misinformation, or harassment

## Troubleshooting

**"RuntimeError: CUDA out of memory"**  
XTTS v2 requires ~4GB VRAM. Solutions:
- Use CPU mode (automatic in V2 if CUDA unavailable)
- Close other GPU applications
- Reduce batch size if processing multiple samples

**"Poor voice similarity despite clean input"**  
Try these adjustments:
1. Use longer reference audio (60-90 seconds optimal)
2. Ensure reference audio is monotone/consistent
3. Lower temperature to 0.5 for more consistent mimicry
4. Verify the cleaned audio has minimal background noise

**"Robotic or unnatural output"**  
Increase temperature and adjust penalties:
```python
audio = cloner.clone_voice(
    text=text,
    temperature=0.75,          # More natural variation
    repetition_penalty=3.0,    # Allow some repetition
    speed=1.05                 # Slightly faster
)
```

**"Words repeating excessively"**  
Increase repetition_penalty:
```python
audio = cloner.clone_voice(text=text, repetition_penalty=7.0)
```

**"Git push rejected - file too large"**  
XTTS model files are downloaded to cache, not tracked by git. If you've added model files:
```bash
git lfs install
git lfs track "*.pt" "*.pth"
```

**"ModuleNotFoundError: TTS"**  
Install Coqui TTS:
```bash
pip install TTS
```

**"Audio too quiet/loud after cleaning"**  
Adjust `target_db` in VoiceCleaning.py:
- `-18 dBFS`: Louder output
- `-20 dBFS`: Standard (recommended)
- `-22 dBFS`: Quieter output

## Contributing

Suggestions and improvements welcome, especially around:
- Novel preprocessing techniques
- Parameter tuning strategies for different voice types
- Evaluation methodology
- Documentation clarity

---

**Project Status:** Active learning project | Educational use only  
**Tech Stack:** Python, Coqui TTS (XTTS v2), PyTorch, librosa, noisereduce  
**Contact:** Open to discussing voice synthesis, audio ML, and signal processing