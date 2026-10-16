# Speech Enhancement Lab

**A modular Python research engine for audio analysis, pretrained model inference, and explainable task-based model selection.**

[![Python Tests](https://github.com/desouz78/speech-enhancement-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/desouz78/speech-enhancement-lab/actions/workflows/tests.yml)

**Status:** Python V1 — pilot-validated; browser demonstration and automatic acoustic-condition-based selection are future work.

## Overview

Speech Enhancement Lab investigates how specialized pretrained models can be combined to analyze, transform, and transcribe audio. Its current release concentrates on a testable **Python engine** rather than a browser application.

The project provides validated audio primitives; deterministic global and temporal acoustic features; an explainable, task-based model catalog; optional adapters for pretrained neural models; and two explicit, executable processing pipelines. **Model selection currently requires the caller to specify the processing task.** The engine does not infer the best task or model from an audio recording, benchmark competing models automatically, or guarantee output quality.

## Python V1: model inventory

| Model | Task | Python V1 status |
|---|---|---|
| DeepFilterNet3 | Speech enhancement | Integrated; real-checkpoint pilot run |
| HTDemucs (`htdemucs`) | Four-stem music separation | Integrated; real-checkpoint pilot run |
| SepFormer (`speechbrain/sepformer-wsj02mix`) | Two-speaker speech separation | Integrated; real-checkpoint pilot run |
| Whisper | Speech transcription | Integrated; real-checkpoint pilot run |
| Silero VAD | Speech activity detection | Integrated; real-checkpoint pilot run |
| CLAP (`laion/clap-htsat-unfused`) | Zero-shot audio description ranking | Integrated; real-checkpoint pilot run |
| YAMNet (`google/yamnet/1`) | Fixed-vocabulary audio-event classification | Integrated; real-checkpoint pilot run |
| pyannote.audio Community-1 | Speaker diarization | Adapter and tests present; **not integrated** in V1 |

Each integrated adapter has been exercised against a pretrained checkpoint in a preliminary Google Colab experiment. These are **integration smoke tests**, not systematic accuracy, robustness, latency, or cross-dataset evaluations. The model registry reflects Python integration and preliminary experimental evidence; **browser compatibility is not assessed** for these models.

The pyannote integration is deferred to V2 because the intended model requires Hugging Face access conditions and token-based setup. The presence of its adapter and mock-backed tests does not establish working checkpoint integration.

## Implemented components

- **Audio processing:** immutable `AudioClip` objects, file/byte decoding, validated sample rates and channels, resampling, and controlled signal-to-interference mixing (`audio.py`).
- **Acoustic features:** deterministic global spectral and signal descriptors (`features.py`), with time-windowed summaries (`temporal.py`). These descriptors are not automatic speech/music classifiers.
- **Explainable selection:** `models.py` and `selector.py` match an explicitly requested task against declared capabilities at the `catalog`, `integrated`, or `browser` tier. Selection explains eligibility and exclusions without silently choosing among multiple matching models.
- **Pipeline planning:** `pipeline.py` describes compatible stage sequences, but planning alone does not perform inference.
- **Executable pipelines:** `execution.py` runs DeepFilterNet3 → Whisper; `separate_execution.py` runs SepFormer → Whisper once per separated output track. Callers supply already-loaded model adapters.
- **Optional pretrained adapters:** `deepfilternet.py`, `htdemucs.py`, `sepformer.py`, `whisper.py`, `silero_vad.py`, `clap.py`, and `yamnet.py`. Imports do not eagerly download checkpoints.

### Supported executable pipelines

| Pipeline ID | Stages | Result |
|---|---|---|
| `enhance_transcribe` | DeepFilterNet3 → Whisper | Enhanced audio and predicted transcription |
| `separate_transcribe` | SepFormer → Whisper × 2 | Two aligned source estimates and two predicted transcriptions |

Both pipelines have been executed in Google Colab with real checkpoints. The separation pipeline's output order is **not a persistent speaker identity** and is **not speaker diarization**.

## Installation and tests

Requires **Python 3.11+**.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The core Python package uses NumPy, SciPy, and SoundFile. Neural-network frameworks and pretrained checkpoints are **optional and not installed by default**. This allows the unit suite to run without downloading model weights. The GitHub Actions workflow executes the repository's tests on Python 3.11; individual model checkpoints are evaluated separately in Colab.

For real-model inference, install the dependencies associated with the specific backend in a compatible environment. For example, for **SepFormer → Whisper**, the Colab pilot used:

```bash
python -m pip install "speechbrain==1.0.3" "huggingface_hub>=0.24,<1" openai-whisper
```

This is an experimental setup, not a complete, platform-independent dependency lockfile. PyTorch and compatible audio packages are also needed; Colab may already provide some of them. Other adapters use their respective model ecosystems (DeepFilterNet, Demucs, Silero VAD, Hugging Face Transformers, or TensorFlow Hub). Avoid installing all heavy backends in one environment without checking version compatibility.

## Quick start: inspect catalog availability

```python
from audio_lab.models import AudioTask
from audio_lab.selector import SelectionTier, select_models
from audio_lab.pipeline import PipelinePreset, plan_pipeline

selection = select_models(AudioTask.AUDIO_CLASSIFICATION, tier=SelectionTier.INTEGRATED)
print([model.model_id for model in selection.eligible_models])
# ['clap', 'yamnet'] — both are eligible; no automatic ranking is implied.

plan = plan_pipeline(PipelinePreset.SEPARATE_TRANSCRIBE, tier=SelectionTier.INTEGRATED)
print(plan.available)  # True, based on registered adapter availability.
```

`plan.available` reports registry-level capability, **not** whether optional packages, cached weights, hardware, or a particular input are ready for execution.

## Example: real two-speaker separation and transcription

Install compatible SepFormer and Whisper dependencies before running this example. Loading models may download large weights; CPU execution can be slow.

```python
from audio_lab.audio import load_audio
from audio_lab.sepformer import SepFormerAdapter
from audio_lab.whisper import WhisperAdapter
from audio_lab.separate_execution import execute_separate_transcribe

# Replace with a real recording containing two overlapping speakers.
# The amplitude must be within [-1, 1] after decoding and resampling.
clip = load_audio("two_speakers.wav", sample_rate=8_000, channels="mono")

separator = SepFormerAdapter.load(device="cpu")
transcriber = WhisperAdapter.load(model_name="base.en", device="cpu")
result = execute_separate_transcribe(
    clip,
    separator=separator,
    transcriber=transcriber,
    language="en",
    normalize_for_transcription=True,
)

print("Shared ASR attenuation gain:", result.transcription_gain)
for index, transcript in enumerate(result.transcriptions, start=1):
    print(f"Output track {index}:", transcript.text)
```

`normalize_for_transcription=True` computes **one common attenuation factor** when separated samples exceed `[-1, 1]`. The gain is recorded in `result.transcription_gain`, and the source estimates in `result.separation` are preserved unchanged. No clipping is performed. The default `False` mode strictly rejects out-of-range separated audio before transcription.

The SepFormer checkpoint was trained for two-speaker mixtures in a particular data domain; separation and transcription quality on other inputs are not guaranteed. The Whisper output is a model prediction, not verified reference text.

## Example: speech enhancement and transcription

With DeepFilterNet and Whisper dependencies installed, the other reviewed workflow is:

```python
from audio_lab.audio import load_audio
from audio_lab.deepfilternet import DeepFilterNet3Adapter
from audio_lab.whisper import WhisperAdapter
from audio_lab.execution import execute_enhance_transcribe

clip = load_audio("noisy_speech.wav", channels="mono")
result = execute_enhance_transcribe(
    clip,
    enhancer=DeepFilterNet3Adapter.load(),
    transcriber=WhisperAdapter.load(model_name="tiny", device="cpu"),
    language="en",
)
print(result.transcription.text)
```

Enhancement retains the original input audio and gives a separate enhanced `AudioClip`; it does not promise better recognition on every recording.

## Limits and research interpretation

- **Short-form inference:** model-specific limits apply (generally up to 30 seconds for the speech pipelines; CLAP accepts up to 10 seconds per request). Audio must meet the validation requirements of each adapter.
- **No automatic content-driven routing:** acoustic features can be computed, but are not yet used to infer a task, rank competing models, or dynamically choose a pipeline.
- **No browser release in V1:** GitHub Pages and an interactive frontend are future goals; the Python integration results do not demonstrate client-side inference feasibility.
- **No diarization guarantee:** pyannote checkpoint integration is deferred; SepFormer produces two unordered tracks, not speaker labels over time.
- **Non-equivalent classification scores:** CLAP's zero-shot scores are relative to the submitted text descriptions. YAMNet returns independent per-class scores for its fixed AudioSet vocabulary; these do not sum to one.
- **Pilot evidence:** preliminary timing and sample outputs demonstrate that checkpoints and interfaces run, not accuracy across speakers, languages, noise conditions, or hardware.

## Reproducibility and third-party licenses

The repository's regression tests use controlled backends to validate contracts and error cases without downloading weights. Real-checkpoint experiments were run separately in Google Colab. For comparable replication, record the Git commit SHA, Python and optional-package versions, checkpoint revisions, hardware, input audio and preprocessing, elapsed time, and model outputs. A fully locked multi-backend environment and consolidated quantitative benchmark are **not yet provided**.

This repository's source code is released under the **MIT License** (see `LICENSE`). Third-party packages, model weights, and datasets have their **own licenses, attribution requirements, and usage conditions**. In particular, publicly accessible or downloadable weights should not be assumed to share the repository's MIT license. Verify the applicable upstream terms before redistributing or commercializing any weights or datasets.

## Roadmap

**Python V1:** validated audio and feature processing; explainable task-based selection and planning; seven pilot-validated pretrained adapters; two real-checkpoint-tested execution pipelines.

**Future work:** reproducible evaluation datasets and quantitative metrics; acoustic-content-aware ranking and routing; pyannote access and integration; scalable pipeline presets; browser feasibility assessments and an interactive visualization experience.

## Research motivation

This project builds on an earlier Digital Signal Processing experiment in acoustic interference attenuation, developed during undergraduate Telecommunications Engineering studies. It explores the research question: *How can an audio system select and combine appropriate pretrained models, with transparent evidence about capability, processing requirements, and limitations?*

## Author

**Luciano Stork**  
PhD Student, Rowan University  
[GitHub profile](https://github.com/desouz78)
