# Speech Enhancement Lab

**An AI-driven framework for intelligent audio analysis, model selection, and processing.**

[![Python Tests](https://github.com/desouz78/speech-enhancement-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/desouz78/speech-enhancement-lab/actions/workflows/tests.yml)

**Project Status:** Active Development

## Overview

Different acoustic scenarios require different AI models. A neural network designed to suppress background noise may not be suitable for separating overlapping speakers, extracting musical instruments, or transcribing speech.

**Speech Enhancement Lab** explores how audio characteristics and processing objectives can guide the automatic selection and orchestration of pretrained AI models.

The project aims to develop an interactive audio laboratory where users can introduce acoustic interference, explore different processing scenarios, and understand how specialized neural networks interpret and transform audio signals.

Rather than developing new neural architectures, the focus is on building an explainable, modular system that connects audio analysis, intelligent model selection, and AI-powered processing.

## Project Objectives

- **Audio Characterization:** Extract acoustic features and identify relevant properties of incoming audio.
- **Intelligent Model Selection:** Recommend appropriate pretrained AI models based on acoustic conditions, processing objectives, and computational requirements.
- **Speech Enhancement:** Suppress unwanted acoustic interference while preserving speech.
- **Source Separation:** Separate overlapping voices, vocals, and musical instruments.
- **Speech Recognition:** Generate text transcriptions from spoken audio.
- **Speaker Diarization:** Determine when different speakers are active in a recording.
- **Explainable AI:** Present the selected models, the reasoning behind their selection, and their processing results.
- **Interactive Visualization:** Explore audio signals and compare their spectrograms before and after processing.

## AI Model Ecosystem

The following pretrained models are candidates for integration:

| Model | Primary Application | Status |
|---|---|---|
| DeepFilterNet3 | Speech enhancement and noise suppression | Experimentally validated |
| HTDemucs | Music source separation | Experimentally validated |
| SepFormer | Overlapping speech separation | Planned |
| Whisper | Automatic speech recognition and transcription | Planned |
| Silero VAD | Voice activity detection | Planned |
| pyannote.audio | Speaker diarization | Under consideration |
| CLAP | Semantic audio classification | Under consideration |
| YAMNet | Environmental sound classification | Under consideration |

DeepFilterNet3 and HTDemucs have been evaluated in preliminary Google Colab experiments. Their integration into the modular Python framework is still in development.

Other models will be evaluated based on compatibility, processing requirements, and relevance to the intended demonstration.

## Intelligent Model Selection

The central component of the planned system is an **Adaptive AI Model Selector**.

Its purpose is to analyze an incoming audio signal and determine which processing model, or combination of models, is appropriate for the selected task.

The decision process will consider:

- Acoustic characteristics of the input signal.
- Presence of speech, music, and environmental sounds.
- Noise and interference conditions.
- Potential speaker overlap.
- Intended processing task.
- Model capabilities and computational requirements.

Rather than treating every recording identically, the system will provide context-aware processing recommendations.

Model selection decisions will be accompanied by explanations of the relevant acoustic characteristics and expected processing behavior.

## Current Implementation

The Python audio processing core currently provides:

- Validated audio loading from supported files and encoded audio bytes.
- Mono and multichannel signal handling.
- Sample-rate conversion.
- Controlled signal-to-interference mixing.
- Handling of short interference recordings.
- Peak-safe normalization while preserving relative signal power.
- Input validation and resource limits.
- Automated regression testing through GitHub Actions.

The implementation is located in `src/audio_lab/audio.py`.

This module establishes the signal-processing infrastructure required for subsequent AI integration.

## Planned Interactive Demonstration

The interactive laboratory is designed around five stages.

### 1. Audio Input

Users will be able to:

- Select audio recordings from an available sample library.
- Upload their own audio files.
- Record audio directly from a microphone.
- Inspect the corresponding waveform and spectrogram.

### 2. Acoustic Interference

Users will be able to:

- Select environmental noise, music, or overlapping speech from an interference library.
- Upload or record custom interference signals.
- Adjust the signal-to-interference ratio.
- Visualize the interference independently and its effect on the original audio.

### 3. AI Analysis and Selection

The system will analyze the resulting audio and recommend suitable AI processing models.

The demonstration will present:

- Detected acoustic characteristics.
- Recommended model or processing pipeline.
- Explanation of the selection.
- Expected capabilities and limitations.

### 4. AI Processing

Depending on the selected task, the system may perform:

- Speech enhancement.
- Music source separation.
- Overlapping speaker separation.
- Speech transcription.
- Speaker diarization.

Multiple models may be combined when the task requires sequential processing.

### 5. Results and Visualization

Users will be able to explore:

- Original, interference, and mixed audio.
- Processed audio and separated sources.
- Spectrograms before and after processing.
- Visual comparisons highlighting attenuated and preserved signal components.
- Speech transcripts when applicable.
- Model selection explanations and processing information.

Spectrograms will be available throughout the interaction, including when audio sources and interference signals are introduced.

## Deployment Strategy

The project targets a static, browser-accessible demonstration hosted through GitHub Pages.

The intended architecture avoids dedicated backend infrastructure and recurring server costs.

Python will be used for algorithm development, experimentation, and validation. Browser-compatible inference technologies will be evaluated for the interactive implementation.

Because pretrained models differ in memory requirements, runtime dependencies, and computational complexity, not every model can be assumed to support real-time browser execution.

The demonstration will prioritize lightweight processing and practical compatibility, with clearly identified limitations.

## Development

**Requirements:** Python 3.11 or newer.

Install the project and development dependencies:

`python -m pip install -e ".[dev]"`

Run automated tests:

`python -m pytest -v`

The repository uses GitHub Actions for continuous integration and automated validation.

## Research Motivation

This project revisits an earlier Digital Signal Processing experiment developed during my undergraduate studies in Telecommunications Engineering.

The original experiment investigated acoustic interference attenuation using a conventional digital filtering approach.

The current project extends that motivation toward AI-driven audio processing, exploring how specialized pretrained models can be selected and combined to address different acoustic scenarios.

The guiding question is:

**How can an audio processing system identify and select suitable pretrained AI models based on acoustic characteristics, processing objectives, and computational constraints?**

The emphasis is on explainable decision-making, modular software design, experimental evaluation, and an accessible interactive demonstration.

## Author

**Luciano Stork**

PhD Student, Rowan University

[GitHub Profile](https://github.com/desouz78)

## License

This project is distributed under the MIT License.

Third-party models, pretrained weights, and datasets are subject to their respective licenses and usage restrictions.
