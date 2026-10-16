"""Execute the validated DeepFilterNet3 -> Whisper short-form Python pipeline.

This module owns orchestration only. It does not load checkpoints, perform
automatic task classification, or imply browser compatibility. Callers supply
already-loaded model adapters; their optional dependencies remain isolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from audio_lab.audio import AudioClip
from audio_lab.pipeline import PipelinePreset
from audio_lab.whisper import TranscriptionResult


_MAX_SECONDS = 30.0


class PipelineExecutionError(RuntimeError):
    """An inference stage returned data violating the pipeline contract."""


class SpeechEnhancer(Protocol):
    """Minimal contract implemented by DeepFilterNet3Adapter."""

    def enhance(self, clip: AudioClip) -> AudioClip: ...


class SpeechTranscriber(Protocol):
    """Minimal contract implemented by WhisperAdapter."""

    def transcribe(
        self, clip: AudioClip, *, language: str | None = None
    ) -> TranscriptionResult: ...


@dataclass(frozen=True, eq=False, slots=True)
class EnhanceTranscribeResult:
    """Immutable references to the input, enhanced audio and ASR prediction."""

    source_audio: AudioClip
    enhanced_audio: AudioClip
    transcription: TranscriptionResult

    def __post_init__(self) -> None:
        if not isinstance(self.source_audio, AudioClip):
            raise TypeError("source_audio must be an AudioClip.")
        if not isinstance(self.enhanced_audio, AudioClip):
            raise TypeError("enhanced_audio must be an AudioClip.")
        if not isinstance(self.transcription, TranscriptionResult):
            raise TypeError("transcription must be a TranscriptionResult.")

    @property
    def pipeline_id(self) -> PipelinePreset:
        return PipelinePreset.ENHANCE_TRANSCRIBE


def execute_enhance_transcribe(
    clip: AudioClip,
    *,
    enhancer: SpeechEnhancer,
    transcriber: SpeechTranscriber,
    language: str | None = None,
) -> EnhanceTranscribeResult:
    """Enhance one mono recording, then transcribe the enhanced recording.

    Preconditions are checked before invoking either model. Returned enhanced
    audio must retain the exact sample rate, channel count and frame count.
    Output gain is not silently modified. Model failures propagate unchanged,
    allowing callers to handle them with accurate stage-specific diagnostics.

    This is an explicit pipeline, not an automatic classification or selector.
    """
    if not isinstance(clip, AudioClip):
        raise TypeError("clip must be an AudioClip.")
    if not callable(getattr(enhancer, "enhance", None)):
        raise TypeError("enhancer must expose a callable enhance(clip) method.")
    if not callable(getattr(transcriber, "transcribe", None)):
        raise TypeError("transcriber must expose a callable transcribe(clip, ...) method.")
    if language is not None and not isinstance(language, str):
        raise TypeError("language must be a string or None.")
    if clip.channels != 1:
        raise ValueError("The pipeline requires mono audio; downmix explicitly.")
    if clip.duration_seconds > _MAX_SECONDS:
        raise ValueError("The pipeline supports recordings of at most 30 seconds.")
    if float(np.max(np.abs(clip.samples))) > 1.0:
        raise ValueError("Input amplitude must be within [-1, 1].")

    enhanced = enhancer.enhance(clip)
    if not isinstance(enhanced, AudioClip):
        raise PipelineExecutionError("Enhancer must return an AudioClip.")
    if enhanced.sample_rate != clip.sample_rate:
        raise PipelineExecutionError("Enhancer changed the audio sample rate.")
    if enhanced.channels != clip.channels:
        raise PipelineExecutionError("Enhancer changed the channel count.")
    if enhanced.frames != clip.frames:
        raise PipelineExecutionError("Enhancer changed the audio frame count.")
    if float(np.max(np.abs(enhanced.samples))) > 1.0:
        raise PipelineExecutionError(
            "Enhanced audio exceeds [-1, 1]; no implicit clipping or normalization is applied."
        )

    transcription = transcriber.transcribe(enhanced, language=language)
    if not isinstance(transcription, TranscriptionResult):
        raise PipelineExecutionError("Transcriber must return a TranscriptionResult.")

    return EnhanceTranscribeResult(clip, enhanced, transcription)
