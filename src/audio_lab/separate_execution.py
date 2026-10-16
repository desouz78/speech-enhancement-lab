"""Run the explicit SepFormer -> Whisper two-speaker Python pipeline.

The caller supplies loaded adapters. The two separated tracks are transcribed
independently in model-output order; their indices are NOT speaker identities.
No checkpoints, audio normalization or automatic model selection occur here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from audio_lab.audio import AudioClip
from audio_lab.execution import PipelineExecutionError
from audio_lab.pipeline import PipelinePreset
from audio_lab.sepformer import SpeechSeparationResult
from audio_lab.whisper import TranscriptionResult


_MAX_SECONDS = 30.0


class SpeakerSeparator(Protocol):
    """Minimal contract implemented by SepFormerAdapter."""

    def separate(self, clip: AudioClip) -> SpeechSeparationResult: ...


class SpeechTranscriber(Protocol):
    """Minimal contract implemented by WhisperAdapter."""

    def transcribe(self, clip: AudioClip, *, language: str | None = None) -> TranscriptionResult: ...


@dataclass(frozen=True, eq=False, slots=True)
class SeparateTranscribeResult:
    """Two aligned separated tracks and their respective ASR predictions."""

    source_audio: AudioClip
    separation: SpeechSeparationResult
    transcriptions: tuple[TranscriptionResult, TranscriptionResult]

    def __post_init__(self) -> None:
        if not isinstance(self.source_audio, AudioClip):
            raise TypeError("source_audio must be an AudioClip.")
        if not isinstance(self.separation, SpeechSeparationResult):
            raise TypeError("separation must be a SpeechSeparationResult.")
        if self.separation.source_audio is not self.source_audio:
            raise ValueError("Separation must reference the original source audio.")
        if not isinstance(self.transcriptions, tuple) or len(self.transcriptions) != 2:
            raise TypeError("transcriptions must be a tuple containing two results.")
        if any(not isinstance(item, TranscriptionResult) for item in self.transcriptions):
            raise TypeError("Each transcription must be a TranscriptionResult.")

    @property
    def pipeline_id(self) -> PipelinePreset:
        return PipelinePreset.SEPARATE_TRANSCRIBE

    @property
    def speaker_1_transcription(self) -> TranscriptionResult:
        """First model-output track, not a persistent speaker identity."""
        return self.transcriptions[0]

    @property
    def speaker_2_transcription(self) -> TranscriptionResult:
        """Second model-output track, not a persistent speaker identity."""
        return self.transcriptions[1]


def execute_separate_transcribe(
    clip: AudioClip,
    *,
    separator: SpeakerSeparator,
    transcriber: SpeechTranscriber,
    language: str | None = None,
) -> SeparateTranscribeResult:
    """Separate a mono recording into two tracks and transcribe each in order.

    The separator must preserve source metadata. Each separated track must be
    aligned with the input and have peak amplitude within [-1, 1] for Whisper.
    No silent clipping, rescaling, speaker identification or diarization occurs.
    Stage exceptions propagate unchanged, as with execute_enhance_transcribe.
    """
    if not isinstance(clip, AudioClip):
        raise TypeError("clip must be an AudioClip.")
    if not callable(getattr(separator, "separate", None)):
        raise TypeError("separator must expose a callable separate(clip) method.")
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

    separation = separator.separate(clip)
    if not isinstance(separation, SpeechSeparationResult):
        raise PipelineExecutionError("Separator must return a SpeechSeparationResult.")
    if separation.source_audio is not clip:
        raise PipelineExecutionError("Separator did not retain the original source audio.")

    # SpeechSeparationResult already guarantees channel and time alignment.
    for index, source in enumerate(separation.sources, start=1):
        if float(np.max(np.abs(source.samples))) > 1.0:
            raise PipelineExecutionError(f"Separated track {index} exceeds [-1, 1]; reduce gain explicitly.")

    transcriptions: list[TranscriptionResult] = []
    for index, source in enumerate(separation.sources, start=1):
        transcription = transcriber.transcribe(source, language=language)
        if not isinstance(transcription, TranscriptionResult):
            raise PipelineExecutionError(f"Transcription for separated track {index} is invalid.")
        transcriptions.append(transcription)

    return SeparateTranscribeResult(clip, separation, (transcriptions[0], transcriptions[1]))
