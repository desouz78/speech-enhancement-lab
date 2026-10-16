"""Run the explicit SepFormer -> Whisper two-speaker Python pipeline.

The caller supplies loaded adapters. The two separated tracks are transcribed
independently in model-output order; their indices are NOT speaker identities.
No checkpoints or automatic model selection occur here. An explicit option
provides shared-peak normalization for ASR without modifying separated sources.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    transcription_gain: float = 1.0

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
        if (type(self.transcription_gain) is not float or not math.isfinite(self.transcription_gain)
                or not 0.0 < self.transcription_gain <= 1.0):
            raise ValueError("transcription_gain must be a finite float in (0, 1].")

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
    normalize_for_transcription: bool = False,
) -> SeparateTranscribeResult:
    """Separate a mono recording into two tracks and transcribe each in order.

    The separator must preserve source metadata. Each separated track must be
    aligned with the input. By default, unbounded source amplitudes are rejected.
    Explicit ``normalize_for_transcription=True`` applies one shared attenuation
    gain across both ASR inputs, preserving relative levels and keeping the
    original separated sources unchanged. The applied gain is recorded in the
    result; no clipping, speaker identification or diarization occurs.
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
    if type(normalize_for_transcription) is not bool:
        raise TypeError("normalize_for_transcription must be a boolean.")
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
    peaks = tuple(float(np.max(np.abs(source.samples))) for source in separation.sources)
    if not normalize_for_transcription:
        for index, peak in enumerate(peaks, start=1):
            if peak > 1.0:
                raise PipelineExecutionError(f"Separated track {index} exceeds [-1, 1]; reduce gain explicitly.")
    gain = 1.0 / max(1.0, *peaks) if normalize_for_transcription else 1.0
    asr_sources = (
        tuple(AudioClip(source.samples.astype(np.float64) * gain, source.sample_rate) for source in separation.sources)
        if gain < 1.0 else separation.sources
    )

    transcriptions: list[TranscriptionResult] = []
    for index, source in enumerate(asr_sources, start=1):
        transcription = transcriber.transcribe(source, language=language)
        if not isinstance(transcription, TranscriptionResult):
            raise PipelineExecutionError(f"Transcription for separated track {index} is invalid.")
        transcriptions.append(transcription)

    return SeparateTranscribeResult(clip, separation, (transcriptions[0], transcriptions[1]), float(gain))
