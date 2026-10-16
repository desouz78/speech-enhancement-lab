"""Optional OpenAI Whisper adapter for short-form speech transcription.

The adapter accepts validated mono AudioClip objects and explicitly resamples
input to 16 kHz. Importing this module does not import Whisper, download model
weights, or initialize a neural network. Browser inference is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Mapping

import numpy as np

from audio_lab.audio import AudioClip, resample_audio


_MODEL_SAMPLE_RATE = 16_000
_MAX_INFERENCE_SECONDS = 30.0
_SUPPORTED_MODELS = frozenset({
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large", "large-v2", "large-v3", "turbo",
})


class WhisperError(RuntimeError):
    """Raised when Whisper dependencies or the inference contract are invalid."""


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """One timestamped phrase; boundaries are model predictions, not ground truth."""

    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Text and optional timestamped segments returned by an ASR model."""

    text: str
    language: str
    segments: tuple[TranscriptionSegment, ...]
    model_name: str


def _parse_language(language: str | None) -> str | None:
    """Accept lowercase 2-3 letter language codes or automatic detection."""
    if language is None:
        return None
    if not isinstance(language, str):
        raise TypeError("language must be a lowercase language code or None.")
    if not (2 <= len(language) <= 3 and language.isascii() and language.isalpha()
            and language.islower()):
        raise ValueError("language must be a lowercase ISO-style code (2-3 letters).")
    return language


def _validate_model_name(model_name: str) -> str:
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string.")
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError("Unsupported Whisper model name.")
    return model_name


def _validate_device(device: str) -> str:
    if not isinstance(device, str):
        raise TypeError("device must be a string.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")
    return device


def _validate_output(raw: Any, model_name: str) -> TranscriptionResult:
    """Validate the backend result without inventing absent transcriptions."""
    if not isinstance(raw, Mapping):
        raise WhisperError("Whisper must return a mapping with transcription data.")
    text = raw.get("text")
    language = raw.get("language")
    segments = raw.get("segments")
    if not isinstance(text, str):
        raise WhisperError("Whisper returned an invalid transcription text.")
    if not isinstance(language, str) or not language or len(language) > 40:
        raise WhisperError("Whisper returned an invalid detected language.")
    if not isinstance(segments, (tuple, list)):
        raise WhisperError("Whisper returned an invalid segments collection.")
    if len(segments) > 3_000:
        raise WhisperError("Whisper returned too many transcription segments.")

    validated = []
    previous_start = 0.0
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise WhisperError("Every transcription segment must be a mapping.")
        start = segment.get("start")
        end = segment.get("end")
        segment_text = segment.get("text")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               for value in (start, end)):
            raise WhisperError("Transcription timestamps must be real numbers.")
        start, end = float(start), float(end)
        if (not isfinite(start) or not isfinite(end) or start < 0.0
                or end < start or start < previous_start):
            raise WhisperError("Transcription timestamps are invalid or out of order.")
        if not isinstance(segment_text, str):
            raise WhisperError("Transcription segment text must be a string.")
        validated.append(TranscriptionSegment(start, end, segment_text.strip()))
        previous_start = start

    return TranscriptionResult(text.strip(), language, tuple(validated), model_name)


class WhisperAdapter:
    """Transcribe up to 30 seconds of mono audio using pretrained Whisper.

    Model weights are loaded explicitly with :meth:`load`. The ``tiny`` model
    is the default for lightweight integration experiments, not guaranteed
    recognition accuracy. The output is an ASR prediction, not verified speech.
    """

    def __init__(self, model: Any, model_name: str = "tiny") -> None:
        self._model_name = _validate_model_name(model_name)
        if model is None or not callable(getattr(model, "transcribe", None)):
            raise TypeError("model must provide a callable transcribe() method.")
        self._model = model

    @classmethod
    def load(cls, model_name: str = "tiny", *, device: str = "cpu") -> WhisperAdapter:
        """Load an explicitly selected pretrained model (may download weights)."""
        name = _validate_model_name(model_name)
        target_device = _validate_device(device)
        try:
            import whisper
        except ImportError as exc:
            raise WhisperError("Install 'openai-whisper' to enable speech transcription.") from exc
        loader = getattr(whisper, "load_model", None)
        if not callable(loader):
            raise WhisperError("The installed 'whisper' package lacks load_model().")
        model = loader(name, device=target_device)
        return cls(model, name)

    def transcribe(self, clip: AudioClip, *, language: str | None = None) -> TranscriptionResult:
        """Transcribe validated mono audio, preserving the original samples.

        ``language=None`` requests automatic language detection. A fixed
        zero-temperature decoding configuration improves comparability, but
        does not guarantee identical inference across different hardware.
        """
        if not isinstance(clip, AudioClip):
            raise TypeError("clip must be an AudioClip instance.")
        requested_language = _parse_language(language)
        if clip.channels != 1:
            raise ValueError("Whisper requires mono audio; downmix explicitly.")
        if clip.duration_seconds > _MAX_INFERENCE_SECONDS:
            raise ValueError("Whisper inference is limited to 30 seconds per recording.")
        if float(np.max(np.abs(clip.samples))) > 1.0:
            raise ValueError("Whisper input must be within the [-1, 1] range.")

        prepared = resample_audio(clip, _MODEL_SAMPLE_RATE)
        # A writable, private waveform avoids exposing AudioClip's read-only buffer.
        waveform = np.array(prepared.samples[:, 0], dtype=np.float32, copy=True)
        if not np.all(np.isfinite(waveform)):
            raise WhisperError("Prepared audio contains nonfinite values.")
        raw = self._model.transcribe(
            waveform,
            language=requested_language,
            task="transcribe",
            fp16=False,
            verbose=None,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        return _validate_output(raw, self._model_name)
