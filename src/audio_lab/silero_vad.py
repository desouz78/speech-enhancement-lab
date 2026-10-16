"""Optional speech-activity detection with the official Silero VAD Python package.

Audio is mono and evaluated at 8 or 16 kHz. Model loading is explicit and
imports neither PyTorch nor Silero at module import time. The result contains
speech intervals, not transcripts, separated voices, or speaker identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from audio_lab.audio import AudioClip, resample_audio


_MODEL_RATE = 16_000
_NATIVE_RATES = (8_000, 16_000)
_SILENCE_FLOOR = 1e-8
_MAX_MS = 180_000


class SileroVADError(RuntimeError):
    """Raised when dependencies, inference, or model outputs violate the contract."""


@dataclass(frozen=True, slots=True)
class SpeechInterval:
    """Half-open speech interval in samples at the source audio's sample rate."""

    start_sample: int
    end_sample: int
    sample_rate: int

    def __post_init__(self) -> None:
        if type(self.start_sample) is not int or type(self.end_sample) is not int:
            raise TypeError("Interval boundaries must be integer samples.")
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("Interval sample rate must be a positive integer.")
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("Speech intervals must have positive length and a nonnegative start.")

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


@dataclass(frozen=True, eq=False, slots=True)
class VoiceActivityResult:
    """Immutable, ordered, nonoverlapping speech intervals aligned to input."""

    source_audio: AudioClip
    intervals: tuple[SpeechInterval, ...]
    model_name: str = "silero-vad"

    def __post_init__(self) -> None:
        if not isinstance(self.source_audio, AudioClip):
            raise TypeError("source_audio must be an AudioClip.")
        if not isinstance(self.intervals, tuple):
            raise TypeError("intervals must be a tuple.")
        if self.model_name != "silero-vad":
            raise ValueError("Unexpected voice activity model name.")
        previous_end = 0
        for interval in self.intervals:
            if not isinstance(interval, SpeechInterval):
                raise TypeError("Every interval must be a SpeechInterval.")
            if (interval.sample_rate != self.source_audio.sample_rate
                    or interval.start_sample < previous_end
                    or interval.end_sample > self.source_audio.frames):
                raise ValueError("Speech intervals must be ordered, disjoint and within the source audio.")
            previous_end = interval.end_sample

    @property
    def speech_duration_seconds(self) -> float:
        return sum(interval.end_sample - interval.start_sample for interval in self.intervals) / self.source_audio.sample_rate

    @property
    def speech_fraction(self) -> float:
        return sum(interval.end_sample - interval.start_sample for interval in self.intervals) / self.source_audio.frames


class SileroVADAdapter:
    """Detect speech intervals without classifying or identifying speakers."""

    def __init__(self, model: Any, get_timestamps: Callable[..., Any]) -> None:
        if model is None or not callable(get_timestamps):
            raise TypeError("A model and a callable timestamp extractor are required.")
        self._model = model
        self._get_timestamps = get_timestamps

    @classmethod
    def load(cls) -> SileroVADAdapter:
        """Load the packaged Silero VAD PyTorch checkpoint on explicit request."""
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            raise SileroVADError("Install 'silero-vad' with compatible PyTorch dependencies.") from exc
        try:
            model = load_silero_vad()
        except Exception as exc:
            raise SileroVADError("Silero VAD model loading failed.") from exc
        return cls(model, get_speech_timestamps)

    @staticmethod
    def _validate_parameters(threshold: float, min_speech_duration_ms: int,
                             min_silence_duration_ms: int, speech_pad_ms: int) -> None:
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.integer, np.floating)):
            raise TypeError("threshold must be a numeric value.")
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and between 0 and 1.")
        for name, value in (("min_speech_duration_ms", min_speech_duration_ms),
                            ("min_silence_duration_ms", min_silence_duration_ms),
                            ("speech_pad_ms", speech_pad_ms)):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer number of milliseconds.")
            if not 0 <= value <= _MAX_MS:
                raise ValueError(f"{name} must be between 0 and {_MAX_MS} milliseconds.")

    def detect(self, clip: AudioClip, *, threshold: float = 0.5, min_speech_duration_ms: int = 250,
               min_silence_duration_ms: int = 100, speech_pad_ms: int = 30) -> VoiceActivityResult:
        """Return half-open speech intervals at the original sample rate.

        Backends receive float32 mono audio and return sample-index intervals,
        not rounded timestamps in seconds. Invalid output fails explicitly.
        """
        if not isinstance(clip, AudioClip):
            raise TypeError("clip must be an AudioClip.")
        self._validate_parameters(threshold, min_speech_duration_ms, min_silence_duration_ms, speech_pad_ms)
        if clip.channels != 1:
            raise ValueError("Silero VAD requires mono audio; downmix explicitly.")
        if float(np.max(np.abs(clip.samples))) > 1.0:
            raise ValueError("Silero VAD expects input within [-1, 1].")
        if float(np.max(np.abs(clip.samples))) < _SILENCE_FLOOR:
            return VoiceActivityResult(clip, ())

        rate = clip.sample_rate if clip.sample_rate in _NATIVE_RATES else _MODEL_RATE
        prepared = resample_audio(clip, rate)
        if float(np.max(np.abs(prepared.samples))) > 1.0:
            raise ValueError("Resampled audio exceeds [-1, 1]; reduce input gain explicitly.")

        try:
            import torch
        except ImportError as exc:
            raise SileroVADError("PyTorch is required for Silero VAD inference.") from exc
        waveform = np.array(prepared.samples[:, 0], dtype=np.float32, copy=True)
        try:
            with torch.inference_mode():
                output = self._get_timestamps(
                    torch.from_numpy(waveform), self._model, sampling_rate=rate,
                    threshold=float(threshold), min_speech_duration_ms=min_speech_duration_ms,
                    min_silence_duration_ms=min_silence_duration_ms, speech_pad_ms=speech_pad_ms,
                    return_seconds=False,
                )
        except Exception as exc:
            raise SileroVADError("Silero VAD inference failed.") from exc

        if not isinstance(output, (list, tuple)):
            raise SileroVADError("Silero VAD must return a sequence of timestamp mappings.")
        intervals: list[SpeechInterval] = []
        previous_end = 0
        for item in output:
            if not isinstance(item, dict) or "start" not in item or "end" not in item:
                raise SileroVADError("Silero VAD returned a malformed speech interval.")
            start, end = item["start"], item["end"]
            if (isinstance(start, (bool, np.bool_)) or isinstance(end, (bool, np.bool_))
                    or not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer))
                    or not 0 <= start < end <= prepared.frames or start < previous_end):
                raise SileroVADError("Silero VAD returned invalid, overlapping or unordered boundaries.")
            previous_end = int(end)
            mapped_start = min(clip.frames, round(int(start) * clip.sample_rate / rate))
            mapped_end = min(clip.frames, round(int(end) * clip.sample_rate / rate))
            if mapped_end <= mapped_start or (intervals and mapped_start < intervals[-1].end_sample):
                raise SileroVADError("Speech intervals lost alignment during resampling.")
            intervals.append(SpeechInterval(mapped_start, mapped_end, clip.sample_rate))
        return VoiceActivityResult(clip, tuple(intervals))
