"""Optional four-stem music separation using official Demucs 4.0.1 interfaces.

Model checkpoints are loaded explicitly with ``HTDemucsAdapter.load``.
Importing this module does not import torch or Demucs or download weights.
The supported checkpoint is the official ``htdemucs`` four-source model.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from audio_lab.audio import AudioClip, resample_audio


_MODEL_SAMPLE_RATE = 44_100
_MAX_INFERENCE_SECONDS = 30.0
_STEM_NAMES = ("drums", "bass", "other", "vocals")
_MIN_STD = 1e-8


class HTDemucsError(RuntimeError):
    """Raised for invalid model contracts, dependencies, or inference output."""


@dataclass(frozen=True, eq=False, slots=True)
class MusicSeparationResult:
    """An immutable map of four stems, each aligned with the original clip."""

    source_audio: AudioClip
    stems: Mapping[str, AudioClip]
    model_name: str = "htdemucs"

    def __post_init__(self) -> None:
        if not isinstance(self.source_audio, AudioClip):
            raise TypeError("source_audio must be an AudioClip.")
        if not isinstance(self.stems, Mapping) or set(self.stems) != set(_STEM_NAMES):
            raise ValueError("stems must contain exactly drums, bass, other and vocals.")
        if self.model_name != "htdemucs":
            raise ValueError("model_name must be 'htdemucs'.")
        ordered: dict[str, AudioClip] = {}
        for name in _STEM_NAMES:
            stem = self.stems[name]
            if not isinstance(stem, AudioClip):
                raise TypeError(f"Stem {name!r} must be an AudioClip.")
            if (stem.sample_rate != self.source_audio.sample_rate or
                    stem.frames != self.source_audio.frames or
                    stem.channels != self.source_audio.channels):
                raise ValueError(f"Stem {name!r} does not match source audio metadata.")
            ordered[name] = stem
        object.__setattr__(self, "stems", MappingProxyType(ordered))

    @property
    def vocals(self) -> AudioClip:
        return self.stems["vocals"]


class HTDemucsAdapter:
    """Separate up to 30 seconds of mono/stereo audio into four Demucs stems.

    A mono recording is duplicated for the stereo model and restored to mono
    afterwards. Stems preserve the input sample rate and frame count; sample
    amplitudes are NOT clipped or normalized, and may exceed one after mixing.
    Model execution is intentionally limited to the official Python backend.
    """

    def __init__(self, model: Any, apply_fn: Callable[..., Any], *, device: str = "cpu") -> None:
        self._device = self._validate_device(device)
        if model is None or not callable(apply_fn):
            raise TypeError("model and callable apply_fn are required.")
        if type(getattr(model, "samplerate", None)) is not int or model.samplerate != _MODEL_SAMPLE_RATE:
            raise HTDemucsError("HTDemucs requires a model sample rate of 44,100 Hz.")
        if type(getattr(model, "audio_channels", None)) is not int or model.audio_channels != 2:
            raise HTDemucsError("HTDemucs requires a stereo model.")
        sources = getattr(model, "sources", None)
        if (not isinstance(sources, (tuple, list)) or len(sources) != len(_STEM_NAMES)
                or any(not isinstance(name, str) for name in sources)
                or set(sources) != set(_STEM_NAMES)):
            raise HTDemucsError("Model must expose the four expected source names.")
        self._model = model
        self._apply_fn = apply_fn
        self._source_indices = {name: sources.index(name) for name in _STEM_NAMES}

    @staticmethod
    def _validate_device(device: str) -> str:
        if not isinstance(device, str):
            raise TypeError("device must be a string.")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'.")
        return device

    @classmethod
    def load(cls, *, device: str = "cpu") -> HTDemucsAdapter:
        """Load the official htdemucs checkpoint (potential network download)."""
        cls._validate_device(device)
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
        except ImportError as exc:
            raise HTDemucsError(
                "Install 'demucs==4.0.1' with compatible PyTorch and torchaudio."
            ) from exc
        return cls(get_model("htdemucs"), apply_model, device=device)

    def separate(self, clip: AudioClip) -> MusicSeparationResult:
        """Execute the Demucs 4.0.1 batch tensor interface and validate its output."""
        if not isinstance(clip, AudioClip):
            raise TypeError("clip must be an AudioClip.")
        if clip.channels not in (1, 2):
            raise ValueError("HTDemucs accepts mono or stereo audio only.")
        if clip.duration_seconds > _MAX_INFERENCE_SECONDS:
            raise ValueError("HTDemucs inference is limited to 30 seconds per recording.")
        if float(np.max(np.abs(clip.samples))) > 1.0:
            raise ValueError("HTDemucs input must be within [-1, 1].")

        # Reject meaningless DC-only content before resampling introduces edge
        # transients that could turn a constant signal into apparent activity.
        if float(np.max(np.ptp(clip.samples.astype(np.float64), axis=0))) < _MIN_STD:
            silence = {name: AudioClip(np.zeros_like(clip.samples), clip.sample_rate) for name in _STEM_NAMES}
            return MusicSeparationResult(clip, silence)

        prepared = resample_audio(clip, _MODEL_SAMPLE_RATE)
        if float(np.max(np.abs(prepared.samples))) > 1.0:
            raise ValueError("Resampled input exceeds [-1, 1].")
        stereo = np.repeat(prepared.samples, 2, axis=1) if clip.channels == 1 else prepared.samples
        # Follow Demucs' own reference-channel normalization. A silent or
        # constant signal cannot be normalized safely and returns silent stems.
        reference = stereo.mean(axis=1, dtype=np.float64)
        mean = float(reference.mean())
        std = float(reference.std(ddof=1)) if len(reference) > 1 else 0.0
        if std < _MIN_STD:
            # Opposite-phase stereo channels can cancel in the mono reference;
            # use full stereo power rather than discarding meaningful content.
            std = float(stereo.astype(np.float64).std(ddof=1)) if stereo.size > 1 else 0.0
            if std < _MIN_STD:
                silence = {name: AudioClip(np.zeros_like(clip.samples), clip.sample_rate) for name in _STEM_NAMES}
                return MusicSeparationResult(clip, silence)

        waveform = np.array(((stereo.astype(np.float64) - mean) / std).T[np.newaxis, :, :],
                            dtype=np.float32, copy=True, order="C")
        if not np.isfinite(waveform).all():
            raise HTDemucsError("Normalized model input contains nonfinite values.")
        try:
            import torch
        except ImportError as exc:
            raise HTDemucsError("PyTorch is required for HTDemucs inference.") from exc

        estimates = self._apply_fn(
            self._model, torch.from_numpy(waveform),
            shifts=0, split=True, overlap=0.25, progress=False,
            device=self._device, num_workers=0,
        )
        try:
            array = estimates.detach().cpu().numpy()
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise HTDemucsError("HTDemucs must return a readable PyTorch tensor.") from exc
        if (not isinstance(array, np.ndarray) or array.shape != (1, 4, 2, prepared.frames)
                or array.dtype.kind != "f" or not np.isfinite(array).all()):
            raise HTDemucsError("HTDemucs output has invalid shape, dtype or values.")

        results: dict[str, AudioClip] = {}
        for name in _STEM_NAMES:
            prediction = array[0, self._source_indices[name]]
            with np.errstate(over="ignore", invalid="ignore"):
                samples = np.array((prediction.astype(np.float64) * std + mean).T,
                                   dtype=np.float32, copy=True)
            if not np.isfinite(samples).all():
                raise HTDemucsError(f"Stem {name!r} exceeds the float32 amplitude range.")
            if clip.channels == 1:
                samples = samples.mean(axis=1, keepdims=True)
            stem = resample_audio(AudioClip(samples, _MODEL_SAMPLE_RATE), clip.sample_rate)
            if abs(stem.frames - clip.frames) > 1:
                raise HTDemucsError(f"Stem {name!r} has an unexpected frame count.")
            if stem.frames != clip.frames:
                adjusted = np.zeros((clip.frames, clip.channels), dtype=np.float32)
                count = min(stem.frames, clip.frames)
                adjusted[:count] = stem.samples[:count]
                stem = AudioClip(adjusted, clip.sample_rate)
            results[name] = stem
        return MusicSeparationResult(clip, results)
