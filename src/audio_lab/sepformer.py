"""Optional two-speaker separation with the public SpeechBrain SepFormer checkpoint.

This adapter uses the ``speechbrain/sepformer-wsj02mix`` model, trained for
mono 8-kHz mixtures of two speakers. Model downloads and heavy imports occur
only when explicitly loading or invoking the model. It does not establish
persistent speaker identity, perform diarization, or promise multilingual
separation quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_lab.audio import AudioClip, resample_audio


_MODEL_SAMPLE_RATE = 8_000
_MODEL_ID = "speechbrain/sepformer-wsj02mix"
_MAX_SECONDS = 30.0
_SILENCE_THRESHOLD = 1e-8


class SepFormerError(RuntimeError):
    """Raised when a pretrained model violates the speech-separation contract."""


@dataclass(frozen=True, eq=False, slots=True)
class SpeechSeparationResult:
    """Two time-aligned speech estimates with arbitrary source ordering."""

    source_audio: AudioClip
    sources: tuple[AudioClip, AudioClip]
    model_name: str = "sepformer-wsj02mix"

    def __post_init__(self) -> None:
        if not isinstance(self.source_audio, AudioClip):
            raise TypeError("source_audio must be an AudioClip.")
        if not isinstance(self.sources, tuple) or len(self.sources) != 2:
            raise TypeError("sources must be a tuple of exactly two AudioClip objects.")
        if self.model_name != "sepformer-wsj02mix":
            raise ValueError("Unexpected SepFormer model name.")
        for source in self.sources:
            if not isinstance(source, AudioClip):
                raise TypeError("Every separated source must be an AudioClip.")
            if (source.channels != 1 or source.sample_rate != self.source_audio.sample_rate
                    or source.frames != self.source_audio.frames):
                raise ValueError("Separated sources must be mono and aligned with the input.")

    @property
    def speaker_1(self) -> AudioClip:
        """First estimated source; numbering is not a stable speaker identity."""
        return self.sources[0]

    @property
    def speaker_2(self) -> AudioClip:
        """Second estimated source; numbering is not a stable speaker identity."""
        return self.sources[1]


class SepFormerAdapter:
    """Run pretrained two-speaker separation on up to 30 seconds of mono audio."""

    def __init__(self, model: Any) -> None:
        if model is None or not callable(getattr(model, "separate_batch", None)):
            raise TypeError("model must expose a callable separate_batch() method.")
        hparams = getattr(model, "hparams", None)
        if (type(getattr(hparams, "sample_rate", None)) is not int
                or hparams.sample_rate != _MODEL_SAMPLE_RATE):
            raise SepFormerError("The SepFormer checkpoint must operate at 8,000 Hz.")
        if type(getattr(hparams, "num_spks", None)) is not int or hparams.num_spks != 2:
            raise SepFormerError("The SepFormer checkpoint must produce exactly two speakers.")
        self._model = model

    @classmethod
    def load(
        cls, *, device: str = "cpu", savedir: str | Path | None = None
    ) -> SepFormerAdapter:
        """Load the public checkpoint; downloads weights only on explicit call.

        ``savedir`` is a local cache path, not a model identifier or network URL.
        PyTorch, torchaudio and SpeechBrain are optional runtime dependencies.
        """
        if not isinstance(device, str):
            raise TypeError("device must be 'cpu' or 'cuda'.")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'.")
        if savedir is None:
            cache_dir = Path.home() / ".cache" / "speech-enhancement-lab" / "sepformer-wsj02mix"
        elif isinstance(savedir, (str, Path)):
            if not str(savedir).strip():
                raise ValueError("savedir must be a nonempty local directory path.")
            cache_dir = Path(savedir)
        else:
            raise TypeError("savedir must be a directory path or None.")
        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError as exc:
            raise SepFormerError("Install compatible 'speechbrain', 'torch' and 'torchaudio'.") from exc
        model = SepformerSeparation.from_hparams(
            source=_MODEL_ID, savedir=str(cache_dir), run_opts={"device": device}
        )
        return cls(model)

    def separate(self, clip: AudioClip) -> SpeechSeparationResult:
        """Return two estimates at the original rate, without amplitude clipping.

        A near-silent signal produces two silent sources without invoking the
        model. For other input, SpeechBrain receives a private float32 mono
        tensor with shape (1, frames). Its output must have shape (1, frames, 2).
        """
        if not isinstance(clip, AudioClip):
            raise TypeError("clip must be an AudioClip.")
        if clip.channels != 1:
            raise ValueError("SepFormer requires mono audio; downmix explicitly.")
        if clip.duration_seconds > _MAX_SECONDS:
            raise ValueError("SepFormer inference is limited to 30 seconds per recording.")
        if float(np.max(np.abs(clip.samples))) > 1.0:
            raise ValueError("Input audio must be within [-1, 1].")

        if float(np.max(np.abs(clip.samples))) < _SILENCE_THRESHOLD:
            silence = AudioClip(np.zeros_like(clip.samples), clip.sample_rate)
            return SpeechSeparationResult(clip, (silence, silence))

        prepared = resample_audio(clip, _MODEL_SAMPLE_RATE)
        if float(np.max(np.abs(prepared.samples))) > 1.0:
            raise ValueError("Resampled audio exceeds [-1, 1]; reduce input gain explicitly.")
        waveform = np.array(prepared.samples[:, 0][None, :], dtype=np.float32, copy=True)
        try:
            import torch
        except ImportError as exc:
            raise SepFormerError("PyTorch is required for SepFormer inference.") from exc
        try:
            with torch.inference_mode():
                estimates = self._model.separate_batch(torch.from_numpy(waveform))
        except Exception as exc:
            raise SepFormerError("SepFormer inference failed.") from exc
        try:
            array = estimates.detach().cpu().numpy()
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise SepFormerError("SepFormer must return a readable PyTorch tensor.") from exc
        if (not isinstance(array, np.ndarray) or array.shape != (1, prepared.frames, 2)
                or array.dtype.kind != "f" or not np.isfinite(array).all()):
            raise SepFormerError("SepFormer returned an invalid tensor shape, dtype or values.")
        if float(np.max(np.abs(array))) > float(np.finfo(np.float32).max):
            raise SepFormerError("SepFormer output exceeds the float32 amplitude range.")

        sources: list[AudioClip] = []
        for source_index in range(2):
            estimate = AudioClip(np.array(array[0, :, source_index], dtype=np.float32), _MODEL_SAMPLE_RATE)
            restored = resample_audio(estimate, clip.sample_rate)
            if abs(restored.frames - clip.frames) > 1:
                raise SepFormerError("Resampled source has an unexpected frame count.")
            if restored.frames != clip.frames:
                adjusted = np.zeros((clip.frames, 1), dtype=np.float32)
                count = min(clip.frames, restored.frames)
                adjusted[:count] = restored.samples[:count]
                restored = AudioClip(adjusted, clip.sample_rate)
            sources.append(restored)
        return SpeechSeparationResult(clip, (sources[0], sources[1]))
