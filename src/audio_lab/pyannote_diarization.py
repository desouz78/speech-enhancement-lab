"""Optional speaker diarization using pyannote Community-1 (CC BY 4.0).

The model requires accepting its Hugging Face conditions and supplying a token.
No credentials or checkpoints are stored in the repository. Imports and
checkpoint downloads occur only during explicit model loading. Speaker labels
are recording-local hypotheses, not verified personal identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from audio_lab.audio import AudioClip, resample_audio


_MODEL_ID = "pyannote/speaker-diarization-community-1"
_MODEL_RATE = 16_000
_MAX_SECONDS = 30.0
_SILENCE_FLOOR = 1e-8
_MAX_SPEAKERS = 8


class PyannoteDiarizationError(RuntimeError):
    """Raised when pyannote loading or inference violates the adapter contract."""


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """Half-open speaker turn at the original audio sample rate."""

    start_sample: int
    end_sample: int
    sample_rate: int
    speaker_id: str

    def __post_init__(self) -> None:
        if type(self.start_sample) is not int or type(self.end_sample) is not int:
            raise TypeError("Turn boundaries must be integer samples.")
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("Turn sample rate must be a positive integer.")
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("Speaker turns must have positive length and nonnegative start.")
        if (not isinstance(self.speaker_id, str) or not self.speaker_id.strip()
                or self.speaker_id != self.speaker_id.strip() or len(self.speaker_id) > 80
                or not self.speaker_id.isprintable()):
            raise ValueError("speaker_id must be a short, nonempty printable label.")

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate


@dataclass(frozen=True, eq=False, slots=True)
class SpeakerDiarizationResult:
    """Immutable diarization, with an additional nonoverlapping transcription view."""

    source_audio: AudioClip
    turns: tuple[SpeakerTurn, ...]
    exclusive_turns: tuple[SpeakerTurn, ...]
    model_name: str = "speaker-diarization-community-1"

    def __post_init__(self) -> None:
        if not isinstance(self.source_audio, AudioClip):
            raise TypeError("source_audio must be an AudioClip.")
        if self.model_name != "speaker-diarization-community-1":
            raise ValueError("Unexpected diarization model name.")
        for name, collection in (("turns", self.turns), ("exclusive_turns", self.exclusive_turns)):
            if not isinstance(collection, tuple):
                raise TypeError(f"{name} must be a tuple.")
            previous_start = -1
            previous_end = 0
            for turn in collection:
                if not isinstance(turn, SpeakerTurn):
                    raise TypeError(f"{name} must contain SpeakerTurn objects.")
                if (turn.sample_rate != self.source_audio.sample_rate or turn.end_sample > self.source_audio.frames
                        or turn.start_sample < previous_start):
                    raise ValueError(f"{name} must be ordered and aligned to the source audio.")
                if name == "exclusive_turns" and turn.start_sample < previous_end:
                    raise ValueError("Exclusive diarization turns must not overlap.")
                previous_start, previous_end = turn.start_sample, turn.end_sample
        regular_ids = {turn.speaker_id for turn in self.turns}
        if any(turn.speaker_id not in regular_ids for turn in self.exclusive_turns):
            raise ValueError("Exclusive diarization cannot introduce new speaker identifiers.")

    @property
    def speakers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(turn.speaker_id for turn in self.turns))


class PyannoteDiarizationAdapter:
    """Diarize up to 30 seconds of mono audio with the gated Community-1 model."""

    def __init__(self, pipeline: Any) -> None:
        if not callable(pipeline):
            raise TypeError("pipeline must be callable.")
        self._pipeline = pipeline

    @classmethod
    def load(cls, *, token: str | None = None, device: str = "cpu") -> PyannoteDiarizationAdapter:
        """Load after Hugging Face terms are accepted; read HF_TOKEN if omitted."""
        if not isinstance(device, str):
            raise TypeError("device must be a string.")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'.")
        if token is None:
            token = os.environ.get("HF_TOKEN")
        if not isinstance(token, str) or not token.strip() or token != token.strip():
            raise ValueError("Accept the model conditions and provide an HF_TOKEN or token parameter.")
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise PyannoteDiarizationError("Install compatible 'pyannote.audio' and 'torch'.") from exc
        try:
            pipeline = Pipeline.from_pretrained(_MODEL_ID, token=token)
            if pipeline is None:
                raise PyannoteDiarizationError("Community-1 model could not be loaded; check access permissions.")
            pipeline.to(torch.device(device))
        except Exception as exc:
            raise PyannoteDiarizationError("Unable to load Community-1; check token, model access and dependencies.") from exc
        return cls(pipeline)

    @staticmethod
    def _validate_num_speakers(num_speakers: int | None) -> None:
        if num_speakers is None:
            return
        if type(num_speakers) is not int or not 1 <= num_speakers <= _MAX_SPEAKERS:
            raise ValueError(f"num_speakers must be an integer from 1 to {_MAX_SPEAKERS} or None.")

    @staticmethod
    def _extract_turns(annotation: Any, clip: AudioClip, *, exclusive: bool) -> tuple[SpeakerTurn, ...]:
        if not callable(getattr(annotation, "itertracks", None)):
            raise PyannoteDiarizationError("Community-1 must return a diarization annotation.")
        turns: list[SpeakerTurn] = []
        tolerance = max(1 / _MODEL_RATE, 1 / clip.sample_rate)
        try:
            for segment, _, speaker in annotation.itertracks(yield_label=True):
                start, end = segment.start, segment.end
                if (isinstance(start, (bool, np.bool_)) or isinstance(end, (bool, np.bool_))
                        or not isinstance(start, (int, float, np.integer, np.floating))
                        or not isinstance(end, (int, float, np.integer, np.floating))):
                    raise ValueError("Non-numeric diarization boundary.")
                start, end = float(start), float(end)
                if (not np.isfinite(start) or not np.isfinite(end) or start < -tolerance
                        or end > clip.duration_seconds + tolerance or end <= start):
                    raise ValueError("Out-of-range diarization boundary.")
                first = max(0, round(start * clip.sample_rate))
                last = min(clip.frames, round(end * clip.sample_rate))
                turns.append(SpeakerTurn(first, last, clip.sample_rate, speaker))
        except (TypeError, ValueError, AttributeError, OverflowError) as exc:
            raise PyannoteDiarizationError("Community-1 returned malformed speaker turns.") from exc
        turns.sort(key=lambda item: (item.start_sample, item.end_sample, item.speaker_id))
        if exclusive and any(left.end_sample > right.start_sample for left, right in zip(turns, turns[1:])):
            raise PyannoteDiarizationError("Exclusive diarization output contains overlapping turns.")
        return tuple(turns)

    def diarize(self, clip: AudioClip, *, num_speakers: int | None = None) -> SpeakerDiarizationResult:
        """Return overlapping and exclusive speaker turns at the original sample rate."""
        if not isinstance(clip, AudioClip):
            raise TypeError("clip must be an AudioClip.")
        self._validate_num_speakers(num_speakers)
        if clip.channels != 1:
            raise ValueError("Pyannote diarization requires mono input; downmix explicitly.")
        if clip.duration_seconds > _MAX_SECONDS:
            raise ValueError("Diarization is limited to 30 seconds per recording.")
        if float(np.max(np.abs(clip.samples))) > 1.0:
            raise ValueError("Input audio must be within [-1, 1].")
        if float(np.max(np.abs(clip.samples))) < _SILENCE_FLOOR:
            return SpeakerDiarizationResult(clip, (), ())

        prepared = resample_audio(clip, _MODEL_RATE)
        if float(np.max(np.abs(prepared.samples))) > 1.0:
            raise ValueError("Resampled audio exceeds [-1, 1]; reduce gain explicitly.")
        try:
            import torch
        except ImportError as exc:
            raise PyannoteDiarizationError("PyTorch is required for pyannote diarization.") from exc
        waveform = np.array(prepared.samples.T, dtype=np.float32, copy=True)
        arguments = {"num_speakers": num_speakers} if num_speakers is not None else {}
        try:
            with torch.inference_mode():
                output = self._pipeline({"waveform": torch.from_numpy(waveform), "sample_rate": _MODEL_RATE}, **arguments)
        except Exception as exc:
            raise PyannoteDiarizationError("Community-1 diarization inference failed.") from exc
        if not hasattr(output, "speaker_diarization") or not hasattr(output, "exclusive_speaker_diarization"):
            raise PyannoteDiarizationError("Community-1 returned an unsupported output format.")
        turns = self._extract_turns(output.speaker_diarization, clip, exclusive=False)
        exclusive = self._extract_turns(output.exclusive_speaker_diarization, clip, exclusive=True)
        try:
            return SpeakerDiarizationResult(clip, turns, exclusive)
        except (TypeError, ValueError) as exc:
            raise PyannoteDiarizationError("Community-1 returned inconsistent speaker identifiers.") from exc
