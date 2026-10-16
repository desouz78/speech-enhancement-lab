"""Optional CLAP zero-shot audio classification with a public Apache-2.0 checkpoint.

This adapter ranks caller-provided English audio descriptions using the
``laion/clap-htsat-unfused`` model. Scores are relative to the candidate set,
not calibrated event probabilities or evidence of any particular sound.
Imports and model downloads occur only during explicit model loading.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence
import unicodedata

import numpy as np

from audio_lab.audio import AudioClip, resample_audio


_MODEL_ID = "laion/clap-htsat-unfused"
_MODEL_RATE = 48_000
_MAX_SECONDS = 10.0
_MAX_LABELS = 32
_MIN_PEAK = 1e-8
_TEMPLATE = "This is the sound of {}."


class CLAPError(RuntimeError):
    """Raised when CLAP loading or inference violates the adapter contract."""


@dataclass(frozen=True, slots=True)
class AudioLabelScore:
    """Candidate-relative score, not an independently calibrated probability."""

    label: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise TypeError("label must be a nonempty string.")
        if type(self.score) is not float or not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("score must be a finite float between 0 and 1.")


@dataclass(frozen=True, eq=False, slots=True)
class CLAPClassificationResult:
    """Immutable ranked descriptions for a particular audio and candidate set."""

    source_audio: AudioClip
    candidate_labels: tuple[str, ...]
    predictions: tuple[AudioLabelScore, ...]
    model_name: str = "clap-htsat-unfused"

    def __post_init__(self) -> None:
        if not isinstance(self.source_audio, AudioClip):
            raise TypeError("source_audio must be an AudioClip.")
        CLAPAdapter._validate_labels(self.candidate_labels)
        if self.model_name != "clap-htsat-unfused":
            raise ValueError("Unexpected CLAP model name.")
        if not isinstance(self.predictions, tuple) or len(self.predictions) != len(self.candidate_labels):
            raise ValueError("predictions must contain one score per candidate label.")
        if any(not isinstance(prediction, AudioLabelScore) for prediction in self.predictions):
            raise TypeError("predictions must contain AudioLabelScore objects.")
        if {prediction.label for prediction in self.predictions} != set(self.candidate_labels):
            raise ValueError("predictions must match candidate labels exactly.")
        if any(left.score < right.score for left, right in zip(self.predictions, self.predictions[1:])):
            raise ValueError("predictions must be ranked by descending score.")
        if not math.isclose(sum(item.score for item in self.predictions), 1.0, abs_tol=0.002):
            raise ValueError("candidate-relative scores must sum to approximately one.")

    @property
    def top_prediction(self) -> AudioLabelScore:
        return self.predictions[0]


class CLAPAdapter:
    """Rank 2-32 user-defined audio descriptions from a mono clip up to 10 s."""

    def __init__(self, backend: Any) -> None:
        if not callable(backend):
            raise TypeError("backend must be callable.")
        self._backend = backend

    @classmethod
    def load(cls, *, device: str = "cpu") -> CLAPAdapter:
        """Load the Transformers zero-shot audio classification pipeline."""
        if not isinstance(device, str):
            raise TypeError("device must be a string.")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'.")
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise CLAPError("Install compatible 'transformers' and PyTorch packages.") from exc
        try:
            backend = pipeline(
                task="zero-shot-audio-classification", model=_MODEL_ID,
                device=0 if device == "cuda" else -1, framework="pt", trust_remote_code=False,
            )
            if getattr(getattr(backend, "feature_extractor", None), "sampling_rate", None) != _MODEL_RATE:
                raise CLAPError("CLAP backend must expect 48,000 Hz input.")
        except Exception as exc:
            raise CLAPError("Unable to load the CLAP checkpoint or pipeline.") from exc
        return cls(backend)

    @staticmethod
    def _validate_labels(labels: Sequence[str]) -> None:
        if not isinstance(labels, (tuple, list)) or not 2 <= len(labels) <= _MAX_LABELS:
            raise ValueError("candidate_labels must be a list or tuple of 2-32 descriptions.")
        for label in labels:
            if (not isinstance(label, str) or not label or label != label.strip()
                    or len(label) > 120 or not label.isprintable()
                    or any(unicodedata.category(ch) in {"Cc", "Cf", "Cs", "Zl", "Zp"} for ch in label)):
                raise ValueError("Each candidate must be a short, trimmed, printable description.")
        if len(set(labels)) != len(labels):
            raise ValueError("candidate_labels must be unique.")

    def classify(self, clip: AudioClip, *, candidate_labels: Sequence[str]) -> CLAPClassificationResult:
        """Classify mono audio without assuming a fixed vocabulary of sounds.

        The Transformers pipeline expects raw 48-kHz audio; it cannot infer
        the input sample rate from a NumPy waveform. Audio is resampled here.
        By limiting duration to 10 s, no long recording is silently truncated.
        """
        if not isinstance(clip, AudioClip):
            raise TypeError("clip must be an AudioClip.")
        self._validate_labels(candidate_labels)
        if clip.channels != 1:
            raise ValueError("CLAP requires mono audio; downmix explicitly.")
        if clip.duration_seconds > _MAX_SECONDS:
            raise ValueError("CLAP accepts at most 10 seconds of audio per request.")
        if float(np.max(np.abs(clip.samples))) > 1.0:
            raise ValueError("CLAP input must lie within [-1, 1].")
        if float(np.max(np.abs(clip.samples))) < _MIN_PEAK:
            raise ValueError("CLAP cannot meaningfully classify silent audio.")

        prepared = resample_audio(clip, _MODEL_RATE)
        if float(np.max(np.abs(prepared.samples))) > 1.0:
            raise ValueError("Resampled audio exceeds [-1, 1]; reduce gain explicitly.")
        waveform = np.array(prepared.samples[:, 0], dtype=np.float32, copy=True, order="C")
        labels = tuple(candidate_labels)
        try:
            output = self._backend(waveform, candidate_labels=list(labels), hypothesis_template=_TEMPLATE)
        except Exception as exc:
            raise CLAPError("CLAP zero-shot inference failed.") from exc
        if not isinstance(output, list) or len(output) != len(labels):
            raise CLAPError("CLAP must return one prediction per candidate label.")

        scores: dict[str, float] = {}
        for item in output:
            if not isinstance(item, dict) or set(item) != {"label", "score"}:
                raise CLAPError("CLAP returned a malformed label-score record.")
            name, raw_score = item["label"], item["score"]
            if (not isinstance(name, str) or name not in labels or name in scores
                    or isinstance(raw_score, (bool, np.bool_))
                    or not isinstance(raw_score, (int, float, np.integer, np.floating))):
                raise CLAPError("CLAP returned an invalid candidate label or score type.")
            score = float(raw_score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise CLAPError("CLAP returned a nonfinite or out-of-range score.")
            scores[name] = score
        if set(scores) != set(labels) or not math.isclose(sum(scores.values()), 1.0, abs_tol=0.002):
            raise CLAPError("CLAP scores do not form a candidate-relative distribution.")

        ranking = sorted(labels, key=lambda label: -scores[label])
        predictions = tuple(AudioLabelScore(label, scores[label]) for label in ranking)
        return CLAPClassificationResult(clip, labels, predictions)
