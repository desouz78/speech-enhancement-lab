"""Optional YAMNet environmental sound classification with the official TF Hub model.

Classify mono audio using 521 fixed AudioSet labels. Per-class scores are
averaged across YAMNet analysis patches; they are NOT mutually exclusive and
do not sum to one. TensorFlow and TensorFlow Hub are optional dependencies,
imported only when the user explicitly loads the pretrained checkpoint.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from audio_lab.audio import AudioClip, resample_audio


_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
_MODEL_RATE = 16_000
_CLASS_COUNT = 521
_EMBEDDING_SIZE = 1024
_MEL_BINS = 64
_MAX_SECONDS = 30.0
_MAX_TOP_K = 20
_SILENCE_FLOOR = 1e-8


class YAMNetError(RuntimeError):
    """Raised when loading or inference violates the YAMNet model contract."""


@dataclass(frozen=True, slots=True)
class YAMNetLabelScore:
    """Mean patch-level score for a fixed AudioSet class (not a softmax)."""

    class_index: int
    label: str
    score: float

    def __post_init__(self) -> None:
        if type(self.class_index) is not int or not 0 <= self.class_index < _CLASS_COUNT:
            raise ValueError("class_index must be an integer within the YAMNet label vocabulary.")
        if not isinstance(self.label, str) or not self.label.strip() or self.label != self.label.strip():
            raise ValueError("label must be a nonempty, trimmed class name.")
        if type(self.score) is not float or not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be a finite float between zero and one.")


@dataclass(frozen=True, eq=False, slots=True)
class YAMNetClassificationResult:
    """Top fixed-vocabulary classes ordered by mean patch-level score."""

    source_audio: AudioClip
    predictions: tuple[YAMNetLabelScore, ...]
    patch_count: int
    model_name: str = "yamnet"

    def __post_init__(self) -> None:
        if not isinstance(self.source_audio, AudioClip):
            raise TypeError("source_audio must be an AudioClip.")
        if not isinstance(self.predictions, tuple) or not 1 <= len(self.predictions) <= _MAX_TOP_K:
            raise ValueError("predictions must be a tuple of 1-20 label scores.")
        if any(not isinstance(item, YAMNetLabelScore) for item in self.predictions):
            raise TypeError("predictions must contain YAMNetLabelScore instances.")
        if len({item.class_index for item in self.predictions}) != len(self.predictions):
            raise ValueError("Class indices must be unique.")
        if len({item.label for item in self.predictions}) != len(self.predictions):
            raise ValueError("Class labels must be unique.")
        if any(a.score < b.score for a, b in zip(self.predictions, self.predictions[1:])):
            raise ValueError("predictions must be sorted by descending score.")
        if type(self.patch_count) is not int or self.patch_count < 1:
            raise ValueError("patch_count must be a positive integer.")
        if self.model_name != "yamnet":
            raise ValueError("Unexpected YAMNet model name.")

    @property
    def top_prediction(self) -> YAMNetLabelScore:
        return self.predictions[0]


class YAMNetAdapter:
    """Classify up to 30 seconds of mono audio with the official 521-class model."""

    def __init__(self, model: Any, class_names: Sequence[str]) -> None:
        if not callable(model):
            raise TypeError("model must be callable.")
        if not isinstance(class_names, (tuple, list)) or len(class_names) != _CLASS_COUNT:
            raise ValueError("class_names must contain exactly 521 AudioSet labels.")
        if any(not isinstance(name, str) or not name.strip() or name != name.strip()
               or not name.isprintable() for name in class_names):
            raise ValueError("class_names must contain nonempty, trimmed, printable labels.")
        if len(set(class_names)) != _CLASS_COUNT:
            raise ValueError("class_names must not contain duplicates.")
        self._model = model
        self._class_names = tuple(class_names)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Read-only class labels ordered to match the checkpoint outputs."""
        return self._class_names

    @classmethod
    def load(cls) -> YAMNetAdapter:
        """Load the official public TF Hub checkpoint and its embedded class map."""
        try:
            import tensorflow_hub as hub
        except ImportError as exc:
            raise YAMNetError("Install compatible 'tensorflow' and 'tensorflow-hub' packages.") from exc
        try:
            model = hub.load(_MODEL_URL)
            path_value = model.class_map_path().numpy()
            if not isinstance(path_value, (str, bytes)):
                raise ValueError("YAMNet class-map path must be a string or bytes.")
            path = Path(path_value.decode("utf-8") if isinstance(path_value, bytes) else path_value)
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or "display_name" not in reader.fieldnames:
                    raise ValueError("YAMNet class map is missing the display_name column.")
                class_names = [row["display_name"] for row in reader]
            return cls(model, class_names)
        except (OSError, UnicodeError, csv.Error, ValueError, TypeError, AttributeError) as exc:
            raise YAMNetError("Unable to load YAMNet checkpoint or its 521-class label map.") from exc
        except Exception as exc:
            raise YAMNetError("Unable to load the YAMNet checkpoint from TensorFlow Hub.") from exc

    @staticmethod
    def _to_numpy(tensor: Any, *, name: str) -> np.ndarray:
        try:
            array = np.asarray(tensor.numpy())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise YAMNetError(f"YAMNet {name} must be a readable TensorFlow tensor.") from exc
        if array.dtype.kind != "f" or not np.isfinite(array).all():
            raise YAMNetError(f"YAMNet {name} must contain finite floating-point values.")
        return array

    def classify(self, clip: AudioClip, *, top_k: int = 5) -> YAMNetClassificationResult:
        """Rank the top AudioSet classes by their average frame-level scores.

        The returned scores are independent per-class confidences, not a
        probability distribution across the 521 event classes.
        """
        if not isinstance(clip, AudioClip):
            raise TypeError("clip must be an AudioClip.")
        if type(top_k) is not int or not 1 <= top_k <= _MAX_TOP_K:
            raise ValueError("top_k must be an integer between 1 and 20.")
        if clip.channels != 1:
            raise ValueError("YAMNet requires mono audio; downmix explicitly.")
        if clip.duration_seconds > _MAX_SECONDS:
            raise ValueError("YAMNet accepts at most 30 seconds of audio.")
        if float(np.max(np.abs(clip.samples))) > 1.0:
            raise ValueError("YAMNet input must lie within [-1, 1].")
        if float(np.max(np.abs(clip.samples))) < _SILENCE_FLOOR:
            raise ValueError("YAMNet cannot meaningfully classify silent audio.")
        prepared = resample_audio(clip, _MODEL_RATE)
        if float(np.max(np.abs(prepared.samples))) > 1.0:
            raise ValueError("Resampled YAMNet input exceeds [-1, 1]; reduce gain explicitly.")
        waveform = np.array(prepared.samples[:, 0], dtype=np.float32, copy=True, order="C")
        try:
            output = self._model(waveform)
        except Exception as exc:
            raise YAMNetError("YAMNet inference failed.") from exc
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise YAMNetError("YAMNet must return scores, embeddings and a spectrogram.")
        scores = self._to_numpy(output[0], name="scores")
        embeddings = self._to_numpy(output[1], name="embeddings")
        spectrogram = self._to_numpy(output[2], name="spectrogram")
        if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] != _CLASS_COUNT:
            raise YAMNetError("YAMNet scores must have shape (patches, 521).")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise YAMNetError("YAMNet class scores must lie within [0, 1].")
        if embeddings.shape != (scores.shape[0], _EMBEDDING_SIZE):
            raise YAMNetError("YAMNet embeddings must have shape (patches, 1024).")
        if spectrogram.ndim != 2 or spectrogram.shape[0] < 1 or spectrogram.shape[1] != _MEL_BINS:
            raise YAMNetError("YAMNet spectrogram must have shape (frames, 64).")
        means = scores.astype(np.float64).mean(axis=0)
        ranking = np.argsort(-means, kind="stable")[:top_k]
        predictions = tuple(YAMNetLabelScore(int(index), self._class_names[index], float(means[index])) for index in ranking)
        return YAMNetClassificationResult(clip, predictions, int(scores.shape[0]))
