"""Declarative registry of candidate AI models for audio processing.

Capability metadata describes intended tasks, not measured performance.
Research evaluation, code integration, and browser feasibility are tracked
separately. No inference or automatic audio classification occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
import unicodedata


class AudioTask(StrEnum):
    SPEECH_ENHANCEMENT = "speech_enhancement"
    MUSIC_SOURCE_SEPARATION = "music_source_separation"
    SPEAKER_SEPARATION = "speaker_separation"
    SPEECH_TRANSCRIPTION = "speech_transcription"
    VOICE_ACTIVITY_DETECTION = "voice_activity_detection"
    SPEAKER_DIARIZATION = "speaker_diarization"
    AUDIO_CLASSIFICATION = "audio_classification"


class EvidenceStatus(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    PILOT_EVALUATED = "pilot_evaluated"


class IntegrationStatus(StrEnum):
    NOT_INTEGRATED = "not_integrated"
    INTEGRATED = "integrated"


class BrowserStatus(StrEnum):
    NOT_ASSESSED = "not_assessed"
    FEASIBLE = "feasible"
    NOT_FEASIBLE = "not_feasible"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable metadata describing one model or named inference pipeline.

    A declared capability is not evidence of quality for a given recording.
    Browser feasibility is unknown until independently assessed.
    """

    model_id: str
    display_name: str
    tasks: tuple[AudioTask, ...]
    evidence_status: EvidenceStatus = EvidenceStatus.NOT_EVALUATED
    integration_status: IntegrationStatus = IntegrationStatus.NOT_INTEGRATED
    browser_status: BrowserStatus = BrowserStatus.NOT_ASSESSED

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]{0,63}", self.model_id
        ):
            raise ValueError("model_id must be a lowercase identifier of at most 64 characters.")
        if (
            not isinstance(self.display_name, str)
            or not self.display_name.strip()
            or self.display_name != self.display_name.strip()
            or len(self.display_name) > 120
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
                   for char in self.display_name)
        ):
            raise ValueError("display_name must be a valid, trimmed, printable string (up to 120 characters).")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise ValueError("tasks must be a nonempty tuple of AudioTask values.")
        if any(not isinstance(task, AudioTask) for task in self.tasks):
            raise TypeError("Every task must be an AudioTask value.")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("A model cannot declare the same task twice.")
        for name, enum_type in (
            ("evidence_status", EvidenceStatus),
            ("integration_status", IntegrationStatus),
            ("browser_status", BrowserStatus),
        ):
            if not isinstance(getattr(self, name), enum_type):
                raise TypeError(f"{name} must be a {enum_type.__name__} value.")


MODEL_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("deepfilternet3", "DeepFilterNet3", (AudioTask.SPEECH_ENHANCEMENT,), EvidenceStatus.PILOT_EVALUATED, IntegrationStatus.INTEGRATED),
    ModelSpec("htdemucs", "HTDemucs", (AudioTask.MUSIC_SOURCE_SEPARATION,), EvidenceStatus.PILOT_EVALUATED, IntegrationStatus.INTEGRATED),
    ModelSpec("sepformer", "SepFormer", (AudioTask.SPEAKER_SEPARATION,)),
    ModelSpec("whisper", "Whisper", (AudioTask.SPEECH_TRANSCRIPTION,), EvidenceStatus.PILOT_EVALUATED, IntegrationStatus.INTEGRATED),
    ModelSpec("silero_vad", "Silero VAD", (AudioTask.VOICE_ACTIVITY_DETECTION,)),
    ModelSpec("pyannote_diarization", "pyannote.audio diarization pipeline", (AudioTask.SPEAKER_DIARIZATION,)),
    ModelSpec("clap", "CLAP", (AudioTask.AUDIO_CLASSIFICATION,)),
    ModelSpec("yamnet", "YAMNet", (AudioTask.AUDIO_CLASSIFICATION,)),
)


def _validate_catalog(catalog: tuple[ModelSpec, ...]) -> None:
    """Ensure the registry is structurally valid and covers every declared task."""
    if not isinstance(catalog, tuple) or not catalog:
        raise ValueError("Model catalog must be a nonempty tuple.")
    if any(not isinstance(model, ModelSpec) for model in catalog):
        raise TypeError("Every catalog entry must be a ModelSpec.")

    identifiers = [model.model_id for model in catalog]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Model catalog contains duplicate model identifiers.")

    names = [model.display_name.casefold() for model in catalog]
    if len(set(names)) != len(names):
        raise ValueError("Model catalog contains duplicate display names.")

    supported_tasks = {task for model in catalog for task in model.tasks}
    missing_tasks = set(AudioTask) - supported_tasks
    if missing_tasks:
        missing = ", ".join(sorted(task.value for task in missing_tasks))
        raise ValueError(f"Model catalog has no candidates for tasks: {missing}.")


_validate_catalog(MODEL_CATALOG)


def models_for_task(task: AudioTask | str) -> tuple[ModelSpec, ...]:
    """Return candidates for an explicitly specified task, without ranking.

    Input classification, performance prediction, and model execution are
    intentionally outside this registry's scope.
    """
    if isinstance(task, str):
        try:
            task = AudioTask(task)
        except ValueError as exc:
            raise ValueError(f"Unknown audio task: {task!r}.") from exc
    if not isinstance(task, AudioTask):
        raise TypeError("task must be an AudioTask or its exact string value.")
    return tuple(model for model in MODEL_CATALOG if task in model.tasks)


def get_model(model_id: str) -> ModelSpec:
    """Look up one model by its exact identifier."""
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string.")
    for model in MODEL_CATALOG:
        if model.model_id == model_id:
            return model
    raise KeyError(f"Unknown model identifier: {model_id!r}.")
