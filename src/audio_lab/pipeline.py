"""Declarative, explainable planning for multi-model audio processing.

Plans describe intended data flow and model capabilities. They do not execute
inference, certify model interfaces, or predict output quality. In particular,
CATALOG-level eligibility does not mean that a pipeline can run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from audio_lab.models import AudioTask
from audio_lab.selector import SelectionResult, SelectionTier, select_models


class MediaType(StrEnum):
    """Logical data passed between planned processing stages."""

    AUDIO = "audio"
    AUDIO_COLLECTION = "audio_collection"
    TEXT = "text"
    TEXT_COLLECTION = "text_collection"


class ExecutionMode(StrEnum):
    """Whether a stage operates on one signal or each separated source."""

    SINGLE = "single"
    EACH_SOURCE = "each_source"


class PipelinePreset(StrEnum):
    """Reviewed processing workflows; names are stable API identifiers."""

    ENHANCE_TRANSCRIBE = "enhance_transcribe"
    SEPARATE_TRANSCRIBE = "separate_transcribe"


# Only tasks with a reviewed conceptual audio-to-output contract belong here.
# Actual tensor formats, sample rates, and model adapters remain unverified.
_TASK_MEDIA = MappingProxyType({
    AudioTask.SPEECH_ENHANCEMENT: (MediaType.AUDIO, MediaType.AUDIO),
    AudioTask.SPEAKER_SEPARATION: (MediaType.AUDIO, MediaType.AUDIO_COLLECTION),
    AudioTask.SPEECH_TRANSCRIPTION: (MediaType.AUDIO, MediaType.TEXT),
})


@dataclass(frozen=True, slots=True)
class PipelineStage:
    """One logical task and its data-flow execution mode."""

    task: AudioTask
    mode: ExecutionMode = ExecutionMode.SINGLE

    def __post_init__(self) -> None:
        if not isinstance(self.task, AudioTask):
            raise TypeError("task must be an AudioTask value.")
        if not isinstance(self.mode, ExecutionMode):
            raise TypeError("mode must be an ExecutionMode value.")
        if self.task not in _TASK_MEDIA:
            raise ValueError(f"No reviewed media contract for task: {self.task.value}.")
        if self.mode is ExecutionMode.EACH_SOURCE and self.task is not AudioTask.SPEECH_TRANSCRIPTION:
            raise ValueError("Only transcription is supported in each-source mode.")

    @property
    def input_type(self) -> MediaType:
        """Required logical input; each-source mode consumes separated audio."""
        if self.mode is ExecutionMode.EACH_SOURCE:
            return MediaType.AUDIO_COLLECTION
        return _TASK_MEDIA[self.task][0]

    @property
    def output_type(self) -> MediaType:
        """Logical output, preserving fan-out in each-source mode."""
        if self.mode is ExecutionMode.EACH_SOURCE:
            return MediaType.TEXT_COLLECTION
        return _TASK_MEDIA[self.task][1]


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """Reviewed preset with compatible adjacent logical media."""

    pipeline_id: PipelinePreset
    stages: tuple[PipelineStage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_id, PipelinePreset):
            raise TypeError("pipeline_id must be a PipelinePreset value.")
        if not isinstance(self.stages, tuple) or not 1 <= len(self.stages) <= 8:
            raise ValueError("stages must be a tuple containing 1 to 8 stages.")
        if any(not isinstance(stage, PipelineStage) for stage in self.stages):
            raise TypeError("Every stage must be a PipelineStage.")
        if self.stages[0].input_type is not MediaType.AUDIO:
            raise ValueError("A pipeline must accept one input audio signal.")
        for previous, current in zip(self.stages, self.stages[1:]):
            if previous.output_type is not current.input_type:
                raise ValueError(
                    f"Incompatible stages: {previous.output_type.value} cannot feed "
                    f"{current.input_type.value}."
                )

        expected = {
            PipelinePreset.ENHANCE_TRANSCRIBE: (
                (AudioTask.SPEECH_ENHANCEMENT, ExecutionMode.SINGLE),
                (AudioTask.SPEECH_TRANSCRIPTION, ExecutionMode.SINGLE),
            ),
            PipelinePreset.SEPARATE_TRANSCRIBE: (
                (AudioTask.SPEAKER_SEPARATION, ExecutionMode.SINGLE),
                (AudioTask.SPEECH_TRANSCRIPTION, ExecutionMode.EACH_SOURCE),
            ),
        }
        actual = tuple((stage.task, stage.mode) for stage in self.stages)
        if actual != expected[self.pipeline_id]:
            raise ValueError("Stages do not match the declared pipeline preset.")


PIPELINE_CATALOG: tuple[PipelineSpec, ...] = (
    PipelineSpec(
        PipelinePreset.ENHANCE_TRANSCRIBE,
        (
            PipelineStage(AudioTask.SPEECH_ENHANCEMENT),
            PipelineStage(AudioTask.SPEECH_TRANSCRIPTION),
        ),
    ),
    PipelineSpec(
        PipelinePreset.SEPARATE_TRANSCRIBE,
        (
            PipelineStage(AudioTask.SPEAKER_SEPARATION),
            PipelineStage(AudioTask.SPEECH_TRANSCRIPTION, ExecutionMode.EACH_SOURCE),
        ),
    ),
)


def _validate_catalog() -> None:
    identifiers = [spec.pipeline_id for spec in PIPELINE_CATALOG]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(PipelinePreset):
        raise ValueError("Pipeline catalog must define each preset exactly once.")


_validate_catalog()


@dataclass(frozen=True, slots=True)
class PlannedStage:
    """One stage and the candidate assessments returned by the selector."""

    definition: PipelineStage
    selection: SelectionResult

    def __post_init__(self) -> None:
        if not isinstance(self.definition, PipelineStage):
            raise TypeError("definition must be a PipelineStage.")
        if not isinstance(self.selection, SelectionResult):
            raise TypeError("selection must be a SelectionResult.")
        if self.definition.task is not self.selection.task:
            raise ValueError("Stage task does not match its model selection.")

    @property
    def available(self) -> bool:
        """At least one model satisfies the requested selection tier."""
        return bool(self.selection.eligible_models)


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    """Planning result, not a runnable or scientifically evaluated pipeline."""

    definition: PipelineSpec
    tier: SelectionTier
    stages: tuple[PlannedStage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.definition, PipelineSpec):
            raise TypeError("definition must be a PipelineSpec.")
        if not isinstance(self.tier, SelectionTier):
            raise TypeError("tier must be a SelectionTier.")
        if not isinstance(self.stages, tuple) or len(self.stages) != len(self.definition.stages):
            raise ValueError("Planned stages must match the pipeline definition.")
        for planned, defined in zip(self.stages, self.definition.stages):
            if not isinstance(planned, PlannedStage):
                raise TypeError("Every planned stage must be a PlannedStage.")
            if planned.definition is not defined or planned.selection.tier is not self.tier:
                raise ValueError("Planned stage definition or selection tier does not match.")

    @property
    def available(self) -> bool:
        """All stages have at least one eligible candidate at this tier."""
        return all(stage.available for stage in self.stages)

    @property
    def unavailable_tasks(self) -> tuple[AudioTask, ...]:
        """Tasks without eligible candidates, in stage order."""
        return tuple(stage.definition.task for stage in self.stages if not stage.available)


def _parse_preset(preset: PipelinePreset | str) -> PipelinePreset:
    if isinstance(preset, PipelinePreset):
        return preset
    if not isinstance(preset, str):
        raise TypeError("preset must be a PipelinePreset or its exact string value.")
    try:
        return PipelinePreset(preset)
    except ValueError as exc:
        raise ValueError("Unknown pipeline preset.") from exc


def get_pipeline(preset: PipelinePreset | str) -> PipelineSpec:
    """Return a reviewed pipeline specification by exact identifier."""
    requested = _parse_preset(preset)
    for specification in PIPELINE_CATALOG:
        if specification.pipeline_id is requested:
            return specification
    raise RuntimeError("A declared pipeline preset is missing from the catalog.")


def plan_pipeline(
    preset: PipelinePreset | str,
    *,
    tier: SelectionTier | str = SelectionTier.CATALOG,
) -> PipelinePlan:
    """Assess model availability for each stage without executing any model.

    CATALOG assesses declared tasks only. INTEGRATED requires marked adapters.
    BROWSER additionally requires declared browser feasibility. When multiple
    candidates exist, no automatic ranking or choice is made.
    """
    specification = get_pipeline(preset)
    selections = tuple(
        PlannedStage(stage, select_models(stage.task, tier=tier))
        for stage in specification.stages
    )
    return PipelinePlan(specification, selections[0].selection.tier, selections)
