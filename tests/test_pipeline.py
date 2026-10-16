"""Contract and regression tests for declarative AI pipeline planning."""

from dataclasses import FrozenInstanceError

import pytest

import audio_lab.pipeline as pipeline
import audio_lab.selector as selector
from audio_lab.models import (
    AudioTask, BrowserStatus, IntegrationStatus, ModelSpec,
)
from audio_lab.pipeline import (
    ExecutionMode, MediaType, PipelinePlan, PipelinePreset, PipelineSpec,
    PipelineStage, PlannedStage, get_pipeline, plan_pipeline,
)
from audio_lab.selector import SelectionReason, SelectionTier, select_models


@pytest.mark.parametrize("preset", list(PipelinePreset))
def test_every_preset_has_a_valid_specification(preset):
    specification = get_pipeline(preset)
    assert specification.pipeline_id is preset
    assert get_pipeline(preset.value) is specification
    assert specification.stages[0].input_type is MediaType.AUDIO
    assert len(specification.stages) == 2
    assert all(isinstance(stage, PipelineStage) for stage in specification.stages)
    for left, right in zip(specification.stages, specification.stages[1:]):
        assert left.output_type is right.input_type


def test_each_declared_preset_appears_exactly_once():
    assert {p.pipeline_id for p in pipeline.PIPELINE_CATALOG} == set(PipelinePreset)
    assert len(pipeline.PIPELINE_CATALOG) == len(PipelinePreset)


def test_enhancement_transcription_dataflow():
    first, second = get_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE).stages
    assert (first.task, first.mode, first.input_type, first.output_type) == (
        AudioTask.SPEECH_ENHANCEMENT, ExecutionMode.SINGLE,
        MediaType.AUDIO, MediaType.AUDIO,
    )
    assert (second.task, second.mode, second.input_type, second.output_type) == (
        AudioTask.SPEECH_TRANSCRIPTION, ExecutionMode.SINGLE,
        MediaType.AUDIO, MediaType.TEXT,
    )


def test_separated_sources_are_transcribed_independently():
    first, second = get_pipeline(PipelinePreset.SEPARATE_TRANSCRIBE).stages
    assert (first.task, first.input_type, first.output_type) == (
        AudioTask.SPEAKER_SEPARATION, MediaType.AUDIO, MediaType.AUDIO_COLLECTION,
    )
    assert (second.task, second.mode, second.input_type, second.output_type) == (
        AudioTask.SPEECH_TRANSCRIPTION, ExecutionMode.EACH_SOURCE,
        MediaType.AUDIO_COLLECTION, MediaType.TEXT_COLLECTION,
    )


@pytest.mark.parametrize("task, expected", [
    (AudioTask.SPEECH_ENHANCEMENT, (MediaType.AUDIO, MediaType.AUDIO)),
    (AudioTask.SPEAKER_SEPARATION, (MediaType.AUDIO, MediaType.AUDIO_COLLECTION)),
    (AudioTask.SPEECH_TRANSCRIPTION, (MediaType.AUDIO, MediaType.TEXT)),
])
def test_reviewed_task_io_contracts(task, expected):
    stage = PipelineStage(task)
    assert (stage.input_type, stage.output_type) == expected


@pytest.mark.parametrize("bad", [None, 42, False, True, [], (), {}, 3.14, "speech_transcription"])
def test_stage_rejects_invalid_task_types(bad):
    with pytest.raises(TypeError, match="task must"):
        PipelineStage(bad)


@pytest.mark.parametrize("task", [
    AudioTask.MUSIC_SOURCE_SEPARATION,
    AudioTask.VOICE_ACTIVITY_DETECTION,
    AudioTask.SPEAKER_DIARIZATION,
    AudioTask.AUDIO_CLASSIFICATION,
])
def test_unreviewed_dataflow_tasks_are_rejected(task):
    with pytest.raises(ValueError, match="No reviewed media contract"):
        PipelineStage(task)


@pytest.mark.parametrize("bad", [None, 12, False, True, [], {}, "single", "each_source"])
def test_stage_rejects_invalid_mode_types(bad):
    with pytest.raises(TypeError, match="mode must"):
        PipelineStage(AudioTask.SPEECH_TRANSCRIPTION, bad)


@pytest.mark.parametrize("task", [AudioTask.SPEECH_ENHANCEMENT, AudioTask.SPEAKER_SEPARATION])
def test_each_source_mode_only_available_for_transcription(task):
    with pytest.raises(ValueError, match="Only transcription"):
        PipelineStage(task, ExecutionMode.EACH_SOURCE)


@pytest.mark.parametrize("bad", [None, True, False, 1, 0, [], {}, (), 2.2, "nope", " ENHANCE_TRANSCRIBE "])
def test_get_pipeline_rejects_invalid_preset(bad):
    with pytest.raises((TypeError, ValueError)):
        get_pipeline(bad)


def test_preset_strings_are_case_sensitive():
    with pytest.raises(ValueError, match="Unknown pipeline preset"):
        get_pipeline("ENHANCE_TRANSCRIBE")


@pytest.mark.parametrize("bad", [None, False, [], {}, "enhance_transcribe"])
def test_spec_requires_enum_identifier(bad):
    with pytest.raises(TypeError, match="pipeline_id must"):
        PipelineSpec(bad, (PipelineStage(AudioTask.SPEECH_TRANSCRIPTION),))


@pytest.mark.parametrize("bad", [
    None, [], {}, "wrong", (),
    tuple(PipelineStage(AudioTask.SPEECH_TRANSCRIPTION) for _ in range(9)),
])
def test_spec_rejects_bad_stage_containers(bad):
    with pytest.raises(ValueError, match="stages must"):
        PipelineSpec(PipelinePreset.ENHANCE_TRANSCRIBE, bad)


@pytest.mark.parametrize("bad", [None, 3, "speech_transcription", {}])
def test_spec_rejects_nonstage_entries(bad):
    with pytest.raises(TypeError, match="Every stage"):
        PipelineSpec(PipelinePreset.ENHANCE_TRANSCRIBE, (PipelineStage(AudioTask.SPEECH_ENHANCEMENT), bad))


def test_pipeline_rejects_fanout_as_first_stage():
    with pytest.raises(ValueError, match="must accept one input"):
        PipelineSpec(
            PipelinePreset.SEPARATE_TRANSCRIBE,
            (PipelineStage(AudioTask.SPEECH_TRANSCRIPTION, ExecutionMode.EACH_SOURCE),),
        )


@pytest.mark.parametrize("first, second", [
    (AudioTask.SPEECH_TRANSCRIPTION, AudioTask.SPEECH_ENHANCEMENT),
    (AudioTask.SPEAKER_SEPARATION, AudioTask.SPEECH_TRANSCRIPTION),
    (AudioTask.SPEAKER_SEPARATION, AudioTask.SPEECH_ENHANCEMENT),
])
def test_pipeline_rejects_incompatible_adjacent_types(first, second):
    with pytest.raises(ValueError, match="Incompatible stages"):
        PipelineSpec(
            PipelinePreset.ENHANCE_TRANSCRIBE,
            (PipelineStage(first), PipelineStage(second)),
        )


@pytest.mark.parametrize("preset", list(PipelinePreset))
@pytest.mark.parametrize("tier", list(SelectionTier))
def test_planning_preserves_task_order_and_selected_tier(preset, tier):
    result = plan_pipeline(preset, tier=tier)
    spec = get_pipeline(preset)
    assert result.definition is spec
    assert result.tier is tier
    assert len(result.stages) == len(spec.stages)
    assert isinstance(result.stages, tuple)
    for result_stage, spec_stage in zip(result.stages, spec.stages):
        assert result_stage.definition is spec_stage
        assert result_stage.selection.task is spec_stage.task
        assert result_stage.selection.tier is tier


@pytest.mark.parametrize("preset", list(PipelinePreset))
@pytest.mark.parametrize("tier", list(SelectionTier))
def test_planning_accepts_exact_string_identifiers(preset, tier):
    plan = plan_pipeline(preset.value, tier=tier.value)
    assert plan.definition.pipeline_id is preset
    assert plan.tier is tier


@pytest.mark.parametrize("preset", list(PipelinePreset))
def test_catalog_plans_only_claim_capability_availability(preset):
    plan = plan_pipeline(preset)
    assert plan.tier is SelectionTier.CATALOG
    assert plan.available
    assert plan.unavailable_tasks == ()
    for stage in plan.stages:
        assert stage.available
        assert stage.selection.eligible_models
        assert all(
            SelectionReason.TASK_SUPPORTED in assessment.reasons
            for assessment in stage.selection.assessments
        )


@pytest.mark.parametrize("preset", list(PipelinePreset))
@pytest.mark.parametrize("tier", [SelectionTier.INTEGRATED, SelectionTier.BROWSER])
def test_pipeline_planning_reflects_partial_python_integration(preset, tier):
    plan = plan_pipeline(preset, tier=tier)

    supported_tasks = {AudioTask.SPEECH_ENHANCEMENT, AudioTask.SPEAKER_SEPARATION, AudioTask.SPEECH_TRANSCRIPTION}
    expected = tuple(tier is SelectionTier.INTEGRATED and stage.definition.task in supported_tasks for stage in plan.stages)

    assert tuple(stage.available for stage in plan.stages) == expected
    assert plan.unavailable_tasks == tuple(stage.definition.task for stage, available in zip(plan.stages, expected) if not available)
    assert plan.available is all(expected)


def test_planning_retains_model_rejection_reasons():
    plan = plan_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE, tier=SelectionTier.BROWSER)
    for stage in plan.stages:
        assert stage.selection.assessments
        for item in stage.selection.assessments:
            assert (SelectionReason.ADAPTER_NOT_INTEGRATED in item.reasons) == (item.model.integration_status is IntegrationStatus.NOT_INTEGRATED)
            assert SelectionReason.BROWSER_NOT_ASSESSED in item.reasons


@pytest.mark.parametrize("bad", [None, False, True, 1, [], {}, "wrong", "browser "])
def test_planning_rejects_invalid_tier(bad):
    with pytest.raises((TypeError, ValueError)):
        plan_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE, tier=bad)


def test_planning_does_not_mutate_catalog():
    original_specs = pipeline.PIPELINE_CATALOG
    for preset in PipelinePreset:
        for tier in SelectionTier:
            plan_pipeline(preset, tier=tier)
    assert pipeline.PIPELINE_CATALOG == original_specs


def test_output_results_are_frozen():
    specification = get_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE)
    plan = plan_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE)
    with pytest.raises(FrozenInstanceError):
        specification.pipeline_id = PipelinePreset.SEPARATE_TRANSCRIBE
    with pytest.raises(FrozenInstanceError):
        specification.stages[0].task = AudioTask.SPEECH_TRANSCRIPTION
    with pytest.raises(FrozenInstanceError):
        plan.tier = SelectionTier.BROWSER
    with pytest.raises(FrozenInstanceError):
        plan.stages[0].definition = specification.stages[1]


def test_missing_candidates_are_reported_and_not_selected(monkeypatch):
    monkeypatch.setattr(selector, "MODEL_CATALOG", ())
    for tier in SelectionTier:
        plan = plan_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE, tier=tier)
        assert not plan.available
        assert plan.unavailable_tasks == (
            AudioTask.SPEECH_ENHANCEMENT, AudioTask.SPEECH_TRANSCRIPTION,
        )
        assert all(not step.selection.assessments for step in plan.stages)


def test_partial_availability_preserves_stage_information(monkeypatch):
    only_enhancement = ModelSpec(
        "unit_enhancer", "Unit enhancer", (AudioTask.SPEECH_ENHANCEMENT,),
        integration_status=IntegrationStatus.INTEGRATED,
        browser_status=BrowserStatus.FEASIBLE,
    )
    monkeypatch.setattr(selector, "MODEL_CATALOG", (only_enhancement,))
    plan = plan_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE, tier=SelectionTier.BROWSER)
    assert not plan.available
    assert plan.stages[0].available
    assert not plan.stages[1].available
    assert plan.unavailable_tasks == (AudioTask.SPEECH_TRANSCRIPTION,)


def test_all_integrated_models_produce_integrated_plan(monkeypatch):
    candidates = (
        ModelSpec(
            "unit_enhancer", "Unit enhancer", (AudioTask.SPEECH_ENHANCEMENT,),
            integration_status=IntegrationStatus.INTEGRATED,
        ),
        ModelSpec(
            "unit_transcriber", "Unit transcriber", (AudioTask.SPEECH_TRANSCRIPTION,),
            integration_status=IntegrationStatus.INTEGRATED,
        ),
    )
    monkeypatch.setattr(selector, "MODEL_CATALOG", candidates)
    assert plan_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE, tier=SelectionTier.INTEGRATED).available
    assert not plan_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE, tier=SelectionTier.BROWSER).available


def test_multiple_candidate_models_are_not_automatically_ranked(monkeypatch):
    candidates = (
        ModelSpec("zeta", "Zeta", (AudioTask.SPEECH_TRANSCRIPTION,)),
        ModelSpec("alpha", "Alpha", (AudioTask.SPEECH_TRANSCRIPTION,)),
        ModelSpec("enhancer", "Enhancer", (AudioTask.SPEECH_ENHANCEMENT,)),
    )
    monkeypatch.setattr(selector, "MODEL_CATALOG", candidates)
    plan = plan_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE)
    assert [m.model_id for m in plan.stages[1].selection.eligible_models] == ["zeta", "alpha"]
    assert plan.available


def test_catalog_validator_detects_missing_preset(monkeypatch):
    monkeypatch.setattr(pipeline, "PIPELINE_CATALOG", pipeline.PIPELINE_CATALOG[:1])
    with pytest.raises(ValueError, match="each preset exactly once"):
        pipeline._validate_catalog()


def test_catalog_validator_detects_duplicate_preset(monkeypatch):
    monkeypatch.setattr(pipeline, "PIPELINE_CATALOG", pipeline.PIPELINE_CATALOG * 2)
    with pytest.raises(ValueError, match="each preset exactly once"):
        pipeline._validate_catalog()


def test_missing_preset_lookup_fails_explicitly(monkeypatch):
    monkeypatch.setattr(pipeline, "PIPELINE_CATALOG", ())
    with pytest.raises(RuntimeError, match="missing from the catalog"):
        get_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE)


def test_media_contract_mapping_is_immutable():
    with pytest.raises(TypeError):
        pipeline._TASK_MEDIA[AudioTask.SPEECH_ENHANCEMENT] = (MediaType.TEXT, MediaType.TEXT)


def test_preset_cannot_be_mislabeled_with_unrelated_stages():
    with pytest.raises(ValueError, match="do not match the declared pipeline preset"):
        PipelineSpec(
            PipelinePreset.ENHANCE_TRANSCRIBE,
            (PipelineStage(AudioTask.SPEECH_TRANSCRIPTION),),
        )


def test_planned_stage_validates_model_selection_correspondence():
    stage = PipelineStage(AudioTask.SPEECH_ENHANCEMENT)
    correct = select_models(AudioTask.SPEECH_ENHANCEMENT)
    wrong = select_models(AudioTask.SPEECH_TRANSCRIPTION)
    assert PlannedStage(stage, correct).available
    with pytest.raises(ValueError, match="task does not match"):
        PlannedStage(stage, wrong)
    with pytest.raises(TypeError, match="definition must"):
        PlannedStage(None, correct)
    with pytest.raises(TypeError, match="selection must"):
        PlannedStage(stage, None)


def test_plan_constructor_rejects_mismatched_stages_and_tiers():
    result = plan_pipeline(PipelinePreset.ENHANCE_TRANSCRIBE)
    with pytest.raises(TypeError, match="definition must"):
        PipelinePlan(None, result.tier, result.stages)
    with pytest.raises(TypeError, match="tier must"):
        PipelinePlan(result.definition, "catalog", result.stages)
    with pytest.raises(ValueError, match="must match"):
        PipelinePlan(result.definition, result.tier, ())
    with pytest.raises(TypeError, match="Every planned stage"):
        PipelinePlan(result.definition, result.tier, (None, result.stages[1]))
    with pytest.raises(ValueError, match="definition or selection tier"):
        PipelinePlan(result.definition, result.tier, result.stages[::-1])
    other_tier_stage = PlannedStage(
        result.stages[0].definition,
        select_models(AudioTask.SPEECH_ENHANCEMENT, tier=SelectionTier.BROWSER),
    )
    with pytest.raises(ValueError, match="definition or selection tier"):
        PipelinePlan(result.definition, result.tier, (other_tier_stage, result.stages[1]))
