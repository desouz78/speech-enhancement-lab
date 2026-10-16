"""Contract and regression tests for explainable model selection.

The tests distinguish declared capabilities from adapter integration and
browser feasibility. They do not validate neural-model accuracy or inference.
"""

from dataclasses import FrozenInstanceError

import pytest

import audio_lab.selector as selector
from audio_lab.models import (
    AudioTask,
    BrowserStatus,
    EvidenceStatus,
    IntegrationStatus,
    MODEL_CATALOG,
    ModelSpec,
)
from audio_lab.selector import (
    ModelAssessment,
    SelectionReason,
    SelectionResult,
    SelectionTier,
    select_models,
)


def make_model(model_id="example", *, tasks=(AudioTask.SPEECH_ENHANCEMENT,), integrated=False, browser=BrowserStatus.NOT_ASSESSED):
    return ModelSpec(
        model_id=model_id,
        display_name=model_id.capitalize(),
        tasks=tasks,
        integration_status=IntegrationStatus.INTEGRATED if integrated else IntegrationStatus.NOT_INTEGRATED,
        browser_status=browser,
    )


@pytest.mark.parametrize("task", list(AudioTask))
def test_catalog_discovers_every_supported_task(task):
    result = select_models(task)
    expected = tuple(model for model in MODEL_CATALOG if task in model.tasks)

    assert isinstance(result, SelectionResult)
    assert result.task is task
    assert result.tier is SelectionTier.CATALOG
    assert result.eligible_models == expected
    assert result.excluded_models == ()
    assert len(result.assessments) == len(expected)


@pytest.mark.parametrize("task", list(AudioTask))
def test_catalog_accepts_exact_task_strings(task):
    assert select_models(task.value) == select_models(task)


@pytest.mark.parametrize("tier", list(SelectionTier))
def test_tier_accepts_exact_string_values(tier):
    assert select_models(AudioTask.SPEECH_ENHANCEMENT, tier=tier.value) == select_models(AudioTask.SPEECH_ENHANCEMENT, tier=tier)


@pytest.mark.parametrize("task, expected_ids", [
    (AudioTask.SPEECH_ENHANCEMENT, ("deepfilternet3",)),
    (AudioTask.SPEECH_TRANSCRIPTION, ("whisper",)),
    (AudioTask.MUSIC_SOURCE_SEPARATION, ("htdemucs",)),
    (AudioTask.SPEAKER_SEPARATION, ("sepformer",)),
    (AudioTask.VOICE_ACTIVITY_DETECTION, ("silero_vad",)),
    (AudioTask.AUDIO_CLASSIFICATION, ("clap", "yamnet")),
])
def test_catalog_reports_python_integration_without_browser_readiness(task, expected_ids):
    result = select_models(task, tier=SelectionTier.INTEGRATED)

    assert tuple(model.model_id for model in result.eligible_models) == expected_ids
    assert all(assessment.reasons == (SelectionReason.TASK_SUPPORTED, SelectionReason.BROWSER_NOT_ASSESSED) for assessment in result.assessments)
    assert all(assessment.model.evidence_status is EvidenceStatus.PILOT_EVALUATED for assessment in result.assessments)


@pytest.mark.parametrize("task", list(AudioTask))
@pytest.mark.parametrize("tier", [SelectionTier.INTEGRATED, SelectionTier.BROWSER])
def test_catalog_availability_matches_adapter_status(task, tier):
    result = select_models(task, tier=tier)

    integrated = {
        AudioTask.SPEECH_ENHANCEMENT: ("deepfilternet3",),
        AudioTask.SPEECH_TRANSCRIPTION: ("whisper",),
        AudioTask.MUSIC_SOURCE_SEPARATION: ("htdemucs",),
        AudioTask.SPEAKER_SEPARATION: ("sepformer",),
        AudioTask.VOICE_ACTIVITY_DETECTION: ("silero_vad",),
        AudioTask.AUDIO_CLASSIFICATION: ("clap",),
        AudioTask.AUDIO_CLASSIFICATION: ("clap", "yamnet"),
    }
    expected = integrated.get(task, ()) if tier is SelectionTier.INTEGRATED else ()

    assert tuple(model.model_id for model in result.eligible_models) == expected
    assert len(result.assessments) == len(result.eligible_models) + len(result.excluded_models)
    assert result.assessments


def test_audio_classification_is_unranked_and_stably_ordered():
    result = select_models(AudioTask.AUDIO_CLASSIFICATION)

    assert tuple(model.model_id for model in result.eligible_models) == ("clap", "yamnet")
    assert tuple(assessment.model for assessment in result.assessments) == result.eligible_models
    assert all(assessment.eligible for assessment in result.assessments)


@pytest.mark.parametrize("value", [None, False, True, 0, 1, 0.0, 1.2, [], {}, (), object()])
def test_invalid_task_types_rejected(value):
    with pytest.raises(TypeError, match="task must"):
        select_models(value)


@pytest.mark.parametrize("value", ["", "Speech Enhancement", "SPEECH_ENHANCEMENT", " speech_enhancement", "speech_enhancement ", "unknown", "whisper", "x" * 100_000])
def test_unknown_tasks_rejected_without_echoing_inputs(value):
    with pytest.raises(ValueError, match="^Unknown audio task\\.$"):
        select_models(value)


@pytest.mark.parametrize("value", [None, False, True, 0, 1, 0.0, 1.5, [], {}, (), object()])
def test_invalid_tier_types_rejected(value):
    with pytest.raises(TypeError, match="tier must"):
        select_models(AudioTask.SPEECH_ENHANCEMENT, tier=value)


@pytest.mark.parametrize("value", ["", "CATALOG", "Browser", " browser", "browser ", "x" * 100_000])
def test_unknown_tiers_rejected_without_echoing_inputs(value):
    with pytest.raises(ValueError, match="^Unknown selection tier\\.$"):
        select_models(AudioTask.SPEECH_ENHANCEMENT, tier=value)


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("browser", list(BrowserStatus))
@pytest.mark.parametrize("tier", list(SelectionTier))
def test_all_availability_combinations(monkeypatch, integrated, browser, tier):
    model = make_model(integrated=integrated, browser=browser)
    monkeypatch.setattr(selector, "MODEL_CATALOG", (model,))
    result = select_models(AudioTask.SPEECH_ENHANCEMENT, tier=tier)
    assessment, = result.assessments

    expected = (
        tier is SelectionTier.CATALOG
        or (tier is SelectionTier.INTEGRATED and integrated)
        or (tier is SelectionTier.BROWSER and integrated and browser is BrowserStatus.FEASIBLE)
    )

    assert assessment.eligible is expected
    assert (assessment.model,) == (result.eligible_models if expected else result.excluded_models)
    assert assessment.reasons[0] is SelectionReason.TASK_SUPPORTED
    assert (SelectionReason.ADAPTER_NOT_INTEGRATED in assessment.reasons) is (not integrated)
    assert (SelectionReason.BROWSER_NOT_ASSESSED in assessment.reasons) is (browser is BrowserStatus.NOT_ASSESSED)
    assert (SelectionReason.BROWSER_NOT_FEASIBLE in assessment.reasons) is (browser is BrowserStatus.NOT_FEASIBLE)


def test_mixed_eligibility_retains_registry_order(monkeypatch):
    models = (
        make_model("first", integrated=False),
        make_model("second", integrated=True, browser=BrowserStatus.FEASIBLE),
        make_model("third", integrated=True, browser=BrowserStatus.NOT_ASSESSED),
        make_model("fourth", integrated=True, browser=BrowserStatus.NOT_FEASIBLE),
    )
    monkeypatch.setattr(selector, "MODEL_CATALOG", models)
    result = select_models(AudioTask.SPEECH_ENHANCEMENT, tier="browser")

    assert tuple(item.model.model_id for item in result.assessments) == ("first", "second", "third", "fourth")
    assert tuple(model.model_id for model in result.eligible_models) == ("second",)
    assert tuple(model.model_id for model in result.excluded_models) == ("first", "third", "fourth")


def test_multitask_model_matches_each_explicit_task(monkeypatch):
    model = make_model(tasks=(AudioTask.SPEECH_ENHANCEMENT, AudioTask.SPEECH_TRANSCRIPTION))
    monkeypatch.setattr(selector, "MODEL_CATALOG", (model,))

    assert select_models(AudioTask.SPEECH_ENHANCEMENT).eligible_models == (model,)
    assert select_models(AudioTask.SPEECH_TRANSCRIPTION).eligible_models == (model,)
    assert select_models(AudioTask.SPEAKER_DIARIZATION).assessments == ()


def test_unrelated_models_never_appear(monkeypatch):
    speech = make_model("speech")
    music = make_model("music", tasks=(AudioTask.MUSIC_SOURCE_SEPARATION,))
    monkeypatch.setattr(selector, "MODEL_CATALOG", (speech, music))
    result = select_models(AudioTask.SPEECH_ENHANCEMENT)

    assert result.eligible_models == (speech,)
    assert result.excluded_models == ()


def test_empty_registry_returns_empty_outcome(monkeypatch):
    monkeypatch.setattr(selector, "MODEL_CATALOG", ())

    for tier in SelectionTier:
        result = select_models(AudioTask.SPEECH_TRANSCRIPTION, tier=tier)
        assert result.assessments == ()
        assert result.eligible_models == ()
        assert result.excluded_models == ()


def test_results_are_frozen_and_use_immutable_collections():
    result = select_models(AudioTask.AUDIO_CLASSIFICATION)

    assert isinstance(result.assessments, tuple)
    assert isinstance(result.eligible_models, tuple)
    assert isinstance(result.assessments[0].reasons, tuple)

    with pytest.raises(FrozenInstanceError):
        result.task = AudioTask.SPEECH_ENHANCEMENT

    with pytest.raises(FrozenInstanceError):
        result.assessments[0].eligible = False

    with pytest.raises(AttributeError):
        result.assessments.append(None)


def test_two_identical_calls_are_deterministic():
    left = select_models("audio_classification", tier="catalog")
    right = select_models("audio_classification", tier="catalog")

    assert left == right
    assert left is not right


def test_assessment_is_associated_with_original_registry_model():
    result = select_models(AudioTask.SPEECH_TRANSCRIPTION)

    assert len(result.assessments) == 1
    assert result.assessments[0].model is next(model for model in MODEL_CATALOG if model.model_id == "whisper")


def test_browser_availability_requires_both_declarations(monkeypatch):
    model = make_model(browser=BrowserStatus.FEASIBLE, integrated=False)
    monkeypatch.setattr(selector, "MODEL_CATALOG", (model,))
    result = select_models(AudioTask.SPEECH_ENHANCEMENT, tier="browser")

    assert result.eligible_models == ()
    assert result.excluded_models == (model,)


def test_reason_values_are_stable_machine_readable_strings():
    for reason in SelectionReason:
        assert reason.value == str(reason)
        assert reason.value.isascii()
        assert all(char.islower() or char == "_" for char in reason.value)


def test_selection_tiers_are_distinct_and_stable():
    assert tuple(tier.value for tier in SelectionTier) == ("catalog", "integrated", "browser")


def test_result_with_zero_eligible_models_still_reports_exclusions():
    result = select_models("speech_transcription", tier="browser")

    assert not result.eligible_models
    assert len(result.excluded_models) == 1
    assert result.excluded_models[0].model_id == "whisper"
