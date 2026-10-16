"""Regression and contract tests for the AI model registry."""

from dataclasses import FrozenInstanceError

import pytest

from audio_lab.models import (
    AudioTask,
    BrowserStatus,
    EvidenceStatus,
    IntegrationStatus,
    ModelSpec,
    MODEL_CATALOG,
    get_model,
    models_for_task,
    _validate_catalog,
)


@pytest.mark.parametrize("task", list(AudioTask))
def test_every_declared_task_has_at_least_one_candidate(task):
    candidates = models_for_task(task)
    assert candidates
    assert all(task in model.tasks for model in candidates)


@pytest.mark.parametrize("task", list(AudioTask))
def test_string_and_enum_task_queries_are_identical(task):
    assert models_for_task(task.value) == models_for_task(task)


def test_audio_classification_returns_both_declared_candidates():
    assert {model.model_id for model in models_for_task(AudioTask.AUDIO_CLASSIFICATION)} == {
        "clap", "yamnet"
    }


def test_diarization_and_separation_are_distinct_tasks():
    diarization = {model.model_id for model in models_for_task(AudioTask.SPEAKER_DIARIZATION)}
    separation = {model.model_id for model in models_for_task(AudioTask.SPEAKER_SEPARATION)}
    assert diarization == {"pyannote_diarization"}
    assert separation == {"sepformer"}
    assert diarization.isdisjoint(separation)


def log_is_immutable_and_has_unique_identifiers():
    assert isinstance(MODEL_CATALOG, tuple)
    assert len(MODEL_CATALOG) == len({model.model_id for model in MODEL_CATALOG})
    assert all(isinstance(model.tasks, tuple) for model in MODEL_CATALOG)
    with pytest.raises(FrozenInstanceError):
        MODEL_CATALOG[0].display_name = "changed"


def test_catalog_progress_is_not_misrepresented():
    assert {model.model_id for model in MODEL_CATALOG if model.evidence_status == EvidenceStatus.PILOT_EVALUATED} == {"deepfilternet3", "htdemucs", "sepformer", "whisper", "silero_vad", "clap", "yamnet"}
    assert {model.model_id for model in MODEL_CATALOG if model.integration_status == IntegrationStatus.INTEGRATED} == {"deepfilternet3", "htdemucs", "sepformer", "whisper", "silero_vad", "clap", "yamnet"}

    assert all(model.browser_status == BrowserStatus.NOT_ASSESSED for model in MODEL_CATALOG)


@pytest.mark.parametrize("model", MODEL_CATALOG)
def test_lookup_round_trip(model):
    assert get_model(model.model_id) is model


@pytest.mark.parametrize("invalid", [None, True, False, 0, 5.0, [], {}, object()])
def test_invalid_task_types_raise_type_error(invalid):
    with pytest.raises(TypeError):
        models_for_task(invalid)


@pytest.mark.parametrize("invalid", ["", "INVALID", "speech enhancement", " speech_enhancement", "speech_enhancement "])
def test_unknown_task_names_raise_value_error(invalid):
    with pytest.raises(ValueError, match="Unknown audio task"):
        models_for_task(invalid)


@pytest.mark.parametrize("invalid", [None, True, 42, [], {}])
def test_invalid_model_lookup_types_raise_type_error(invalid):
    with pytest.raises(TypeError):
        get_model(invalid)


@pytest.mark.parametrize("invalid", ["", "unknown", "DeepFilterNet3", "htdemucs "])
def test_unknown_model_identifiers_raise_key_error(invalid):
    with pytest.raises(KeyError, match="Unknown model identifier"):
        get_model(invalid)


@pytest.mark.parametrize("invalid", ["", "Uppercase", "spaces not allowed", "_starts_underscore", 12, None])
def test_invalid_catalog_model_ids_are_rejected(invalid):
    with pytest.raises(ValueError, match="model_id"):
        ModelSpec(invalid, "Example", (AudioTask.SPEECH_ENHANCEMENT,))


@pytest.mark.parametrize("invalid", ["", "  ", None, 7])
def test_invalid_display_names_are_rejected(invalid):
    with pytest.raises(ValueError, match="display_name"):
        ModelSpec("valid_model", invalid, (AudioTask.SPEECH_ENHANCEMENT,))


@pytest.mark.parametrize("invalid", [(), [], None, "speech_enhancement"])
def test_invalid_task_collections_are_rejected(invalid):
    with pytest.raises(ValueError, match="tasks"):
        ModelSpec("valid_model", "Example", invalid)


def test_non_enum_capability_is_rejected():
    with pytest.raises(TypeError, match="Every task"):
        ModelSpec("valid_model", "Example", ("speech_enhancement",))


def test_duplicate_capabilities_are_rejected():
    with pytest.raises(ValueError, match="same task twice"):
        ModelSpec("valid_model", "Example", (AudioTask.SPEECH_ENHANCEMENT,) * 2)


@pytest.mark.parametrize("field", ["evidence_status", "integration_status", "browser_status"])
def test_non_enum_progress_metadata_is_rejected(field):
    with pytest.raises(TypeError, match=field):
        ModelSpec("valid_model", "Example", (AudioTask.SPEECH_ENHANCEMENT,), **{field: "not_integrated"})


@pytest.mark.parametrize("invalid", [" A", "A ", "A\x00B", "A\nB", "A\u202eB", "A" * 121])
def test_unsafe_or_unbounded_display_names_are_rejected(invalid):
    with pytest.raises(ValueError, match="display_name"):
        ModelSpec("valid_model", invalid, (AudioTask.SPEECH_ENHANCEMENT,))


@pytest.mark.parametrize("invalid", ["a" * 65, "a-b", "a\n", "1abc"])
def test_invalid_or_unbounded_model_identifiers_are_rejected(invalid):
    with pytest.raises(ValueError, match="model_id"):
        ModelSpec(invalid, "Example", (AudioTask.SPEECH_ENHANCEMENT,))


def test_model_identifier_boundary_is_accepted():
    model = ModelSpec("a" * 64, "Example", (AudioTask.SPEECH_ENHANCEMENT,))
    assert len(model.model_id) == 64


def test_display_name_boundary_is_accepted():
    model = ModelSpec("example", "A" * 120, (AudioTask.SPEECH_ENHANCEMENT,))
    assert len(model.display_name) == 120


def test_catalog_structural_validation_accepts_declared_catalog():
    _validate_catalog(MODEL_CATALOG)


@pytest.mark.parametrize("invalid", [None, [], (), "invalid"])
def test_catalog_rejects_empty_or_non_tuple_collections(invalid):
    with pytest.raises(ValueError, match="nonempty tuple"):
        _validate_catalog(invalid)


def test_catalog_rejects_non_model_entries():
    with pytest.raises(TypeError, match="ModelSpec"):
        _validate_catalog((object(),))


def test_catalog_rejects_duplicate_model_identifiers():
    with pytest.raises(ValueError, match="duplicate model identifiers"):
        _validate_catalog(MODEL_CATALOG + (MODEL_CATALOG[0],))


def test_catalog_rejects_duplicate_case_insensitive_display_names():
    duplicate = ModelSpec("unique_id", "deepfilternet3", (AudioTask.SPEECH_ENHANCEMENT,))
    with pytest.raises(ValueError, match="duplicate display names"):
        _validate_catalog(MODEL_CATALOG + (duplicate,))


def test_catalog_rejects_uncovered_tasks():
    with pytest.raises(ValueError, match="no candidates for tasks"):
        _validate_catalog((MODEL_CATALOG[0],))
