"""Regression tests for the optional CLAP zero-shot audio classifier.

Deterministic fake backends verify contracts, shapes and dependency isolation.
They do not establish accuracy on pretrained checkpoints or real recordings.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace
import builtins
import sys

import numpy as np
import pytest

from audio_lab.audio import AudioClip
from audio_lab.clap import AudioLabelScore, CLAPAdapter, CLAPClassificationResult, CLAPError


LABELS = ("dog barking", "human speech", "music playing")


class FakeBackend:
    def __init__(self, *, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def __call__(self, waveform, *, candidate_labels, hypothesis_template):
        self.calls.append((waveform, candidate_labels, hypothesis_template))
        if self.error is not None:
            raise self.error
        if self.output is not None:
            return self.output
        return [{"label": label, "score": score} for label, score in zip(candidate_labels, (0.7, 0.2, 0.1)[:len(candidate_labels)])]


def make_clip(rate=16_000, duration=0.1, amplitude=0.2, channels=1):
    time = np.arange(round(rate * duration), dtype=np.float64) / rate
    signal = amplitude * np.sin(2 * np.pi * 237 * time)
    return AudioClip(np.repeat(signal[:, None], channels, axis=1), rate)


def test_classification_returns_ranked_immutable_predictions():
    backend = FakeBackend()
    clip = make_clip()
    result = CLAPAdapter(backend).classify(clip, candidate_labels=LABELS)
    assert result.source_audio is clip
    assert result.candidate_labels == LABELS
    assert result.model_name == "clap-htsat-unfused"
    assert result.predictions == (AudioLabelScore("dog barking", 0.7), AudioLabelScore("human speech", 0.2), AudioLabelScore("music playing", 0.1))
    assert result.top_prediction is result.predictions[0]
    assert len(backend.calls) == 1
    with pytest.raises(FrozenInstanceError):
        result.model_name = "other"
    with pytest.raises(FrozenInstanceError):
        result.predictions[0].score = 0.5


@pytest.mark.parametrize("rate", [8_000, 16_000, 22_050, 44_100, 48_000, 96_000, 192_000])
def test_backend_always_receives_mono_48khz_audio(rate):
    clip = make_clip(rate=rate)
    backend = FakeBackend()
    CLAPAdapter(backend).classify(clip, candidate_labels=LABELS)
    waveform, labels, template = backend.calls[0]
    assert waveform.ndim == 1
    assert waveform.dtype == np.float32
    assert waveform.flags.c_contiguous
    assert waveform.flags.writeable
    assert abs(len(waveform) / 48_000 - clip.duration_seconds) < 1 / 48_000 + 1e-9
    assert labels == list(LABELS)
    assert template == "This is the sound of {}."
    assert np.isfinite(waveform).all()
    assert not np.shares_memory(waveform, clip.samples)


@pytest.mark.parametrize("duration", [0.02, 0.1, 0.5, 1.0, 10.0])
def test_supported_durations(duration):
    result = CLAPAdapter(FakeBackend()).classify(make_clip(duration=duration), candidate_labels=LABELS)
    assert len(result.predictions) == 3


def test_backend_input_mutation_cannot_change_original_audio():
    clip = make_clip(rate=48_000)
    original = clip.samples.copy()

    def mutate(waveform, *, candidate_labels, hypothesis_template):
        waveform[:] = 0
        return [{"label": label, "score": score} for label, score in zip(candidate_labels, (0.7, 0.2, 0.1))]

    CLAPAdapter(mutate).classify(clip, candidate_labels=LABELS)
    assert np.array_equal(clip.samples, original)


def test_backend_return_order_is_not_assumed():
    output = [
        {"label": "human speech", "score": 0.6},
        {"label": "music playing", "score": 0.3},
        {"label": "dog barking", "score": 0.1},
    ]
    result = CLAPAdapter(FakeBackend(output=output)).classify(make_clip(), candidate_labels=LABELS)
    assert tuple(item.label for item in result.predictions) == ("human speech", "music playing", "dog barking")


def test_equal_scores_preserve_candidate_order():
    output = [{"label": label, "score": 0.5} for label in ("second", "first")]
    result = CLAPAdapter(FakeBackend(output=output)).classify(make_clip(), candidate_labels=["first", "second"])
    assert tuple(item.label for item in result.predictions) == ("first", "second")


@pytest.mark.parametrize("invalid", [None, False, 1, "dog, speech", (), ("one",), ("a",) * 33, {"a": "x"}])
def test_bad_candidate_containers_rejected(invalid):
    backend = FakeBackend()
    with pytest.raises(ValueError, match="candidate_labels"):
        CLAPAdapter(backend).classify(make_clip(), candidate_labels=invalid)
    assert not backend.calls


@pytest.mark.parametrize("invalid", ["", " ", " dog", "dog ", "\nline", "long" * 31, "text\u200bzero", None, False, 10])
def test_bad_candidate_descriptions_rejected(invalid):
    with pytest.raises(ValueError, match="description"):
        CLAPAdapter(FakeBackend()).classify(make_clip(), candidate_labels=["speech", invalid])


def test_duplicate_candidate_descriptions_rejected():
    with pytest.raises(ValueError, match="unique"):
        CLAPAdapter(FakeBackend()).classify(make_clip(), candidate_labels=["speech", "speech"])


def test_maximum_32_labels_supported():
    labels = tuple(f"event {i}" for i in range(32))
    score = 1.0 / len(labels)
    output = [{"label": label, "score": score} for label in labels]
    result = CLAPAdapter(FakeBackend(output=output)).classify(make_clip(), candidate_labels=labels)
    assert len(result.predictions) == 32


@pytest.mark.parametrize("invalid", [None, "audio", 3.14, [], {}])
def test_bad_audio_rejected(invalid):
    with pytest.raises(TypeError, match="AudioClip"):
        CLAPAdapter(FakeBackend()).classify(invalid, candidate_labels=LABELS)


@pytest.mark.parametrize("channels", [2, 3, 8])
def test_multichannel_input_rejected(channels):
    with pytest.raises(ValueError, match="mono"):
        CLAPAdapter(FakeBackend()).classify(make_clip(channels=channels), candidate_labels=LABELS)


def test_audio_longer_than_model_context_rejected():
    clip = make_clip(rate=8_000, duration=10.1)
    with pytest.raises(ValueError, match="10 seconds"):
        CLAPAdapter(FakeBackend()).classify(clip, candidate_labels=LABELS)


@pytest.mark.parametrize("amplitude", [1.2, 3.0])
def test_out_of_range_input_rejected(amplitude):
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        CLAPAdapter(FakeBackend()).classify(make_clip(amplitude=amplitude), candidate_labels=LABELS)


@pytest.mark.parametrize("value", [0.0, 1e-10])
def test_silent_input_rejected(value):
    clip = AudioClip(np.full((1600, 1), value, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="silent"):
        CLAPAdapter(FakeBackend()).classify(clip, candidate_labels=LABELS)


def test_resampled_peak_out_of_range_rejected(monkeypatch):
    import audio_lab.clap as clap
    monkeypatch.setattr(clap, "resample_audio", lambda clip, rate: AudioClip(np.full((100, 1), 1.01), rate))
    with pytest.raises(ValueError, match="Resampled"):
        CLAPAdapter(FakeBackend()).classify(make_clip(), candidate_labels=LABELS)


def test_backend_failure_wrapped_without_leaking_internal_details():
    with pytest.raises(CLAPError, match="inference failed"):
        CLAPAdapter(FakeBackend(error=RuntimeError("internal details"))).classify(make_clip(), candidate_labels=LABELS)


@pytest.mark.parametrize("output", [None, "invalid", {}, (), [], [{"label": "dog barking", "score": 1}]])
def test_invalid_output_collection_rejected(output):
    with pytest.raises(CLAPError, match="one prediction"):
        CLAPAdapter(FakeBackend(output=output if output is not None else "bad")).classify(make_clip(), candidate_labels=LABELS)


@pytest.mark.parametrize("record", [None, [], ("a", 0.1), {"label": "dog barking"}, {"label": "dog barking", "score": 0.7, "extra": 1}])
def test_malformed_backend_records_rejected(record):
    output = [record, {"label": "human speech", "score": 0.2}, {"label": "music playing", "score": 0.1}]
    with pytest.raises(CLAPError, match="malformed"):
        CLAPAdapter(FakeBackend(output=output)).classify(make_clip(), candidate_labels=LABELS)


@pytest.mark.parametrize("label", ["unknown", "dog barking", 5, None])
def test_duplicate_unknown_and_bad_output_labels_rejected(label):
    output = [{"label": label, "score": 0.7}, {"label": "dog barking", "score": 0.2}, {"label": "music playing", "score": 0.1}]
    with pytest.raises(CLAPError, match="candidate label"):
        CLAPAdapter(FakeBackend(output=output)).classify(make_clip(), candidate_labels=LABELS)


@pytest.mark.parametrize("score", [None, "0.7", True, np.bool_(True), [], {}])
def test_invalid_output_score_type_rejected(score):
    output = [{"label": "dog barking", "score": score}, {"label": "human speech", "score": 0.2}, {"label": "music playing", "score": 0.1}]
    with pytest.raises(CLAPError, match="score type"):
        CLAPAdapter(FakeBackend(output=output)).classify(make_clip(), candidate_labels=LABELS)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1])
def test_nonfinite_out_of_range_scores_rejected(score):
    output = [{"label": "dog barking", "score": score}, {"label": "human speech", "score": 0.2}, {"label": "music playing", "score": 0.1}]
    with pytest.raises(CLAPError, match="nonfinite or out-of-range"):
        CLAPAdapter(FakeBackend(output=output)).classify(make_clip(), candidate_labels=LABELS)


@pytest.mark.parametrize("scores", [(0.2, 0.2, 0.2), (0.8, 0.6, 0.1)])
def test_invalid_relative_distribution_rejected(scores):
    output = [{"label": label, "score": score} for label, score in zip(LABELS, scores)]
    with pytest.raises(CLAPError, match="candidate-relative"):
        CLAPAdapter(FakeBackend(output=output)).classify(make_clip(), candidate_labels=LABELS)


@pytest.mark.parametrize("backend", [None, False, {}, [], "not callable"])
def test_constructor_rejects_noncallable_backend(backend):
    with pytest.raises(TypeError, match="callable"):
        CLAPAdapter(backend)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1.0, 1.1, 1, True, np.float64(0.5)])
def test_prediction_rejects_invalid_score(score):
    with pytest.raises(ValueError, match="finite float"):
        AudioLabelScore("label", score)


@pytest.mark.parametrize("label", [None, "", 5, False])
def test_prediction_rejects_invalid_label(label):
    with pytest.raises(TypeError, match="nonempty"):
        AudioLabelScore(label, 0.5)


@pytest.mark.parametrize("field,value,error", [
    ("source_audio", None, TypeError),
    ("candidate_labels", ("one",), ValueError),
    ("model_name", "other", ValueError),
    ("predictions", [AudioLabelScore("one", 0.5), AudioLabelScore("two", 0.5)], ValueError),
    ("predictions", (AudioLabelScore("one", 0.5),), ValueError),
    ("predictions", (None, AudioLabelScore("two", 0.5)), TypeError),
    ("predictions", (AudioLabelScore("one", 0.5), AudioLabelScore("extra", 0.5)), ValueError),
    ("predictions", (AudioLabelScore("one", 0.3), AudioLabelScore("two", 0.7)), ValueError),
    ("predictions", (AudioLabelScore("one", 0.7), AudioLabelScore("two", 0.2)), ValueError),
])
def test_result_rejects_inconsistent_fields(field, value, error):
    kwargs = dict(source_audio=make_clip(), candidate_labels=("one", "two"), predictions=(AudioLabelScore("one", 0.7), AudioLabelScore("two", 0.3)))
    kwargs[field] = value
    with pytest.raises(error):
        CLAPClassificationResult(**kwargs)


@pytest.mark.parametrize("device", [None, 1, False, [], {}, "auto", "mps", "CUDA"])
def test_load_rejects_invalid_device(device):
    with pytest.raises((TypeError, ValueError), match="device"):
        CLAPAdapter.load(device=device)


@pytest.mark.parametrize("device,expected", [("cpu", -1), ("cuda", 0)])
def test_load_uses_official_pipeline_and_sampling_rate(monkeypatch, device, expected):
    calls = []
    module = ModuleType("transformers")
    backend = FakeBackend()
    backend.feature_extractor = SimpleNamespace(sampling_rate=48_000)

    def factory(**kwargs):
        calls.append(kwargs)
        return backend

    module.pipeline = factory
    monkeypatch.setitem(sys.modules, "transformers", module)
    adapter = CLAPAdapter.load(device=device)
    assert adapter._backend is backend
    assert calls == [{"task": "zero-shot-audio-classification", "model": "laion/clap-htsat-unfused", "device": expected, "framework": "pt", "trust_remote_code": False}]


def test_load_rejects_incompatible_feature_extractor(monkeypatch):
    module = ModuleType("transformers")
    module.pipeline = lambda **kwargs: SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16_000))
    monkeypatch.setitem(sys.modules, "transformers", module)
    with pytest.raises(CLAPError, match="Unable to load"):
        CLAPAdapter.load()


def test_load_wraps_pipeline_error(monkeypatch):
    module = ModuleType("transformers")
    def factory(**kwargs):
        raise RuntimeError("internal token or path")
    module.pipeline = factory
    monkeypatch.setitem(sys.modules, "transformers", module)
    with pytest.raises(CLAPError, match="Unable to load"):
        CLAPAdapter.load()


def test_load_without_transformers_reports_optional_dependency(monkeypatch):
    original_import = builtins.__import__

    def block_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_import)
    with pytest.raises(CLAPError, match="Install compatible"):
        CLAPAdapter.load()


def test_module_import_does_not_download_or_import_transformers():
    import audio_lab.clap as clap
    assert "transformers" not in clap.__dict__
    assert "torch" not in clap.__dict__
