"""Dependency-free contract tests for the optional OpenAI Whisper adapter.

A controlled model double verifies interface behavior without downloading
checkpoints. These tests do not measure word error rate or model accuracy.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import ModuleType
import sys

import numpy as np
import pytest

from audio_lab.audio import AudioClip
from audio_lab.whisper import (
    TranscriptionResult,
    TranscriptionSegment,
    WhisperAdapter,
    WhisperError,
)


_DEFAULT_RESULT = object()


class FakeModel:
    def __init__(self, result=_DEFAULT_RESULT):
        self.result = {
            "text": " Hello, world! ",
            "language": "en",
            "segments": [{"start": 0.0, "end": 0.15, "text": " Hello, world! "}],
        }
        if result is not _DEFAULT_RESULT:
            self.result = result
        self.calls = []

    def transcribe(self, waveform, **kwargs):
        self.calls.append((waveform, kwargs))
        return self.result


def make_clip(rate=16000, duration=0.2, amplitude=0.3):
    time = np.arange(round(rate * duration), dtype=np.float64) / rate
    return AudioClip(amplitude * np.sin(2 * np.pi * 440 * time), rate)


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 96000, 192000])
def test_adapter_resamples_to_exactly_16khz(rate):
    model = FakeModel()
    clip = make_clip(rate=rate)
    result = WhisperAdapter(model).transcribe(clip)
    waveform, opts = model.calls[0]
    assert waveform.shape == (3200,)
    assert waveform.dtype == np.float32
    assert waveform.flags.writeable
    assert result.text == "Hello, world!"
    assert result.language == "en"
    assert result.model_name == "tiny"
    assert opts == {
        "language": None,
        "task": "transcribe",
        "fp16": False,
        "verbose": None,
        "temperature": 0.0,
        "condition_on_previous_text": False,
    }


def test_model_receives_private_copy_not_original_buffer():
    model = FakeModel()
    clip = make_clip()
    previous = clip.samples.copy()
    WhisperAdapter(model).transcribe(clip)
    waveform = model.calls[0][0]
    assert not np.shares_memory(waveform, clip.samples)
    waveform[:] = 0.0
    np.testing.assert_array_equal(clip.samples, previous)


def test_result_segment_is_immutable_and_sorted():
    model = FakeModel({
        "text": " hello there ", "language": "pt",
        "segments": [
            {"start": 0.01, "end": 0.04, "text": " hello "},
            {"start": 0.04, "end": 0.19, "text": " there "},
        ],
    })
    result = WhisperAdapter(model, "base").transcribe(make_clip())
    assert result == TranscriptionResult(
        "hello there", "pt",
        (TranscriptionSegment(0.01, 0.04, "hello"),
         TranscriptionSegment(0.04, 0.19, "there")), "base"
    )
    with pytest.raises(FrozenInstanceError):
        result.text = "replaced"
    with pytest.raises(FrozenInstanceError):
        result.segments[0].text = "replaced"
    assert isinstance(result.segments, tuple)


@pytest.mark.parametrize("language", [None, "en", "pt", "de", "yue"])
def test_valid_language_reaches_backend(language):
    model = FakeModel()
    WhisperAdapter(model).transcribe(make_clip(), language=language)
    assert model.calls[0][1]["language"] == language


@pytest.mark.parametrize("language", [1, True, False, [], {}, 3.0, object()])
def test_invalid_language_types_rejected(language):
    model = FakeModel()
    with pytest.raises(TypeError, match="language"):
        WhisperAdapter(model).transcribe(make_clip(), language=language)
    assert model.calls == []


@pytest.mark.parametrize("language", ["", "a", "engl", "EN", "en-US", " pt", "pt ", "1a", "中文", "a\nb"])
def test_invalid_language_strings_rejected(language):
    model = FakeModel()
    with pytest.raises(ValueError, match="language"):
        WhisperAdapter(model).transcribe(make_clip(), language=language)
    assert model.calls == []


@pytest.mark.parametrize("value", [None, True, 123, "audio", np.ones(5), object()])
def test_invalid_clip_types_rejected(value):
    model = FakeModel()
    with pytest.raises(TypeError, match="AudioClip"):
        WhisperAdapter(model).transcribe(value)
    assert model.calls == []


def test_stereo_rejected_without_downmix():
    model = FakeModel()
    clip = AudioClip(np.zeros((1600, 2), dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="mono"):
        WhisperAdapter(model).transcribe(clip)
    assert model.calls == []


def test_duration_limit_includes_exact_boundary():
    model = FakeModel()
    at_limit = AudioClip(np.zeros(30 * 8000, dtype=np.float32), 8000)
    over_limit = AudioClip(np.zeros(30 * 8000 + 1, dtype=np.float32), 8000)
    WhisperAdapter(model).transcribe(at_limit)
    assert len(model.calls) == 1
    with pytest.raises(ValueError, match="30 seconds"):
        WhisperAdapter(model).transcribe(over_limit)
    assert len(model.calls) == 1


@pytest.mark.parametrize("value", [-1.01, 1.01, 6.0])
def test_invalid_amplitude_rejected(value):
    model = FakeModel()
    with pytest.raises(ValueError, match="range"):
        WhisperAdapter(model).transcribe(AudioClip([value, 0.0], 16000))
    assert model.calls == []


def test_zero_signal_is_passed_without_claiming_a_transcription():
    model = FakeModel({"text": "", "language": "en", "segments": []})
    result = WhisperAdapter(model).transcribe(AudioClip(np.zeros(100), 16000))
    assert result.text == ""
    assert result.segments == ()
    assert len(model.calls) == 1


@pytest.mark.parametrize("model_name", ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v2", "large-v3", "turbo"])
def test_supported_checkpoint_names(model_name):
    assert WhisperAdapter(FakeModel(), model_name).transcribe(make_clip()).model_name == model_name


@pytest.mark.parametrize("model_name", ["", "Tiny", "../weights", "large-v4", "turbo "])
def test_unsupported_checkpoint_names(model_name):
    with pytest.raises(ValueError, match="Unsupported"):
        WhisperAdapter(FakeModel(), model_name)


@pytest.mark.parametrize("model_name", [None, True, 3, [], {}])
def test_bad_checkpoint_name_types(model_name):
    with pytest.raises(TypeError, match="model_name"):
        WhisperAdapter(FakeModel(), model_name)


@pytest.mark.parametrize("model", [None, 5, object(), "fake"])
def test_invalid_model_objects(model):
    with pytest.raises(TypeError, match="transcribe"):
        WhisperAdapter(model)


def test_model_loader_uses_explicit_device_and_checkpoint(monkeypatch):
    spy = []
    fake_whisper = ModuleType("whisper")
    fake_whisper.load_model = lambda name, *, device: (spy.append((name, device)) or FakeModel())
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    adapter = WhisperAdapter.load("base", device="cpu")
    assert spy == [("base", "cpu")]
    assert adapter.transcribe(make_clip()).model_name == "base"


def test_cuda_is_allowed_without_prechecking_device(monkeypatch):
    spy = []
    fake_whisper = ModuleType("whisper")
    fake_whisper.load_model = lambda name, *, device: (spy.append(device) or FakeModel())
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    WhisperAdapter.load(device="cuda")
    assert spy == ["cuda"]


@pytest.mark.parametrize("device", ["", "metal", "gpu", "cpu "])
def test_invalid_device_values(device):
    with pytest.raises(ValueError, match="device"):
        WhisperAdapter.load(device=device)


@pytest.mark.parametrize("device", [None, False, [], 3])
def test_invalid_device_types(device):
    with pytest.raises(TypeError, match="device"):
        WhisperAdapter.load(device=device)


def test_bad_loader_api(monkeypatch):
    fake_whisper = ModuleType("whisper")
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    with pytest.raises(WhisperError, match="load_model"):
        WhisperAdapter.load()


def test_loader_with_invalid_model_object(monkeypatch):
    fake_whisper = ModuleType("whisper")
    fake_whisper.load_model = lambda name, *, device: object()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    with pytest.raises(TypeError, match="transcribe"):
        WhisperAdapter.load()


def test_missing_dependency_reports_install_hint(monkeypatch):
    import builtins
    native_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "whisper":
            raise ImportError("simulated missing optional dependency")
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(WhisperError, match="openai-whisper"):
        WhisperAdapter.load()


@pytest.mark.parametrize("raw", [None, [], 123, "hello", True])
def test_non_mapping_result_rejected(raw):
    with pytest.raises(WhisperError, match="mapping"):
        WhisperAdapter(FakeModel(raw)).transcribe(make_clip())


@pytest.mark.parametrize("bad", [None, True, 55, ["hello"]])
def test_bad_transcription_text_rejected(bad):
    raw = {"text": bad, "language": "en", "segments": []}
    with pytest.raises(WhisperError, match="text"):
        WhisperAdapter(FakeModel(raw)).transcribe(make_clip())


@pytest.mark.parametrize("bad", [None, False, [], "", "x" * 41])
def test_invalid_detected_language_rejected(bad):
    raw = {"text": "example", "language": bad, "segments": []}
    with pytest.raises(WhisperError, match="language"):
        WhisperAdapter(FakeModel(raw)).transcribe(make_clip())


@pytest.mark.parametrize("bad", [None, False, {}, 7, "invalid"])
def test_invalid_segment_container_rejected(bad):
    raw = {"text": "", "language": "en", "segments": bad}
    with pytest.raises(WhisperError, match="segments"):
        WhisperAdapter(FakeModel(raw)).transcribe(make_clip())


@pytest.mark.parametrize("bad", [None, 5, [0], "bad"])
def test_invalid_segment_object_rejected(bad):
    raw = {"text": "", "language": "en", "segments": [bad]}
    with pytest.raises(WhisperError, match="segment"):
        WhisperAdapter(FakeModel(raw)).transcribe(make_clip())


@pytest.mark.parametrize("start,end", [
    (None, 1.0), (False, 1.0), ("0", 1.0), (0.0, True),
    (0.0, "0.1"), (float("nan"), 1.0), (0.0, float("nan")),
    (float("inf"), 1.0), (0.0, float("inf")), (-0.1, 0.1), (0.5, 0.4),
])
def test_invalid_segment_timestamps_rejected(start, end):
    raw = {"text": "", "language": "en", "segments": [{"start": start, "end": end, "text": "Hi"}]}
    with pytest.raises(WhisperError, match="timestamps"):
        WhisperAdapter(FakeModel(raw)).transcribe(make_clip())


def test_out_of_order_segment_starts_rejected():
    raw = {"text": "", "language": "en", "segments": [
        {"start": 0.3, "end": 0.4, "text": "One"},
        {"start": 0.2, "end": 0.5, "text": "Two"},
    ]}
    with pytest.raises(WhisperError, match="out of order"):
        WhisperAdapter(FakeModel(raw)).transcribe(make_clip())


@pytest.mark.parametrize("bad", [None, 1, False, ["hello"]])
def test_invalid_segment_text_rejected(bad):
    raw = {"text": "", "language": "en", "segments": [{"start": 0.0, "end": 0.1, "text": bad}]}
    with pytest.raises(WhisperError, match="segment text"):
        WhisperAdapter(FakeModel(raw)).transcribe(make_clip())


def test_large_segment_list_rejected():
    segment = {"start": 0.0, "end": 0.01, "text": "Hi"}
    raw = {"text": "", "language": "en", "segments": [segment] * 3001}
    with pytest.raises(WhisperError, match="too many"):
        WhisperAdapter(FakeModel(raw)).transcribe(make_clip())


def test_inference_exceptions_are_not_silently_swallowed():
    class FailedModel:
        def transcribe(self, waveform, **kwargs):
            raise RuntimeError("backend failed")

    with pytest.raises(RuntimeError, match="backend failed"):
        WhisperAdapter(FailedModel()).transcribe(make_clip())


def test_nonfinite_prepared_audio_rejected_before_backend(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "audio_lab.whisper.resample_audio",
        lambda clip, rate: SimpleNamespace(samples=np.array([[float("nan")]], dtype=np.float32)),
    )
    model = FakeModel()
    with pytest.raises(WhisperError, match="nonfinite"):
        WhisperAdapter(model).transcribe(make_clip())
    assert model.calls == []
