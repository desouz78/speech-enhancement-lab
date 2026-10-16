"""Contract and regression tests for optional Silero voice activity detection.

Controlled backends test shape, time alignment, validation and dependency
isolation; these tests do not claim pretrained accuracy or voice recognition.
"""

from __future__ import annotations

import builtins
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import ModuleType
import sys

import numpy as np
import pytest

import audio_lab.silero_vad as vad
from audio_lab.audio import AudioClip
from audio_lab.silero_vad import SileroVADAdapter, SileroVADError, SpeechInterval, VoiceActivityResult


class FakeTensor:
    def __init__(self, data):
        self.data = data


class FakeExtractor:
    def __init__(self, output=(), error=None):
        self.output = output
        self.error = error
        self.calls = []

    def __call__(self, tensor, model, **kwargs):
        self.calls.append((tensor, model, kwargs))
        if self.error is not None:
            raise self.error
        return self.output


@pytest.fixture
def fake_torch(monkeypatch):
    torch = ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.from_numpy = FakeTensor
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def make_clip(rate=16_000, duration=0.5, amplitude=0.1, channels=1):
    indices = np.arange(round(rate * duration), dtype=np.float64)
    tone = amplitude * np.sin(2 * np.pi * 211 * indices / rate)
    return AudioClip(np.repeat(tone[:, None], channels, axis=1), rate)


def make_adapter(output=(), error=None):
    extractor = FakeExtractor(output, error)
    return SileroVADAdapter(object(), extractor), extractor


@pytest.mark.parametrize("rate", [8_000, 16_000, 22_050, 44_100, 48_000, 96_000, 192_000])
def test_sampling_rate_conversion_and_original_alignment(fake_torch, rate):
    clip = make_clip(rate=rate)
    expected_rate = rate if rate in (8_000, 16_000) else 16_000
    adapter, extractor = make_adapter([{"start": 0, "end": expected_rate // 4}])
    result = adapter.detect(clip)
    tensor, _, kwargs = extractor.calls[0]
    assert kwargs["sampling_rate"] == expected_rate
    assert tensor.data.shape == (round(expected_rate * 0.5),)
    assert tensor.data.dtype == np.float32
    assert tensor.data.flags.writeable
    assert result.source_audio is clip
    assert result.intervals[0].start_sample == 0
    assert abs(result.intervals[0].end_seconds - 0.25) <= 1 / rate
    assert result.intervals[0].sample_rate == rate


@pytest.mark.parametrize("rate", [8_000, 16_000, 44_100, 192_000])
def test_end_of_recording_maps_to_original_duration(fake_torch, rate):
    clip = make_clip(rate=rate, duration=0.127)
    target = rate if rate in (8_000, 16_000) else 16_000
    prepared = vad.resample_audio(clip, target)
    adapter, _ = make_adapter([{"start": 0, "end": prepared.frames}])
    result = adapter.detect(clip)
    assert result.intervals[0].start_sample == 0
    assert result.intervals[0].end_sample == clip.frames
    assert result.speech_fraction == pytest.approx(1)


def test_default_parameters_match_official_api(fake_torch):
    clip = make_clip()
    adapter, extractor = make_adapter()
    result = adapter.detect(clip)
    assert result.intervals == ()
    assert extractor.calls[0][2] == {
        "sampling_rate": 16_000,
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
        "speech_pad_ms": 30,
        "return_seconds": False,
    }


def test_parameters_are_forwarded_explicitly(fake_torch):
    adapter, extractor = make_adapter()
    adapter.detect(make_clip(), threshold=0.67, min_speech_duration_ms=120,
                   min_silence_duration_ms=160, speech_pad_ms=15)
    kwargs = extractor.calls[0][2]
    assert kwargs["threshold"] == 0.67
    assert kwargs["min_speech_duration_ms"] == 120
    assert kwargs["min_silence_duration_ms"] == 160
    assert kwargs["speech_pad_ms"] == 15


@pytest.mark.parametrize("threshold", [0, 0.2, 0.5, 1, np.float32(0.65), np.int32(1)])
def test_valid_thresholds(fake_torch, threshold):
    adapter, extractor = make_adapter()
    adapter.detect(make_clip(), threshold=threshold)
    assert extractor.calls[0][2]["threshold"] == float(threshold)


@pytest.mark.parametrize("value", [None, False, True, "0.5", [], {}, object()])
def test_invalid_threshold_types(value):
    adapter, extractor = make_adapter()
    with pytest.raises(TypeError, match="threshold"):
        adapter.detect(make_clip(), threshold=value)
    assert not extractor.calls


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_threshold_values(value):
    adapter, _ = make_adapter()
    with pytest.raises(ValueError, match="threshold"):
        adapter.detect(make_clip(), threshold=value)


@pytest.mark.parametrize("field", ["min_speech_duration_ms", "min_silence_duration_ms", "speech_pad_ms"])
@pytest.mark.parametrize("value", [None, True, False, 1.3, "15", [], np.int32(5)])
def test_invalid_millisecond_parameter_types(field, value):
    adapter, extractor = make_adapter()
    with pytest.raises(TypeError, match=field):
        adapter.detect(make_clip(), **{field: value})
    assert not extractor.calls


@pytest.mark.parametrize("field", ["min_speech_duration_ms", "min_silence_duration_ms", "speech_pad_ms"])
@pytest.mark.parametrize("value", [-1, 180_001])
def test_invalid_millisecond_parameter_values(field, value):
    adapter, extractor = make_adapter()
    with pytest.raises(ValueError, match=field):
        adapter.detect(make_clip(), **{field: value})
    assert not extractor.calls


def test_zero_millisecond_parameters_are_valid(fake_torch):
    adapter, extractor = make_adapter()
    adapter.detect(make_clip(), min_speech_duration_ms=0, min_silence_duration_ms=0, speech_pad_ms=0)
    assert extractor.calls[0][2]["min_speech_duration_ms"] == 0
    assert extractor.calls[0][2]["min_silence_duration_ms"] == 0
    assert extractor.calls[0][2]["speech_pad_ms"] == 0


@pytest.mark.parametrize("value", [None, object(), [], "audio", np.zeros(100)])
def test_non_audio_clip_rejected(value):
    adapter, _ = make_adapter()
    with pytest.raises(TypeError, match="AudioClip"):
        adapter.detect(value)


@pytest.mark.parametrize("channels", [2, 3, 8])
def test_multichannel_rejected_explicitly(channels):
    adapter, extractor = make_adapter()
    with pytest.raises(ValueError, match="mono"):
        adapter.detect(make_clip(channels=channels))
    assert not extractor.calls


def test_input_amplitude_above_one_rejected():
    adapter, extractor = make_adapter()
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        adapter.detect(make_clip(amplitude=1.2))
    assert not extractor.calls


def test_resampled_amplitude_above_one_rejected(monkeypatch):
    adapter, extractor = make_adapter()
    monkeypatch.setattr(vad, "resample_audio", lambda clip, rate: AudioClip(np.ones(1600) * 1.1, rate))
    with pytest.raises(ValueError, match="Resampled audio"):
        adapter.detect(make_clip(rate=44_100))
    assert not extractor.calls


@pytest.mark.parametrize("amplitude", [0.0, 1e-10])
def test_silence_does_not_import_or_invoke_model(amplitude):
    adapter, extractor = make_adapter()
    clip = make_clip(amplitude=amplitude)
    result = adapter.detect(clip)
    assert result.source_audio is clip
    assert result.intervals == ()
    assert result.speech_duration_seconds == 0
    assert result.speech_fraction == 0
    assert not extractor.calls


def test_original_audio_is_not_modified(fake_torch):
    clip = make_clip(rate=44_100)
    reference = clip.samples.copy()
    adapter, extractor = make_adapter([{"start": 100, "end": 500}])
    adapter.detect(clip)
    assert np.array_equal(clip.samples, reference)
    assert not clip.samples.flags.writeable
    assert not np.shares_memory(extractor.calls[0][0].data, clip.samples)


def test_multiple_speech_intervals_are_ordered_and_half_open(fake_torch):
    clip = make_clip()
    adapter, _ = make_adapter([{"start": 100, "end": 1200}, {"start": 1600, "end": 3000}])
    result = adapter.detect(clip)
    assert len(result.intervals) == 2
    assert [(s.start_sample, s.end_sample) for s in result.intervals] == [(100, 1200), (1600, 3000)]
    assert result.speech_duration_seconds == pytest.approx(2500 / 16_000)
    assert result.speech_fraction == pytest.approx(2500 / clip.frames)
    assert result.intervals[0].start_seconds == pytest.approx(100 / 16_000)
    assert result.intervals[1].end_seconds == pytest.approx(3000 / 16_000)


def test_adjacent_intervals_are_valid(fake_torch):
    adapter, _ = make_adapter([{"start": 20, "end": 100}, {"start": 100, "end": 400}])
    assert len(adapter.detect(make_clip()).intervals) == 2


@pytest.mark.parametrize("output", [
    None, {}, 3, "timestamps", (i for i in range(2)),
    [None], [1], [[1, 2]], [{"start": 2}], [{"end": 3}],
    [{"start": 4, "end": 4}], [{"start": 6, "end": 2}],
    [{"start": -1, "end": 2}], [{"start": 0, "end": 9_000}],
    [{"start": 0.0, "end": 10}], [{"start": 0, "end": 10.0}],
    [{"start": True, "end": 10}], [{"start": 0, "end": False}],
    [{"start": "0", "end": 10}],
    [{"start": 0, "end": 500}, {"start": 300, "end": 700}],
    [{"start": 600, "end": 700}, {"start": 100, "end": 300}],
])
def test_malformed_backend_output_rejected(fake_torch, output):
    adapter, _ = make_adapter(output)
    with pytest.raises(SileroVADError):
        adapter.detect(make_clip())


def test_numpy_integer_timestamps_are_supported(fake_torch):
    adapter, _ = make_adapter([{"start": np.int64(10), "end": np.int32(100)}])
    assert adapter.detect(make_clip()).intervals[0] == SpeechInterval(10, 100, 16_000)


def test_backend_exception_is_wrapped(fake_torch):
    adapter, _ = make_adapter(error=ValueError("unsafe internal detail"))
    with pytest.raises(SileroVADError, match="inference failed"):
        adapter.detect(make_clip())


def test_missing_torch_dependency_is_actionable(monkeypatch):
    original_import = builtins.__import__
    def missing_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing_torch)
    adapter, _ = make_adapter()
    with pytest.raises(SileroVADError, match="PyTorch"):
        adapter.detect(make_clip())


def test_missing_silero_dependency_is_actionable(monkeypatch):
    original_import = builtins.__import__
    def missing_silero(name, *args, **kwargs):
        if name == "silero_vad":
            raise ImportError("silero unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing_silero)
    with pytest.raises(SileroVADError, match="Install 'silero-vad'"):
        SileroVADAdapter.load()


def test_explicit_load_uses_official_packaged_api(monkeypatch):
    module = ModuleType("silero_vad")
    model = object()
    calls = []
    def load_model():
        calls.append("load")
        return model
    module.load_silero_vad = load_model
    module.get_speech_timestamps = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "silero_vad", module)
    adapter = SileroVADAdapter.load()
    assert adapter._model is model
    assert adapter._get_timestamps is module.get_speech_timestamps
    assert calls == ["load"]


def test_model_load_exception_is_wrapped(monkeypatch):
    module = ModuleType("silero_vad")
    def load_model():
        raise RuntimeError("checkpoint failed")
    module.load_silero_vad = load_model
    module.get_speech_timestamps = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "silero_vad", module)
    with pytest.raises(SileroVADError, match="model loading failed"):
        SileroVADAdapter.load()


@pytest.mark.parametrize("model,extractor", [(None, lambda *args: []), (object(), None), (object(), 1), (None, None)])
def test_invalid_adapter_dependencies(model, extractor):
    with pytest.raises(TypeError, match="model and a callable"):
        SileroVADAdapter(model, extractor)


@pytest.mark.parametrize("start,end,rate,error", [
    (-1, 1, 16_000, ValueError), (0, 0, 16_000, ValueError), (2, 1, 16_000, ValueError),
    (0, 1, 0, ValueError), (0, 1, -1, ValueError),
    (True, 1, 16_000, TypeError), (0.5, 1, 16_000, TypeError),
    (0, False, 16_000, TypeError), (0, 1, "16000", ValueError),
])
def test_invalid_speech_interval(start, end, rate, error):
    with pytest.raises(error):
        SpeechInterval(start, end, rate)


def test_result_is_frozen_and_stores_original_audio():
    clip = make_clip()
    result = VoiceActivityResult(clip, (SpeechInterval(100, 1000, clip.sample_rate),))
    assert result.source_audio is clip
    assert result.model_name == "silero-vad"
    with pytest.raises(FrozenInstanceError):
        result.intervals = ()
    with pytest.raises(FrozenInstanceError):
        result.intervals[0].start_sample = 0
    with pytest.raises(AttributeError):
        result.intervals.append(None)


@pytest.mark.parametrize("clip,intervals,error", [
    (None, (), TypeError),
    ("audio", (), TypeError),
    (make_clip(), [], TypeError),
    (make_clip(), (None,), TypeError),
    (make_clip(), (SpeechInterval(1, 10, 8_000),), ValueError),
    (make_clip(), (SpeechInterval(1, 9_000, 16_000),), ValueError),
    (make_clip(), (SpeechInterval(50, 100, 16_000), SpeechInterval(40, 200, 16_000)), ValueError),
])
def test_invalid_result_contract(clip, intervals, error):
    with pytest.raises(error):
        VoiceActivityResult(clip, intervals)


def test_invalid_result_model_name():
    with pytest.raises(ValueError, match="model name"):
        VoiceActivityResult(make_clip(), (), model_name="other")


def test_interval_duration_and_seconds():
    interval = SpeechInterval(160, 800, 8_000)
    assert interval.start_seconds == pytest.approx(0.02)
    assert interval.end_seconds == pytest.approx(0.1)
    assert interval.duration_seconds == pytest.approx(0.08)


def test_boundary_rounding_collapse_is_rejected(fake_torch, monkeypatch):
    clip = make_clip(rate=8_000)
    # Force the 16-kHz backend path to exercise a sub-sample mapping collision.
    monkeypatch.setattr(vad, "_NATIVE_RATES", ())
    adapter, _ = make_adapter([{"start": 3, "end": 4}])
    with pytest.raises(SileroVADError, match="lost alignment"):
        adapter.detect(clip)
