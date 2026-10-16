"""Dependency-free contract tests for the optional DeepFilterNet3 adapter.

These tests use a controlled PyTorch/DeepFilterNet test double; they do not
measure real model quality, installation reliability, or checkpoint inference.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from audio_lab.audio import AudioClip
from audio_lab.deepfilternet import DeepFilterNet3Adapter, DeepFilterNetError


class FakeTensor:
    def __init__(self, array, *, copy=True):
        self.data = np.array(array, copy=copy)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeState:
    def __init__(self, rate=48000):
        self.rate = rate

    def sr(self):
        return self.rate


@pytest.fixture
def fake_torch(monkeypatch):
    module = ModuleType("torch")
    module.from_numpy = lambda array: FakeTensor(array, copy=False)
    module.Tensor = FakeTensor
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def make_clip(rate=16000, duration=0.25, amplitude=0.3):
    samples = amplitude * np.sin(2 * np.pi * 440 * np.arange(round(rate * duration)) / rate)
    return AudioClip(samples, rate)


def adapter_with(enhancer=None, state=None):
    if enhancer is None:
        enhancer = lambda model, df_state, tensor, *, pad: FakeTensor(tensor.data)
    return DeepFilterNet3Adapter(object(), state or FakeState(), enhancer)


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 96000, 192000])
def test_identity_backend_preserves_rate_shape_and_audio(rate, fake_torch):
    source = make_clip(rate)
    output = adapter_with().enhance(source)
    assert output.sample_rate == rate
    assert output.frames == source.frames
    assert output.channels == 1
    assert output.samples.flags.writeable is False
    np.testing.assert_allclose(output.samples, source.samples, atol=0.006)


def test_backend_receives_mono_48khz_tensor_with_padding(fake_torch):
    seen = []

    def backend(model, state, tensor, *, pad):
        seen.append((state.sr(), tensor.data.shape, tensor.data.dtype, pad))
        return FakeTensor(tensor.data * 0.5)

    source = make_clip()
    enhanced = adapter_with(backend).enhance(source)
    assert seen == [(48000, (1, 12000), np.dtype("float32"), True)]
    np.testing.assert_allclose(enhanced.samples, source.samples / 2, atol=0.002)


def test_original_clip_unchanged(fake_torch):
    source = make_clip()
    original = source.samples.copy()
    adapter_with().enhance(source)
    np.testing.assert_array_equal(source.samples, original)


def test_silence_is_preserved(fake_torch):
    clip = AudioClip(np.zeros(2400), 48000)
    result = adapter_with().enhance(clip)
    np.testing.assert_array_equal(result.samples, clip.samples)


@pytest.mark.parametrize("bad", [None, 1, "sound", np.zeros(10), object()])
def test_invalid_audio_rejected_before_inference(bad, fake_torch):
    with pytest.raises(TypeError, match="AudioClip"):
        adapter_with().enhance(bad)


def test_stereo_rejected_before_inference(fake_torch):
    stereo = AudioClip(np.zeros((1600, 2), dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="mono"):
        adapter_with().enhance(stereo)


@pytest.mark.parametrize("amplitude", [1.01, 7.0, 1000.0])
def test_out_of_range_amplitude_rejected(amplitude, fake_torch):
    clip = AudioClip(np.array([amplitude, 0.0]), 16000)
    with pytest.raises(ValueError, match="amplitude"):
        adapter_with().enhance(clip)


@pytest.mark.parametrize("rate", [44100, 0, 16000, 96000, 48000.0, True, None])
def test_invalid_model_sample_rate_rejected(rate):
    with pytest.raises(DeepFilterNetError, match="sample rate|48,000"):
        adapter_with(state=FakeState(rate))


def test_absent_state_interface_rejected():
    with pytest.raises(DeepFilterNetError, match=r"sr\(\)"):
        DeepFilterNet3Adapter(object(), object(), lambda *args: None)


def test_state_rate_failure_rejected():
    def fails():
        raise RuntimeError("native failure")

    with pytest.raises(DeepFilterNetError, match="sample rate"):
        DeepFilterNet3Adapter(object(), SimpleNamespace(sr=fails), lambda *args: None)


@pytest.mark.parametrize("model,state,backend", [
    (None, FakeState(), lambda *args: None),
    (object(), None, lambda *args: None),
    (object(), FakeState(), None),
    (object(), FakeState(), object()),
])
def test_bad_adapter_construction(model, state, backend):
    with pytest.raises(TypeError):
        DeepFilterNet3Adapter(model, state, backend)


@pytest.mark.parametrize("output", [
    np.zeros((2, 2400), dtype=np.float32),
    np.zeros((2400, 1), dtype=np.float32),
    np.zeros((1, 2401), dtype=np.float32),
    np.zeros((2400,), dtype=np.float32),
    np.zeros((1, 2400), dtype=np.int32),
    np.full((1, 2400), np.nan, dtype=np.float32),
    np.full((1, 2400), np.inf, dtype=np.float32),
])
def test_invalid_output_rejected(output, fake_torch):
    adapter = adapter_with(lambda model, state, tensor, *, pad: FakeTensor(output))
    clip = AudioClip(np.zeros(2400), 48000)
    with pytest.raises(DeepFilterNetError, match="output shape|nonfinite|nonfloating"):
        adapter.enhance(clip)


def test_non_tensor_output_rejected(fake_torch):
    adapter = adapter_with(lambda model, state, tensor, *, pad: np.zeros((1, 2400)))
    with pytest.raises(DeepFilterNetError, match="PyTorch tensor"):
        adapter.enhance(AudioClip(np.zeros(2400), 48000))


def test_backend_exception_not_hidden(fake_torch):
    def fails(*args, **kwargs):
        raise RuntimeError("DF engine failure")

    adapter = adapter_with(fails)
    with pytest.raises(RuntimeError, match="DF engine failure"):
        adapter.enhance(AudioClip(np.zeros(2400), 48000))


def test_missing_torch_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(DeepFilterNetError, match="PyTorch"):
        adapter_with().enhance(AudioClip(np.zeros(2400), 48000))


def test_load_explicitly_requests_deepfilternet3(monkeypatch):
    df = ModuleType("df")
    df.__path__ = []
    enhance_module = ModuleType("df.enhance")
    calls = []

    def init_df(**kwargs):
        calls.append(kwargs)
        return object(), FakeState(), "DeepFilterNet3", 10

    def enhance(*args, **kwargs):
        return None

    enhance_module.init_df = init_df
    enhance_module.enhance = enhance
    monkeypatch.setitem(sys.modules, "df", df)
    monkeypatch.setitem(sys.modules, "df.enhance", enhance_module)
    adapter = DeepFilterNet3Adapter.load()
    assert isinstance(adapter, DeepFilterNet3Adapter)
    assert calls == [{"model_base_dir": "DeepFilterNet3", "log_file": None}]


@pytest.mark.parametrize("response", [(object(), FakeState()), None, [object(), FakeState(), "v3"]])
def test_load_rejects_unexpected_response(monkeypatch, response):
    df = ModuleType("df")
    df.__path__ = []
    module = ModuleType("df.enhance")
    module.init_df = lambda **kwargs: response
    module.enhance = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "df", df)
    monkeypatch.setitem(sys.modules, "df.enhance", module)
    with pytest.raises(DeepFilterNetError, match="initialization"):
        DeepFilterNet3Adapter.load()


def test_load_missing_dependency_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "df", None)
    monkeypatch.delitem(sys.modules, "df.enhance", raising=False)
    with pytest.raises(DeepFilterNetError, match="dependencies"):
        DeepFilterNet3Adapter.load()


def test_torch_input_does_not_alias_source_samples(fake_torch):
    source = make_clip(rate=48000)
    before = source.samples.copy()

    def destructive_backend(model, state, tensor, *, pad):
        tensor.data[:] = 0.0
        return FakeTensor(tensor.data)

    processed = adapter_with(destructive_backend).enhance(source)
    np.testing.assert_array_equal(source.samples, before)
    assert np.count_nonzero(processed.samples) == 0


def test_inference_length_limit_rejects_oversized_input(fake_torch):
    source = AudioClip(np.zeros(31 * 8000, dtype=np.float32), 8000)
    with pytest.raises(ValueError, match="30 seconds"):
        adapter_with().enhance(source)


def test_30_second_input_is_accepted(fake_torch):
    source = AudioClip(np.zeros(30 * 48000, dtype=np.float32), 48000)
    assert adapter_with().enhance(source).frames == source.frames


def test_resampling_overshoot_is_rejected(fake_torch):
    from audio_lab.audio import resample_audio

    # A bandlimited resampler may overshoot even for bounded PCM input.
    source = AudioClip(np.tile(np.array([1.0, -1.0, -1.0, 1.0]), 300), 16000)
    upsampled = resample_audio(source, 48000)
    if np.max(np.abs(upsampled.samples)) <= 1.0:
        pytest.skip("Current SciPy version does not overshoot for this signal.")
    with pytest.raises(ValueError, match="Resampled"):
        adapter_with().enhance(source)
