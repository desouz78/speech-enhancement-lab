"""Contract and regression tests for optional SpeechBrain SepFormer inference.

Tests exercise validation, output reconstruction and dependency isolation with
controlled backends; they do not measure real-checkpoint separation accuracy.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

import audio_lab.sepformer as sepformer
from audio_lab.audio import AudioClip
from audio_lab.sepformer import SepFormerAdapter, SepFormerError, SpeechSeparationResult


_DEFAULT = object()


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeModel:
    def __init__(self, output=_DEFAULT, *, rate=8000, speakers=2, error=None):
        self.hparams = SimpleNamespace(sample_rate=rate, num_spks=speakers)
        self.output = output
        self.error = error
        self.calls = []

    def separate_batch(self, tensor):
        self.calls.append(tensor)
        if self.error is not None:
            raise self.error
        if self.output is _DEFAULT:
            waveform = tensor.data
            return FakeTensor(np.stack((waveform, waveform * 0.25), axis=-1))
        return self.output(tensor) if callable(self.output) else self.output


@pytest.fixture
def fake_torch(monkeypatch):
    torch = ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.from_numpy = lambda data: FakeTensor(data)
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def make_clip(rate=16000, duration=0.12, amplitude=0.2, channels=1):
    time = np.arange(round(rate * duration)) / rate
    signal = amplitude * np.sin(2 * np.pi * 237 * time)
    return AudioClip(np.repeat(signal[:, None], channels, axis=1), rate)


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000, 96000, 192000])
def test_realistic_contract_preserves_metadata_and_input(rate, fake_torch):
    clip = make_clip(rate=rate)
    before = clip.samples.copy()
    backend = FakeModel()
    result = SepFormerAdapter(backend).separate(clip)

    assert result.source_audio is clip
    assert result.model_name == "sepformer-wsj02mix"
    assert isinstance(result.sources, tuple) and len(result.sources) == 2
    assert result.speaker_1 is result.sources[0]
    assert result.speaker_2 is result.sources[1]
    assert len(backend.calls) == 1
    input_array = backend.calls[0].data
    assert input_array.shape == (1, round(clip.duration_seconds * 8000))
    assert input_array.dtype == np.float32
    assert input_array.flags.writeable
    assert not np.shares_memory(input_array, clip.samples)
    for source in result.sources:
        assert source.sample_rate == clip.sample_rate
        assert source.frames == clip.frames
        assert source.channels == 1
        assert not source.samples.flags.writeable
        assert np.all(np.isfinite(source.samples))
    interior = slice(round(rate * 0.02), -round(rate * 0.02))
    np.testing.assert_allclose(result.speaker_1.samples[interior], before[interior], atol=3e-3)
    np.testing.assert_allclose(result.speaker_2.samples[interior], before[interior] * 0.25, atol=3e-3)
    np.testing.assert_array_equal(clip.samples, before)


def test_source_numbering_does_not_imply_identity(fake_torch):
    result = SepFormerAdapter(FakeModel()).separate(make_clip())
    assert result.speaker_1 is not result.speaker_2
    assert result.speaker_1.frames == result.speaker_2.frames


def test_result_dataclass_is_immutable(fake_torch):
    result = SepFormerAdapter(FakeModel()).separate(make_clip())
    with pytest.raises(FrozenInstanceError):
        result.model_name = "changed"
    with pytest.raises(TypeError):
        result.sources[0] = make_clip()


@pytest.mark.parametrize("bad", [None, True, 1, "not audio", np.zeros(4), object()])
def test_invalid_input_type_prevents_inference(bad, fake_torch):
    model = FakeModel()
    with pytest.raises(TypeError, match="AudioClip"):
        SepFormerAdapter(model).separate(bad)
    assert model.calls == []


@pytest.mark.parametrize("channels", [2, 3, 8])
def test_multichannel_rejected(channels, fake_torch):
    model = FakeModel()
    with pytest.raises(ValueError, match="mono"):
        SepFormerAdapter(model).separate(make_clip(channels=channels))
    assert model.calls == []


def test_exact_duration_limit_accepted(fake_torch):
    clip = make_clip(rate=8000, duration=30)
    result = SepFormerAdapter(FakeModel()).separate(clip)
    assert result.source_audio.duration_seconds == 30


def test_more_than_thirty_seconds_rejected(fake_torch):
    model = FakeModel()
    with pytest.raises(ValueError, match="30 seconds"):
        SepFormerAdapter(model).separate(make_clip(rate=8000, duration=30.01))
    assert model.calls == []


@pytest.mark.parametrize("amplitude", [1.0001, -1.1, 4.0])
def test_amplitude_out_of_range_rejected(amplitude, fake_torch):
    model = FakeModel()
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        SepFormerAdapter(model).separate(make_clip(rate=8000, amplitude=amplitude))
    assert model.calls == []


@pytest.mark.parametrize("amplitude", [0.0, 1e-12])
def test_silence_returns_two_silent_sources_without_torch(amplitude, monkeypatch):
    model = FakeModel()
    monkeypatch.setitem(sys.modules, "torch", None)
    result = SepFormerAdapter(model).separate(make_clip(amplitude=amplitude))
    assert model.calls == []
    for source in result.sources:
        assert not source.samples.any()


def test_resampled_clipping_rejected(monkeypatch):
    clip = make_clip(rate=16000)
    def clipping_resampler(audio, rate):
        return AudioClip(np.full(960, 1.01, dtype=np.float32), rate)
    monkeypatch.setattr(sepformer, "resample_audio", clipping_resampler)
    with pytest.raises(ValueError, match="Resampled"):
        SepFormerAdapter(FakeModel()).separate(clip)


def test_torch_dependency_is_lazy(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SepFormerError, match="PyTorch"):
        SepFormerAdapter(FakeModel()).separate(make_clip())


def test_backend_error_preserves_cause(fake_torch):
    error = ValueError("unsupported backend")
    with pytest.raises(SepFormerError, match="inference failed") as exc:
        SepFormerAdapter(FakeModel(error=error)).separate(make_clip())
    assert exc.value.__cause__ is error


@pytest.mark.parametrize("invalid", [None, 7, "model", object(), lambda clip: clip])
def test_invalid_model_rejected(invalid):
    with pytest.raises(TypeError, match="separate_batch"):
        SepFormerAdapter(invalid)


@pytest.mark.parametrize("rate", [0, 16000, "8000", 8000.0, True, None])
def test_model_rate_contract(rate):
    with pytest.raises(SepFormerError, match="8,000 Hz"):
        SepFormerAdapter(FakeModel(rate=rate))


@pytest.mark.parametrize("speakers", [0, 1, 3, "2", 2.0, True, None])
def test_model_two_speaker_contract(speakers):
    with pytest.raises(SepFormerError, match="two speakers"):
        SepFormerAdapter(FakeModel(speakers=speakers))


@pytest.mark.parametrize("invalid", [None, 123, "broken", np.zeros((1, 960, 2))])
def test_unreadable_tensor_rejected(invalid, fake_torch):
    model = FakeModel(output=invalid)
    with pytest.raises(SepFormerError, match="readable PyTorch tensor"):
        SepFormerAdapter(model).separate(make_clip())


class BadTensor:
    def detach(self):
        raise RuntimeError("backend tensor conversion failed")


def test_unreadable_tensor_runtime_error(fake_torch):
    with pytest.raises(SepFormerError, match="readable PyTorch tensor"):
        SepFormerAdapter(FakeModel(output=BadTensor())).separate(make_clip())


@pytest.mark.parametrize("bad_data", [
    np.zeros((960, 2), dtype=np.float32),
    np.zeros((1, 960, 3), dtype=np.float32),
    np.zeros((2, 960, 2), dtype=np.float32),
    np.zeros((1, 959, 2), dtype=np.float32),
    np.zeros((1, 961, 2), dtype=np.float32),
    np.zeros((1, 960, 2), dtype=np.int32),
    np.ones((1, 960, 2), dtype=bool),
    np.full((1, 960, 2), np.nan, dtype=np.float32),
    np.full((1, 960, 2), np.inf, dtype=np.float32),
    np.full((1, 960, 2), -np.inf, dtype=np.float32),
    "bad tensor",
])
def test_invalid_output_contract_rejected(bad_data, fake_torch):
    with pytest.raises(SepFormerError, match="invalid tensor"):
        SepFormerAdapter(FakeModel(output=FakeTensor(bad_data))).separate(make_clip())


def test_output_overflow_float32_rejected(fake_torch):
    output = np.full((1, 960, 2), np.finfo(np.float64).max, dtype=np.float64)
    with pytest.raises(SepFormerError, match="float32"):
        SepFormerAdapter(FakeModel(output=FakeTensor(output))).separate(make_clip())


def test_result_constructor_strict_types(fake_torch):
    clip = make_clip()
    source = SepFormerAdapter(FakeModel()).separate(clip).speaker_1
    with pytest.raises(TypeError, match="source_audio"):
        SpeechSeparationResult(None, (source, source))
    with pytest.raises(TypeError, match="exactly two"):
        SpeechSeparationResult(clip, [source, source])
    with pytest.raises(TypeError, match="exactly two"):
        SpeechSeparationResult(clip, (source,))
    with pytest.raises(TypeError, match="AudioClip"):
        SpeechSeparationResult(clip, (source, None))
    with pytest.raises(ValueError, match="model name"):
        SpeechSeparationResult(clip, (source, source), "unknown")
    with pytest.raises(ValueError, match="aligned"):
        SpeechSeparationResult(clip, (source, make_clip(rate=8000)))
    with pytest.raises(ValueError, match="aligned"):
        SpeechSeparationResult(clip, (source, make_clip(duration=0.1)))
    with pytest.raises(ValueError, match="aligned"):
        SpeechSeparationResult(clip, (source, make_clip(channels=2)))


def test_load_validates_device_type_and_value_without_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "speechbrain", None)
    for bad in [None, True, 0, 3.0, [], {}]:
        with pytest.raises(TypeError, match="device"):
            SepFormerAdapter.load(device=bad)
    for bad in ["", "gpu", "CPU", "cuda:0", " cpu"]:
        with pytest.raises(ValueError, match="device"):
            SepFormerAdapter.load(device=bad)


def test_load_validates_savedir_without_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "speechbrain", None)
    for bad in [1, 2.0, [], {}, False, object()]:
        with pytest.raises(TypeError, match="savedir"):
            SepFormerAdapter.load(savedir=bad)
    for bad in ["", "   "]:
        with pytest.raises(ValueError, match="savedir"):
            SepFormerAdapter.load(savedir=bad)


def test_missing_speechbrain_dependency_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "speechbrain", None)
    with pytest.raises(SepFormerError, match="speechbrain"):
        SepFormerAdapter.load()


def test_loading_uses_fixed_public_checkpoint_and_explicit_device(monkeypatch, tmp_path):
    observed = []
    inference = ModuleType("speechbrain.inference")
    separation = ModuleType("speechbrain.inference.separation")

    class Loader:
        @staticmethod
        def from_hparams(*, source, savedir, run_opts):
            observed.append((source, savedir, run_opts))
            return FakeModel()

    separation.SepformerSeparation = Loader
    monkeypatch.setitem(sys.modules, "speechbrain", ModuleType("speechbrain"))
    monkeypatch.setitem(sys.modules, "speechbrain.inference", inference)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.separation", separation)

    loaded = SepFormerAdapter.load(device="cuda", savedir=tmp_path)
    assert isinstance(loaded, SepFormerAdapter)
    assert observed == [("speechbrain/sepformer-wsj02mix", str(tmp_path), {"device": "cuda"})]
    loaded_default = SepFormerAdapter.load()
    assert len(observed) == 2
    assert Path(observed[1][1]).name == "sepformer-wsj02mix"
    assert observed[1][2] == {"device": "cpu"}


def test_loading_rejects_bad_backend_model(monkeypatch):
    separation = ModuleType("speechbrain.inference.separation")
    class Loader:
        @staticmethod
        def from_hparams(**kwargs):
            return object()
    separation.SepformerSeparation = Loader
    monkeypatch.setitem(sys.modules, "speechbrain", ModuleType("speechbrain"))
    monkeypatch.setitem(sys.modules, "speechbrain.inference", ModuleType("speechbrain.inference"))
    monkeypatch.setitem(sys.modules, "speechbrain.inference.separation", separation)
    with pytest.raises(TypeError, match="separate_batch"):
        SepFormerAdapter.load()


def test_two_speaker_outputs_may_exceed_unit_amplitude_without_clipping(fake_torch):
    output = np.full((1, 960, 2), 2.5, dtype=np.float32)
    result = SepFormerAdapter(FakeModel(output=FakeTensor(output))).separate(make_clip())
    assert np.max(result.speaker_1.samples) > 1.0
    assert np.max(result.speaker_2.samples) > 1.0


@pytest.mark.parametrize("delta", [-1, 1, 2, -2])
def test_resampling_frame_rounding_and_tolerance(delta, fake_torch, monkeypatch):
    source = make_clip(rate=16000)
    original_resample = sepformer.resample_audio

    def adjusted_resample(audio, rate):
        converted = original_resample(audio, rate)
        if audio.sample_rate == 8000 and rate == 16000:
            if delta > 0:
                samples = np.pad(converted.samples, ((0, delta), (0, 0)))
            else:
                samples = converted.samples[:delta]
            return AudioClip(samples, rate)
        return converted

    monkeypatch.setattr(sepformer, "resample_audio", adjusted_resample)
    if abs(delta) > 1:
        with pytest.raises(SepFormerError, match="frame count"):
            SepFormerAdapter(FakeModel()).separate(source)
    else:
        result = SepFormerAdapter(FakeModel()).separate(source)
        assert all(item.frames == source.frames for item in result.sources)
        if delta == -1:
            assert all(item.samples[-1, 0] == 0.0 for item in result.sources)


def test_model_class_does_not_download_on_import():
    assert SepFormerAdapter.__module__ == "audio_lab.sepformer"
    assert sepformer._MODEL_ID == "speechbrain/sepformer-wsj02mix"
