"""Dependency-free contract/regression tests for the optional HTDemucs adapter.

Model doubles exercise the documented Demucs 4.0.1 tensor interface, not
checkpoint quality. No weights or heavy neural dependencies are downloaded.
"""

from dataclasses import FrozenInstanceError
import builtins
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from audio_lab.audio import AudioClip
from audio_lab.htdemucs import HTDemucsAdapter, HTDemucsError, MusicSeparationResult


STEMS = ("drums", "bass", "other", "vocals")


class FakeTensor:
    def __init__(self, data, *, broken=False):
        self.data = data
        self.broken = broken

    def detach(self):
        if self.broken:
            raise RuntimeError("fake backend failure")
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeModel:
    def __init__(self, *, rate=44100, channels=2, sources=STEMS):
        self.samplerate = rate
        self.audio_channels = channels
        self.sources = sources


class FakeBackend:
    def __init__(self, model=None, outputs=None, error=None):
        self.model = model or FakeModel()
        self.outputs = outputs
        self.error = error
        self.calls = []

    def apply(self, model, mixture, **options):
        assert model is self.model
        assert isinstance(mixture, FakeTensor)
        self.calls.append((mixture.data.copy(), options))
        if self.error is not None:
            raise self.error
        if self.outputs is not None:
            return self.outputs
        batch, channels, frames = mixture.data.shape
        data = np.stack([
            mixture.data * scale for scale in [0.2, 0.3, 0.4, 0.5]
        ], axis=1)
        assert data.shape == (batch, 4, channels, frames)
        mixture.data[:] = 0.0  # Simulate a destructive model backend.
        return FakeTensor(data)


@pytest.fixture
def torch_stub(monkeypatch):
    stub = ModuleType("torch")
    stub.from_numpy = lambda data: FakeTensor(data)
    stub.Tensor = FakeTensor
    monkeypatch.setitem(sys.modules, "torch", stub)
    return stub


def make_clip(rate=44100, duration=0.12, channels=2, amplitude=0.3):
    length = round(rate * duration)
    t = np.arange(length, dtype=np.float64) / rate
    left = amplitude * np.sin(2.0 * np.pi * 241.0 * t)
    channels_data = [left, amplitude * np.sin(2.0 * np.pi * 359.0 * t)]
    data = np.column_stack(channels_data[:channels]) if channels <= 2 else np.tile(left[:, None], (1, channels))
    return AudioClip(data, rate)


def make_adapter(backend=None, *, device="cpu"):
    backend = backend or FakeBackend()
    return HTDemucsAdapter(backend.model, backend.apply, device=device)


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 96000, 192000])
@pytest.mark.parametrize("channels", [1, 2])
def test_round_trip_metadata_and_tensor_contract(rate, channels, torch_stub):
    source = make_clip(rate, channels=channels)
    before = source.samples.copy()
    backend = FakeBackend()
    result = make_adapter(backend).separate(source)
    assert result.source_audio is source
    assert result.model_name == "htdemucs"
    assert tuple(result.stems) == STEMS
    assert result.vocals is result.stems["vocals"]
    np.testing.assert_array_equal(source.samples, before)
    assert len(backend.calls) == 1
    input_array, kwargs = backend.calls[0]
    assert input_array.shape == (1, 2, round(source.duration_seconds * 44100)) or abs(input_array.shape[-1] - round(source.duration_seconds * 44100)) <= 1
    assert input_array.dtype == np.float32
    assert input_array.flags.c_contiguous
    assert np.isfinite(input_array).all()
    assert kwargs == dict(shifts=0, split=True, overlap=0.25, progress=False, device="cpu", num_workers=0)
    if channels == 1:
        np.testing.assert_array_equal(input_array[0, 0], input_array[0, 1])
    for name in STEMS:
        stem = result.stems[name]
        assert stem.frames == source.frames
        assert stem.channels == source.channels
        assert stem.sample_rate == rate
        assert not stem.samples.flags.writeable
        assert np.isfinite(stem.samples).all()
        assert float(np.max(np.abs(stem.samples))) > 0.0


def test_model_source_order_is_respected(torch_stub):
    sources = tuple(reversed(STEMS))
    model = FakeModel(sources=sources)
    backend = FakeBackend(model)
    result = make_adapter(backend).separate(make_clip())
    # Fake backend emits scale in source order; reversed source IDs change mapping.
    assert np.mean(abs(result.stems["vocals"].samples)) < np.mean(abs(result.stems["drums"].samples))


@pytest.mark.parametrize("bad", [None, False, 1, "audio", np.zeros(3), object()])
def test_invalid_input_rejected_before_inference(bad, torch_stub):
    backend = FakeBackend()
    with pytest.raises(TypeError, match="AudioClip"):
        make_adapter(backend).separate(bad)
    assert not backend.calls


@pytest.mark.parametrize("channels", [3, 8])
def test_nonmono_nonstereo_rejected(channels, torch_stub):
    backend = FakeBackend()
    with pytest.raises(ValueError, match="mono or stereo"):
        make_adapter(backend).separate(make_clip(channels=channels))
    assert not backend.calls


def test_duration_30s_is_allowed_without_checkpoint(torch_stub):
    source = make_clip(rate=8000, duration=30.0, channels=1)
    assert make_adapter().separate(source).source_audio is source


def test_too_long_rejected_before_inference(torch_stub):
    backend = FakeBackend()
    with pytest.raises(ValueError, match="30 seconds"):
        make_adapter(backend).separate(make_clip(rate=8000, duration=30.001))
    assert not backend.calls


@pytest.mark.parametrize("amplitude", [1.1, -1.1, 10.0])
def test_out_of_range_input_rejected(amplitude, torch_stub):
    samples = np.full((100, 2), amplitude, dtype=np.float32)
    backend = FakeBackend()
    with pytest.raises(ValueError, match="\u005c\u005b-1, 1\u005c\u005d"):
        make_adapter(backend).separate(AudioClip(samples, 44100))
    assert not backend.calls


def test_resampling_overshoot_rejected(monkeypatch):
    import audio_lab.htdemucs as module
    backend = FakeBackend()
    monkeypatch.setattr(module, "resample_audio", lambda clip, rate: AudioClip(np.ones((300, 2), dtype=np.float32) * 1.2, rate))
    with pytest.raises(ValueError, match="Resampled input"):
        make_adapter(backend).separate(make_clip(rate=8000))
    assert not backend.calls


@pytest.mark.parametrize("rate", [44101, 48000, "44100", True, None])
def test_model_sample_rate_validation(rate):
    with pytest.raises(HTDemucsError, match="44,100"):
        HTDemucsAdapter(FakeModel(rate=rate), lambda *a, **k: None)


@pytest.mark.parametrize("channels", [0, 1, 3, "2", True, None])
def test_model_channel_validation(channels):
    with pytest.raises(HTDemucsError, match="stereo"):
        HTDemucsAdapter(FakeModel(channels=channels), lambda *a, **k: None)


@pytest.mark.parametrize("sources", [(), ["vocals"], ["vocals"] * 4, "vocals", None, [[], "bass", "other", "vocals"], [1, "bass", "other", "vocals"]])
def test_model_stem_names_validation(sources):
    with pytest.raises(HTDemucsError, match="source names"):
        HTDemucsAdapter(FakeModel(sources=sources), lambda *a, **k: None)


@pytest.mark.parametrize("model,fn", [(None, lambda: None), (FakeModel(), None), (FakeModel(), 5), (FakeModel(), "callable")])
def test_invalid_constructor_arguments(model, fn):
    with pytest.raises(TypeError, match="model and callable"):
        HTDemucsAdapter(model, fn)


@pytest.mark.parametrize("device", [None, True, False, 1, {}, object()])
def test_device_wrong_types(device):
    with pytest.raises(TypeError, match="device"):
        HTDemucsAdapter(FakeModel(), lambda *a, **k: None, device=device)


@pytest.mark.parametrize("device", ["auto", "mps", "gpu", "cuda:0", "CPU", ""])
def test_unsupported_devices(device):
    with pytest.raises(ValueError, match="device"):
        HTDemucsAdapter(FakeModel(), lambda *a, **k: None, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_load_uses_real_official_modules(device, monkeypatch):
    calls = []
    model = FakeModel()
    fake_pretrained = ModuleType("demucs.pretrained")
    fake_apply = ModuleType("demucs.apply")
    def get_model(name):
        calls.append(name)
        return model
    def apply_model(*args, **kwargs):
        return None
    fake_pretrained.get_model = get_model
    fake_apply.apply_model = apply_model
    monkeypatch.setitem(sys.modules, "demucs", ModuleType("demucs"))
    monkeypatch.setitem(sys.modules, "demucs.pretrained", fake_pretrained)
    monkeypatch.setitem(sys.modules, "demucs.apply", fake_apply)
    adapter = HTDemucsAdapter.load(device=device)
    assert isinstance(adapter, HTDemucsAdapter)
    assert adapter._model is model
    assert adapter._apply_fn is apply_model
    assert adapter._device == device
    assert calls == ["htdemucs"]


@pytest.mark.parametrize("device", [None, 1, "gpu"])
def test_load_rejects_invalid_device_before_import(device):
    with pytest.raises((TypeError, ValueError)):
        HTDemucsAdapter.load(device=device)


@pytest.mark.parametrize("which", ["demucs.pretrained", "demucs.apply"])
def test_load_missing_optional_dependencies(which, monkeypatch):
    real_import = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == which:
            raise ImportError("Simulated missing package")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(HTDemucsError, match="demucs==4.0.1"):
        HTDemucsAdapter.load()


def test_missing_torch_is_actionable(monkeypatch):
    real_import = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("Simulated torch dependency missing")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(HTDemucsError, match="PyTorch"):
        make_adapter().separate(make_clip())


@pytest.mark.parametrize("array", [None, "bad", 1, np.zeros((4, 2, 500)), np.zeros((1, 4, 2, 3)), np.zeros((1, 4, 2, 5292), dtype=np.int64), np.full((1, 4, 2, 5292), np.inf), np.full((1, 4, 2, 5292), np.nan)])
def test_bad_estimate_tensor_contract(array, torch_stub):
    backend = FakeBackend(outputs=FakeTensor(array))
    with pytest.raises(HTDemucsError, match="invalid shape|readable PyTorch"):
        make_adapter(backend).separate(make_clip())


def test_bad_tensor_interface_and_backend_failure(torch_stub):
    for result in (object(), FakeTensor(np.zeros((1,4,2,5292)), broken=True)):
        backend = FakeBackend(outputs=result)
        with pytest.raises(HTDemucsError, match="readable PyTorch"):
            make_adapter(backend).separate(make_clip())
    backend = FakeBackend(error=RuntimeError("upstream failed"))
    with pytest.raises(RuntimeError, match="upstream failed"):
        make_adapter(backend).separate(make_clip())


def test_overflow_on_denormalization_rejected(torch_stub):
    backend = FakeBackend(outputs=FakeTensor(np.full((1,4,2,5292), 1e100)))
    with pytest.raises(HTDemucsError, match="float32 amplitude"):
        make_adapter(backend).separate(make_clip())


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_zeros_or_constant_signals_return_silence(rate):
    for amplitude in (0.0, 0.2):
        samples = np.full((256, 2), amplitude, dtype=np.float32)
        backend = FakeBackend()
        result = make_adapter(backend).separate(AudioClip(samples, rate))
        assert not backend.calls
        assert tuple(result.stems) == STEMS
        for stem in result.stems.values():
            assert stem.frames == 256
            assert np.count_nonzero(stem.samples) == 0


def test_single_sample_does_not_divide_by_zero():
    clip = AudioClip(np.array([[0.2, 0.2]], dtype=np.float32), 44100)
    result = make_adapter().separate(clip)
    assert np.count_nonzero(result.vocals.samples) == 0


def test_immutable_stems_and_result(torch_stub):
    result = make_adapter().separate(make_clip())
    with pytest.raises(TypeError):
        result.stems["vocals"] = make_clip()
    with pytest.raises(FrozenInstanceError):
        result.model_name = "invalid"


@pytest.mark.parametrize("source", [None, True, "wrong", object()])
def test_result_invalid_source(source):
    with pytest.raises(TypeError, match="source_audio"):
        MusicSeparationResult(source, {name:make_clip() for name in STEMS})


@pytest.mark.parametrize("stems", [None, False, {}, {"vocals":make_clip()}, {"drums":make_clip()}])
def test_result_invalid_stems(stems):
    with pytest.raises(ValueError, match="exactly"):
        MusicSeparationResult(make_clip(), stems)


def test_result_invalid_stem_type():
    stems = {name:make_clip() for name in STEMS}
    stems["vocals"] = False
    with pytest.raises(TypeError, match="Stem"):
        MusicSeparationResult(make_clip(), stems)


@pytest.mark.parametrize("broken", [make_clip(rate=16000), make_clip(duration=0.14), make_clip(channels=1)])
def test_result_invalid_stem_metadata(broken):
    stems = {name:make_clip() for name in STEMS}
    stems["vocals"] = broken
    with pytest.raises(ValueError, match="metadata"):
        MusicSeparationResult(make_clip(), stems)


@pytest.mark.parametrize("name", ["wrong", None, False, 5])
def test_result_invalid_model_name(name):
    with pytest.raises(ValueError, match="model_name"):
        MusicSeparationResult(make_clip(), {name:make_clip() for name in STEMS}, name)


def test_out_of_range_stem_amplitude_is_not_clipped(torch_stub):
    source = make_clip()
    backend = FakeBackend(outputs=FakeTensor(np.full((1,4,2,source.frames), 8.0, dtype=np.float32)))
    result = make_adapter(backend).separate(source)
    assert np.max(result.vocals.samples) > 1.0


@pytest.mark.parametrize("delta", [-2, 2])
def test_resampling_length_discrepancy_gt_one_is_an_error(delta, torch_stub, monkeypatch):
    import audio_lab.htdemucs as module
    original = module.resample_audio
    calls = []
    def resampled(clip, rate):
        calls.append(rate)
        if len(calls) >= 2:
            return AudioClip(np.zeros((make_clip().frames + delta, 2), dtype=np.float32), rate)
        return original(clip, rate)
    monkeypatch.setattr(module, "resample_audio", resampled)
    with pytest.raises(HTDemucsError, match="frame count"):
        make_adapter().separate(make_clip())


@pytest.mark.parametrize("delta", [-1, 1])
def test_one_frame_resampling_discrepancy_is_repaired(delta, torch_stub, monkeypatch):
    import audio_lab.htdemucs as module
    original = module.resample_audio
    calls = []
    def resampled(clip, rate):
        calls.append(rate)
        if len(calls) >= 2:
            return AudioClip(np.ones((make_clip().frames + delta, 2), dtype=np.float32), rate)
        return original(clip, rate)
    monkeypatch.setattr(module, "resample_audio", resampled)
    result = make_adapter().separate(make_clip())
    assert result.vocals.frames == make_clip().frames
    assert result.vocals.samples[-1, 0] == pytest.approx(0 if delta == -1 else 1)


def test_nonfinite_normalized_input_is_caught(monkeypatch):
    # Artificially force unstable normalization to verify the guard independently.
    import audio_lab.htdemucs as module
    original = module.np.array
    def changed(data, *args, **kwargs):
        if kwargs.get("order") == "C":
            result = original(data, *args, **kwargs)
            result.reshape(-1)[0] = np.nan
            return result
        return original(data, *args, **kwargs)
    monkeypatch.setattr(module.np, "array", changed)
    with pytest.raises(HTDemucsError, match="nonfinite"):
        make_adapter().separate(make_clip())


def test_antiphase_stereo_not_discarded(torch_stub):
    source = make_clip(channels=2)
    antiphase = AudioClip(np.column_stack((source.samples[:, 0], -source.samples[:, 0])), source.sample_rate)
    backend = FakeBackend()
    result = make_adapter(backend).separate(antiphase)
    assert len(backend.calls) == 1
    assert np.count_nonzero(result.vocals.samples) > 0


def test_post_resampling_below_noise_floor_does_not_call_model(monkeypatch):
    import audio_lab.htdemucs as module
    backend = FakeBackend()
    monkeypatch.setattr(module, "resample_audio", lambda clip, rate: AudioClip(np.zeros((512, 2), dtype=np.float32), rate))
    result = make_adapter(backend).separate(make_clip())
    assert not backend.calls
    assert result.vocals.frames == make_clip().frames
    assert np.count_nonzero(result.vocals.samples) == 0
