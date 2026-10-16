"""Contract tests for gated pyannote Community-1 speaker diarization.

All model outputs are simulated. These tests do not validate pretrained speaker
identification, diarization accuracy, or Hugging Face checkpoint access.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace
import builtins
import sys

import numpy as np
import pytest

import audio_lab.pyannote_diarization as diarization
from audio_lab.audio import AudioClip
from audio_lab.pyannote_diarization import (
    PyannoteDiarizationAdapter,
    PyannoteDiarizationError,
    SpeakerDiarizationResult,
    SpeakerTurn,
)


class FakeTensor:
    def __init__(self, data):
        self.data = data


class FakeAnnotation:
    def __init__(self, turns=()):
        self.turns = turns

    def itertracks(self, *, yield_label=False):
        assert yield_label is True
        for index, (start, end, speaker) in enumerate(self.turns):
            yield SimpleNamespace(start=start, end=end), index, speaker


class FakePipeline:
    def __init__(self, normal=((0.1, 0.3, "SPEAKER_00"), (0.2, 0.4, "SPEAKER_01")),
                 exclusive=((0.1, 0.2, "SPEAKER_00"), (0.2, 0.4, "SPEAKER_01")), error=None):
        self.normal = normal
        self.exclusive = exclusive
        self.error = error
        self.calls = []
        self.devices = []

    def __call__(self, audio, **options):
        self.calls.append((audio, options))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(speaker_diarization=FakeAnnotation(self.normal),
                               exclusive_speaker_diarization=FakeAnnotation(self.exclusive))

    def to(self, device):
        self.devices.append(device)
        return self


@pytest.fixture
def fake_torch(monkeypatch):
    torch = ModuleType("torch")
    torch.from_numpy = FakeTensor
    torch.Tensor = FakeTensor
    torch.inference_mode = nullcontext
    torch.device = lambda name: name
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def make_clip(rate=16000, duration=0.5, amplitude=0.25, channels=1):
    time = np.arange(round(rate * duration)) / rate
    signal = amplitude * np.sin(2 * np.pi * 317 * time)
    return AudioClip(np.repeat(signal[:, None], channels, axis=1), rate)


def test_normal_and_exclusive_diarization(fake_torch):
    clip = make_clip()
    model = FakePipeline()
    result = PyannoteDiarizationAdapter(model).diarize(clip)
    assert isinstance(result, SpeakerDiarizationResult)
    assert result.source_audio is clip
    assert result.model_name == "speaker-diarization-community-1"
    assert result.speakers == ("SPEAKER_00", "SPEAKER_01")
    assert [(turn.start_seconds, turn.end_seconds, turn.speaker_id) for turn in result.turns] == [
        (0.1, 0.3, "SPEAKER_00"), (0.2, 0.4, "SPEAKER_01")]
    assert len(result.exclusive_turns) == 2
    assert result.turns[0].end_sample > result.turns[1].start_sample
    assert result.exclusive_turns[0].end_sample <= result.exclusive_turns[1].start_sample
    audio, options = model.calls[0]
    assert options == {}
    assert audio["sample_rate"] == 16000
    assert audio["waveform"].data.shape == (1, 8000)
    assert audio["waveform"].data.dtype == np.float32
    assert np.isfinite(audio["waveform"].data).all()


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000, 96000, 192000])
def test_resampling_preserves_original_sample_timestamps(fake_torch, rate):
    clip = make_clip(rate=rate)
    result = PyannoteDiarizationAdapter(FakePipeline()).diarize(clip)
    assert result.turns[0].sample_rate == rate
    assert result.turns[0].start_sample == round(0.1 * rate)
    assert result.turns[0].end_sample == round(0.3 * rate)
    assert result.exclusive_turns[1].end_sample == round(0.4 * rate)


@pytest.mark.parametrize("num", [1, 2, 3, 8])
def test_num_speakers_is_forwarded_when_set(fake_torch, num):
    model = FakePipeline()
    PyannoteDiarizationAdapter(model).diarize(make_clip(), num_speakers=num)
    assert model.calls[0][1] == {"num_speakers": num}


@pytest.mark.parametrize("num", [0, -1, 9, True, 1.0, "2", [], {}, object()])
def test_invalid_num_speakers_is_rejected_without_inference(num, fake_torch):
    model = FakePipeline()
    with pytest.raises(ValueError, match="num_speakers"):
        PyannoteDiarizationAdapter(model).diarize(make_clip(), num_speakers=num)
    assert model.calls == []


def test_silent_input_skips_model(fake_torch):
    clip = AudioClip(np.zeros(8000, dtype=np.float32), 16000)
    model = FakePipeline()
    result = PyannoteDiarizationAdapter(model).diarize(clip)
    assert result.turns == () and result.exclusive_turns == () and result.speakers == ()
    assert not model.calls


@pytest.mark.parametrize("amplitude", [0.0, 1e-10, 1e-9])
def test_near_silence_skips_model(fake_torch, amplitude):
    model = FakePipeline()
    clip = make_clip(amplitude=amplitude)
    assert PyannoteDiarizationAdapter(model).diarize(clip).turns == ()
    assert not model.calls


@pytest.mark.parametrize("bad", [None, True, "audio", 4, [], np.zeros(5)])
def test_invalid_audio_types_are_rejected(bad, fake_torch):
    with pytest.raises(TypeError, match="AudioClip"):
        PyannoteDiarizationAdapter(FakePipeline()).diarize(bad)


@pytest.mark.parametrize("channels", [2, 3, 8])
def test_multichannel_audio_is_rejected(fake_torch, channels):
    with pytest.raises(ValueError, match="mono"):
        PyannoteDiarizationAdapter(FakePipeline()).diarize(make_clip(channels=channels))


def test_long_audio_is_rejected(fake_torch):
    with pytest.raises(ValueError, match="30 seconds"):
        PyannoteDiarizationAdapter(FakePipeline()).diarize(make_clip(rate=8000, duration=30.1))


def test_amplitude_out_of_range_rejected(fake_torch):
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        PyannoteDiarizationAdapter(FakePipeline()).diarize(make_clip(amplitude=1.2))


def test_resampled_amplitude_out_of_range_rejected(fake_torch, monkeypatch):
    original = diarization.resample_audio

    def amplified(clip, rate):
        prepared = original(clip, rate)
        return AudioClip(prepared.samples * 10, rate)

    monkeypatch.setattr(diarization, "resample_audio", amplified)
    with pytest.raises(ValueError, match="Resampled audio"):
        PyannoteDiarizationAdapter(FakePipeline()).diarize(make_clip(rate=44100, amplitude=0.2))


def test_input_samples_are_never_modified(fake_torch):
    clip = make_clip(rate=44100)
    before = clip.samples.copy()
    model = FakePipeline()
    PyannoteDiarizationAdapter(model).diarize(clip)
    np.testing.assert_array_equal(clip.samples, before)
    assert not np.shares_memory(model.calls[0][0]["waveform"].data, clip.samples)


def test_inference_failure_is_sanitized(fake_torch):
    model = FakePipeline(error=RuntimeError("private token should not appear"))
    with pytest.raises(PyannoteDiarizationError, match="inference failed") as exc:
        PyannoteDiarizationAdapter(model).diarize(make_clip())
    assert "private token" not in str(exc.value)


def test_missing_torch_is_reported(monkeypatch):
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(PyannoteDiarizationError, match="PyTorch"):
        PyannoteDiarizationAdapter(FakePipeline()).diarize(make_clip())


def test_unsupported_output_is_rejected(fake_torch):
    model = FakePipeline()
    original = model.__class__.__call__

    def invalid(self, audio, **options):
        return object()

    try:
        model.__class__.__call__ = invalid
        with pytest.raises(PyannoteDiarizationError, match="unsupported output"):
            PyannoteDiarizationAdapter(model).diarize(make_clip())
    finally:
        model.__class__.__call__ = original


@pytest.mark.parametrize("normal, exclusive", [
    (None, ()), ((), None),
    (((0.1, 0.2, "SPEAKER_00"),), ((0.1, 0.2, "UNKNOWN"),)),
])
def test_invalid_output_annotations_or_labels_rejected(fake_torch, normal, exclusive):
    model = FakePipeline(normal=normal, exclusive=exclusive)
    with pytest.raises(PyannoteDiarizationError):
        PyannoteDiarizationAdapter(model).diarize(make_clip())


@pytest.mark.parametrize("start, end", [
    (float("nan"), 0.3), (0.1, float("inf")), (-0.1, 0.3),
    (0.1, 0.7), (0.4, 0.4), (0.4, 0.1), (True, 0.2),
    ("0.1", 0.2), (0.1, None), (0.1, 0.10000000001),
])
def test_invalid_turn_boundaries_rejected(fake_torch, start, end):
    model = FakePipeline(normal=((start, end, "SPEAKER_00"),), exclusive=())
    with pytest.raises(PyannoteDiarizationError, match="malformed speaker turns"):
        PyannoteDiarizationAdapter(model).diarize(make_clip())


@pytest.mark.parametrize("speaker", [None, "", "  ", " SPEAKER_00", "x" * 81, "\n", True])
def test_invalid_speaker_labels_rejected(fake_torch, speaker):
    model = FakePipeline(normal=((0.1, 0.3, speaker),), exclusive=())
    with pytest.raises(PyannoteDiarizationError, match="malformed speaker turns"):
        PyannoteDiarizationAdapter(model).diarize(make_clip())


def test_exclusive_annotation_must_not_overlap(fake_torch):
    model = FakePipeline(exclusive=((0.1, 0.25, "SPEAKER_00"), (0.2, 0.4, "SPEAKER_01")))
    with pytest.raises(PyannoteDiarizationError, match="overlapping"):
        PyannoteDiarizationAdapter(model).diarize(make_clip())


def test_unordered_turns_are_sorted_by_time(fake_torch):
    model = FakePipeline(normal=((0.3, 0.4, "SPEAKER_01"), (0.1, 0.2, "SPEAKER_00")), exclusive=())
    result = PyannoteDiarizationAdapter(model).diarize(make_clip())
    assert [turn.speaker_id for turn in result.turns] == ["SPEAKER_00", "SPEAKER_01"]


def test_boundaries_one_sample_outside_are_clamped(fake_torch):
    model = FakePipeline(normal=((-0.00005, 0.50005, "SPEAKER_00"),), exclusive=())
    result = PyannoteDiarizationAdapter(model).diarize(make_clip())
    assert result.turns[0].start_sample == 0
    assert result.turns[0].end_sample == 8000


@pytest.mark.parametrize("start, end, rate, speaker", [
    (-1, 1, 8000, "SPEAKER_00"), (0, 0, 8000, "SPEAKER_00"),
    (1, 0, 8000, "SPEAKER_00"), (True, 1, 8000, "SPEAKER_00"),
    (0.1, 2, 8000, "SPEAKER_00"), (0, 1, 0, "SPEAKER_00"),
    (0, 1, 8000, ""), (0, 1, 8000, " x"),
])
def test_speaker_turn_rejects_invalid_metadata(start, end, rate, speaker):
    with pytest.raises((TypeError, ValueError)):
        SpeakerTurn(start, end, rate, speaker)


def test_speaker_turn_time_properties():
    turn = SpeakerTurn(800, 2400, 8000, "SPEAKER_00")
    assert turn.start_seconds == 0.1 and turn.end_seconds == 0.3
    with pytest.raises(FrozenInstanceError):
        turn.speaker_id = "changed"


@pytest.mark.parametrize("field,value", [
    ("source_audio", "bad"), ("turns", []), ("turns", (None,)),
    ("exclusive_turns", []), ("model_name", "wrong"),
])
def test_result_rejects_invalid_fields(field, value):
    data = dict(source_audio=make_clip(), turns=(), exclusive_turns=())
    data[field] = value
    with pytest.raises((TypeError, ValueError)):
        SpeakerDiarizationResult(**data)


def test_result_rejects_out_of_bounds_wrong_rate_and_out_of_order_turns():
    clip = make_clip()
    for turns in [(SpeakerTurn(0, 100, 8000, "SPEAKER_00"),),
                  (SpeakerTurn(0, clip.frames + 1, 16000, "SPEAKER_00"),),
                  (SpeakerTurn(100, 200, 16000, "SPEAKER_00"), SpeakerTurn(50, 100, 16000, "SPEAKER_01"))]:
        with pytest.raises(ValueError):
            SpeakerDiarizationResult(clip, turns, ())


def test_result_rejects_exclusive_overlap_and_unknown_speaker():
    clip = make_clip()
    a = SpeakerTurn(100, 500, 16000, "SPEAKER_00")
    b = SpeakerTurn(300, 600, 16000, "SPEAKER_00")
    with pytest.raises(ValueError, match="overlap"):
        SpeakerDiarizationResult(clip, (a,), (a, b))
    with pytest.raises(ValueError, match="new speaker"):
        SpeakerDiarizationResult(clip, (a,), (SpeakerTurn(100, 200, 16000, "OTHER"),))


def test_result_is_frozen():
    result = SpeakerDiarizationResult(make_clip(), (), ())
    with pytest.raises(FrozenInstanceError):
        result.turns = ()


def test_pipeline_must_be_callable():
    for value in [None, {}, [], "model", object()]:
        with pytest.raises(TypeError, match="callable"):
            PyannoteDiarizationAdapter(value)


@pytest.mark.parametrize("token", [None, "", " ", "abc ", [], 42, True])
def test_load_requires_valid_token_without_importing_models(monkeypatch, token):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="HF_TOKEN"):
        PyannoteDiarizationAdapter.load(token=token)


@pytest.mark.parametrize("device", [None, 1, False, "mps", "GPU", "", "cuda:0"])
def test_load_rejects_invalid_devices(device, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy")
    with pytest.raises((TypeError, ValueError), match="device"):
        PyannoteDiarizationAdapter.load(device=device)


def install_fake_pyannote(monkeypatch, loader):
    library = ModuleType("pyannote")
    api = ModuleType("pyannote.audio")
    api.Pipeline = SimpleNamespace(from_pretrained=loader)
    library.audio = api
    monkeypatch.setitem(sys.modules, "pyannote", library)
    monkeypatch.setitem(sys.modules, "pyannote.audio", api)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_load_uses_real_api_shape_without_leaking_token(monkeypatch, fake_torch, device):
    backend = FakePipeline()
    calls = []

    def loader(source, *, token):
        calls.append((source, token))
        return backend

    install_fake_pyannote(monkeypatch, loader)
    monkeypatch.setenv("HF_TOKEN", "secret")
    adapter = PyannoteDiarizationAdapter.load(device=device)
    assert isinstance(adapter, PyannoteDiarizationAdapter)
    assert calls == [("pyannote/speaker-diarization-community-1", "secret")]
    assert backend.devices == [device]
    assert "secret" not in repr(adapter)


def test_explicit_token_overrides_environment(monkeypatch, fake_torch):
    calls = []

    def loader(source, *, token):
        calls.append(token)
        return FakePipeline()

    install_fake_pyannote(monkeypatch, loader)
    monkeypatch.setenv("HF_TOKEN", "from_environment")
    PyannoteDiarizationAdapter.load(token="explicit")
    assert calls == ["explicit"]


@pytest.mark.parametrize("kind", ["returns_none", "raises"])
def test_load_failure_is_sanitized(monkeypatch, fake_torch, kind):
    def loader(source, *, token):
        if kind == "raises":
            raise RuntimeError("very-secret")
        return None

    install_fake_pyannote(monkeypatch, loader)
    with pytest.raises(PyannoteDiarizationError, match="Unable to load") as exc:
        PyannoteDiarizationAdapter.load(token="very-secret")
    assert "very-secret" not in str(exc.value)


def test_load_missing_dependency_reports_action(monkeypatch):
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pyannote.audio":
            raise ImportError("No pyannote")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(PyannoteDiarizationError, match="Install compatible"):
        PyannoteDiarizationAdapter.load(token="demo")


def test_annotation_without_itertracks_is_rejected():
    with pytest.raises(PyannoteDiarizationError, match="annotation"):
        PyannoteDiarizationAdapter._extract_turns(None, make_clip(), exclusive=False)
