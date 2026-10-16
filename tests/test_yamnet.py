"""YAMNet adapter contract tests using a controlled TensorFlow Hub backend.

Tests cover input safety, label-map integrity, class-score aggregation, stable
ranking, output validation and optional dependency isolation. They do not
measure accuracy or execute the pretrained YAMNet checkpoint.
"""

from __future__ import annotations

import builtins
import csv
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

import audio_lab.yamnet as yamnet
from audio_lab.audio import AudioClip
from audio_lab.yamnet import YAMNetAdapter, YAMNetClassificationResult, YAMNetError, YAMNetLabelScore


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def numpy(self):
        return self.data


class FakeModel:
    def __init__(self, output=None, *, error=None, class_map=None):
        self.output = output
        self.error = error
        self.class_map = class_map
        self.calls = []

    def __call__(self, waveform):
        self.calls.append(waveform)
        if self.error is not None:
            raise self.error
        return self.output if self.output is not None else make_output()

    def class_map_path(self):
        return FakeTensor(self.class_map)


def labels():
    return tuple(f"Event {index:03d}" for index in range(521))


def make_output(*, patches=3, classes=521, score=0.1, index=5, special=0.75):
    scores = np.full((patches, classes), score, dtype=np.float32)
    if classes > index and patches > 0:
        scores[:, index] = special
    embeddings = np.zeros((patches, 1024), dtype=np.float32)
    spectrogram = np.zeros((max(1, patches * 48), 64), dtype=np.float32)
    return FakeTensor(scores), FakeTensor(embeddings), FakeTensor(spectrogram)


def make_clip(*, rate=16_000, duration=1.0, amplitude=0.2, channels=1):
    frames = max(1, round(rate * duration))
    time = np.arange(frames) / rate
    signal = amplitude * np.sin(2 * np.pi * 211 * time)
    return AudioClip(np.repeat(signal[:, None], channels, axis=1), rate)


@pytest.fixture
def adapter():
    model = FakeModel()
    return YAMNetAdapter(model, labels()), model


@pytest.mark.parametrize("rate", [8_000, 11_025, 16_000, 22_050, 44_100, 48_000, 96_000, 192_000])
def test_supported_sample_rates(rate, adapter):
    classifier, backend = adapter
    clip = make_clip(rate=rate, duration=0.16)
    result = classifier.classify(clip)
    assert result.source_audio is clip
    assert result.top_prediction.class_index == 5
    assert backend.calls[0].dtype == np.float32
    assert backend.calls[0].ndim == 1
    assert backend.calls[0].shape == (round(clip.frames * 16_000 / rate),)
    assert backend.calls[0].flags.writeable


@pytest.mark.parametrize("count", [1, 2, 3, 5, 10, 20])
def test_top_k_controls_result_length(count, adapter):
    classifier, _ = adapter
    result = classifier.classify(make_clip(), top_k=count)
    assert len(result.predictions) == count
    assert result.predictions[0].label == "Event 005"
    assert result.patch_count == 3
    expected = [5] + [index for index in range(521) if index != 5][:count - 1]
    assert [entry.class_index for entry in result.predictions] == expected


@pytest.mark.parametrize("top_k", [None, False, True, 0, -1, 21, 521, 1.0, "5", [], {}, np.int64(2)])
def test_invalid_top_k_is_rejected(top_k, adapter):
    classifier, backend = adapter
    with pytest.raises(ValueError, match="top_k"):
        classifier.classify(make_clip(), top_k=top_k)
    assert not backend.calls


@pytest.mark.parametrize("clip", [None, False, True, 3, "audio", [], {}, object()])
def test_invalid_audio_input_is_rejected(clip, adapter):
    classifier, backend = adapter
    with pytest.raises(TypeError, match="AudioClip"):
        classifier.classify(clip)
    assert not backend.calls


def test_stereo_is_rejected(adapter):
    classifier, backend = adapter
    with pytest.raises(ValueError, match="mono"):
        classifier.classify(make_clip(channels=2))
    assert not backend.calls


def test_long_audio_is_rejected(adapter):
    classifier, backend = adapter
    with pytest.raises(ValueError, match="30 seconds"):
        classifier.classify(make_clip(rate=8000, duration=30.1))
    assert not backend.calls


@pytest.mark.parametrize("amplitude", [1.01, 2.0, 10.0])
def test_out_of_range_audio_is_rejected(amplitude, adapter):
    classifier, backend = adapter
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        classifier.classify(make_clip(amplitude=amplitude))
    assert not backend.calls


@pytest.mark.parametrize("amplitude", [0.0, 1e-10, 1e-9])
def test_silence_is_rejected(amplitude, adapter):
    classifier, backend = adapter
    with pytest.raises(ValueError, match="silent"):
        classifier.classify(make_clip(amplitude=amplitude))
    assert not backend.calls


def test_resampling_overshoot_is_not_silently_clipped(adapter, monkeypatch):
    classifier, backend = adapter
    clip = make_clip(rate=48_000)
    monkeypatch.setattr(yamnet, "resample_audio", lambda _clip, _rate: AudioClip(np.full((16_000, 1), 1.1), 16_000))
    with pytest.raises(ValueError, match="Resampled"):
        classifier.classify(clip)
    assert not backend.calls


def test_model_error_is_wrapped(adapter):
    classifier, backend = adapter
    backend.error = RuntimeError("private failure")
    with pytest.raises(YAMNetError, match="inference failed"):
        classifier.classify(make_clip())


@pytest.mark.parametrize("output", [None, (), (1,), (1, 2), (1, 2, 3, 4), [1, 2], "bad", {}, 42])
def test_invalid_output_structure(output, adapter):
    classifier, backend = adapter
    backend.output = output if output is not None else "bad"
    with pytest.raises(YAMNetError, match="scores, embeddings"):
        classifier.classify(make_clip())


@pytest.mark.parametrize("name", ["scores", "embeddings", "spectrogram"])
@pytest.mark.parametrize("bad_value", [None, [], 2, "bad", object()])
def test_unreadable_tensor_is_rejected(name, bad_value, adapter):
    classifier, backend = adapter
    parts = list(make_output())
    parts[{"scores": 0, "embeddings": 1, "spectrogram": 2}[name]] = bad_value
    backend.output = tuple(parts)
    with pytest.raises(YAMNetError, match="readable TensorFlow tensor"):
        classifier.classify(make_clip())


@pytest.mark.parametrize("name", ["scores", "embeddings", "spectrogram"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_tensors_are_rejected(name, bad_value, adapter):
    classifier, backend = adapter
    parts = list(make_output())
    index = {"scores": 0, "embeddings": 1, "spectrogram": 2}[name]
    array = parts[index].data.copy()
    array.flat[0] = bad_value
    parts[index] = FakeTensor(array)
    backend.output = tuple(parts)
    with pytest.raises(YAMNetError, match="finite floating-point"):
        classifier.classify(make_clip())


@pytest.mark.parametrize("name", ["scores", "embeddings", "spectrogram"])
@pytest.mark.parametrize("kind", [np.int32, np.bool_])
def test_nonfloat_tensor_is_rejected(name, kind, adapter):
    classifier, backend = adapter
    parts = list(make_output())
    index = {"scores": 0, "embeddings": 1, "spectrogram": 2}[name]
    parts[index] = FakeTensor(parts[index].data.astype(kind))
    backend.output = tuple(parts)
    with pytest.raises(YAMNetError, match="finite floating-point"):
        classifier.classify(make_clip())


@pytest.mark.parametrize("shape", [(0, 521), (3, 520), (3, 522), (521,), (1, 1, 521)])
def test_invalid_score_shape(shape, adapter):
    classifier, backend = adapter
    parts = list(make_output())
    parts[0] = FakeTensor(np.zeros(shape, dtype=np.float32))
    backend.output = tuple(parts)
    with pytest.raises(YAMNetError, match="scores must have shape"):
        classifier.classify(make_clip())


@pytest.mark.parametrize("score", [-0.001, 1.001, 1e5])
def test_out_of_range_scores_are_rejected(score, adapter):
    classifier, backend = adapter
    output = list(make_output())
    values = output[0].data.copy()
    values[0, 0] = score
    output[0] = FakeTensor(values)
    backend.output = tuple(output)
    with pytest.raises(YAMNetError, match=r"\[0, 1\]"):
        classifier.classify(make_clip())


@pytest.mark.parametrize("shape", [(0, 1024), (2, 1024), (3, 1023), (3, 1025), (3072,), (3, 1024, 1)])
def test_invalid_embedding_shape(shape, adapter):
    classifier, backend = adapter
    parts = list(make_output())
    parts[1] = FakeTensor(np.zeros(shape, dtype=np.float32))
    backend.output = tuple(parts)
    with pytest.raises(YAMNetError, match="embeddings must have shape"):
        classifier.classify(make_clip())


@pytest.mark.parametrize("shape", [(0, 64), (144, 63), (144, 65), (9216,), (144, 64, 1)])
def test_invalid_spectrogram_shape(shape, adapter):
    classifier, backend = adapter
    parts = list(make_output())
    parts[2] = FakeTensor(np.zeros(shape, dtype=np.float32))
    backend.output = tuple(parts)
    with pytest.raises(YAMNetError, match="spectrogram must have shape"):
        classifier.classify(make_clip())


def test_average_patch_scores_without_softmax(adapter):
    classifier, backend = adapter
    parts = list(make_output(patches=2))
    values = np.zeros((2, 521), dtype=np.float32)
    values[:, 1] = (0.2, 0.8)
    values[:, 2] = (0.6, 0.6)
    parts[0] = FakeTensor(values)
    parts[1] = FakeTensor(np.zeros((2, 1024), dtype=np.float32))
    backend.output = tuple(parts)
    result = classifier.classify(make_clip(), top_k=2)
    assert [item.class_index for item in result.predictions] == [2, 1]
    assert result.predictions[0].score == pytest.approx(0.6)
    assert result.predictions[1].score == pytest.approx(0.5)
    assert sum(item.score for item in result.predictions) == pytest.approx(1.1)
    assert result.patch_count == 2


def test_tie_breaking_preserves_official_label_order(adapter):
    classifier, backend = adapter
    parts = list(make_output())
    parts[0] = FakeTensor(np.full((3, 521), 0.4, dtype=np.float32))
    backend.output = tuple(parts)
    result = classifier.classify(make_clip(), top_k=6)
    assert [entry.class_index for entry in result.predictions] == list(range(6))


def test_source_audio_is_unchanged(adapter):
    classifier, backend = adapter
    clip = make_clip()
    original = clip.samples.copy()
    result = classifier.classify(clip)
    np.testing.assert_array_equal(clip.samples, original)
    assert result.source_audio is clip
    assert not clip.samples.flags.writeable
    assert not np.shares_memory(backend.calls[0], clip.samples)


@pytest.mark.parametrize("class_names", [None, "label", [], (), labels()[:-1], labels() + ("extra",), [*labels()[:-1], "Event 000"], [*labels()[:-1], ""], [*labels()[:-1], " bad"], [*labels()[:-1], "bad\nname"], [*labels()[:-1], 3]])
def test_invalid_class_labels_rejected(class_names):
    with pytest.raises(ValueError, match="class_names"):
        YAMNetAdapter(FakeModel(), class_names)


@pytest.mark.parametrize("backend", [None, True, False, 3, "bad", [], {}, object()])
def test_non_callable_backend_rejected(backend):
    with pytest.raises(TypeError, match="callable"):
        YAMNetAdapter(backend, labels())


def test_class_names_are_immutable(adapter):
    classifier, _ = adapter
    assert classifier.class_names == labels()
    with pytest.raises(TypeError):
        classifier.class_names[0] = "changed"
    with pytest.raises(AttributeError):
        classifier.class_names = ()


@pytest.mark.parametrize("index", [-1, 521, True, None, 1.0, "0", np.int32(2)])
def test_invalid_label_index(index):
    with pytest.raises(ValueError, match="class_index"):
        YAMNetLabelScore(index, "Speech", 0.5)


@pytest.mark.parametrize("label", [None, "", "  ", " leading", "trailing "])
def test_invalid_prediction_label(label):
    with pytest.raises(ValueError, match="label"):
        YAMNetLabelScore(1, label, 0.5)


@pytest.mark.parametrize("score", [None, False, True, -0.1, 1.1, 3, np.float32(0.5), float("nan"), float("inf")])
def test_invalid_prediction_score(score):
    with pytest.raises(ValueError, match="score"):
        YAMNetLabelScore(1, "Speech", score)


@pytest.mark.parametrize("predictions", [None, [], (), [YAMNetLabelScore(0, "Speech", 0.5)], (None,), (YAMNetLabelScore(0, "Speech", 0.5),) * 21])
def test_invalid_predictions_collection(predictions):
    with pytest.raises((ValueError, TypeError)):
        YAMNetClassificationResult(make_clip(), predictions, 1)


def test_duplicate_result_indices_rejected():
    predictions = (YAMNetLabelScore(1, "Speech", 0.9), YAMNetLabelScore(1, "Music", 0.8))
    with pytest.raises(ValueError, match="indices"):
        YAMNetClassificationResult(make_clip(), predictions, 1)


def test_duplicate_result_labels_rejected():
    predictions = (YAMNetLabelScore(1, "Speech", 0.9), YAMNetLabelScore(2, "Speech", 0.8))
    with pytest.raises(ValueError, match="labels"):
        YAMNetClassificationResult(make_clip(), predictions, 1)


def test_wrong_result_order_rejected():
    predictions = (YAMNetLabelScore(1, "Speech", 0.1), YAMNetLabelScore(2, "Music", 0.8))
    with pytest.raises(ValueError, match="descending"):
        YAMNetClassificationResult(make_clip(), predictions, 1)


@pytest.mark.parametrize("patch_count", [0, -1, True, None, 1.0, np.int64(2)])
def test_invalid_patch_count_rejected(patch_count):
    with pytest.raises(ValueError, match="patch_count"):
        YAMNetClassificationResult(make_clip(), (YAMNetLabelScore(1, "Speech", 0.5),), patch_count)


def test_invalid_result_audio_and_model_name_rejected():
    prediction = (YAMNetLabelScore(1, "Speech", 0.5),)
    with pytest.raises(TypeError, match="source_audio"):
        YAMNetClassificationResult(None, prediction, 2)
    with pytest.raises(ValueError, match="model name"):
        YAMNetClassificationResult(make_clip(), prediction, 2, model_name="other")


def test_result_objects_are_immutable(adapter):
    classifier, _ = adapter
    result = classifier.classify(make_clip())
    with pytest.raises(FrozenInstanceError):
        result.patch_count = 99
    with pytest.raises(FrozenInstanceError):
        result.predictions[0].score = 0.0
    assert isinstance(result.predictions, tuple)


def write_class_map(path, *, headers=("index", "display_name"), count=521):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(count):
            writer.writerow((index, f"Event {index:03d}"))


def test_load_real_api_contract_from_fake_tensorflow_hub(tmp_path, monkeypatch):
    path = tmp_path / "class-map.csv"
    write_class_map(path)
    backend = FakeModel(class_map=str(path).encode("utf-8"))
    hub = ModuleType("tensorflow_hub")
    urls = []
    hub.load = lambda url: (urls.append(url), backend)[1]
    monkeypatch.setitem(sys.modules, "tensorflow_hub", hub)
    classifier = YAMNetAdapter.load()
    assert urls == ["https://tfhub.dev/google/yamnet/1"]
    assert classifier.class_names == labels()
    assert classifier.classify(make_clip()).top_prediction.label == "Event 005"


def test_load_accepts_unicode_string_class_map_path(tmp_path, monkeypatch):
    path = tmp_path / "classes.csv"
    write_class_map(path)
    hub = ModuleType("tensorflow_hub")
    hub.load = lambda url: FakeModel(class_map=str(path))
    monkeypatch.setitem(sys.modules, "tensorflow_hub", hub)
    assert len(YAMNetAdapter.load().class_names) == 521


@pytest.mark.parametrize("case", ["missing", "wrong_column", "short", "duplicate", "broken_label", "invalid_utf8", "invalid_path", "no_class_map", "hub_error"])
def test_load_rejects_bad_checkpoint_metadata(tmp_path, monkeypatch, case):
    path = tmp_path / "bad.csv"
    write_class_map(path, headers=("index", "value") if case == "wrong_column" else ("index", "display_name"), count=520 if case == "short" else 521)
    if case == "duplicate":
        path.write_text('index,display_name\n' + ''.join(f'{i},Same\n' for i in range(521)), encoding="utf-8")
    if case == "broken_label":
        path.write_text('index,display_name\n' + ''.join(f'{i},{"" if i == 8 else f"Event {i:03d}"}\n' for i in range(521)), encoding="utf-8")
    if case == "invalid_utf8":
        path.write_bytes(b'\xff\xfe')
    backend = FakeModel(class_map=(tmp_path / "missing.csv" if case == "missing" else path))
    if case != "invalid_path":
        backend.class_map = str(backend.class_map).encode("utf-8")
    else:
        backend.class_map = 32
    if case == "no_class_map":
        backend.class_map_path = None
    hub = ModuleType("tensorflow_hub")
    def load(url):
        if case == "hub_error":
            raise RuntimeError("backend network failed")
        return backend
    hub.load = load
    monkeypatch.setitem(sys.modules, "tensorflow_hub", hub)
    with pytest.raises(YAMNetError, match="Unable to load"):
        YAMNetAdapter.load()


def test_missing_tensorflow_hub_dependency_is_reported(monkeypatch):
    original_import = builtins.__import__
    def fail_import(name, *args, **kwargs):
        if name == "tensorflow_hub":
            raise ImportError("test missing dependency")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fail_import)
    with pytest.raises(YAMNetError, match="Install compatible"):
        YAMNetAdapter.load()


def test_module_does_not_import_tensorflow_on_import(monkeypatch):
    assert "tensorflow" not in yamnet.__dict__
    assert "tensorflow_hub" not in yamnet.__dict__


def test_model_error_not_leaked_to_user(adapter):
    classifier, backend = adapter
    backend.error = RuntimeError("SECRET BACKEND DETAIL")
    with pytest.raises(YAMNetError, match="YAMNet inference failed") as exc_info:
        classifier.classify(make_clip())
    assert "SECRET" not in str(exc_info.value)
