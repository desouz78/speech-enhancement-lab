"""Regression tests for temporal audio segmentation and feature extraction."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
from math import ceil

import numpy as np
import pytest

from audio_lab.audio import AudioClip
from audio_lab.features import AudioFeatures, extract_features
from audio_lab.temporal import TemporalFeatureFrame, extract_temporal_features


def make_clip(frames=32_000, sample_rate=16_000, channels=1, seed=17):
    """Create deterministic, independently varying audio channels."""
    samples = np.random.default_rng(seed).uniform(-0.8, 0.8, (frames, channels))
    return AudioClip(samples.astype(np.float32), sample_rate)


def test_default_window_duration_and_metadata():
    clip = make_clip(frames=40_000, channels=2)
    result = extract_temporal_features(clip)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert [(frame.start_seconds, frame.end_seconds) for frame in result] == [
        (0.0, 1.0), (1.0, 2.0), (2.0, 2.5)
    ]
    assert all(isinstance(frame, TemporalFeatureFrame) for frame in result)
    assert all(isinstance(frame.features, AudioFeatures) for frame in result)
    assert all(frame.features.channels == 2 for frame in result)
    assert all(frame.features.sample_rate == clip.sample_rate for frame in result)


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 44_100, 48_000, 192_000])
@pytest.mark.parametrize("duration", [0.1, 0.333, 1.0, 10.0])
def test_sample_aligned_coverage_across_rates_and_windows(sample_rate, duration, monkeypatch):
    window_frames = round(duration * sample_rate)
    total_frames = 2 * window_frames + 37
    clip = make_clip(total_frames, sample_rate, channels=2)
    captured = []
    original_extract = extract_features

    def capture(segment):
        captured.append(segment.samples.copy())
        return original_extract(segment)

    monkeypatch.setattr("audio_lab.temporal.extract_features", capture)
    result = extract_temporal_features(clip, duration)

    assert len(result) == 3
    assert [block.shape[0] for block in captured] == [window_frames, window_frames, 37]
    np.testing.assert_array_equal(np.concatenate(captured, axis=0), clip.samples)
    for index, frame in enumerate(result):
        start = index * window_frames
        end = min(start + window_frames, total_frames)
        assert frame.start_seconds == pytest.approx(start / sample_rate, abs=1e-12)
        assert frame.end_seconds == pytest.approx(end / sample_rate, abs=1e-12)
        assert frame.features.duration_seconds == pytest.approx((end - start) / sample_rate)
    assert result[-1].end_seconds == pytest.approx(clip.duration_seconds)


@pytest.mark.parametrize("frames", [1, 2, 3, 7, 799, 800, 801, 1599, 1600, 1601])
def test_short_and_boundary_length_recordings(frames):
    clip = make_clip(frames, sample_rate=8_000)
    result = extract_temporal_features(clip, 0.1)
    assert len(result) == ceil(frames / 800)
    assert result[0].start_seconds == 0
    assert result[-1].end_seconds == pytest.approx(frames / 8_000)
    assert sum(round(f.features.duration_seconds * 8_000) for f in result) == frames
    assert all(f.start_seconds < f.end_seconds for f in result)


def test_exact_multiple_has_no_extra_window():
    clip = make_clip(frames=48_000)
    result = extract_temporal_features(clip, 1.0)
    assert len(result) == 3
    assert all(frame.features.duration_seconds == pytest.approx(1.0) for frame in result)
    assert result[-1].end_seconds == pytest.approx(3.0)


def test_segment_features_equal_standalone_reference():
    clip = make_clip(frames=23_123, channels=2)
    window_frames = round(0.31 * clip.sample_rate)
    result = extract_temporal_features(clip, 0.31)
    for index, frame in enumerate(result):
        start = index * window_frames
        end = min(start + window_frames, clip.frames)
        expected = extract_features(AudioClip(clip.samples[start:end], clip.sample_rate))
        assert frame.features == expected


def test_nonstationary_tones_keep_temporal_contrast():
    sample_rate = 16_000
    t = np.arange(sample_rate) / sample_rate
    low = 0.5 * np.sin(2 * np.pi * 250 * t)
    high = 0.5 * np.sin(2 * np.pi * 2_000 * t)
    clip = AudioClip(np.concatenate((low, high)), sample_rate)
    result = extract_temporal_features(clip, 1.0)
    assert len(result) == 2
    assert result[0].features.spectral_centroid_hz == pytest.approx(250, abs=8)
    assert result[1].features.spectral_centroid_hz == pytest.approx(2_000, abs=8)
    assert result[1].features.spectral_centroid_hz > result[0].features.spectral_centroid_hz


def test_silent_segments_remain_well_defined():
    samples = np.concatenate((np.zeros(16_000), np.ones(16_000) * 0.25))
    result = extract_temporal_features(AudioClip(samples, 16_000), 1.0)
    assert len(result) == 2
    assert result[0].features.rms_dbfs == float("-inf")
    assert result[0].features.peak_amplitude == 0
    assert result[0].features.spectral_flatness == 0
    assert result[1].features.rms_dbfs == pytest.approx(20 * np.log10(0.25))
    assert result[1].features.spectral_centroid_hz == 0


def test_amplitude_changes_are_preserved_across_windows():
    clip = AudioClip(np.concatenate((np.full(16_000, 0.125), np.full(16_000, 0.5))), 16_000)
    a, b = extract_temporal_features(clip)
    assert a.features.rms_dbfs == pytest.approx(20 * np.log10(0.125))
    assert b.features.rms_dbfs == pytest.approx(20 * np.log10(0.5))
    assert b.features.rms_dbfs - a.features.rms_dbfs == pytest.approx(20 * np.log10(4))


def test_stereo_antiphase_is_not_silently_downmixed():
    time = np.arange(16_000, dtype=np.float64) / 16_000
    mono = (0.4 * np.sin(2 * np.pi * 1_000 * time)).astype(np.float32)
    clip = AudioClip(np.column_stack((mono, -mono)), 16_000)
    result = extract_temporal_features(clip)
    assert len(result) == 1
    assert result[0].features.channels == 2
    assert result[0].features.rms_dbfs > -20
    assert result[0].features.spectral_centroid_hz == pytest.approx(1_000, abs=8)


def test_no_cross_window_zero_crossings():
    # Each half has constant sign; their boundary is intentionally excluded.
    clip = AudioClip(np.concatenate((np.ones(16_000), -np.ones(16_000))), 16_000)
    result = extract_temporal_features(clip)
    assert len(result) == 2
    assert all(frame.features.zero_crossing_rate == 0 for frame in result)
    assert extract_features(clip).zero_crossing_rate > 0


def test_fractional_duration_rounds_to_nearest_sample():
    clip = make_clip(frames=17_001, sample_rate=16_000)
    duration = 0.333
    window_frames = round(duration * 16_000)
    result = extract_temporal_features(clip, duration)
    assert result[0].end_seconds == pytest.approx(window_frames / 16_000)
    assert result[1].start_seconds == pytest.approx(window_frames / 16_000)
    assert result[-1].end_seconds == pytest.approx(clip.duration_seconds)


@pytest.mark.parametrize(
    "duration", [0.1, 10.0, 1, np.float32(0.5), np.float64(0.5), Fraction(1, 2)]
)
def test_valid_window_types_and_boundaries(duration):
    clip = make_clip(frames=8_001, sample_rate=8_000)
    result = extract_temporal_features(clip, duration)
    assert len(result) == ceil(clip.frames / round(float(duration) * clip.sample_rate))
    assert result[-1].end_seconds == pytest.approx(clip.duration_seconds)


@pytest.mark.parametrize(
    "duration", [None, True, False, np.bool_(True), "1.0", b"1.0", 1j, [], {}, np.array(1.0)]
)
def test_invalid_window_types_rejected(duration):
    clip = make_clip(frames=800, sample_rate=8_000)
    with pytest.raises(TypeError, match="window_seconds must be a real number"):
        extract_temporal_features(clip, duration)


@pytest.mark.parametrize(
    "duration",
    [0, -1, -0.1, 0.0999, 10.0001, 100, float("nan"), float("inf"), float("-inf"), 10 ** 400],
)
def test_invalid_window_values_rejected(duration):
    clip = make_clip(frames=800, sample_rate=8_000)
    with pytest.raises(ValueError, match="window_seconds"):
        extract_temporal_features(clip, duration)


@pytest.mark.parametrize("clip", [None, [], "recording.wav", np.ones(16_000)])
def test_invalid_audio_inputs_rejected(clip):
    with pytest.raises(TypeError, match="clip must be an AudioClip instance"):
        extract_temporal_features(clip)


def test_input_is_not_modified_and_outputs_are_deterministic():
    clip = make_clip(frames=17_555, channels=2)
    before = clip.samples.copy()
    first = extract_temporal_features(clip, 0.35)
    second = extract_temporal_features(clip, 0.35)
    np.testing.assert_array_equal(clip.samples, before)
    assert first == second
    assert not clip.samples.flags.writeable


def test_results_are_immutable():
    clip = make_clip(frames=8_000, sample_rate=8_000)
    result = extract_temporal_features(clip)
    assert isinstance(result, tuple)
    with pytest.raises(FrozenInstanceError):
        result[0].start_seconds = 99.0
    with pytest.raises(FrozenInstanceError):
        result[0].features.peak_amplitude = 0.0


def test_window_metadata_supports_exact_sample_reconstruction():
    sample_rate = 44_100
    clip = make_clip(frames=45_321, sample_rate=sample_rate)
    result = extract_temporal_features(clip, 0.17)
    windows = [
        (round(f.start_seconds * sample_rate), round(f.end_seconds * sample_rate))
        for f in result
    ]
    assert windows[0][0] == 0
    assert windows[-1][1] == clip.frames
    assert all(right[0] == left[1] for left, right in zip(windows, windows[1:]))
    assert all(end > start for start, end in windows)
