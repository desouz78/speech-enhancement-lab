"""Numerical and edge-case regression tests for acoustic feature extraction."""

import numpy as np
import pytest

from audio_lab.audio import AudioClip
from audio_lab.features import extract_features


def tone(frequency=1000, rate=16000, duration=1.0, amplitude=0.5):
    frames = round(rate * duration)
    t = np.arange(frames, dtype=np.float64) / rate
    return AudioClip(amplitude * np.sin(2 * np.pi * frequency * t), rate)


def test_known_sinusoidal_rms_peak_and_centroid():
    features = extract_features(tone())
    assert features.duration_seconds == pytest.approx(1)
    assert features.sample_rate == 16000
    assert features.channels == 1
    assert features.rms_dbfs == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=1e-5)
    assert features.peak_amplitude == pytest.approx(0.5, abs=1e-6)
    assert features.spectral_centroid_hz == pytest.approx(1000, abs=5)
    assert features.spectral_bandwidth_hz < 35
    assert features.spectral_flatness < 1e-5


def test_zero_crossing_rate_on_alternating_samples():
    samples = np.array([1, -1, 1, -1, 1], dtype=np.float32)
    features = extract_features(AudioClip(samples, 16000))
    assert features.zero_crossing_rate == pytest.approx(1)
    assert features.rms_dbfs == pytest.approx(0)


def test_silence_semantics():
    features = extract_features(AudioClip(np.zeros(16000), 16000))
    assert features.rms_dbfs == float("-inf")
    assert features.peak_amplitude == 0
    assert features.zero_crossing_rate == 0
    assert features.spectral_centroid_hz == 0
    assert features.spectral_bandwidth_hz == 0
    assert features.spectral_flatness == 0


@pytest.mark.parametrize("length", [1, 2, 3, 4, 127, 511, 1024, 1025, 65536, 65537])
def test_very_short_and_block_boundary_lengths(length):
    features = extract_features(AudioClip(np.ones(length, dtype=np.float32), 16000))
    assert features.duration_seconds == pytest.approx(length / 16000)
    assert features.rms_dbfs == pytest.approx(0)
    assert features.peak_amplitude == 1
    assert features.zero_crossing_rate == 0
    assert features.spectral_centroid_hz >= 0
    assert 0 <= features.spectral_flatness <= 1


def test_dc_offset_is_removed_from_spectral_features():
    features = extract_features(AudioClip(np.full(16000, 0.25), 16000))
    assert features.rms_dbfs == pytest.approx(20 * np.log10(0.25))
    assert features.spectral_centroid_hz == 0
    assert features.spectral_flatness == 0


def test_spectrum_is_channel_phase_invariant():
    mono = tone().samples[:, 0]
    stereo = AudioClip(np.column_stack((mono, -mono)), 16000)
    stereo_features = extract_features(stereo)
    mono_features = extract_features(AudioClip(mono, 16000))
    assert stereo_features.channels == 2
    assert stereo_features.rms_dbfs == pytest.approx(mono_features.rms_dbfs)
    assert stereo_features.spectral_centroid_hz == pytest.approx(mono_features.spectral_centroid_hz)
    assert stereo_features.spectral_bandwidth_hz == pytest.approx(mono_features.spectral_bandwidth_hz)
    assert stereo_features.spectral_flatness == pytest.approx(mono_features.spectral_flatness)


def test_amplitude_scaling_preserves_shape_descriptors():
    reference = extract_features(tone(amplitude=0.4))
    quiet = extract_features(tone(amplitude=0.04))
    assert quiet.rms_dbfs - reference.rms_dbfs == pytest.approx(-20, abs=1e-4)
    assert quiet.spectral_centroid_hz == pytest.approx(reference.spectral_centroid_hz, abs=0.01)
    assert quiet.spectral_flatness == pytest.approx(reference.spectral_flatness, abs=1e-7)


def test_low_amplitude_does_not_corrupt_flatness():
    random = np.random.default_rng(101).normal(size=16384)
    full = extract_features(AudioClip(random * 0.1, 16000))
    low = extract_features(AudioClip(random * 1e-10, 16000))
    assert low.spectral_flatness == pytest.approx(full.spectral_flatness, abs=1e-6)
    assert low.spectral_centroid_hz == pytest.approx(full.spectral_centroid_hz, abs=1e-5)


def test_noise_spectrum_has_higher_flatness_than_tone():
    rng = np.random.default_rng(42)
    noise = extract_features(AudioClip(rng.normal(size=16000), 16000))
    pure = extract_features(tone())
    assert noise.spectral_flatness > 0.4
    assert pure.spectral_flatness < 1e-5


def test_zero_crossings_are_counted_across_block_boundaries():
    samples = np.ones(65540, dtype=np.float32)
    samples[65536:] *= -1
    features = extract_features(AudioClip(samples, 16000))
    assert features.zero_crossing_rate == pytest.approx(1 / (len(samples) - 1))


def test_late_transient_is_not_dropped():
    samples = np.zeros(16384, dtype=np.float32)
    samples[-400:] = tone(1600, duration=400 / 16000).samples[:, 0]
    features = extract_features(AudioClip(samples, 16000))
    assert features.peak_amplitude > 0.4
    assert 1200 < features.spectral_centroid_hz < 2000


def test_input_unchanged_and_determinism():
    clip = tone()
    original = clip.samples.copy()
    a = extract_features(clip)
    b = extract_features(clip)
    assert a == b
    np.testing.assert_array_equal(clip.samples, original)


@pytest.mark.parametrize("bad", [None, 12, "some.wav", np.zeros(100)])
def test_invalid_input_rejected(bad):
    with pytest.raises(TypeError, match="AudioClip"):
        extract_features(bad)


@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100, 48000, 96000, 192000])
def test_tone_centroid_across_supported_sample_rates(sample_rate):
    features = extract_features(tone(rate=sample_rate, duration=0.1))
    assert features.spectral_centroid_hz == pytest.approx(1000, abs=60)


def test_spectral_metrics_in_valid_physical_range():
    rng = np.random.default_rng(42)
    for rate in (8000, 16000, 48000):
        for channels in (1, 2):
            clip = AudioClip(rng.normal(size=(4097, channels)), rate)
            features = extract_features(clip)
            assert 0 <= features.zero_crossing_rate <= 1
            assert 0 <= features.spectral_centroid_hz <= rate / 2
            assert 0 <= features.spectral_bandwidth_hz <= rate / 2
            assert 0 <= features.spectral_flatness <= 1

@pytest.mark.parametrize("tail_length", [256, 512])
def test_partial_frame_is_weighted_by_actual_duration(tail_length):
    rate = 16000
    full_length = 1024
    first = np.sin(2 * np.pi * 1000 * np.arange(full_length) / rate)
    tail = np.sin(2 * np.pi * 3000 * np.arange(tail_length) / rate)
    features = extract_features(AudioClip(np.concatenate((first, tail)), rate))
    expected = (1000 * full_length + 3000 * tail_length) / (full_length + tail_length)
    assert features.spectral_centroid_hz == pytest.approx(expected, abs=2)


def test_zero_crossings_are_not_counted_between_channels():
    samples = np.column_stack((np.ones(1024), -np.ones(1024)))
    features = extract_features(AudioClip(samples, 16000))
    assert features.channels == 2
    assert features.zero_crossing_rate == 0
