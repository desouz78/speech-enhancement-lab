"""Regression tests for audio loading, validation, and mixing."""

from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from audio_lab.audio import AudioClip, AudioInputError, load_audio, mix_at_snr, resample_audio


def make_clip(rate=16000, duration=0.5, channels=1, frequency=250):
    times = np.arange(round(rate * duration)) / rate
    wave = 0.35 * np.sin(2 * np.pi * frequency * times)
    return AudioClip(np.tile(wave[:, None], (1, channels)), rate)


@pytest.mark.parametrize("target_snr", [-20, -5, 0, 5, 20, 60])
def test_target_snr_is_accurate_and_no_clipping(target_snr):
    rng = np.random.default_rng(42)
    clean = make_clip()
    noise = AudioClip(rng.normal(0, 0.5, size=clean.frames), clean.sample_rate)
    result = mix_at_snr(clean, noise, target_snr)
    assert result.actual_snr_db == pytest.approx(target_snr, abs=0.001)
    assert result.mixture.samples.shape == clean.samples.shape
    assert max(np.max(abs(c.samples)) for c in (result.clean, result.interference, result.mixture)) <= 1.0
    np.testing.assert_allclose(result.mixture.samples, result.clean.samples + result.interference.samples, atol=1e-6)


def test_resamples_interference_from_different_rate():
    result = mix_at_snr(make_clip(rate=16000), make_clip(rate=48000), 5)
    assert result.mixture.sample_rate == 16000
    assert result.actual_snr_db == pytest.approx(5, abs=0.001)


def test_short_interference_loops_with_crossfade():
    clean = make_clip(duration=1)
    short = make_clip(duration=0.08, frequency=320)
    result = mix_at_snr(clean, short, 0)
    assert result.mixture.frames == clean.frames
    assert result.actual_snr_db == pytest.approx(0, abs=0.001)


def test_short_interference_can_fail_explicitly():
    with pytest.raises(AudioInputError, match="shorter"):
        mix_at_snr(make_clip(duration=1), make_clip(duration=0.2), 5, short_interference="error")


def test_tiny_interference_cannot_loop():
    with pytest.raises(AudioInputError, match="10 ms"):
        mix_at_snr(make_clip(duration=1), make_clip(duration=0.005), 5)


def test_channel_conversion_and_preservation():
    stereo = make_clip(channels=2)
    mono = make_clip()
    stereo_mix = mix_at_snr(stereo, mono, 5)
    mono_mix = mix_at_snr(mono, stereo, 5)
    assert stereo_mix.mixture.channels == 2
    assert mono_mix.mixture.channels == 1
    assert stereo_mix.actual_snr_db == pytest.approx(5, abs=0.001)
    assert mono_mix.actual_snr_db == pytest.approx(5, abs=0.001)


def test_rejects_incompatible_multi_channel_layouts():
    with pytest.raises(AudioInputError, match="layouts"):
        mix_at_snr(make_clip(channels=2), make_clip(channels=3), 5)


@pytest.mark.parametrize("bad", [np.array([]), np.array([np.nan]), np.array([np.inf]), np.ones((2, 2, 2))])
def test_invalid_audio_arrays_are_rejected(bad):
    with pytest.raises(AudioInputError):
        AudioClip(bad, 16000)


@pytest.mark.parametrize("rate", [0, -1, 1000, 200000, 16.5, True])
def test_invalid_sample_rates_are_rejected(rate):
    with pytest.raises(AudioInputError):
        AudioClip(np.ones(1024), rate)


@pytest.mark.parametrize("snr", [float("nan"), float("inf"), -61, 81, "nope"])
def test_invalid_snr_is_rejected(snr):
    with pytest.raises(AudioInputError):
        mix_at_snr(make_clip(), make_clip(), snr)


def test_silent_and_cancelled_interference_are_rejected():
    with pytest.raises(AudioInputError, match="silent"):
        mix_at_snr(make_clip(), AudioClip(np.zeros(16000), 16000), 5)
    stereo = np.column_stack((make_clip().samples[:, 0], -make_clip().samples[:, 0]))
    with pytest.raises(AudioInputError, match="silent"):
        mix_at_snr(make_clip(), AudioClip(stereo, 16000), 5)


def test_audio_arrays_are_copied_and_read_only():
    original = np.ones(100, dtype=np.float32)
    clip = AudioClip(original, 16000)
    original[:] = 0
    assert np.all(clip.samples == 1)
    with pytest.raises(ValueError):
        clip.samples[0, 0] = 0


def test_integer_pcm_conversion():
    samples = np.array([-32768, 0, 32767], dtype=np.int16)
    clip = AudioClip(samples, 16000)
    np.testing.assert_allclose(clip.samples[:, 0], [-1, 0, 32767 / 32768])
    clip8 = AudioClip(np.array([0, 128, 255], dtype=np.uint8), 16000)
    np.testing.assert_allclose(clip8.samples[:, 0], [-1, 0, 127 / 128])


def test_load_wav_from_path_and_bytes(tmp_path):
    wav = make_clip(duration=0.25, channels=2)
    path = tmp_path / "audio.wav"
    sf.write(path, wav.samples, wav.sample_rate, subtype="PCM_16")
    loaded = load_audio(path)
    assert loaded.channels == 2
    assert loaded.sample_rate == wav.sample_rate
    decoded = load_audio(path.read_bytes(), sample_rate=8000, channels="mono")
    assert decoded.channels == 1
    assert decoded.sample_rate == 8000
    assert abs(decoded.duration_seconds - 0.25) < 1 / 8000


def test_corrupted_audio_is_rejected():
    with pytest.raises(AudioInputError, match="decode"):
        load_audio(b"This is not an audio file")


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "missing.wav")


def test_rejects_overly_long_audio_before_decoding(tmp_path):
    path = tmp_path / "long.wav"
    with sf.SoundFile(path, "w", samplerate=8000, channels=1, format="WAV", subtype="PCM_16") as file:
        file.seek(8000 * 181 - 1)
        file.write(np.zeros(1))
    with pytest.raises(AudioInputError, match="180-second"):
        load_audio(path)


def test_resample_same_rate_is_noop():
    clip = make_clip()
    assert resample_audio(clip, clip.sample_rate) is clip


def test_rejects_too_many_decoded_samples(monkeypatch):
    from audio_lab import audio
    monkeypatch.setattr(audio, "MAX_DECODED_SAMPLES", 100)
    with pytest.raises(AudioInputError, match="decoded-sample"):
        AudioClip(np.ones((101, 1)), 16000)


def test_rejects_non_clip_inputs():
    with pytest.raises(TypeError, match="AudioClip"):
        mix_at_snr(np.ones(16), make_clip(), 5)


def test_randomized_mixing_properties():
    rng = np.random.default_rng(2026)
    rates = [8000, 16000, 44100, 48000]
    for _ in range(50):
        clean_rate = int(rng.choice(rates))
        noise_rate = int(rng.choice(rates))
        clean_channels = int(rng.choice([1, 2]))
        noise_channels = int(rng.choice([1, 2]))
        duration = float(rng.uniform(0.06, 0.30))
        noisy_duration = float(rng.uniform(0.015, 0.32))
        desired_snr = float(rng.uniform(-25, 45))
        clean = AudioClip(rng.normal(size=(round(clean_rate * duration), clean_channels)), clean_rate)
        noise = AudioClip(rng.normal(size=(round(noise_rate * noisy_duration), noise_channels)), noise_rate)
        result = mix_at_snr(clean, noise, desired_snr)
        assert result.actual_snr_db == pytest.approx(desired_snr, abs=0.01)
        assert result.mixture.frames == clean.frames
        assert result.mixture.channels == clean_channels
        assert result.mixture.sample_rate == clean_rate
        assert np.all(np.isfinite(result.mixture.samples))
        assert np.max(np.abs(result.mixture.samples)) <= 1.0
