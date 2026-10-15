"""Validated audio I/O and reproducible signal-to-interference mixing.

This module is independent of neural networks and any web interface.
Audio arrays use shape (frames, channels), with float32 samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import gcd
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_DURATION_SECONDS = 180.0
MAX_CHANNELS = 8
MAX_DECODED_SAMPLES = 20_000_000
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 192_000
MIN_RMS = 1e-8


class AudioInputError(ValueError):
    """Raised when audio input violates the processing contract."""


def _validate_sample_rate(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise AudioInputError("Sample rate must be an integer in Hz.")
    if not MIN_SAMPLE_RATE <= value <= MAX_SAMPLE_RATE:
        raise AudioInputError(f"Sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz.")
    return int(value)


def _float_audio(samples: np.ndarray) -> np.ndarray:
    """Convert floating-point samples or signed/unsigned PCM to float32."""
    array = np.asarray(samples)
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)):
            raise AudioInputError("Audio contains NaN or infinite samples.")
        with np.errstate(over="ignore", invalid="ignore"):
            return np.array(array, dtype=np.float32, copy=True)
    if array.dtype.kind in "iu":
        bounds = np.iinfo(array.dtype)
        if array.dtype.kind == "u":
            midpoint = (bounds.max + 1) / 2
            converted = (array.astype(np.float64) - midpoint) / midpoint
        else:
            scale = float(max(abs(int(bounds.min)), abs(int(bounds.max))))
            converted = array.astype(np.float64) / scale
        return converted.astype(np.float32)
    raise AudioInputError("Samples must be floating-point audio or integer PCM.")


@dataclass(frozen=True, eq=False, slots=True)
class AudioClip:
    """Audio with explicit channel layout and sample-rate metadata.

    One-dimensional arrays are mono; two-dimensional arrays are (frames, channels).
    Frozen instances hold read-only copies of their samples.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        sr = _validate_sample_rate(self.sample_rate)
        raw = np.asarray(self.samples)
        if raw.ndim not in (1, 2):
            raise AudioInputError("Audio must be mono or shaped (frames, channels).")
        frames = raw.shape[0]
        channels = 1 if raw.ndim == 1 else raw.shape[1]
        if frames == 0 or not 1 <= channels <= MAX_CHANNELS:
            raise AudioInputError(f"Audio must be non-empty with 1-{MAX_CHANNELS} channels.")
        if frames * channels > MAX_DECODED_SAMPLES:
            raise AudioInputError("Audio exceeds the decoded-sample memory limit.")
        if frames / sr > MAX_DURATION_SECONDS:
            raise AudioInputError(f"Audio exceeds the {MAX_DURATION_SECONDS:g}-second safety limit.")

        data = _float_audio(raw)
        if data.ndim == 1:
            data = data[:, None]
        if not np.all(np.isfinite(data)):
            raise AudioInputError("Audio amplitude exceeds the supported float32 range.")
        data.setflags(write=False)
        object.__setattr__(self, "samples", data)
        object.__setattr__(self, "sample_rate", sr)

    @property
    def frames(self) -> int:
        return self.samples.shape[0]

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


@dataclass(frozen=True, eq=False, slots=True)
class AudioMix:
    """Synchronized, peak-safe components and their SNR metadata."""

    clean: AudioClip
    interference: AudioClip
    mixture: AudioClip
    target_snr_db: float
    actual_snr_db: float
    applied_gain: float
    peak_scale: float


def load_audio(
    source: str | Path | bytes,
    *,
    sample_rate: int | None = None,
    channels: Literal["preserve", "mono"] = "preserve",
) -> AudioClip:
    """Decode a local audio file or bytes using supported libsndfile codecs.

    The file size, decoded duration, and channel count are checked before reading.
    Unsupported formats fail with an actionable error. No network I/O occurs.
    """
    if sample_rate is not None:
        _validate_sample_rate(sample_rate)
    if channels not in ("preserve", "mono"):
        raise AudioInputError("channels must be 'preserve' or 'mono'.")

    if isinstance(source, bytes):
        if len(source) > MAX_FILE_BYTES:
            raise AudioInputError("Audio file exceeds the 64 MiB upload limit.")
        stream = BytesIO(source)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise AudioInputError("Audio file exceeds the 64 MiB upload limit.")
        stream = str(path)
    else:
        raise TypeError("source must be a file path or encoded audio bytes.")

    try:
        with sf.SoundFile(stream, mode="r") as decoder:
            _validate_sample_rate(decoder.samplerate)
            if decoder.frames == 0 or decoder.frames / decoder.samplerate > MAX_DURATION_SECONDS:
                raise AudioInputError("Audio is empty or exceeds the 180-second safety limit.")
            if not 1 <= decoder.channels <= MAX_CHANNELS:
                raise AudioInputError(f"Audio must contain 1-{MAX_CHANNELS} channels.")
            if decoder.frames * decoder.channels > MAX_DECODED_SAMPLES:
                raise AudioInputError("Audio exceeds the decoded-sample memory limit.")
            samples = decoder.read(dtype="float32", always_2d=True)
            decoded_rate = decoder.samplerate
    except (sf.LibsndfileError, RuntimeError, OSError) as exc:
        raise AudioInputError("Unable to decode audio; use a supported WAV, FLAC, OGG, or MP3 file.") from exc

    if channels == "mono":
        samples = samples.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    clip = AudioClip(samples, decoded_rate)
    return resample_audio(clip, sample_rate) if sample_rate is not None else clip


def resample_audio(clip: AudioClip, target_rate: int) -> AudioClip:
    """Bandlimited sample-rate conversion preserving channels and duration."""
    rate = _validate_sample_rate(target_rate)
    if clip.sample_rate == rate:
        return clip
    divisor = gcd(clip.sample_rate, rate)
    result = resample_poly(clip.samples, rate // divisor, clip.sample_rate // divisor, axis=0)
    return AudioClip(result, rate)


def _fit_interference(noise: np.ndarray, frames: int, sample_rate: int, policy: str) -> np.ndarray:
    """Crop or smoothly repeat short interference using overlap crossfades."""
    if noise.shape[0] >= frames:
        return np.array(noise[:frames], dtype=np.float64, copy=True)
    if policy == "error":
        raise AudioInputError("Interference is shorter than clean audio; use short_interference='loop'.")
    if policy != "loop":
        raise AudioInputError("short_interference must be 'loop' or 'error'.")
    if noise.shape[0] < max(8, int(0.01 * sample_rate)):
        raise AudioInputError("Interference must be at least 10 ms long to loop smoothly.")

    source = np.asarray(noise, dtype=np.float64)
    fade_frames = min(int(0.02 * sample_rate), source.shape[0] // 4)
    fade = np.linspace(0.0, 1.0, fade_frames + 2, dtype=np.float64)[1:-1, None]
    output = np.empty((frames, source.shape[1]), dtype=np.float64)
    output[:source.shape[0]] = source
    end = source.shape[0]

    while end < frames:
        output[end - fade_frames:end] = (
            output[end - fade_frames:end] * (1 - fade) + source[:fade_frames] * fade
        )
        count = min(source.shape[0] - fade_frames, frames - end)
        output[end:end + count] = source[fade_frames:fade_frames + count]
        end += count
    return output


def mix_at_snr(
    clean: AudioClip,
    interference: AudioClip,
    snr_db: float,
    *,
    short_interference: Literal["loop", "error"] = "loop",
) -> AudioMix:
    """Produce a synchronized mixture at a specified component-power SNR.

    Interference is resampled to the clean clip's sample rate. Mono interference
    is expanded to match multichannel clean audio. For mono clean audio, any
    multichannel interference is downmixed. Other channel mismatches are rejected.
    A common peak gain is applied to *all* components to avoid clipping without
    altering their relative SNR. This SNR is known because sources are available.
    """
    if not isinstance(clean, AudioClip) or not isinstance(interference, AudioClip):
        raise TypeError("clean and interference must be AudioClip instances.")
    try:
        target = float(snr_db)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AudioInputError("snr_db must be a finite number.") from exc
    if not np.isfinite(target) or not -60 <= target <= 80:
        raise AudioInputError("snr_db must be finite and between -60 and 80 dB.")
    if short_interference not in ("loop", "error"):
        raise AudioInputError("short_interference must be 'loop' or 'error'.")

    matched = resample_audio(interference, clean.sample_rate)
    signal = clean.samples.astype(np.float64)
    noise = matched.samples.astype(np.float64)

    if noise.shape[1] == 1 and clean.channels > 1:
        noise = np.repeat(noise, clean.channels, axis=1)
    elif clean.channels == 1 and noise.shape[1] > 1:
        noise = noise.mean(axis=1, keepdims=True)
    elif noise.shape[1] != clean.channels:
        raise AudioInputError("Channel layouts are incompatible; explicitly convert to mono first.")

    noise = _fit_interference(noise, clean.frames, clean.sample_rate, short_interference)
    signal_rms = float(np.sqrt(np.mean(np.square(signal))))
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    if signal_rms < MIN_RMS or noise_rms < MIN_RMS:
        raise AudioInputError("Clean or interference signal is silent or below the minimum RMS level.")

    gain = (signal_rms / noise_rms) / (10 ** (target / 20))
    scaled_noise = noise * gain
    mixture = signal + scaled_noise
    peak = max(1.0, float(np.max(np.abs(signal))), float(np.max(np.abs(scaled_noise))),
               float(np.max(np.abs(mixture))))

    clean_result = AudioClip(signal / peak, clean.sample_rate)
    noise_result = AudioClip(scaled_noise / peak, clean.sample_rate)
    mixture_result = AudioClip(mixture / peak, clean.sample_rate)
    measured = 20 * np.log10(
        np.sqrt(np.mean(clean_result.samples.astype(np.float64)**2)) /
        np.sqrt(np.mean(noise_result.samples.astype(np.float64)**2))
    )
    return AudioMix(clean_result, noise_result, mixture_result, target, float(measured),
                    float(gain), float(peak))
