"""Deterministic, bounded-memory acoustic descriptors for validated audio.

All channels contribute independently (no mono downmix). Spectral descriptors
summarize the channel-averaged, DC-removed, Hann-windowed power spectrum over
all nonoverlapping analysis frames. They are not speech/music classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import hann

from audio_lab.audio import AudioClip


_BLOCK_FRAMES = 65_536
_FFT_BATCH_FRAMES = 16
_MIN_FFT_SIZE = 512
_MAX_FFT_SIZE = 8192


@dataclass(frozen=True, slots=True)
class AudioFeatures:
    """Full-clip signal descriptors, averaged over all available channels.

    RMS dBFS uses amplitude 1.0 as the digital reference (float signals may
    exceed 0 dBFS). A silent signal has RMS -inf dBFS and zero spectral metrics.
    Zero crossings count transitions between negative and nonnegative samples.
    Spectral flatness is unitless and ranges from 0 to 1 for nonzero spectra.
    """

    duration_seconds: float
    sample_rate: int
    channels: int
    rms_dbfs: float
    peak_amplitude: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_flatness: float


def _analysis_fft_size(sample_rate: int) -> int:
    """Choose a power-of-two FFT spanning approximately 50 ms of audio."""
    target = (sample_rate + 19) // 20
    return max(_MIN_FFT_SIZE, min(_MAX_FFT_SIZE, 1 << (target - 1).bit_length()))


def _power_spectrum(clip: AudioClip, fft_size: int) -> np.ndarray:
    """Time-weighted, channel-averaged, one-sided AC power spectrum.

    Processes every input sample in bounded-memory batches. The final partial
    analysis frame is weighted by its actual length, rather than being dropped.
    Frames shorter than four samples use a rectangular analysis window.
    """
    samples = clip.samples
    frames, channels = samples.shape
    full_frames, remainder = divmod(frames, fft_size)
    window = hann(fft_size, sym=False)
    window_energy = float(np.dot(window, window))
    aggregate = np.zeros(fft_size // 2 + 1, dtype=np.float64)

    for start in range(0, full_frames, _FFT_BATCH_FRAMES):
        count = min(_FFT_BATCH_FRAMES, full_frames - start)
        block = samples[start * fft_size:(start + count) * fft_size]
        block = block.reshape(count, fft_size, channels).astype(np.float64)
        block -= np.mean(block, axis=1, keepdims=True)
        block *= window[None, :, None]
        spectrum = np.fft.rfft(block, axis=1)
        aggregate += np.sum(np.abs(spectrum) ** 2, axis=(0, 2)) / (window_energy * channels)

    if remainder:
        block = samples[full_frames * fft_size:].astype(np.float64)
        block -= np.mean(block, axis=0, keepdims=True)
        partial_window = hann(remainder, sym=False) if remainder >= 4 else np.ones(remainder)
        block *= partial_window[:, None]
        spectrum = np.fft.rfft(block, n=fft_size, axis=0)
        aggregate += (remainder / fft_size) * np.mean(np.abs(spectrum) ** 2, axis=1) / np.dot(partial_window, partial_window)

    aggregate /= frames
    if fft_size > 2:
        aggregate[1:-1] *= 2.0
    return aggregate

def extract_features(clip: AudioClip) -> AudioFeatures:
    """Compute global amplitude, zero-crossing and spectral descriptors.

    For silence, the spectral centroid/bandwidth/flatness are defined as zero.
    Spectral features exclude each analysis frame's DC offset. ZCR uses the
    original samples; no cross-channel transitions are counted.
    """
    if not isinstance(clip, AudioClip):
        raise TypeError("clip must be an AudioClip instance.")

    samples = clip.samples
    frames, channels = samples.shape
    energy = 0.0
    peak = 0.0
    crossings = 0
    last_negative = None

    for start in range(0, frames, _BLOCK_FRAMES):
        block = samples[start:start + _BLOCK_FRAMES].astype(np.float64)
        energy += float(np.sum(block * block))
        peak = max(peak, float(np.max(np.abs(block))))
        negative = block < 0.0
        crossings += int(np.count_nonzero(negative[1:] != negative[:-1]))
        if last_negative is not None:
            crossings += int(np.count_nonzero(negative[0] != last_negative))
        last_negative = negative[-1]

    rms = float(np.sqrt(energy / (frames * channels)))
    rms_dbfs = float(20 * np.log10(rms)) if rms > 0 else float("-inf")
    zcr = crossings / ((frames - 1) * channels) if frames > 1 else 0.0

    fft_size = _analysis_fft_size(clip.sample_rate)
    power = _power_spectrum(clip, fft_size)
    total_power = float(np.sum(power))

    if total_power > 0:
        frequencies = np.fft.rfftfreq(fft_size, d=1.0 / clip.sample_rate)
        centroid = float(np.dot(frequencies, power) / total_power)
        bandwidth = float(np.sqrt(np.dot((frequencies - centroid) ** 2, power) / total_power))

        # Normalize before taking logs: flatness must be scale invariant.
        normalized = power / float(np.max(power))
        positive = np.maximum(normalized, np.finfo(np.float64).tiny)
        flatness = float(np.exp(np.mean(np.log(positive))) / np.mean(positive))
        flatness = float(np.clip(flatness, 0.0, 1.0))
    else:
        centroid = bandwidth = flatness = 0.0

    return AudioFeatures(
        duration_seconds=clip.duration_seconds,
        sample_rate=clip.sample_rate,
        channels=channels,
        rms_dbfs=rms_dbfs,
        peak_amplitude=peak,
        zero_crossing_rate=float(zcr),
        spectral_centroid_hz=centroid,
        spectral_bandwidth_hz=bandwidth,
        spectral_flatness=flatness,
    )
