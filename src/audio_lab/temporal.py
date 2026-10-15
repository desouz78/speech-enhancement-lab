"""Deterministic temporal acoustic feature extraction for validated audio.

Audio is divided into consecutive, nonoverlapping windows. Every sample is
analyzed exactly once, including samples in the final incomplete window.

Window boundaries are aligned to sample indices, and timestamps are
expressed in seconds relative to the beginning of the recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from audio_lab.audio import AudioClip
from audio_lab.features import AudioFeatures, extract_features


_MIN_WINDOW_SECONDS = 0.1
_MAX_WINDOW_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class TemporalFeatureFrame:
    """Acoustic descriptors associated with a specific audio interval.

    The interval is represented as [start_seconds, end_seconds).
    The final interval may be shorter than the requested window duration.
    """

    start_seconds: float
    end_seconds: float
    features: AudioFeatures


def _validate_window_seconds(window_seconds: Real, sample_rate: int) -> int:
    """Validate the requested duration and convert it to sample frames."""
    if isinstance(window_seconds, bool) or not isinstance(window_seconds, Real):
        raise TypeError("window_seconds must be a real number.")

    try:
        duration = float(window_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "window_seconds cannot be converted to a finite float."
        ) from exc

    if not isfinite(duration):
        raise ValueError("window_seconds must be finite.")

    if not _MIN_WINDOW_SECONDS <= duration <= _MAX_WINDOW_SECONDS:
        raise ValueError(
            f"window_seconds must be between "
            f"{_MIN_WINDOW_SECONDS} and {_MAX_WINDOW_SECONDS}."
        )

    window_frames = round(duration * sample_rate)

    if window_frames < 1:
        raise ValueError(
            "Window duration must correspond to at least one audio sample."
        )

    return window_frames


def extract_temporal_features(
    clip: AudioClip,
    window_seconds: float = 1.0,
) -> tuple[TemporalFeatureFrame, ...]:
    """Extract acoustic features from consecutive audio windows.

    Args:
        clip: Validated audio recording.
        window_seconds: Target window duration in seconds, from 0.1 to 10.0.

    Returns:
        An immutable tuple of temporal feature frames in chronological order.

    Raises:
        TypeError: If the audio or window duration has an invalid type.
        ValueError: If the requested window duration is invalid.

    Notes:
        Window duration is rounded to the nearest sample. All windows except
        the final one contain exactly the same number of samples.

        The final incomplete window is retained. No samples are discarded,
        duplicated, or modified.

        Each window is analyzed using the same acoustic descriptors defined
        in audio_lab.features.
    """
    if not isinstance(clip, AudioClip):
        raise TypeError("clip must be an AudioClip instance.")

    window_frames = _validate_window_seconds(
        window_seconds,
        clip.sample_rate,
    )

    temporal_frames = []

    for start in range(0, clip.frames, window_frames):
        end = min(start + window_frames, clip.frames)

        segment = AudioClip(
            samples=clip.samples[start:end],
            sample_rate=clip.sample_rate,
        )

        temporal_frames.append(
            TemporalFeatureFrame(
                start_seconds=start / clip.sample_rate,
                end_seconds=end / clip.sample_rate,
                features=extract_features(segment),
            )
        )

    return tuple(temporal_frames)
