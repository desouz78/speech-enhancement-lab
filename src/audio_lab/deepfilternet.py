"""Optional, single-channel DeepFilterNet3 inference adapter.

This module does not import neural-network dependencies, initialize weights,
access the network, or run inference when imported. Loading is explicit.
The adapter is for Python inference, not browser deployment.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from audio_lab.audio import AudioClip, resample_audio


_MODEL_NAME = "DeepFilterNet3"
_MODEL_SAMPLE_RATE = 48_000
_MAX_INFERENCE_SECONDS = 30.0


class DeepFilterNetError(RuntimeError):
    """Raised for missing model dependencies or an invalid model interface."""


class DeepFilterNet3Adapter:
    """Run pretrained DeepFilterNet3 on validated mono audio.

    Model input and output use 48 kHz, shape [1, samples]. Calls are limited
    to 30 seconds. The result is resampled to the original rate. No clipping or loudness normalization is
    applied, as either operation would alter the model's output.
    """

    def __init__(self, model: Any, df_state: Any, enhance_fn: Callable[..., Any]) -> None:
        if model is None or df_state is None or not callable(enhance_fn):
            raise TypeError("model, df_state and callable enhance_fn are required.")
        sampling_rate = getattr(df_state, "sr", None)
        if not callable(sampling_rate):
            raise DeepFilterNetError("DF state must expose an sr() method.")
        try:
            rate = sampling_rate()
        except Exception as exc:
            raise DeepFilterNetError("Unable to read the DF state sample rate.") from exc
        if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, (int, np.integer)):
            raise DeepFilterNetError("DF state returned a noninteger sample rate.")
        if rate != _MODEL_SAMPLE_RATE:
            raise DeepFilterNetError("DeepFilterNet3 requires a 48,000 Hz model state.")
        self._model = model
        self._state = df_state
        self._enhance = enhance_fn

    @classmethod
    def load(cls) -> DeepFilterNet3Adapter:
        """Explicitly load the DeepFilterNet3 checkpoint (may download weights).

        Requires the separately installed deepfilternet and torch packages.
        Errors from model initialization are not hidden or silently retried.
        """
        try:
            from df.enhance import enhance, init_df
        except ImportError as exc:
            raise DeepFilterNetError(
                "DeepFilterNet3 dependencies are missing; install deepfilternet and torch."
            ) from exc

        initialized = init_df(model_base_dir=_MODEL_NAME, log_file=None)
        if not isinstance(initialized, tuple) or len(initialized) not in (3, 4):
            raise DeepFilterNetError("Unexpected DeepFilterNet initialization response.")
        return cls(initialized[0], initialized[1], enhance)

    def enhance(self, clip: AudioClip) -> AudioClip:
        """Enhance one mono AudioClip, preserving its nominal sample rate.

        Raises TypeError for invalid inputs and ValueError for unsupported
        channel layouts or out-of-range samples. Invalid model outputs raise
        DeepFilterNetError. Model execution errors propagate to the caller.
        """
        if not isinstance(clip, AudioClip):
            raise TypeError("clip must be an AudioClip instance.")
        if clip.duration_seconds > _MAX_INFERENCE_SECONDS:
            raise ValueError("Inference is limited to 30 seconds per recording.")
        if clip.channels != 1:
            raise ValueError("DeepFilterNet3 adapter requires mono audio; downmix explicitly.")
        if float(np.max(np.abs(clip.samples))) > 1.0:
            raise ValueError("Model input exceeds the supported [-1, 1] amplitude range.")

        try:
            import torch
        except ImportError as exc:
            raise DeepFilterNetError("PyTorch is required to run DeepFilterNet3 inference.") from exc

        model_input = resample_audio(clip, _MODEL_SAMPLE_RATE)
        if float(np.max(np.abs(model_input.samples))) > 1.0:
            raise ValueError("Resampled audio exceeds the supported [-1, 1] amplitude range.")
        # A writable copy avoids sharing the immutable AudioClip buffer with PyTorch.
        model_samples = np.array(model_input.samples.T, dtype=np.float32, order="C", copy=True)
        tensor = torch.from_numpy(model_samples)
        model_output = self._enhance(self._model, self._state, tensor, pad=True)

        try:
            output = model_output.detach().cpu().numpy()
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise DeepFilterNetError("Model output must be a readable PyTorch tensor.") from exc
        if not isinstance(output, np.ndarray) or output.shape != (1, model_input.frames):
            raise DeepFilterNetError("Model returned an unexpected output shape.")
        if output.dtype.kind != "f" or not np.all(np.isfinite(output)):
            raise DeepFilterNetError("Model returned nonfinite or nonfloating audio.")

        result = AudioClip(output.T, _MODEL_SAMPLE_RATE)
        return resample_audio(result, clip.sample_rate)
