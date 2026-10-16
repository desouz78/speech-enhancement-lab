"""Dependency-free regression tests for the explicit enhancement/ASR pipeline.

These tests exercise orchestration and failure contracts with controlled stage
doubles. They do not claim neural-model accuracy or checkpoint compatibility.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from audio_lab.audio import AudioClip
from audio_lab.execution import (
    EnhanceTranscribeResult,
    PipelineExecutionError,
    execute_enhance_transcribe,
)
from audio_lab.pipeline import PipelinePreset
from audio_lab.whisper import TranscriptionResult, TranscriptionSegment


def make_clip(rate=16000, duration=0.25, amplitude=0.25, channels=1):
    samples = np.full((round(rate * duration), channels), amplitude, dtype=np.float32)
    return AudioClip(samples, rate)


def make_result():
    return TranscriptionResult(
        "hello",
        "en",
        (TranscriptionSegment(0.0, 0.2, "hello"),),
        "tiny.en",
    )


_DEFAULT_OUTPUT = object()


class Enhancer:
    def __init__(self, output=_DEFAULT_OUTPUT, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def enhance(self, clip):
        self.calls.append(clip)
        if self.error is not None:
            raise self.error
        return self.output if self.output is not _DEFAULT_OUTPUT else AudioClip(
            clip.samples * 0.5, clip.sample_rate
        )


class Transcriber:
    def __init__(self, output=_DEFAULT_OUTPUT, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def transcribe(self, clip, *, language=None):
        self.calls.append((clip, language))
        if self.error is not None:
            raise self.error
        return self.output if self.output is not _DEFAULT_OUTPUT else make_result()


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 192000])
@pytest.mark.parametrize("language", [None, "en", "pt"])
def test_success_at_supported_sample_rates_and_languages(rate, language):
    source = make_clip(rate=rate)
    enhancer = Enhancer()
    transcriber = Transcriber()
    result = execute_enhance_transcribe(
        source, enhancer=enhancer, transcriber=transcriber, language=language,
    )
    assert result.source_audio is source
    assert result.pipeline_id is PipelinePreset.ENHANCE_TRANSCRIBE
    assert result.transcription is not None
    assert result.transcription.text == "hello"
    assert len(enhancer.calls) == 1
    assert enhancer.calls[0] is source
    assert transcriber.calls == [(result.enhanced_audio, language)]
    assert result.enhanced_audio.frames == source.frames
    assert result.enhanced_audio.sample_rate == source.sample_rate
    assert result.enhanced_audio.channels == 1
    assert result.enhanced_audio.samples.flags.writeable is False


def test_preserves_source_without_mutation():
    source = make_clip()
    original = source.samples.copy()
    result = execute_enhance_transcribe(
        source, enhancer=Enhancer(), transcriber=Transcriber()
    )
    np.testing.assert_array_equal(source.samples, original)
    np.testing.assert_allclose(result.enhanced_audio.samples, original * 0.5)


def test_pipeline_result_is_frozen():
    result = execute_enhance_transcribe(
        make_clip(), enhancer=Enhancer(), transcriber=Transcriber()
    )
    with pytest.raises(FrozenInstanceError):
        result.transcription = make_result()


@pytest.mark.parametrize("field,value", [
    ("source_audio", None),
    ("enhanced_audio", "not audio"),
    ("transcription", object()),
])
def test_result_type_validation(field, value):
    fields = dict(
        source_audio=make_clip(),
        enhanced_audio=make_clip(),
        transcription=make_result(),
    )
    fields[field] = value
    with pytest.raises(TypeError):
        EnhanceTranscribeResult(**fields)


@pytest.mark.parametrize("bad", [None, False, 1, "audio", np.zeros(3), object()])
def test_invalid_input_rejected_before_inference(bad):
    enhancer, transcriber = Enhancer(), Transcriber()
    with pytest.raises(TypeError, match="AudioClip"):
        execute_enhance_transcribe(bad, enhancer=enhancer, transcriber=transcriber)
    assert enhancer.calls == transcriber.calls == []


@pytest.mark.parametrize("enhancer", [None, object(), object, lambda audio: audio])
def test_invalid_enhancer_rejected(enhancer):
    transcriber = Transcriber()
    with pytest.raises(TypeError, match="enhancer"):
        execute_enhance_transcribe(make_clip(), enhancer=enhancer, transcriber=transcriber)
    assert transcriber.calls == []


@pytest.mark.parametrize("transcriber", [None, object(), object, lambda audio: audio])
def test_invalid_transcriber_rejected(transcriber):
    enhancer = Enhancer()
    with pytest.raises(TypeError, match="transcriber"):
        execute_enhance_transcribe(make_clip(), enhancer=enhancer, transcriber=transcriber)
    assert enhancer.calls == []


@pytest.mark.parametrize("language", [False, True, 1, 0.1, [], {}, object()])
def test_invalid_language_type_rejected_before_inference(language):
    enhancer, transcriber = Enhancer(), Transcriber()
    with pytest.raises(TypeError, match="language"):
        execute_enhance_transcribe(
            make_clip(), enhancer=enhancer, transcriber=transcriber, language=language
        )
    assert enhancer.calls == transcriber.calls == []


@pytest.mark.parametrize("channels", [2, 3, 8])
def test_non_mono_input_rejected_before_inference(channels):
    enhancer, transcriber = Enhancer(), Transcriber()
    with pytest.raises(ValueError, match="mono"):
        execute_enhance_transcribe(
            make_clip(channels=channels), enhancer=enhancer, transcriber=transcriber,
        )
    assert enhancer.calls == transcriber.calls == []


def test_exact_duration_limit_passes():
    clip = make_clip(rate=8000, duration=30)
    result = execute_enhance_transcribe(
        clip, enhancer=Enhancer(), transcriber=Transcriber(),
    )
    assert result.source_audio.duration_seconds == 30


def test_duration_over_limit_rejected_before_inference():
    enhancer, transcriber = Enhancer(), Transcriber()
    with pytest.raises(ValueError, match="30 seconds"):
        execute_enhance_transcribe(
            make_clip(rate=8000, duration=30.001),
            enhancer=enhancer, transcriber=transcriber,
        )
    assert enhancer.calls == transcriber.calls == []


@pytest.mark.parametrize("amplitude", [1.00001, 1.2, -1.2])
def test_input_amplitude_out_of_range_rejected(amplitude):
    enhancer, transcriber = Enhancer(), Transcriber()
    with pytest.raises(ValueError, match="amplitude"):
        execute_enhance_transcribe(
            make_clip(amplitude=amplitude), enhancer=enhancer, transcriber=transcriber,
        )
    assert enhancer.calls == transcriber.calls == []


@pytest.mark.parametrize("output", [None, False, 0, "bad", np.zeros(3)])
def test_invalid_enhancer_output_type_blocks_transcription(output):
    enhancer, transcriber = Enhancer(output=output), Transcriber()
    with pytest.raises(PipelineExecutionError, match="AudioClip"):
        execute_enhance_transcribe(
            make_clip(), enhancer=enhancer, transcriber=transcriber
        )
    assert transcriber.calls == []


@pytest.mark.parametrize("output,match", [
    (make_clip(rate=8000), "sample rate"),
    (make_clip(duration=0.3), "frame count"),
    (make_clip(channels=2), "channel count"),
    (make_clip(amplitude=1.2), r"\[-1, 1\]"),
])
def test_invalid_enhancer_output_metadata_blocks_transcription(output, match):
    enhancer, transcriber = Enhancer(output=output), Transcriber()
    with pytest.raises(PipelineExecutionError, match=match):
        execute_enhance_transcribe(
            make_clip(), enhancer=enhancer, transcriber=transcriber,
        )
    assert transcriber.calls == []


def test_enhancer_exception_propagates_and_transcriber_not_called():
    error = RuntimeError("enhancer backend failed")
    transcriber = Transcriber()
    with pytest.raises(RuntimeError, match="enhancer backend failed"):
        execute_enhance_transcribe(
            make_clip(), enhancer=Enhancer(error=error), transcriber=transcriber
        )
    assert transcriber.calls == []


def test_transcriber_exception_propagates_after_enhancement():
    error = RuntimeError("transcriber backend failed")
    enhancer = Enhancer()
    with pytest.raises(RuntimeError, match="transcriber backend failed"):
        execute_enhance_transcribe(
            make_clip(), enhancer=enhancer, transcriber=Transcriber(error=error)
        )
    assert len(enhancer.calls) == 1


@pytest.mark.parametrize("bad", [False, None, 5, "hello", {"text": "hello"}])
def test_wrong_transcriber_return_type_is_rejected(bad):
    enhancer, transcriber = Enhancer(), Transcriber(output=bad)
    with pytest.raises(PipelineExecutionError, match="TranscriptionResult"):
        execute_enhance_transcribe(
            make_clip(), enhancer=enhancer, transcriber=transcriber,
        )
    assert len(enhancer.calls) == len(transcriber.calls) == 1


def test_no_checkpoint_is_loaded_as_a_pipeline_side_effect():
    # Only supplied doubles are invoked; the orchestration layer never loads a model.
    enhancer, transcriber = Enhancer(), Transcriber()
    result = execute_enhance_transcribe(
        make_clip(), enhancer=enhancer, transcriber=transcriber
    )
    assert result.transcription.model_name == "tiny.en"
