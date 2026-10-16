"""Dependency-free regression tests for SepFormer -> Whisper orchestration.

Controlled stage doubles test data flow and failure boundaries; these tests do
not evaluate pretrained inference quality or persistent speaker identities.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from audio_lab.audio import AudioClip
from audio_lab.execution import PipelineExecutionError
from audio_lab.pipeline import PipelinePreset
from audio_lab.sepformer import SpeechSeparationResult
from audio_lab.separate_execution import SeparateTranscribeResult, execute_separate_transcribe
from audio_lab.whisper import TranscriptionResult, TranscriptionSegment


_DEFAULT = object()


def make_clip(rate=16000, duration=0.25, amplitude=0.3, channels=1):
    samples = np.full((round(rate * duration), channels), amplitude, dtype=np.float32)
    return AudioClip(samples, rate)


def make_transcription(text="hello"):
    return TranscriptionResult(text, "en", (TranscriptionSegment(0.0, 0.1, text),), "tiny.en")


def make_separation(clip, *, first=0.5, second=0.25):
    one = AudioClip(clip.samples * first, clip.sample_rate)
    two = AudioClip(clip.samples * second, clip.sample_rate)
    return SpeechSeparationResult(clip, (one, two))


class Separator:
    def __init__(self, output=_DEFAULT, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def separate(self, clip):
        self.calls.append(clip)
        if self.error is not None:
            raise self.error
        return make_separation(clip) if self.output is _DEFAULT else self.output


class Transcriber:
    def __init__(self, output=_DEFAULT, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def transcribe(self, clip, *, language=None):
        self.calls.append((clip, language))
        if self.error is not None:
            raise self.error
        if self.output is _DEFAULT:
            return make_transcription(f"track {len(self.calls)}")
        if callable(self.output):
            return self.output(len(self.calls))
        return self.output


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 192000])
@pytest.mark.parametrize("language", [None, "en", "pt"])
def test_success_with_supported_sample_rates_and_languages(rate, language):
    clip = make_clip(rate=rate)
    separator, transcriber = Separator(), Transcriber()
    result = execute_separate_transcribe(clip, separator=separator, transcriber=transcriber, language=language)

    assert result.source_audio is clip
    assert result.pipeline_id is PipelinePreset.SEPARATE_TRANSCRIBE
    assert result.separation.source_audio is clip
    assert separator.calls == [clip]
    assert transcriber.calls == [(result.separation.speaker_1, language), (result.separation.speaker_2, language)]
    assert tuple(item.text for item in result.transcriptions) == ("track 1", "track 2")
    assert result.speaker_1_transcription is result.transcriptions[0]
    assert result.speaker_2_transcription is result.transcriptions[1]
    assert all(source.frames == clip.frames for source in result.separation.sources)
    assert all(source.sample_rate == clip.sample_rate for source in result.separation.sources)


def test_does_not_modify_input_or_separated_sources():
    clip = make_clip()
    input_samples = clip.samples.copy()
    sep = Separator()
    outcome = execute_separate_transcribe(clip, separator=sep, transcriber=Transcriber())

    np.testing.assert_array_equal(clip.samples, input_samples)
    np.testing.assert_array_equal(outcome.separation.speaker_1.samples, input_samples * 0.5)
    np.testing.assert_array_equal(outcome.separation.speaker_2.samples, input_samples * 0.25)
    assert all(not source.samples.flags.writeable for source in outcome.separation.sources)


def test_result_is_frozen_and_has_explicit_output_order():
    result = execute_separate_transcribe(make_clip(), separator=Separator(), transcriber=Transcriber())
    with pytest.raises(FrozenInstanceError):
        result.transcriptions = ()
    with pytest.raises(AttributeError):
        result.transcriptions.append(make_transcription())


@pytest.mark.parametrize("field,invalid,error", [
    ("source_audio", None, TypeError),
    ("separation", None, TypeError),
    ("transcriptions", None, TypeError),
    ("transcriptions", [], TypeError),
    ("transcriptions", (), TypeError),
    ("transcriptions", (make_transcription(),), TypeError),
    ("transcriptions", (make_transcription(), None), TypeError),
])
def test_result_dataclass_rejects_invalid_fields(field, invalid, error):
    clip = make_clip()
    fields = dict(source_audio=clip, separation=make_separation(clip), transcriptions=(make_transcription(), make_transcription()))
    fields[field] = invalid
    with pytest.raises(error):
        SeparateTranscribeResult(**fields)


def test_result_dataclass_requires_original_source_reference():
    clip, different = make_clip(), make_clip()
    with pytest.raises(ValueError, match="original source audio"):
        SeparateTranscribeResult(clip, make_separation(different), (make_transcription(), make_transcription()))


@pytest.mark.parametrize("invalid", [None, False, 0, "audio", np.zeros((4, 1)), object()])
def test_invalid_input_rejected_before_inference(invalid):
    separator, transcriber = Separator(), Transcriber()
    with pytest.raises(TypeError, match="AudioClip"):
        execute_separate_transcribe(invalid, separator=separator, transcriber=transcriber)
    assert separator.calls == transcriber.calls == []


@pytest.mark.parametrize("invalid", [None, object(), object, lambda clip: clip])
def test_invalid_separator_rejected(invalid):
    transcriber = Transcriber()
    with pytest.raises(TypeError, match="separator"):
        execute_separate_transcribe(make_clip(), separator=invalid, transcriber=transcriber)
    assert transcriber.calls == []


@pytest.mark.parametrize("invalid", [None, object(), object, lambda clip: clip])
def test_invalid_transcriber_rejected(invalid):
    separator = Separator()
    with pytest.raises(TypeError, match="transcriber"):
        execute_separate_transcribe(make_clip(), separator=separator, transcriber=invalid)
    assert separator.calls == []


@pytest.mark.parametrize("invalid", [True, False, 0, 2, 1.3, [], {}, object()])
def test_invalid_language_types_rejected_before_inference(invalid):
    separator, transcriber = Separator(), Transcriber()
    with pytest.raises(TypeError, match="language"):
        execute_separate_transcribe(make_clip(), separator=separator, transcriber=transcriber, language=invalid)
    assert separator.calls == transcriber.calls == []


@pytest.mark.parametrize("channels", [2, 3, 8])
def test_nonmono_input_rejected(channels):
    separator, transcriber = Separator(), Transcriber()
    with pytest.raises(ValueError, match="mono"):
        execute_separate_transcribe(make_clip(channels=channels), separator=separator, transcriber=transcriber)
    assert separator.calls == transcriber.calls == []


def test_overlong_input_rejected():
    separator, transcriber = Separator(), Transcriber()
    with pytest.raises(ValueError, match="30 seconds"):
        execute_separate_transcribe(make_clip(duration=30.001), separator=separator, transcriber=transcriber)
    assert separator.calls == transcriber.calls == []


def test_exact_duration_limit_accepted():
    clip = make_clip(rate=8000, duration=30)
    result = execute_separate_transcribe(clip, separator=Separator(), transcriber=Transcriber())
    assert result.source_audio is clip


def test_out_of_range_input_rejected_before_inference():
    separator, transcriber = Separator(), Transcriber()
    with pytest.raises(ValueError, match="amplitude"):
        execute_separate_transcribe(make_clip(amplitude=1.01), separator=separator, transcriber=transcriber)
    assert separator.calls == transcriber.calls == []


@pytest.mark.parametrize("output", [None, make_clip(), (make_clip(), make_clip()), object()])
def test_separator_wrong_output_type_rejected_without_transcribing(output):
    transcriber = Transcriber()
    with pytest.raises(PipelineExecutionError, match="SpeechSeparationResult"):
        execute_separate_transcribe(make_clip(), separator=Separator(output=output), transcriber=transcriber)
    assert transcriber.calls == []


def test_separator_must_reference_original_audio():
    clip = make_clip()
    transcriber = Transcriber()
    with pytest.raises(PipelineExecutionError, match="original source"):
        execute_separate_transcribe(clip, separator=Separator(output=make_separation(make_clip())), transcriber=transcriber)
    assert transcriber.calls == []


@pytest.mark.parametrize("index", [0, 1])
def test_out_of_range_separated_amplitude_rejected_before_transcription(index):
    clip = make_clip()
    scales = (4.0, 0.2) if index == 0 else (0.2, 4.0)
    transcriber = Transcriber()
    with pytest.raises(PipelineExecutionError, match=f"track {index + 1} exceeds"):
        execute_separate_transcribe(clip, separator=Separator(output=make_separation(clip, first=scales[0], second=scales[1])), transcriber=transcriber)
    assert transcriber.calls == []


@pytest.mark.parametrize("index", [1, 2])
def test_transcription_invalid_result_rejected_at_correct_track(index):
    transcriber = Transcriber(output=lambda call: make_transcription() if call != index else "bad")
    with pytest.raises(PipelineExecutionError, match=f"track {index} is invalid"):
        execute_separate_transcribe(make_clip(), separator=Separator(), transcriber=transcriber)
    assert len(transcriber.calls) == index


@pytest.mark.parametrize("stage", ["separator", "transcriber"])
def test_backend_errors_propagate_without_masking(stage):
    failure = RuntimeError("backend failed")
    sep = Separator(error=failure if stage == "separator" else None)
    transcriber = Transcriber(error=failure if stage == "transcriber" else None)
    with pytest.raises(RuntimeError, match="backend failed") as exc:
        execute_separate_transcribe(make_clip(), separator=sep, transcriber=transcriber)
    assert exc.value is failure
    assert len(transcriber.calls) == (0 if stage == "separator" else 1)


@pytest.mark.parametrize("first,second", [(4.0, 0.2), (0.2, 4.0), (8.0, 5.0)])
def test_explicit_shared_gain_normalizes_asr_without_modifying_stems(first, second):
    clip = make_clip(amplitude=0.3)
    separation = make_separation(clip, first=first, second=second)
    original = tuple(source.samples.copy() for source in separation.sources)
    transcriber = Transcriber()

    result = execute_separate_transcribe(
        clip, separator=Separator(output=separation), transcriber=transcriber,
        language="en", normalize_for_transcription=True,
    )

    expected_gain = 1.0 / max(1.0, *(float(np.max(np.abs(source))) for source in original))
    assert result.transcription_gain == pytest.approx(expected_gain)
    assert result.separation is separation
    assert tuple(item.text for item in result.transcriptions) == ("track 1", "track 2")
    for index, (asr_clip, language) in enumerate(transcriber.calls):
        assert language == "en"
        assert asr_clip is not separation.sources[index]
        assert asr_clip.sample_rate == clip.sample_rate
        assert asr_clip.frames == clip.frames
        assert not asr_clip.samples.flags.writeable
        assert np.max(np.abs(asr_clip.samples)) <= 1.0
        np.testing.assert_allclose(asr_clip.samples, original[index] * expected_gain, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(separation.sources[index].samples, original[index])


def test_normalization_opt_in_leaves_bounded_audio_and_identity_gain_untouched():
    clip = make_clip()
    transcriber = Transcriber()
    result = execute_separate_transcribe(
        clip, separator=Separator(), transcriber=transcriber, normalize_for_transcription=True,
    )
    assert result.transcription_gain == 1.0
    assert transcriber.calls[0][0] is result.separation.speaker_1
    assert transcriber.calls[1][0] is result.separation.speaker_2


@pytest.mark.parametrize("invalid", [None, 0, 1, 0.0, 1.0, "True", [], object()])
def test_invalid_normalization_mode_rejected_before_inference(invalid):
    separator, transcriber = Separator(), Transcriber()
    with pytest.raises(TypeError, match="normalize_for_transcription"):
        execute_separate_transcribe(
            make_clip(), separator=separator, transcriber=transcriber,
            normalize_for_transcription=invalid,
        )
    assert separator.calls == transcriber.calls == []


@pytest.mark.parametrize("invalid", [None, False, True, 0, 1, -0.1, 0.0, 1.00001, float("inf"), float("nan"), "0.5"])
def test_result_rejects_invalid_transcription_gain(invalid):
    clip = make_clip()
    with pytest.raises(ValueError, match="transcription_gain"):
        SeparateTranscribeResult(
            clip, make_separation(clip), (make_transcription(), make_transcription()), invalid,
        )


def test_strict_mode_rejects_overrange_even_if_normalization_option_not_passed():
    clip = make_clip()
    transcriber = Transcriber()
    with pytest.raises(PipelineExecutionError, match="track 1 exceeds"):
        execute_separate_transcribe(
            clip, separator=Separator(output=make_separation(clip, first=4.0)), transcriber=transcriber,
        )
    assert transcriber.calls == []
