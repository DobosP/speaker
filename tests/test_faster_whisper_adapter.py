from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from core.engines._faster_whisper import FasterWhisperEndpointRecognizer


class _FakeModel:
    def __init__(self, segments=(), *, language_probability=0.98):
        self.segments = tuple(segments)
        self.language_probability = language_probability
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        info = SimpleNamespace(language_probability=self.language_probability)
        return (segment for segment in self.segments), info


class _RecordingFactory:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.model


def _segment(
    text,
    *,
    start,
    end,
    avg_logprob,
    no_speech_prob,
    compression_ratio=1.0,
):
    return SimpleNamespace(
        text=text,
        start=start,
        end=end,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
    )


def test_cached_model_load_and_decode_are_lazy_local_cuda_and_deterministic(tmp_path):
    model = _FakeModel(
        (
            _segment(
                " SENTINEL_PRIVATE",
                start=0.0,
                end=1.0,
                avg_logprob=-0.2,
                no_speech_prob=0.1,
                compression_ratio=1.1,
            ),
            _segment(
                " transcript",
                start=1.0,
                end=3.0,
                avg_logprob=-0.4,
                no_speech_prob=0.3,
                compression_ratio=1.4,
            ),
        )
    )
    factory = _RecordingFactory(model)
    recognizer = FasterWhisperEndpointRecognizer(
        tmp_path,
        model_factory=factory,
    )

    assert factory.calls == []
    result = recognizer.transcribe(np.zeros(48_000, dtype=np.float32), 16_000)

    assert factory.calls == [
        (
            (str(tmp_path.resolve()),),
            {
                "device": "cuda",
                "device_index": 0,
                "compute_type": "float16",
                "num_workers": 1,
                "local_files_only": True,
            },
        )
    ]
    assert len(model.calls) == 1
    decoded_audio, options = model.calls[0]
    assert decoded_audio.dtype == np.float32
    assert decoded_audio.flags.c_contiguous
    assert options == {
        "task": "transcribe",
        "language": "en",
        "beam_size": 5,
        "patience": 1.0,
        "temperature": 0.0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "vad_filter": False,
        "vad_parameters": None,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "prefix": None,
        "hotwords": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "word_timestamps": False,
    }
    assert result.text == "SENTINEL_PRIVATE transcript"
    assert result.avg_logprob == pytest.approx(-1.0 / 3.0)
    assert result.no_speech_probability == pytest.approx(7.0 / 30.0)
    assert result.compression_ratio == pytest.approx(1.3)
    assert result.confidence == pytest.approx(math.exp(-1.0 / 3.0) * (23.0 / 30.0))
    assert result.segment_count == 2
    assert result.language == "en"
    assert result.language_probability == 0.98
    assert result.audio_seconds == 3.0
    assert "SENTINEL_PRIVATE" not in repr(result)
    assert "transcript" not in repr(result)


def test_production_factory_bootstraps_wheel_cuda_before_faster_whisper_model(
    tmp_path,
    monkeypatch,
):
    from core.engines import _cuda_wheels

    events = []
    model = _FakeModel()

    def whisper_model(*args, **kwargs):
        events.append(("model", args, kwargs))
        return model

    fake_module = ModuleType("faster_whisper")
    fake_module.WhisperModel = whisper_model
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(
        _cuda_wheels,
        "preload_cuda_wheel_libraries",
        lambda: events.append(("bootstrap",)),
    )
    recognizer = FasterWhisperEndpointRecognizer(tmp_path)

    recognizer.transcribe(np.zeros(100, dtype=np.float32), 16_000)

    assert events[0] == ("bootstrap",)
    assert events[1][0] == "model"


def test_warm_loads_the_model_once_without_decoding(tmp_path):
    model = _FakeModel()
    factory = _RecordingFactory(model)
    recognizer = FasterWhisperEndpointRecognizer(tmp_path, model_factory=factory)

    recognizer.warm()
    recognizer.warm()

    assert len(factory.calls) == 1
    assert model.calls == []

    recognizer.transcribe(np.zeros(100, dtype=np.float32), 16_000)
    assert len(factory.calls) == 1
    assert len(model.calls) == 1


def test_non_16khz_audio_is_resampled_once_and_flushed(tmp_path):
    events = []

    class _Resampler:
        def __init__(self, source_rate, target_rate, *, quality):
            events.append(("init", source_rate, target_rate, quality))

        def process(self, samples, *, last):
            events.append(("process", samples.copy(), last))
            return np.full(samples.size * 2, 0.25, dtype=np.float64)

    model = _FakeModel(
        (
            _segment(
                " resampled",
                start=0.0,
                end=1.0,
                avg_logprob=-0.1,
                no_speech_prob=0.0,
            ),
        )
    )
    recognizer = FasterWhisperEndpointRecognizer(
        tmp_path,
        model_factory=_RecordingFactory(model),
        resampler_factory=_Resampler,
    )
    source = np.linspace(-0.5, 0.5, 8_000, dtype=np.float32)

    result = recognizer.transcribe(source, 8_000)

    assert events[0] == ("init", 8_000, 16_000, "VHQ")
    assert events[1][0] == "process"
    np.testing.assert_array_equal(events[1][1], source)
    assert events[1][2] is True
    decoded_audio, _ = model.calls[0]
    assert decoded_audio.shape == (16_000,)
    assert decoded_audio.dtype == np.float32
    assert result.text == "resampled"
    assert result.audio_seconds == 1.0


def test_empty_audio_does_not_load_the_model(tmp_path):
    factory = _RecordingFactory(_FakeModel())
    recognizer = FasterWhisperEndpointRecognizer(
        tmp_path,
        model_factory=factory,
    )

    result = recognizer.transcribe(np.array([], dtype=np.float32), 16_000)

    assert factory.calls == []
    assert result.text == ""
    assert result.confidence == 0.0
    assert result.segment_count == 0
    assert result.avg_logprob is None
    assert result.compression_ratio is None


def test_model_identifier_and_non_directory_paths_are_rejected_before_loading(
    tmp_path,
):
    file_path = tmp_path / "model.bin"
    file_path.write_bytes(b"cached")

    with pytest.raises(ValueError, match="local directory"):
        FasterWhisperEndpointRecognizer("not-a-local-model-identifier")
    with pytest.raises(ValueError, match="local directory"):
        FasterWhisperEndpointRecognizer(file_path)


@pytest.mark.parametrize("sample_rate", [0, -1, 16_000.0, True])
def test_invalid_sample_rates_are_rejected_without_loading(tmp_path, sample_rate):
    factory = _RecordingFactory(_FakeModel())
    recognizer = FasterWhisperEndpointRecognizer(
        tmp_path,
        model_factory=factory,
    )

    with pytest.raises(ValueError, match="sample_rate"):
        recognizer.transcribe(np.zeros(10, dtype=np.float32), sample_rate)

    assert factory.calls == []


@pytest.mark.parametrize(
    "pcm",
    [
        np.zeros((2, 10), dtype=np.float32),
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([0.0, np.inf], dtype=np.float32),
        np.array([0, 1], dtype=np.int16),
        np.array(["private text"]),
        0.0,
    ],
)
def test_invalid_pcm_is_rejected_without_leaking_or_loading(tmp_path, pcm):
    factory = _RecordingFactory(_FakeModel())
    recognizer = FasterWhisperEndpointRecognizer(
        tmp_path,
        model_factory=factory,
    )

    with pytest.raises(ValueError, match="finite one-dimensional real audio") as raised:
        recognizer.transcribe(pcm, 16_000)

    assert "private text" not in str(raised.value)
    assert factory.calls == []


def test_missing_or_nonfinite_diagnostics_are_repr_safe_and_not_overstated(tmp_path):
    model = _FakeModel(
        (
            SimpleNamespace(
                text=" SECRET",
                start=float("nan"),
                end=float("inf"),
                avg_logprob=float("nan"),
                no_speech_prob=2.0,
                compression_ratio=float("-inf"),
            ),
        ),
        language_probability=float("inf"),
    )
    recognizer = FasterWhisperEndpointRecognizer(
        tmp_path,
        model_factory=_RecordingFactory(model),
    )

    result = recognizer.transcribe([0.0], 16_000)

    assert result.text == "SECRET"
    assert result.avg_logprob is None
    assert result.no_speech_probability == 1.0
    assert result.compression_ratio is None
    assert result.confidence == 0.0
    assert result.language_probability is None
    assert "SECRET" not in repr(result)
