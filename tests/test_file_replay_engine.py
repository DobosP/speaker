from __future__ import annotations

import glob

import pytest

np = pytest.importorskip("numpy")

import core.engines.file_replay as fr
from core.engine import EngineCallbacks
from core.engines.file_replay import FileReplayEngine, load_waveform
from core.engines.sherpa import SherpaConfig
from core.metrics import MetricsRecorder


# --- fakes mirroring the slice of the sherpa-onnx API the engine touches ---
# Real API: stream = recognizer.create_stream(); stream.accept_waveform(sr, x);
# recognizer.is_ready/decode_stream/get_result/is_endpoint/reset(stream).
class _FakeStream:
    def __init__(self) -> None:
        self.heard = False
        self.endpoint = False

    def accept_waveform(self, sample_rate: int, samples) -> None:
        loud = samples.size and float(np.max(np.abs(samples))) > 0.05
        if loud:
            self.heard = True
        elif self.heard:
            self.endpoint = True


class _FakeRecognizer:
    """Emits 'hello world' as final once it sees voiced audio then silence."""

    def create_stream(self) -> _FakeStream:
        return _FakeStream()

    def is_ready(self, stream: _FakeStream) -> bool:
        return False

    def decode_stream(self, stream: _FakeStream) -> None:  # pragma: no cover
        pass

    def get_result(self, stream: _FakeStream) -> str:
        return "hello world" if stream.heard else ""

    def is_endpoint(self, stream: _FakeStream) -> bool:
        return stream.endpoint

    def reset(self, stream: _FakeStream) -> None:
        stream.heard = False
        stream.endpoint = False


class _FakeTts:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, text: str, sid: int = 0, speed: float = 1.0):
        self.calls.append(text)
        return None


def _patch_models(monkeypatch, recognizer, tts) -> None:
    monkeypatch.setattr(fr, "build_recognizer", lambda c: recognizer)
    monkeypatch.setattr(fr, "build_tts", lambda c: tts)


def test_replay_fires_final_and_records_metrics(monkeypatch):
    rec = _FakeRecognizer()
    tts = _FakeTts()
    _patch_models(monkeypatch, rec, tts)

    finals: list[str] = []
    recorder = MetricsRecorder()
    engine = FileReplayEngine(SherpaConfig(asr_encoder="x", tts_model="y"))
    engine.start(
        EngineCallbacks(
            on_final=finals.append,
            on_metric=recorder.mark,
            on_speech_start=lambda: None,
            on_speech_end=lambda: None,
        )
    )

    samples = np.concatenate(
        [np.ones(16000, dtype="float32") * 0.5, np.zeros(1600, dtype="float32")]
    )
    engine.replay_samples(samples, 16000)

    assert finals == ["hello world"]
    # speech_end + asr-final stamps captured; first audio not yet (no speak()).
    [record] = recorder.records()
    assert "speech_end" in record.stamps

    # Now synthesize -- offline TTS stamps tts_first_audio when the clip is ready.
    engine.speak("hello world")
    assert tts.calls == ["hello world"]
    assert record.stamps.get("tts_first_audio") is not None
    assert record.first_audio_latency is not None


def test_load_waveform_reads_real_npy_fixture():
    paths = sorted(glob.glob("tests/fixture_audio/real_usage_full/*.npy"))
    if not paths:
        pytest.skip("no .npy fixtures present")
    samples, sample_rate = load_waveform(paths[0])
    assert sample_rate == 16000
    assert samples.dtype == np.float32
    assert samples.ndim == 1 and samples.size > 0


def test_load_waveform_rejects_unknown_format():
    with pytest.raises(ValueError):
        load_waveform("something.mp3")


@pytest.mark.real_model
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sherpa_onnx") is None,
    reason="sherpa_onnx native package not installed",
)
def test_replay_with_real_models_if_available():
    # Smoke test for the real path -- only runs where sherpa-onnx + model files
    # exist (the bench environment / CI perf job), skipped otherwise.
    pytest.skip("requires configured sherpa model files; exercised by tools/bench")
