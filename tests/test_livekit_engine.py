"""LiveKitEngine logic tests with fakes -- no livekit server / models / audio.

The async room/audio path needs a live LiveKit server and so is out of scope
here; these cover the thread-safe stepping logic (streaming ASR, barge-in) and
the no-op speak path, plus import-safety without the optional deps.
"""
import numpy as np

from core.engine import EngineCallbacks
from core.engines.livekit import STT_SR, LiveKitEngine
from core.engines.sherpa import SherpaConfig


class _FakeStream:
    def __init__(self):
        self.fed = []

    def accept_waveform(self, sr, mono):
        self.fed.append((sr, len(mono)))


class _FakeRecognizer:
    """Emits one partial then an endpoint final on the first feed."""

    def __init__(self, result="hello world", endpoint=True):
        self._result = result
        self._endpoint = endpoint
        self._ready = True

    def create_stream(self):
        return _FakeStream()

    def is_ready(self, stream):
        if self._ready:
            self._ready = False
            return True
        return False

    def decode_stream(self, stream):
        pass

    def get_result(self, stream):
        return self._result

    def is_endpoint(self, stream):
        return self._endpoint

    def reset(self, stream):
        self._ready = True


class _FakeVad:
    def __init__(self, speech=True):
        self._speech = speech
        self.fed = 0

    def accept_waveform(self, mono):
        self.fed += 1

    def is_speech_detected(self):
        return self._speech


def _engine():
    return LiveKitEngine(SherpaConfig(), url="ws://localhost:7880", token="t")


def test_feed_asr_fires_partial_and_final():
    eng = _engine()
    partials, finals = [], []
    eng._cb = EngineCallbacks(on_partial=partials.append, on_final=finals.append)
    eng._recognizer = _FakeRecognizer()
    eng._asr_stream = eng._recognizer.create_stream()

    eng._feed_asr(np.zeros(160, dtype=np.float32))

    assert partials == ["hello world"]
    assert finals == ["hello world"]
    assert eng._asr_stream.fed == [(STT_SR, 160)]


def test_feed_asr_noop_without_recognizer():
    eng = _engine()
    finals = []
    eng._cb = EngineCallbacks(on_final=finals.append)
    eng._feed_asr(np.zeros(160, dtype=np.float32))  # no recognizer/stream
    assert finals == []


def test_watch_barge_in_fires_after_min_speech():
    cfg = SherpaConfig(barge_in_min_speech_sec=0.2)
    eng = LiveKitEngine(cfg, url="ws://localhost:7880", token="t")
    barges = []
    eng._cb = EngineCallbacks(on_barge_in=lambda: barges.append(1))
    eng._vad = _FakeVad(speech=True)

    frame = np.zeros(STT_SR // 10, dtype=np.float32)  # 0.1s
    eng._watch_barge_in(frame)
    assert barges == []  # 0.1s < 0.2s threshold
    eng._watch_barge_in(frame)
    assert barges == [1]  # 0.2s reached


def test_watch_barge_in_silence_resets():
    eng = _engine()
    barges = []
    eng._cb = EngineCallbacks(on_barge_in=lambda: barges.append(1))
    eng._vad = _FakeVad(speech=False)
    eng._watch_barge_in(np.zeros(STT_SR, dtype=np.float32))  # 1s but no speech
    assert barges == []
    assert eng._voiced_run == 0.0


def test_speak_without_tts_or_loop_calls_on_done():
    eng = _engine()
    done = []
    eng.speak("hi", on_done=lambda: done.append(1))
    assert done == [1]
    assert eng.is_speaking is False


def test_construction_does_not_require_optional_deps():
    # Building the engine must not import livekit / sherpa_onnx (models are only
    # built in start()); construction should be cheap and dependency-free.
    eng = _engine()
    assert eng._recognizer is None and eng._tts is None and eng._loop is None
