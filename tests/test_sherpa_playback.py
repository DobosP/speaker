"""Playback-sink + streaming-synthesis logic for the local sherpa engine.

No audio hardware, no sherpa-onnx, no models: a fake TTS is injected and the
pure synthesis/queue logic is exercised directly. The threaded OutputStream
path (``_playback_loop``) needs a real device and is out of scope here, so the
testable pieces (`_synthesize`, queue/flush) are factored out of the thread.
"""
from __future__ import annotations

import numpy as np

from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


class _GenAudio:
    def __init__(self, samples, sample_rate=22050):
        self.samples = samples
        self.sample_rate = sample_rate


class _StreamingTts:
    """Calls the streaming callback with two chunks, like sherpa-onnx."""

    sample_rate = 22050

    def __init__(self):
        self.calls = 0

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.calls += 1
        chunks = [np.array([0.1, 0.2], dtype="float32"), np.array([0.3], dtype="float32")]
        if callback is not None:
            for c in chunks:
                if callback(c, 1.0) == 0:
                    break
            return _GenAudio(np.concatenate(chunks))
        return _GenAudio(np.concatenate(chunks))


class _NonStreamingTts:
    """A build whose generate() has no ``callback`` param (raises TypeError)."""

    sample_rate = 16000

    def generate(self, text, sid=0, speed=1.0):
        return _GenAudio(np.arange(4000, dtype="float32"), sample_rate=16000)


def _engine(tts) -> SherpaOnnxEngine:
    eng = SherpaOnnxEngine(SherpaConfig())
    eng._tts = tts
    return eng


def test_synthesize_streams_chunks_as_produced():
    eng = _engine(_StreamingTts())
    written: list = []
    eng._synthesize("hello there", written.append)
    # Two callback chunks arrived separately (streamed), not one consolidated blob.
    assert len(written) == 2
    assert np.allclose(np.concatenate(written), [0.1, 0.2, 0.3])
    assert eng._tts_can_stream is True


def test_synthesize_stops_early_when_interrupted():
    eng = _engine(_StreamingTts())
    written: list = []
    eng._stop_speaking.set()  # barge-in before synthesis
    eng._synthesize("hello there", written.append)
    # First chunk is handed over, then the callback returns 0 and synthesis stops.
    assert len(written) == 1


def test_synthesize_falls_back_to_chunked_waveform_without_callback():
    eng = _engine(_NonStreamingTts())
    written: list = []
    eng._synthesize("hi", written.append)
    assert eng._tts_can_stream is False  # detected and remembered
    # 4000 samples at 16 kHz => 0.1s (1600-sample) chunks => 3 writes (1600,1600,800).
    assert [len(w) for w in written] == [1600, 1600, 800]
    assert np.concatenate(written).shape[0] == 4000


def test_speak_enqueues_and_stop_speaking_flushes():
    eng = _engine(_StreamingTts())
    eng.speak("one")
    eng.speak("two")
    assert eng._play_q.qsize() == 2
    eng.stop_speaking()
    assert eng._stop_speaking.is_set()
    assert eng._play_q.empty()


def test_speak_without_tts_calls_on_done_immediately():
    eng = SherpaOnnxEngine(SherpaConfig())  # no _tts
    done = []
    eng.speak("hi", on_done=lambda: done.append(1))
    assert done == [1]
    assert eng._play_q.empty()


def test_enqueue_drops_oldest_under_backpressure():
    eng = _engine(_StreamingTts())
    eng._play_q.maxsize = 2
    eng.speak("a")
    eng.speak("b")
    eng.speak("c")  # full -> oldest ("a") dropped, "c" enqueued
    queued = [eng._play_q.get_nowait()[0] for _ in range(eng._play_q.qsize())]
    assert queued == ["b", "c"]


class _FakeOutStream:
    def __init__(self):
        self.aborted = 0

    def abort(self):
        self.aborted += 1


def test_stop_speaking_aborts_live_stream_and_stamps_metric():
    # Barge-in must drop audio already buffered in the device (abort), not just
    # stop feeding it -- and stamp the true audible-stop instant at that moment.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._out_stream = _FakeOutStream()
    eng.stop_speaking()
    assert eng._out_stream.aborted == 1
    assert "barge_in_stop" in metrics


def test_stop_speaking_without_live_stream_is_silent_noop():
    # Nothing playing -> no abort, no spurious barge_in_stop metric.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng.stop_speaking()  # _out_stream is None
    assert "barge_in_stop" not in metrics


# --- clean shutdown: stop() must abort a live stream so a wedged play thread ---
# can exit (fix A). On a dead ALSA device the play thread blocks in the C-level
# out.write(); the queue sentinel can't wake it, so stop() would hang waiting on
# the join. stop() now abort()s the live output stream first, which makes the
# blocked write() return so the daemon play thread exits.


def test_stop_aborts_live_stream_before_join():
    # With a live stream set, stop() abort()s it (mirrors stop_speaking()) but
    # emits NO barge_in_stop metric -- this is teardown, not an interruption.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._out_stream = _FakeOutStream()
    eng._running.set()
    eng.stop()  # no capture/play threads started -> joins are no-ops
    assert eng._out_stream is not None and eng._out_stream.aborted == 1
    assert "barge_in_stop" not in metrics  # teardown, not a barge-in
    assert not eng._running.is_set()
    assert eng._stop_speaking.is_set()


def test_stop_without_live_stream_is_noop():
    # No live stream -> stop() must not raise and must still tear down cleanly.
    eng = _engine(_StreamingTts())
    eng._running.set()
    eng.stop()  # _out_stream is None
    assert not eng._running.is_set()


def test_stop_returns_promptly_when_write_is_blocked():
    # Simulate the dead-device hang: a play thread blocked in out.write() until
    # abort() is called. stop() must abort the stream (unblocking the write) and
    # join the thread within its bounded timeout instead of hanging forever.
    import threading

    eng = _engine(_StreamingTts())

    class _BlockingOut:
        def __init__(self):
            self.released = threading.Event()
            self.aborted = 0

        def write(self, samples):
            # Block like a stalled ALSA device until abort() releases us.
            self.released.wait(timeout=5.0)

        def abort(self):
            self.aborted += 1
            self.released.set()

    out = _BlockingOut()
    eng._out_stream = out
    eng._running.set()
    eng._stop_speaking.clear()

    def fake_play():
        # Mimic the play thread sitting in a blocking write under the closure's
        # except (which swallows the post-abort error because _stop_speaking is
        # set by stop()). The loop checks _running so it exits after the write.
        try:
            out.write(None)
        except Exception:  # noqa: BLE001
            if not eng._stop_speaking.is_set():
                raise

    t = threading.Thread(target=fake_play, daemon=True)
    eng._play_thread = t
    t.start()

    done = threading.Event()

    def call_stop():
        eng.stop()
        done.set()

    threading.Thread(target=call_stop, daemon=True).start()
    # stop() bounds each join at 1.0s; with abort() unblocking the write it
    # should finish well under that.
    assert done.wait(timeout=3.0), "stop() hung instead of returning promptly"
    assert out.aborted == 1
    assert not t.is_alive()
