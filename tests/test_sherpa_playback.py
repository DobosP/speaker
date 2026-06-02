"""Playback-sink + streaming-synthesis logic for the local sherpa engine.

No audio hardware, no sherpa-onnx, no models: a fake TTS is injected and the
pure synthesis/queue logic is exercised directly. The threaded OutputStream
path (``_playback_loop``) needs a real device and is out of scope here, so the
testable pieces (`_synthesize`, queue/flush) are factored out of the thread.
"""
from __future__ import annotations

import numpy as np

from core.engines._aec import FarEndRing, PlaybackFIFO
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.metrics import TTS_FIRST_AUDIO


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


# --- barge-in + shutdown now go through the playback FIFO ---
# The callback-OutputStream rewrite means barge-in no longer abort()s the stream;
# it FLUSHES the playback FIFO (the next PortAudio callback emits silence) while
# the stream stays open. Shutdown stop()/close()s the stream (clean device
# handback; PortAudio joins the callback) after flushing the FIFO. So the fakes
# below model stop()/close() (not abort), and stop_speaking's cut now also
# requires a FIFO + a speaking stream.


class _FakeOutStream:
    def __init__(self):
        self.stopped = 0
        self.closed = 0

    def stop(self):
        self.stopped += 1

    def close(self):
        self.closed += 1


class _FakeFIFO:
    def __init__(self):
        self.flushed = 0

    def flush(self):
        self.flushed += 1

    def count(self):
        return 0


def test_stop_speaking_flushes_live_fifo_and_stamps_metric():
    # Barge-in must drop queued audio (flush the FIFO so the next callback emits
    # silence), not just stop feeding it -- and stamp the true audible-stop
    # instant at that moment. Requires a live stream + FIFO + a speaking run.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._out_stream = _FakeOutStream()
    eng._fifo = _FakeFIFO()
    eng._speaking.set()
    eng.stop_speaking()
    assert eng._fifo.flushed == 1
    assert "barge_in_stop" in metrics


def test_stop_speaking_without_live_stream_is_silent_noop():
    # Nothing playing -> no flush, no spurious barge_in_stop metric.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng.stop_speaking()  # _out_stream/_fifo are None, not speaking
    assert "barge_in_stop" not in metrics


# --- clean shutdown: stop() must tear the live stream down so a wedged play ---
# thread can exit. On a dead device the play thread can block in FIFO.write();
# the queue sentinel can't wake it, so stop() would hang on the join. stop() now
# flushes the FIFO (releasing a blocked producer) and stop()/close()s the live
# stream first (a clean handback; PortAudio joins the callback) so the daemon
# play thread exits.


def test_stop_flushes_and_closes_live_stream_before_join():
    # With a live stream + FIFO set, stop() flushes the FIFO and stop()/close()s
    # the stream (NOT abort) but emits NO barge_in_stop metric -- this is
    # teardown, not an interruption.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._out_stream = _FakeOutStream()
    eng._fifo = _FakeFIFO()
    eng._running.set()
    eng.stop()  # no capture/play threads started -> joins are no-ops
    assert eng._out_stream.stopped == 1 and eng._out_stream.closed == 1
    assert eng._fifo.flushed == 1
    assert "barge_in_stop" not in metrics  # teardown, not a barge-in
    assert not eng._running.is_set()
    assert eng._stop_speaking.is_set()


def test_stop_without_live_stream_is_noop():
    # No live stream -> stop() must not raise and must still tear down cleanly.
    eng = _engine(_StreamingTts())
    eng._running.set()
    eng.stop()  # _out_stream is None
    assert not eng._running.is_set()


def test_stop_returns_promptly_when_producer_is_blocked():
    # Simulate the dead-device hang: a play thread blocked in FIFO.write() until
    # the abort predicate flips. stop() sets _stop_speaking + clears _running
    # (the should_abort predicate) and flushes the FIFO (notify_all), both of
    # which release the blocked producer, then joins within its bounded timeout
    # instead of hanging forever.
    import threading

    eng = _engine(_StreamingTts())
    fifo = PlaybackFIFO(4)  # tiny so the producer blocks immediately when full
    eng._fifo = fifo
    eng._out_stream = _FakeOutStream()
    eng._running.set()
    eng._stop_speaking.clear()

    def fake_play():
        # Block in FIFO.write() like the real producer; the should_abort
        # predicate (set by stop()) releases it and it returns cleanly.
        fifo.write(
            np.ones(100, dtype="float32"),  # >> capacity -> blocks until aborted
            should_abort=lambda: eng._stop_speaking.is_set() or not eng._running.is_set(),
        )

    t = threading.Thread(target=fake_play, daemon=True)
    eng._play_thread = t
    t.start()

    done = threading.Event()

    def call_stop():
        eng.stop()
        done.set()

    threading.Thread(target=call_stop, daemon=True).start()
    # stop() bounds each join at 1.0s; with the abort predicate unblocking the
    # producer it should finish well under that.
    assert done.wait(timeout=3.0), "stop() hung instead of returning promptly"
    assert not t.is_alive()
    # stop() flushes the FIFO but never nulls it (the worker loop owns that, on
    # idle-release/teardown), so the same instance survives and was emptied.
    assert eng._fifo is fifo
    assert eng._fifo.count() == 0


# --- _audio_cb: the core of the callback rewrite (drain + tee + first-audio) ---
# _audio_cb is the SINGLE place the far-end AEC ref / coherence ref / level EWMA
# are teed and TTS_FIRST_AUDIO is stamped -- all off TRUE playback, which is the
# whole point of the rewrite. It is pure Python (no device, no models), so we
# drive it directly. Setting _play_sr == config.sample_rate keeps the far-ring
# tee resample-free so we can assert exact samples; _echo_coherence stays None
# (the callback guards on it) to keep the test scipy-free.


def _outbuf(frames):
    # -1.0 sentinel so any cell the callback fails to write is caught.
    return np.full((frames, 1), -1.0, dtype="float32")


def test_audio_cb_drains_fifo_zero_fills_underrun_and_tees_only_real_samples():
    eng = _engine(_StreamingTts())
    eng._play_sr = eng.config.sample_rate  # no resample -> far ring gets exact samples
    eng._far_ref = FarEndRing()
    eng._echo_coherence = None
    eng._fifo = PlaybackFIFO(8)
    eng._fifo.write(np.array([0.5, 0.25], dtype="float32"), lambda: False)  # float32-exact
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._first_audio_pending = True

    out = _outbuf(5)
    eng._audio_cb(out, 5, None, None)

    # The 2 queued samples play; the 3-sample underrun tail is silent zero-fill.
    np.testing.assert_array_equal(out[:, 0], [0.5, 0.25, 0.0, 0.0, 0.0])
    # ONLY the 2 real samples are teed into the far ring -- NOT the zero tail.
    # (A regression to teeing `view`/`view[:frames]` would push silence into the
    # AEC far-end and stay green without this assertion.)
    assert eng._far_ref._written == 2
    np.testing.assert_array_equal(eng._far_ref.read(2, 0), [0.5, 0.25])
    # TTS_FIRST_AUDIO stamped exactly once, at the first real-audio block.
    assert metrics == [TTS_FIRST_AUDIO]
    assert eng._first_audio_pending is False


def test_audio_cb_does_not_stamp_first_audio_on_an_underrun_only_block():
    eng = _engine(_StreamingTts())
    eng._play_sr = eng.config.sample_rate
    eng._far_ref = FarEndRing()
    eng._fifo = PlaybackFIFO(8)  # empty -> pure underrun, n == 0
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._first_audio_pending = True  # armed...

    out = _outbuf(4)
    eng._audio_cb(out, 4, None, None)

    np.testing.assert_array_equal(out[:, 0], [0.0, 0.0, 0.0, 0.0])  # all silence
    assert eng._far_ref._written == 0  # nothing teed
    assert metrics == []  # ...no real audio played -> no stamp
    assert eng._first_audio_pending is True  # stays armed for the first real block


def test_audio_cb_with_no_fifo_emits_silence_and_does_not_tee_or_stamp():
    eng = _engine(_StreamingTts())
    eng._play_sr = eng.config.sample_rate
    eng._far_ref = FarEndRing()
    eng._fifo = None  # before the first utterance / after teardown
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._first_audio_pending = True

    out = _outbuf(3)
    eng._audio_cb(out, 3, None, None)

    np.testing.assert_array_equal(out[:, 0], [0.0, 0.0, 0.0])
    assert eng._far_ref._written == 0
    assert metrics == []
    assert eng._first_audio_pending is True
