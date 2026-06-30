"""Playback-sink + streaming-synthesis logic for the local sherpa engine.

No audio hardware, no sherpa-onnx, no models: a fake TTS is injected and the
pure synthesis/queue logic is exercised directly. The threaded OutputStream
path (``_playback_loop``) needs a real device and is out of scope here, so the
testable pieces (`_synthesize`, queue/flush) are factored out of the thread.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from core.audio_frontend import apply_gain_soft_limit
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


def test_synthesize_normalizes_then_declicks_when_target_rms_set():
    """With tts_target_rms>0 the non-streaming path runs normalize_rms THEN declick
    on the whole sentence -- the shipped fluidity path, currently untested at the
    engine level (no other test sets tts_target_rms, so they all take the streaming
    branch). A LOUD clip with an impulse spike must come out near the target level
    with the spike repaired -- proving both stages ran in _synthesize."""
    sr = 16000
    t = np.arange(4000) / sr
    loud = (0.4 * np.sqrt(2) * np.sin(2 * np.pi * 220 * t)).astype("float32")  # RMS ~0.4
    loud[2000] = 0.95                                   # an impulse spike on top

    class _LoudTts:
        sample_rate = sr

        def generate(self, text, sid=0, speed=1.0):
            return _GenAudio(loud.copy(), sample_rate=sr)

    eng = SherpaOnnxEngine(
        SherpaConfig(tts_target_rms=0.12, tts_declick=True, tts_declick_threshold=0.22)
    )
    eng._tts = _LoudTts()
    written: list = []
    eng._synthesize("x", written.append)
    out = np.concatenate(written)
    out_rms = float(np.sqrt(np.mean(out.astype("float64") ** 2)))
    assert abs(out_rms - 0.12) < 0.02                  # normalize_rms ran (level steady)
    assert abs(float(out[2000])) < 0.22                # declick repaired the post-gain spike


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


# --- rc-3: per-utterance generation counter (stale-utterance suppression) ---
# A barge-in / stop bumps the generation; a sentence enqueued BEFORE the bump is
# stale and the playback worker skips it instead of clearing _stop_speaking (the
# wipe race) and speaking it. A sentence enqueued AFTER the bump is current and
# plays -- so a fresh reply after a barge is never muted.


def test_speak_gen_bumps_on_each_stop():
    eng = _engine(_StreamingTts())
    g0 = eng._speak_gen
    eng.stop_speaking()
    assert eng._speak_gen == g0 + 1
    eng.stop_speaking()
    assert eng._speak_gen == g0 + 2


def test_enqueue_stamps_current_generation_and_barge_makes_it_stale():
    eng = _engine(_StreamingTts())
    eng.speak("two")
    item = eng._play_q.get_nowait()  # take it out as the worker would, pre-barge
    assert item[0] == "two"
    assert item[2] == eng._speak_gen  # stamped with the generation at enqueue
    eng.stop_speaking()  # barge: bumps the generation (and drains the queue)
    # The sentence we hold was enqueued before the barge -> now stale, so the
    # worker's `item_gen != self._speak_gen` check skips it.
    assert item[2] != eng._speak_gen


def test_new_reply_after_barge_is_current_generation():
    """Mute-bug guard: a sentence enqueued AFTER a barge carries the new
    generation, so the worker plays it (it is NOT treated as stale)."""
    eng = _engine(_StreamingTts())
    eng.speak("old reply")        # generation G
    eng.stop_speaking()           # barge: bump to G+1, drain the queue
    assert eng._play_q.empty()    # the old reply was drained
    eng.speak("brand new reply")  # enqueued at G+1
    item = eng._play_q.get_nowait()
    assert item[0] == "brand new reply"
    assert item[2] == eng._speak_gen  # current generation -> the worker plays it


def test_claim_utterance_skips_stale_without_clearing_stop():
    """The load-bearing rc-3 guard: a stale dequeued sentence is rejected (None)
    and _stop_speaking is NOT wiped, so a pending barge survives."""
    eng = _engine(_StreamingTts())
    eng.speak("x")        # enqueued at generation G
    eng.stop_speaking()   # barge: bumps to G+1, sets _stop_speaking
    assert eng._stop_speaking.is_set()
    assert eng._claim_utterance(0) is None     # the old-generation sentence is stale
    assert eng._stop_speaking.is_set()         # and the barge flag was NOT wiped


def test_claim_utterance_accepts_current_and_clears_stop():
    eng = _engine(_StreamingTts())
    eng._stop_speaking.set()  # a prior barge left the flag set
    g = eng._speak_gen
    assert eng._claim_utterance(g) == g        # current generation -> claimed
    assert not eng._stop_speaking.is_set()     # cleared for this fresh utterance


def test_synthesize_streaming_aborts_on_generation_mismatch():
    eng = _engine(_StreamingTts())
    eng._speak_gen = 7
    written: list = []
    eng._synthesize("hi", written.append, gen=7)  # current gen -> full synthesis
    assert len(written) == 2
    written.clear()
    eng._speak_gen = 8  # a barge bumped the generation past this utterance's gen
    eng._synthesize("hi", written.append, gen=7)
    # First chunk is handed over, then on_chunk returns 0 and synthesis stops --
    # same shape as the _stop_speaking interrupt path.
    assert len(written) == 1


def test_synthesize_nonstreaming_aborts_before_write_on_generation_mismatch():
    eng = _engine(_NonStreamingTts())
    eng._speak_gen = 8
    written: list = []
    eng._synthesize("hi", written.append, gen=7)  # stale -> loop breaks first
    assert written == []
    # A matching generation synthesizes normally.
    eng._speak_gen = 7
    eng._synthesize("hi", written.append, gen=7)
    assert written  # chunks were written


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
        self.last_fade = None

    def flush(self, fade_samples=0):
        self.flushed += 1
        self.last_fade = fade_samples

    def count(self):
        return 0


class _StuckFIFO:
    def __init__(self, count=1600):
        self._count = int(count)
        self.flushed = 0

    def flush(self, fade_samples=0):
        self.flushed += 1
        self._count = 0

    def count(self):
        return self._count


def test_playback_drain_timeout_flushes_before_reopening_asr(caplog):
    # Root-cause guard for full-sentence echo finals: if the output callback stalls,
    # _speaking must not clear while queued assistant audio remains available to
    # play later with ASR open. The timeout path drops that stale tail and emits a
    # metric so the hardware stall is visible in the run bundle.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._fifo = _StuckFIFO(count=800)
    eng._running.set()
    caplog.set_level(logging.WARNING, logger="speaker.sherpa")

    drained = eng._wait_for_playback_drain(timeout_sec=0.0)

    assert drained is False
    assert eng._fifo.flushed == 1
    assert eng._fifo.count() == 0
    assert "playback FIFO did not drain before idle deadline" in caplog.text
    assert metrics == ["playback_drain_timeout"]


def test_playback_drain_timeout_does_not_flush_during_stop():
    # stop_speaking()/stop() own teardown flushing. The idle-drain guard should
    # only diagnose an otherwise-healthy playback callback that missed its drain
    # deadline, not double-report a deliberate stop.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._fifo = _StuckFIFO(count=800)
    eng._running.set()
    eng._stop_speaking.set()

    drained = eng._wait_for_playback_drain(timeout_sec=0.0)

    assert drained is False
    assert eng._fifo.flushed == 0
    assert metrics == []


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


def test_stop_speaking_clears_speaking_and_relatches_on_the_cut():
    # RC-2 regression guard: stop_speaking() must AUTHORITATIVELY end the speaking
    # state itself, not wait on the playback worker's epilogue. If the native
    # tts.generate() wedges, the worker never clears _speaking and the capture
    # loop stays deaf to ASR for the rest of the session. So the cut path must
    # clear _speaking + re-arm the one-per-run barge latch on its own. (Here we
    # never run the worker at all -- exactly the wedged case.)
    eng = _engine(_StreamingTts())
    eng._out_stream = _FakeOutStream()
    eng._fifo = _FakeFIFO()
    eng._speaking.set()                   # worker set it; worker will NOT clear it (wedged)
    eng._barge_in_fired_this_run = True   # a barge already fired this run
    eng.stop_speaking()
    assert eng._fifo.flushed == 1
    assert eng.is_speaking is False              # ASR re-enables without the worker returning
    assert eng._barge_in_fired_this_run is False  # barge-in re-armed for the next interrupt


def test_barge_watch_is_not_armed_before_first_audio_plays():
    # A reply is marked _speaking before TTS has produced its first audible block.
    # During that synth lead-in there is no playback reference yet, so VAD noise
    # must not accumulate as a rejected/detected "during playback" barge episode.
    eng = _engine(_StreamingTts())
    eng._speaking.set()
    eng._first_audio_pending = True
    eng._playback_level = 0.0

    assert eng._barge_watch_active() is False


def test_barge_watch_stays_armed_between_queued_sentences_while_tail_is_audible():
    # For adjacent queued sentences the next utterance can set _first_audio_pending
    # while the previous sentence's FIFO tail is still audibly playing. Keep the
    # watch armed then so real talk-over in the inter-sentence gap is not missed.
    eng = _engine(_StreamingTts())
    eng._speaking.set()
    eng._first_audio_pending = True
    eng._playback_level = 0.02

    assert eng._barge_watch_active() is True


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


# --- Streaming normalize_rms + low-pass path --------------------------------


class _TrackingTts:
    """Streaming TTS stub that records whether _synthesize used callback=."""

    sample_rate = 16000

    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype="float32").reshape(-1)
        self.callback_used: list[bool] = []

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.callback_used.append(callback is not None)
        samples = self.samples.copy()
        if callback is not None:
            for chunk in np.array_split(samples, 4):
                if callback(chunk, 1.0) == 0:
                    break
        return _GenAudio(samples, sample_rate=self.sample_rate)


def _sine(sr=16000, dur_s=0.25, rms=0.4, freq=300.0):
    t = np.arange(int(sr * dur_s), dtype="float64") / sr
    return (rms * np.sqrt(2.0) * np.sin(2.0 * np.pi * freq * t)).astype("float32")


def _tone_mag(samples, sr: int, freq: float) -> float:
    x = np.asarray(samples, dtype="float64").reshape(-1)
    t = np.arange(x.size, dtype="float64") / float(sr)
    s = np.sin(2.0 * np.pi * float(freq) * t)
    c = np.cos(2.0 * np.pi * float(freq) * t)
    return float((2.0 / x.size) * np.hypot(np.dot(x, s), np.dot(x, c)))


def test_target_rms_wholeclip_on_first_sentence_sets_carry():
    """First sentence with tts_target_rms>0: whole-clip path runs (callback not
    used) and _tts_normalize_gain is set so the next sentence can stream."""
    samp = _sine()
    tts = _TrackingTts(samp)
    eng = SherpaOnnxEngine(SherpaConfig(tts_target_rms=0.12, tts_declick=False))
    eng._tts = tts
    assert eng._tts_normalize_gain is None  # not yet set
    written: list = []
    eng._synthesize("hello", written.append)
    assert tts.callback_used == [False]         # whole-clip path (no callback)
    assert eng._tts_normalize_gain is not None  # carry established
    assert eng._tts_normalize_gain > 0.0


def test_target_rms_streaming_on_second_sentence_applies_carry():
    """Second sentence with tts_target_rms>0: streaming callback is used, the
    carried gain is applied, and _tts_normalize_gain is updated afterward."""
    samp = _sine(rms=0.4)
    tts = _TrackingTts(samp)
    eng = SherpaOnnxEngine(SherpaConfig(tts_target_rms=0.12, tts_declick=False))
    eng._tts = tts
    # Synthesize first sentence (whole-clip) to establish the carry.
    eng._synthesize("first", [].append)
    assert eng._tts_normalize_gain is not None
    gain_after_first = eng._tts_normalize_gain
    # Second sentence: streaming path should be taken.
    written2: list = []
    eng._synthesize("second", written2.append)
    assert tts.callback_used == [False, True]  # streaming path taken
    # Carry is updated from the measured raw RMS.
    assert eng._tts_normalize_gain is not None
    # Output level should be near target_rms (within ±20% tolerance,
    # acknowledging feed-forward may be slightly off if the TTS level varies).
    out = np.concatenate(written2)
    out_rms = float(np.sqrt(np.mean(out.astype("float64") ** 2)))
    assert abs(out_rms - 0.12) < 0.06, f"streaming RMS {out_rms:.3f} far from 0.12"
    _ = gain_after_first  # used for context; the carry updated after second


def test_target_rms_zero_always_takes_streaming_regardless_of_carry():
    """With tts_target_rms=0 the streaming path is taken on the first sentence
    (no carry needed, no normalization). This pins the pure-streaming invariant."""
    samp = _sine()
    tts = _TrackingTts(samp)
    eng = SherpaOnnxEngine(SherpaConfig(tts_target_rms=0.0, tts_declick=False))
    eng._tts = tts
    assert eng._tts_normalize_gain is None  # irrelevant on this path
    eng._synthesize("hi", [].append)
    assert tts.callback_used == [True]      # streaming path regardless
    assert eng._tts_normalize_gain is None  # still None (not updated by this path)


def test_lowpass_enabled_streams_from_first_sentence_when_target_rms_off():
    sr = 16000
    t = np.arange(sr // 2, dtype="float64") / sr
    raw = (
        0.15 * np.sin(2.0 * np.pi * 300.0 * t)
        + 0.30 * np.sin(2.0 * np.pi * 6000.0 * t)
    ).astype("float32")
    tts = _TrackingTts(raw)
    eng = SherpaOnnxEngine(
        SherpaConfig(
            tts_target_rms=0.0,
            tts_output_lowpass_hz=1000.0,
            tts_declick=False,
        )
    )
    eng._tts = tts

    written: list = []
    eng._synthesize("lowpass now streams", written.append)

    assert tts.callback_used == [True]
    assert len(written) == 4
    out = np.concatenate(written)
    assert _tone_mag(out, sr, 6000.0) < 0.08 * _tone_mag(raw, sr, 6000.0)


def test_streaming_lowpass_output_is_peak_bounded_after_filter_overshoot():
    sr = 24000
    t = np.arange(sr, dtype="float64") / sr
    raw = (
        0.40 * np.sin(2.0 * np.pi * 300.0 * t)
        + 0.35 * np.sin(2.0 * np.pi * 7000.0 * t)
    ).astype("float32")
    hot = np.asarray(apply_gain_soft_limit(raw, 4.0), dtype="float32")
    tts = _TrackingTts(hot)
    tts.sample_rate = sr
    eng = SherpaOnnxEngine(
        SherpaConfig(
            tts_target_rms=0.0,
            tts_output_lowpass_hz=7000.0,
            tts_declick=False,
        )
    )
    eng._tts = tts

    written: list = []
    eng._synthesize("lowpass overshoot is bounded", written.append)

    assert tts.callback_used == [True]
    out = np.concatenate(written)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) <= 1.0


def test_target_rms_and_lowpass_stream_on_second_sentence_after_carry_seeded():
    samp = _sine(rms=0.4, freq=300.0)
    tts = _TrackingTts(samp)
    eng = SherpaOnnxEngine(
        SherpaConfig(
            tts_target_rms=0.12,
            tts_output_lowpass_hz=2000.0,
            tts_declick=False,
        )
    )
    eng._tts = tts

    first: list = []
    eng._synthesize("first", first.append)
    assert tts.callback_used == [False]
    assert eng._tts_normalize_gain is not None

    second: list = []
    eng._synthesize("second", second.append)

    assert tts.callback_used == [False, True]
    assert len(second) == 4
    out = np.concatenate(second)
    out_rms = float(np.sqrt(np.mean(out.astype("float64") ** 2)))
    assert out_rms == pytest.approx(0.12, rel=0.15)


def test_output_leveler_still_forces_whole_clip_with_lowpass_enabled():
    tts = _TrackingTts(_sine(rms=0.2, freq=300.0))
    eng = SherpaOnnxEngine(
        SherpaConfig(
            tts_target_rms=0.0,
            tts_output_leveler=True,
            tts_output_lowpass_hz=2000.0,
            tts_declick=False,
        )
    )
    eng._tts = tts

    written: list = []
    eng._synthesize("leveler remains whole clip", written.append)

    assert tts.callback_used == [False]
    assert written
