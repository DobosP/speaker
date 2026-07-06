"""Continuous no-duck WORD-CUT barge for the OS echo-cancel path (ADR-0013).

The Phase B live experiment (2026-07-06) proved the OS voice-comm canceller
keeps the near-end user CLEAN during playback, so barge-in becomes a word-content
decision: feed the streaming recognizer every playback block and hard-cut on
>= ``barge_word_cut_min_words`` NEW non-own-speech words (or a bare stop
command) -- no duck, no level gate. Until now the path had ZERO headless
coverage (the exact green-headless/failed-live pattern that sank prior barge
phases), so these pin the no-false-cut gates the three adversarial verifiers
demanded -- the 4-word floor (garbled 2-word echo fragments must NOT cut) and
the per-speech-burst stream reset -- plus the fire path, its suppress guards,
and the ``self._aec is None`` scoping. State-machine fakes only: no audio
device, no models, no threads (Tier 0), mirroring ``test_barge_confirm``.
"""
from __future__ import annotations

import time

import numpy as np

from core.engine import EngineCallbacks
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


class _FakeStream:
    def __init__(self):
        self.fed_blocks = 0

    def accept_waveform(self, sr, samples):
        self.fed_blocks += 1


class _FakeRecognizer:
    """Streaming-recognizer stand-in: yields scripted partials in order."""

    def __init__(self, partials):
        self._partials = list(partials)
        self._i = -1
        self.resets = 0

    def is_ready(self, stream):
        return False  # decode loop is a no-op for the fake

    def decode_stream(self, stream):  # pragma: no cover - never reached
        pass

    def get_result(self, stream):
        # Advance one scripted partial per call, then hold the last.
        if self._i < len(self._partials) - 1:
            self._i += 1
        return self._partials[self._i] if self._partials else ""

    def reset(self, stream):
        self.resets += 1


class _FakeVad:
    """The step feeds it every block (``accept_waveform``) then consults
    ``is_speech_detected()`` -- the live run-20260706-231226 failure was exactly
    a step that consulted WITHOUT feeding, so the fake counts the feeds."""

    def __init__(self, speech: bool = True) -> None:
        self._speech = bool(speech)
        self.accepted = 0

    def accept_waveform(self, samples) -> None:
        self.accepted += 1

    def is_speech_detected(self) -> bool:
        return self._speech

    def set_speech(self, speech: bool) -> None:
        self._speech = bool(speech)


class _Rec:
    def __init__(self):
        self.barges = 0
        self.metrics: list[str] = []

    def callbacks(self) -> EngineCallbacks:
        return EngineCallbacks(
            on_barge_in=self._on_barge, on_metric=self._on_metric
        )

    def _on_barge(self):
        self.barges += 1

    def _on_metric(self, name, **kw):
        self.metrics.append(name)


def _engine(
    rec: _Rec | None = None, *, aec=None, vad_speech: bool = True, **cfg
) -> SherpaOnnxEngine:
    eng = SherpaOnnxEngine(SherpaConfig(barge_word_cut_enabled=True, **cfg))
    eng._aec = aec  # None = OS echo-cancel path -> word-cut live (ADR-0013)
    eng._vad = _FakeVad(vad_speech)
    if rec is not None:
        eng._cb = rec.callbacks()
    return eng


_BLOCK = np.zeros(1600, dtype="float32")  # one 0.1s block at 16kHz


# --- config / scoping -----------------------------------------------------------


def test_word_cut_disabled_by_default():
    c = SherpaConfig()
    assert c.barge_word_cut_enabled is False  # shipped default: legacy paths intact
    assert c.barge_word_cut_min_words == 4    # the garbled-echo no-false-cut floor
    assert c.barge_word_cut_reset_quiet_blocks == 3  # flicker-proof burst reset


def test_config_from_dict_roundtrip():
    c = SherpaConfig.from_dict(
        {
            "barge_word_cut_enabled": True,
            "barge_word_cut_min_words": 3,
            "barge_word_cut_reset_quiet_blocks": 5,
        }
    )
    assert c.barge_word_cut_enabled is True
    assert c.barge_word_cut_min_words == 3
    assert c.barge_word_cut_reset_quiet_blocks == 5


def test_active_only_when_flag_set_and_no_inapp_aec():
    # Flag ON + no in-app AEC/APM (the OS echo-cancel path) -> live.
    assert _engine()._barge_word_cut_active() is True
    # Flag ON but an in-app canceller owns the decision -> inert (byte-identical
    # legacy/APM paths; the acoustic DTD machinery handles barge there).
    assert _engine(aec=object())._barge_word_cut_active() is False
    # Flag OFF (default) -> inert even on the OS path.
    eng = SherpaOnnxEngine(SherpaConfig())
    eng._aec = None
    assert eng._barge_word_cut_active() is False


# --- the cut: word content decides, never level, never a duck --------------------


def test_four_novel_words_cut_with_no_duck():
    rec = _Rec()
    eng = _engine(rec)
    r, s = _FakeRecognizer(["what are", "what are you doing"]), _FakeStream()
    now = time.monotonic()
    # 2 words: below the floor, no cut -- and the volume was never touched.
    assert eng._barge_word_cut_step(r, s, _BLOCK, now) is False
    assert rec.barges == 0
    assert eng._duck_gain == 1.0
    # 4 novel non-own words: hard-cut, still no duck at any point.
    assert eng._barge_word_cut_step(r, s, _BLOCK, now + 0.1) is True
    assert rec.barges == 1
    assert "barge_in_confirmed" in rec.metrics
    assert eng._duck_gain == 1.0                       # no-duck by construction
    assert eng._barge_in_fired_this_run is True        # one-cut-per-run latch
    assert eng._barge_in_suppressed_until > now        # debounce armed
    assert eng._word_cut_fed_stream is False           # user words kept as pre-roll


def test_bare_stop_command_cuts_alone():
    rec = _Rec()
    eng = _engine(rec)
    r, s = _FakeRecognizer(["stop"]), _FakeStream()
    assert eng._barge_word_cut_step(r, s, _BLOCK, time.monotonic()) is True
    assert rec.barges == 1


def test_garbled_two_word_echo_does_not_cut():
    # The verifier-demanded scenario (ADR-0013): nonlinear-speaker residual echo
    # transcribes as garbled 2-word junk ("YOU'RE ANY") that does NOT match the
    # played text -- _reads_like_own_speech can't reject it, so ONLY the 4-word
    # floor stands between it and a false cut on a silent reply.
    rec = _Rec()
    eng = _engine(rec)
    eng._now_playing = "once upon a time there was a mighty dragon in the hills"
    r, s = _FakeRecognizer(["you're any"]), _FakeStream()
    assert eng._barge_word_cut_step(r, s, _BLOCK, time.monotonic()) is False
    assert rec.barges == 0
    assert eng._duck_gain == 1.0
    # Novel (non-own) text is NOT folded into the base -- a real user's words
    # must keep accumulating toward the floor across blocks.
    assert eng._word_cut_base == ""


def test_own_echo_folds_into_base_then_user_words_cut():
    rec = _Rec()
    eng = _engine(rec)
    eng._now_playing = "once upon a time there was"
    r = _FakeRecognizer(
        ["upon a time", "upon a time what are you doing there"]
    )
    s = _FakeStream()
    now = time.monotonic()
    # Pure echo (reads like own speech) folds into the base so it can't pile
    # up toward the floor or swamp the novelty diff...
    assert eng._barge_word_cut_step(r, s, _BLOCK, now) is False
    assert eng._word_cut_base == "upon a time"
    # ...and the user's genuinely new words then clear the floor on their own.
    assert eng._barge_word_cut_step(r, s, _BLOCK, now + 0.1) is True
    assert rec.barges == 1


# --- per-speech-burst stream reset (the second no-false-cut gate) ----------------


def test_vad_quiet_run_resets_stream_and_base_after_debounce():
    # Default debounce = 3 consecutive quiet blocks (~300 ms). A single quiet
    # block is VAD flicker on OS-cancelled double-talk, NOT a burst boundary --
    # the live batch (run-20260706-231226) lost its accumulated talk-over words
    # to exactly that hair trigger.
    rec = _Rec()
    eng = _engine(rec, vad_speech=False)
    eng._word_cut_fed_stream = True     # a prior burst fed the recognizer
    eng._word_cut_base = "stale burst text"
    r, s = _FakeRecognizer(["anything"]), _FakeStream()
    now = time.monotonic()
    # Two quiet blocks: still inside the debounce -> NO reset yet.
    assert eng._barge_word_cut_step(r, s, _BLOCK, now) is False
    assert eng._barge_word_cut_step(r, s, _BLOCK, now + 0.1) is False
    assert r.resets == 0
    assert eng._word_cut_base == "stale burst text"
    # Third consecutive quiet block: a real burst boundary -> reset fires.
    assert eng._barge_word_cut_step(r, s, _BLOCK, now + 0.2) is False
    assert r.resets == 1                # recognizer state cleared...
    assert eng._word_cut_base == ""     # ...and the novelty base with it
    assert eng._word_cut_fed_stream is False
    assert s.fed_blocks == 0            # quiet blocks are never fed
    # Idempotent: further quiet blocks don't keep resetting.
    assert eng._barge_word_cut_step(r, s, _BLOCK, now + 0.3) is False
    assert r.resets == 1


def test_reset_quiet_blocks_one_restores_hair_trigger():
    # Knob parity: 1 restores the original single-quiet-block semantics.
    eng = _engine(vad_speech=False, barge_word_cut_reset_quiet_blocks=1)
    eng._word_cut_fed_stream = True
    eng._word_cut_base = "stale"
    r, s = _FakeRecognizer(["anything"]), _FakeStream()
    assert eng._barge_word_cut_step(r, s, _BLOCK, time.monotonic()) is False
    assert r.resets == 1
    assert eng._word_cut_base == ""


def test_speech_block_between_quiets_defuses_the_debounce():
    # quiet, quiet, SPEECH, quiet, quiet: the run never reaches 3 consecutive
    # -> no reset, and the burst's accumulated text survives the flicker.
    eng = _engine(vad_speech=False)
    eng._word_cut_fed_stream = True
    eng._word_cut_base = "kept"
    r, s = _FakeRecognizer(["kept and more user words here"]), _FakeStream()
    now = time.monotonic()
    eng._barge_word_cut_step(r, s, _BLOCK, now)
    eng._barge_word_cut_step(r, s, _BLOCK, now + 0.1)
    eng._vad.set_speech(True)           # flicker ends: user is audible again
    eng._barge_word_cut_step(r, s, _BLOCK, now + 0.2)
    eng._vad.set_speech(False)
    eng._barge_word_cut_step(r, s, _BLOCK, now + 0.3)
    eng._barge_word_cut_step(r, s, _BLOCK, now + 0.4)
    assert r.resets == 0
    assert eng._word_cut_base == "kept"


# --- the live-failure regression: the step must FEED the VAD it consults ---------


def test_step_feeds_vad_every_block():
    # run-20260706-231226: nothing fed the VAD during playback, so
    # is_speech_detected() stayed frozen quiet and the recognizer was starved
    # for the whole reply (zero words -> the talk-over batch could not cut).
    # The step must accept THIS block into the VAD before consulting it.
    eng = _engine(vad_speech=True)
    r, s = _FakeRecognizer([""]), _FakeStream()
    now = time.monotonic()
    for i in range(3):
        eng._barge_word_cut_step(r, s, _BLOCK, now + i * 0.1)
    assert eng._vad.accepted == 3       # fed on speech blocks...
    eng._vad.set_speech(False)
    eng._barge_word_cut_step(r, s, _BLOCK, now + 0.3)
    assert eng._vad.accepted == 4       # ...and on quiet blocks alike


# --- funnel telemetry (ADR-0013 post-hoc scoring) ---------------------------------


def test_funnel_counters_and_emission(caplog):
    import logging

    eng = _engine(vad_speech=True)
    r = _FakeRecognizer(["what are", "what are you doing today"])
    s = _FakeStream()
    now = time.monotonic()
    with caplog.at_level(logging.INFO, logger="speaker.sherpa"):
        eng._barge_word_cut_step(r, s, _BLOCK, now)          # 2 words: trace
        eng._barge_word_cut_step(r, s, _BLOCK, now + 0.1)    # 5 words: cut
        assert eng._wc_stats.get("fed") == 2
        assert eng._wc_stats.get("max_words") == 5
        assert eng._wc_stats.get("cuts") == 1
        eng._wc_reply_active = True
        eng._emit_word_cut_funnel()
    text = caplog.text
    assert "word-cut trace: 2 word(s)" in text
    assert "word-cut funnel: fed=2" in text
    assert "max_words=5" in text
    assert "cuts=1" in text
    assert eng._wc_stats == {}           # stats are per-reply, cleared on emit


def test_burst_reset_logs_dropped_words(caplog):
    import logging

    # A debounced reset that wipes accumulated NOVEL words must say so -- it is
    # the smoking gun for "user words swallowed by the burst reset".
    eng = _engine(vad_speech=False)
    eng._word_cut_fed_stream = True
    r, s = _FakeRecognizer(["tell me something else"]), _FakeStream()
    now = time.monotonic()
    with caplog.at_level(logging.INFO, logger="speaker.sherpa"):
        for i in range(3):
            eng._barge_word_cut_step(r, s, _BLOCK, now + i * 0.1)
    assert "word-cut burst reset: dropped 4 word(s)" in caplog.text
    assert eng._wc_stats.get("dropped_words") == 4
    assert eng._wc_stats.get("resets") == 1


def test_decode_error_counted_and_warned_once(caplog):
    import logging

    class _BoomRecognizer(_FakeRecognizer):
        def get_result(self, stream):
            raise RuntimeError("decoder crashed")

    eng = _engine(vad_speech=True)
    r, s = _BoomRecognizer([]), _FakeStream()
    now = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="speaker.sherpa"):
        assert eng._barge_word_cut_step(r, s, _BLOCK, now) is False
        assert eng._barge_word_cut_step(r, s, _BLOCK, now + 0.1) is False
    assert eng._wc_stats.get("decode_errors") == 2
    # Warn once per reply, then count silently (no per-block log spam).
    assert caplog.text.count("word-cut: recognizer decode failed") == 1


def test_near_end_window_emits_after_two_seconds(caplog):
    import logging

    eng = _engine(vad_speech=True)
    r, s = _FakeRecognizer([""]), _FakeStream()
    now = time.monotonic()
    with caplog.at_level(logging.INFO, logger="speaker.sherpa"):
        eng._barge_word_cut_step(r, s, _BLOCK, now)         # opens the window
        eng._barge_word_cut_step(r, s, _BLOCK, now + 2.1)   # crosses 2s -> emits
    assert "word-cut near-end: rms_avg=" in caplog.text
    assert "vad_frac=1.00" in caplog.text
    # The window fed the per-reply percentile accumulator and reset itself.
    assert eng._wc_stats.get("win_rms")
    assert eng._wc_win == {}


# --- suppress guards (shared with the acoustic path) ------------------------------


def test_one_cut_per_run_latch_suppresses():
    rec = _Rec()
    eng = _engine(rec)
    eng._barge_in_fired_this_run = True
    r, s = _FakeRecognizer(["what are you doing"]), _FakeStream()
    assert eng._barge_word_cut_step(r, s, _BLOCK, time.monotonic()) is False
    assert rec.barges == 0


def test_suppress_window_and_refractory_block_the_cut():
    rec = _Rec()
    eng = _engine(rec)
    r, s = _FakeRecognizer(["what are you doing"]), _FakeStream()
    now = time.monotonic()
    # Debounce suppress window.
    eng._barge_in_suppressed_until = now + 10.0
    assert eng._barge_word_cut_step(r, s, _BLOCK, now) is False
    eng._barge_in_suppressed_until = 0.0
    # Post-speaking refractory: the just-cancelled reply's echo tail.
    eng._last_speaking_end = now
    assert eng._barge_word_cut_step(r, s, _BLOCK, now + 0.1) is False
    assert rec.barges == 0
    # Both cleared -> the same text cuts.
    eng._last_speaking_end = 0.0
    assert eng._barge_word_cut_step(r, s, _BLOCK, now + 0.2) is True
    assert rec.barges == 1


def test_playback_onset_grace_blocks_early_cut():
    rec = _Rec()
    eng = _engine(rec)
    r, s = _FakeRecognizer(["what are you doing"]), _FakeStream()
    now = time.monotonic()
    eng._playback_onset_at = now  # reply-onset echo transient window
    assert eng._barge_word_cut_step(r, s, _BLOCK, now + 0.1) is False
    assert rec.barges == 0
    # Past the grace window the accumulated words cut normally.
    grace = eng.config.barge_in_playback_onset_grace_sec
    assert eng._barge_word_cut_step(r, s, _BLOCK, now + grace + 0.1) is True
    assert rec.barges == 1
