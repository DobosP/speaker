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
    """The step only calls ``is_speech_detected()`` on it."""

    def __init__(self, speech: bool = True) -> None:
        self._speech = bool(speech)

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


def test_config_from_dict_roundtrip():
    c = SherpaConfig.from_dict(
        {"barge_word_cut_enabled": True, "barge_word_cut_min_words": 3}
    )
    assert c.barge_word_cut_enabled is True
    assert c.barge_word_cut_min_words == 3


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


def test_vad_quiet_block_resets_stream_and_base():
    rec = _Rec()
    eng = _engine(rec, vad_speech=False)
    eng._word_cut_fed_stream = True     # a prior burst fed the recognizer
    eng._word_cut_base = "stale burst text"
    r, s = _FakeRecognizer(["anything"]), _FakeStream()
    assert eng._barge_word_cut_step(r, s, _BLOCK, time.monotonic()) is False
    assert r.resets == 1                # recognizer state cleared...
    assert eng._word_cut_base == ""     # ...and the novelty base with it
    assert eng._word_cut_fed_stream is False
    assert s.fed_blocks == 0            # quiet blocks are never fed
    # Idempotent: further quiet blocks don't keep resetting.
    assert eng._barge_word_cut_step(r, s, _BLOCK, time.monotonic()) is False
    assert r.resets == 1


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
