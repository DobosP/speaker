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
and the OS-path scoping. State-machine fakes only except one finite capture-loop
thread: no audio device or models (Tier 0), mirroring ``test_barge_confirm``.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from core.engine import EngineCallbacks
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


class _FakeStream:
    def __init__(self):
        self.fed_blocks = 0
        self.blocks: list[np.ndarray] = []

    def accept_waveform(self, sr, samples):
        self.fed_blocks += 1
        self.blocks.append(np.asarray(samples, dtype="float32").copy())


class _FakeRecognizer:
    """Streaming-recognizer stand-in: yields scripted partials in order."""

    def __init__(self, partials):
        self._partials = list(partials)
        self._i = -1
        self.resets = 0
        self.reset_streams: list[_FakeStream] = []
        self.endpoint = False

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
        self.reset_streams.append(stream)

    def is_endpoint(self, stream):
        return self.endpoint


class _FakeVad:
    """The step feeds it every block (``accept_waveform``) then consults
    ``is_speech_detected()`` -- the live run-20260706-231226 failure was exactly
    a step that consulted WITHOUT feeding, so the fake counts the feeds."""

    def __init__(self, speech: bool = True) -> None:
        self._speech = bool(speech)
        self.accepted = 0
        self.resets = 0
        self._reactivate_on_accept = False

    def accept_waveform(self, samples) -> None:
        self.accepted += 1
        if self._reactivate_on_accept:
            self._speech = True
            self._reactivate_on_accept = False

    def is_speech_detected(self) -> bool:
        return self._speech

    def set_speech(self, speech: bool) -> None:
        self._speech = bool(speech)
        self._reactivate_on_accept = False

    def reset(self) -> None:
        self.resets += 1
        self._reactivate_on_accept = self._speech
        self._speech = False


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
    eng._word_cut_route_verified = True
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
            "barge_word_cut_vad_preroll_sec": 0.4,
        }
    )
    assert c.barge_word_cut_enabled is True
    assert c.barge_word_cut_min_words == 3
    assert c.barge_word_cut_reset_quiet_blocks == 5
    assert c.barge_word_cut_vad_preroll_sec == 0.4


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


def test_word_cut_fails_closed_when_configured_aec_failed_to_build():
    # Runtime _aec=None can mean either "OS owns cancellation" OR "the selected
    # in-app AEC failed to build". Config intent distinguishes them; never turn a
    # failed APM/DTLN setup into raw-mic word-cut by accident.
    eng = _engine(aec_enabled=True)
    eng._aec = None
    assert eng._barge_word_cut_active() is False


def test_word_cut_fails_closed_without_vad():
    # VAD defines burst boundaries. Without it, short garbled echo fragments
    # could accumulate across a whole reply and defeat the four-word floor.
    eng = _engine()
    eng._vad = None
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


def test_cancel_that_command_cuts_below_four_word_floor():
    rec = _Rec()
    eng = _engine(rec)
    assert eng.config.barge_word_cut_min_words == 4
    r, s = _FakeRecognizer(["cancel that"]), _FakeStream()
    assert eng._barge_word_cut_step(r, s, _BLOCK, time.monotonic()) is True
    assert rec.barges == 1


def test_vad_onset_preroll_preserves_first_word_and_bare_stop():
    class _DelayedVad(_FakeVad):
        def __init__(self):
            super().__init__(False)

        def accept_waveform(self, samples):
            self.accepted += 1
            self._speech = self.accepted >= 3  # model confirms after 300 ms

    rec = _Rec()
    eng = _engine(rec, vad_speech=False)
    eng._vad = _DelayedVad()
    r, stream, normal = _FakeRecognizer(["stop"]), _FakeStream(), _FakeStream()
    blocks = [np.full(1600, value, dtype="float32") for value in (0.1, 0.2, 0.3)]
    now = time.monotonic()

    assert not eng._barge_word_cut_step(r, stream, blocks[0], now, normal_stream=normal)
    assert not eng._barge_word_cut_step(
        r, stream, blocks[1], now + 0.1, normal_stream=normal
    )
    assert eng._barge_word_cut_step(
        r, stream, blocks[2], now + 0.2, normal_stream=normal
    )

    assert len(stream.blocks) == 3
    assert len(normal.blocks) == 3
    for actual, expected in zip(normal.blocks, blocks):
        np.testing.assert_array_equal(actual, expected)
    assert eng._wc_stats["vad_preroll_blocks"] == 2
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


def test_cut_replays_only_candidate_pcm_not_folded_own_echo_prefix():
    rec = _Rec()
    eng = _engine(rec)
    eng._now_playing = "once upon a time there was"
    detect = _FakeStream()
    normal = _FakeStream()
    r = _FakeRecognizer(
        ["upon a time", "upon a time what are you doing there"]
    )
    echo_block = np.full(1600, 0.1, dtype="float32")
    user_block = np.full(1600, 0.7, dtype="float32")
    now = time.monotonic()

    assert eng._barge_word_cut_step(
        r, detect, echo_block, now, normal_stream=normal
    ) is False
    assert eng._word_cut_base == "upon a time"
    assert eng._word_cut_candidate_samples == 0  # own PCM folded out

    assert eng._barge_word_cut_step(
        r, detect, user_block, now + 0.1, normal_stream=normal
    ) is True
    assert rec.barges == 1
    # The throw-away detector saw echo + user, but normal ASR was reset and
    # seeded from ONLY the candidate user block.
    assert len(detect.blocks) == 2
    assert len(normal.blocks) == 1
    np.testing.assert_array_equal(normal.blocks[0], user_block)
    assert normal in r.reset_streams and detect in r.reset_streams
    assert eng._word_cut_pending_samples == user_block.size
    np.testing.assert_array_equal(eng._word_cut_pending_pcm[0], user_block)


def test_cut_replay_failure_is_telemetry_visible_and_keeps_finalizer_pcm(caplog):
    import logging

    class _BrokenNormal(_FakeStream):
        def accept_waveform(self, sr, samples):
            raise RuntimeError("normal stream failed")

    caplog.set_level(logging.WARNING, logger="speaker.sherpa")
    eng = _engine(_Rec())
    detect, normal = _FakeStream(), _BrokenNormal()
    r = _FakeRecognizer(["what are you doing"])
    block = np.full(1600, 0.6, dtype="float32")

    assert eng._barge_word_cut_step(
        r, detect, block, time.monotonic(), normal_stream=normal
    ) is True
    assert eng._wc_stats["handoff_replay_errors"] == 1
    assert eng._word_cut_pending_samples == block.size
    assert "normal-stream replay failed" in caplog.text


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
    eng._append_word_cut_candidate(np.ones(1600, dtype="float32"))
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
    assert eng._word_cut_candidate_samples == 0
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
    r, s = _FakeRecognizer(["kept and more"]), _FakeStream()
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


# --- natural reply-tail handoff + PCM finalizer pre-roll --------------------------


def test_novel_short_tail_requires_postplay_word_before_handoff(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="speaker.sherpa")
    eng = _engine(vad_speech=True)
    detect, normal = _FakeStream(), _FakeStream()
    r = _FakeRecognizer(["please wait", "please wait", "please wait now"])
    tail_block = np.full(1600, 0.35, dtype="float32")
    continuation_block = np.full(1600, 0.45, dtype="float32")
    now = time.monotonic()

    # Two words remain below the four-word MID-PLAYBACK safety floor.
    assert eng._barge_word_cut_step(r, detect, tail_block, now) is False
    assert eng.config.barge_word_cut_min_words == 4
    # Reply end alone is NOT authority: stage it and keep normal ASR clean.
    assert eng._finish_word_cut_reply(r, detect, normal, now=now) is False
    assert eng._word_cut_tail_staged is True
    assert normal.fed_blocks == 0
    # The same VAD-active burst gains one post-playback word -> replay tail +
    # continuation into normal ASR and hand off to ordinary endpointing.
    assert eng._word_cut_tail_probation_step(
        r, detect, normal, continuation_block, now + 0.1
    ) == "promoted"
    assert len(normal.blocks) == 2
    np.testing.assert_array_equal(normal.blocks[0], tail_block)
    np.testing.assert_array_equal(normal.blocks[1], continuation_block)
    assert eng._wc_stats["tail_handoffs"] == 1
    assert eng._wc_stats.get("tail_drops", 0) == 0
    assert eng._wc_stats["tail_continuations"] == 1
    assert "word-cut tail staged: words=2" in caplog.text
    assert "word-cut tail handoff: words=3 pcm_ms=200 replay=True" in caplog.text


def test_garbled_two_word_tail_plus_silence_is_never_handed_to_normal_asr():
    eng = _engine(vad_speech=True)
    eng._now_playing = "once upon a time there was a dragon"
    detect, normal = _FakeStream(), _FakeStream()
    r = _FakeRecognizer(["you're any", "you're any", "you're any"])
    now = time.monotonic()

    # The known nonlinear-speaker failure shape: novel-looking two-word junk.
    assert eng._barge_word_cut_step(r, detect, _BLOCK, now) is False
    assert eng._finish_word_cut_reply(r, detect, normal, now=now) is False
    assert eng._word_cut_tail_staged is True
    assert normal.fed_blocks == 0

    eng._vad.set_speech(False)
    r.endpoint = True
    assert eng._word_cut_tail_probation_step(
        r, detect, normal, np.zeros_like(_BLOCK), now + 0.1
    ) == "dropped"
    assert normal.fed_blocks == 0
    assert eng._word_cut_pending_samples == 0
    assert eng._wc_stats.get("handoffs", 0) == 0
    assert eng._wc_stats["tail_drop_no_continuation"] == 1


def test_short_tail_probation_deadline_drops_even_if_vad_stays_active():
    eng = _engine(vad_speech=True, endpoint_max_silence_sec=0.3)
    detect, normal = _FakeStream(), _FakeStream()
    r = _FakeRecognizer(["please wait", "please wait", "please wait"])
    now = time.monotonic()

    assert eng._barge_word_cut_step(r, detect, _BLOCK, now) is False
    assert eng._finish_word_cut_reply(r, detect, normal, now=now) is False
    assert eng._word_cut_tail_staged
    assert eng._word_cut_tail_probation_step(
        r, detect, normal, _BLOCK, now + 0.31
    ) == "dropped"
    assert not eng._word_cut_tail_staged
    assert normal.fed_blocks == 0
    assert eng._wc_stats["tail_drop_no_continuation"] == 1


def test_tail_requires_fresh_vad_epoch_not_inherited_playback_hangover():
    class _StickyVad(_FakeVad):
        def reset(self):
            self.resets += 1  # broken/hangover state remains active

    eng = _engine(vad_speech=True)
    eng._vad = _StickyVad(True)
    detect, normal = _FakeStream(), _FakeStream()
    r = _FakeRecognizer(["please wait", "please wait", "please wait now"])
    now = time.monotonic()

    assert not eng._barge_word_cut_step(r, detect, _BLOCK, now)
    assert not eng._finish_word_cut_reply(r, detect, normal, now=now)
    assert eng._vad.resets == 1
    assert not eng._word_cut_tail_vad_reset_ok
    assert eng._word_cut_tail_probation_step(
        r, detect, normal, np.zeros_like(_BLOCK), now + 0.1
    ) == "waiting"
    assert normal.fed_blocks == 0
    assert eng._word_cut_pending_samples == 0


def test_floor_clearing_tail_hands_off_immediately_when_cut_was_guarded():
    eng = _engine(vad_speech=True)
    detect, normal = _FakeStream(), _FakeStream()
    r = _FakeRecognizer(["what are you doing", "what are you doing"])
    now = time.monotonic()
    eng._barge_in_suppressed_until = now + 10.0

    # The suppress guard delays the cut, but it does not revoke the four-word
    # transcript's authority once playback naturally ends.
    assert eng._barge_word_cut_step(r, detect, _BLOCK, now) is False
    assert eng._finish_word_cut_reply(r, detect, normal, now=now + 0.1) is True
    assert eng._word_cut_tail_staged is False
    assert normal.fed_blocks == 1
    assert eng._wc_stats["tail_handoffs"] == 1


def test_natural_tail_discards_own_empty_and_stale_bursts():
    # Own speech: it was folded into the detection base and its PCM cleared.
    own = _engine(vad_speech=True)
    own._now_playing = "once upon a time there was"
    r_own = _FakeRecognizer(["upon a time"])
    d_own, n_own = _FakeStream(), _FakeStream()
    own._barge_word_cut_step(r_own, d_own, _BLOCK, time.monotonic())
    assert own._finish_word_cut_reply(r_own, d_own, n_own) is False
    assert n_own.fed_blocks == 0
    assert own._wc_stats["tail_drop_own"] == 1

    # Empty recognizer result: energy alone is never tail authority.
    empty = _engine(vad_speech=True)
    r_empty = _FakeRecognizer([""])
    d_empty, n_empty = _FakeStream(), _FakeStream()
    empty._barge_word_cut_step(r_empty, d_empty, _BLOCK, time.monotonic())
    assert empty._finish_word_cut_reply(r_empty, d_empty, n_empty) is False
    assert n_empty.fed_blocks == 0
    assert empty._wc_stats["tail_drop_empty"] == 1

    # A debounce-expired burst is stale even if a fixture leaves text/PCM behind.
    stale = _engine(vad_speech=True)
    r_stale = _FakeRecognizer(["please wait"])
    d_stale, n_stale = _FakeStream(), _FakeStream()
    stale._barge_word_cut_step(r_stale, d_stale, _BLOCK, time.monotonic())
    stale._word_cut_quiet_run = stale.config.barge_word_cut_reset_quiet_blocks
    assert stale._finish_word_cut_reply(r_stale, d_stale, n_stale) is False
    assert n_stale.fed_blocks == 0
    assert stale._wc_stats["tail_drop_stale"] == 1


def test_pending_preroll_splices_once_into_both_finalizer_segments():
    eng = _engine(vad_speech=True)
    detect, normal = _FakeStream(), _FakeStream()
    r = _FakeRecognizer(["please wait", "please wait", "please wait now"])
    block = np.full(1600, 0.42, dtype="float32")
    now = time.monotonic()
    eng._barge_word_cut_step(r, detect, block, now)
    assert eng._finish_word_cut_reply(r, detect, normal, now=now) is False
    assert eng._word_cut_tail_probation_step(
        r, detect, normal, block, now + 0.1
    ) == "promoted"

    utterance: list[np.ndarray] = []
    asr_utterance: list[np.ndarray] = []
    assert eng._splice_word_cut_preroll(utterance, asr_utterance) == 2 * block.size
    assert eng._splice_word_cut_preroll(utterance, asr_utterance) == 0
    assert len(utterance) == len(asr_utterance) == 2
    np.testing.assert_array_equal(utterance[0], block)
    np.testing.assert_array_equal(asr_utterance[1], block)


def test_preroll_pcm_reaches_second_pass_floor_and_speaker_gate():
    rec = _Rec()
    eng = _engine(rec, vad_speech=True)
    detect, normal = _FakeStream(), _FakeStream()
    r = _FakeRecognizer(["please wait", "please wait", "please wait now"])
    block = np.full(1600, 0.51, dtype="float32")
    now = time.monotonic()
    eng._barge_word_cut_step(r, detect, block, now)
    assert eng._finish_word_cut_reply(r, detect, normal, now=now) is False
    assert eng._word_cut_tail_probation_step(
        r, detect, normal, block, now + 0.1
    ) == "promoted"
    utterance: list[np.ndarray] = []
    eng._splice_word_cut_preroll(utterance)
    seg = np.concatenate(utterance)

    seen: dict[str, np.ndarray] = {}

    def final_transcribe(samples, raw):
        seen["second_pass"] = np.asarray(samples).copy()
        return raw

    def floor(samples):
        seen["floor"] = np.asarray(samples).copy()
        return True

    def speaker(samples):
        seen["speaker"] = np.asarray(samples).copy()
        return True

    eng._final_transcribe = final_transcribe
    eng._final_above_floor = floor
    eng._should_act_on_final = speaker
    eng._finalize_and_dispatch(seg, "please wait", 1.0)

    for consumer in ("second_pass", "floor", "speaker"):
        np.testing.assert_array_equal(
            seen[consumer], np.concatenate([block, block])
        )


def test_word_cut_candidate_pcm_bound_derives_from_rule3_and_endpoint_config():
    eng = _engine(
        vad_speech=True,
        asr_rule3_min_utterance_length=2.0,
        endpoint_max_silence_sec=0.5,
        barge_confirm_window_sec=0.25,
    )
    limit = int(eng.config.sample_rate * 2.5)
    assert eng._asr_utterance_limit_samples() == limit
    # One oversized append exercises partial oldest-block trimming precisely.
    block = np.arange(limit + 123, dtype="float32")
    eng._append_word_cut_candidate(block)
    assert eng._word_cut_candidate_samples == limit
    assert sum(b.size for b in eng._word_cut_candidate_pcm) == limit
    np.testing.assert_array_equal(eng._word_cut_candidate_pcm[0], block[-limit:])
    assert eng._wc_stats["pcm_trimmed_samples"] == 123


def test_default_utterance_bound_covers_configured_twenty_second_rule3():
    eng = _engine(vad_speech=True)
    assert eng.config.asr_rule3_min_utterance_length == 20.0
    assert eng._asr_utterance_limit_samples() > 20 * eng.config.sample_rate


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


def test_capture_feeds_word_cut_before_acoustic_reference_watch():
    class _CaptureRecognizer:
        def __init__(self):
            self.streams: list[_FakeStream] = []

        def create_stream(self, **kwargs):
            stream = _FakeStream()
            self.streams.append(stream)
            return stream

        def is_ready(self, stream):
            return False

        def decode_stream(self, stream):  # pragma: no cover - never ready
            pass

        def get_result(self, stream):
            return ""

        def reset(self, stream):
            pass

        def is_endpoint(self, stream):
            return False

    class _OneBlockInput:
        def __init__(self, engine, block):
            self.engine = engine
            self.block = block

        def read(self, n):
            # The current iteration still processes this block, then exits.
            self.engine._running.clear()
            return self.block.copy(), False

    eng = _engine(vad_speech=True)
    recognizer = _CaptureRecognizer()
    eng._recognizer = recognizer
    eng._capture_sr = eng.config.sample_rate
    eng._stream_in = _OneBlockInput(eng, _BLOCK)
    eng._speaking.set()
    eng._barge_sustain_reset_pending = True
    eng._first_audio_pending = True
    eng._playback_level = 0.0
    assert eng._barge_watch_active() is False  # acoustic DTD has no reference

    eng._running.set()
    thread = threading.Thread(target=eng._capture_loop)
    thread.start()
    thread.join(timeout=3.0)
    assert not thread.is_alive()

    # Stream 0 is normal ASR and remains clean; stream 1 is the dedicated
    # word-cut detector and receives the synth-lead-in block despite ref-empty.
    assert len(recognizer.streams) == 2
    assert recognizer.streams[0].fed_blocks == 0
    assert recognizer.streams[1].fed_blocks == 1
    assert eng._vad.accepted == 1


def test_capture_route_loss_blocks_playback_keyword_authority():
    """A raw fallback must not turn echoed "stop" into a KWS command."""

    class _CaptureRecognizer:
        def create_stream(self, **kwargs):
            return _FakeStream()

        def is_ready(self, stream):
            return False

        def decode_stream(self, stream):  # pragma: no cover - never ready
            pass

        def get_result(self, stream):
            return ""

        def reset(self, stream):
            pass

        def is_endpoint(self, stream):
            return False

    class _OneBlockInput:
        def __init__(self, engine):
            self.engine = engine

        def read(self, _n):
            self.engine._running.clear()
            return _BLOCK.copy(), False

    eng = _engine(vad_speech=True)
    eng._recognizer = _CaptureRecognizer()
    eng._capture_sr = eng.config.sample_rate
    eng._stream_in = _OneBlockInput(eng)
    eng._speaking.set()
    eng._word_cut_route_verified = False
    eng._os_echo_route_verified = False
    commands = []
    eng._poll_keywords = lambda samples: commands.append(samples)

    eng._running.set()
    thread = threading.Thread(target=eng._capture_loop)
    thread.start()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert commands == []


def test_capture_tail_continuation_replays_current_block_once(caplog):
    import logging

    class _LifecycleRecognizer:
        def __init__(self):
            self.streams: list[_FakeStream] = []
            self.detect_results = iter(
                ["please wait", "please wait", "please wait now"]
            )

        def create_stream(self, **kwargs):
            stream = _FakeStream()
            self.streams.append(stream)
            return stream

        def is_ready(self, stream):
            return False

        def decode_stream(self, stream):  # pragma: no cover - never ready
            pass

        def get_result(self, stream):
            if len(self.streams) > 1 and stream is self.streams[1]:
                return next(self.detect_results, "please wait now")
            return "please wait now" if stream.blocks else ""

        def reset(self, stream):
            if stream is self.streams[0]:
                stream.blocks.clear()
                stream.fed_blocks = 0

        def is_endpoint(self, stream):
            return False

    class _TwoBlockInput:
        def __init__(self, engine, first, second):
            self.engine = engine
            self.blocks = [first, second]
            self.index = 0

        def read(self, n):
            block = self.blocks[self.index]
            self.index += 1
            if self.index == 2:
                self.engine._speaking.clear()  # natural reply boundary
                self.engine._running.clear()
            return block.copy(), False

    caplog.set_level(logging.INFO, logger="speaker.sherpa")
    eng = _engine(vad_speech=True)
    recognizer = _LifecycleRecognizer()
    tail = np.full(1600, 0.3, dtype="float32")
    continuation = np.full(1600, 0.5, dtype="float32")
    partials: list[str] = []
    eng._recognizer = recognizer
    eng._cb = EngineCallbacks(on_partial=partials.append)
    eng._capture_sr = eng.config.sample_rate
    eng._stream_in = _TwoBlockInput(eng, tail, continuation)
    eng._speaking.set()
    eng._barge_sustain_reset_pending = True
    eng._first_audio_pending = True

    eng._running.set()
    thread = threading.Thread(target=eng._capture_loop)
    thread.start()
    thread.join(timeout=3.0)
    assert not thread.is_alive()

    normal = recognizer.streams[0]
    assert len(normal.blocks) == 2  # tail + continuation, no duplicate current block
    np.testing.assert_array_equal(normal.blocks[0], tail)
    np.testing.assert_array_equal(normal.blocks[1], continuation)
    assert partials == ["Please wait now"]
    assert "tail_continuations=1" in caplog.text
    assert "spliced_samples=3200" in caplog.text


def test_capture_dropped_tail_feeds_vad_once_per_physical_block():
    class _LifecycleRecognizer:
        def __init__(self):
            self.streams: list[_FakeStream] = []

        def create_stream(self, **kwargs):
            stream = _FakeStream()
            self.streams.append(stream)
            return stream

        def is_ready(self, stream):
            return False

        def decode_stream(self, stream):  # pragma: no cover - never ready
            pass

        def get_result(self, stream):
            return (
                "please wait"
                if len(self.streams) > 1 and stream is self.streams[1]
                else ""
            )

        def reset(self, stream):
            pass

        def is_endpoint(self, stream):
            return len(self.streams) > 1 and stream is self.streams[1]

    class _TwoBlockInput:
        def __init__(self, engine):
            self.engine = engine
            self.index = 0

        def read(self, _n):
            self.index += 1
            if self.index == 2:
                self.engine._speaking.clear()
                self.engine._vad.set_speech(False)
                self.engine._running.clear()
            return _BLOCK.copy(), False

    eng = _engine(vad_speech=True)
    eng._recognizer = _LifecycleRecognizer()
    eng._capture_sr = eng.config.sample_rate
    eng._stream_in = _TwoBlockInput(eng)
    eng._speaking.set()
    eng._barge_sustain_reset_pending = True

    eng._running.set()
    thread = threading.Thread(target=eng._capture_loop)
    thread.start()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert eng._vad.accepted == 2
    assert not eng._word_cut_tail_staged


def test_capture_confirmed_word_cut_handoff_finalizes_candidate_once():
    """The playback candidate survives the real ASRSegment handoff intact.

    This is deliberately a finite capture-loop test, not a direct helper test:
    it covers promotion into the normal recognizer, the one-shot pending splice,
    ASRSegment ownership, endpoint extraction, and the finalizer work item.
    """

    class _LifecycleRecognizer:
        def __init__(self):
            self.streams: list[_FakeStream] = []
            self.detect_results = iter(["please wait", "please wait for me"])
            self.endpoint_blocks: list[np.ndarray] = []

        def create_stream(self, **kwargs):
            stream = _FakeStream()
            self.streams.append(stream)
            return stream

        def is_ready(self, stream):
            return False

        def decode_stream(self, stream):  # pragma: no cover - never ready
            pass

        def get_result(self, stream):
            if len(self.streams) > 1 and stream is self.streams[1]:
                return next(self.detect_results, "please wait for me")
            return "please wait for me" if stream.blocks else ""

        def reset(self, stream):
            # Promotion intentionally throws away any old normal-stream context
            # before replaying only the confirmed user candidate.
            if self.streams and stream is self.streams[0]:
                stream.blocks.clear()
                stream.fed_blocks = 0

        def is_endpoint(self, stream):
            endpoint = bool(
                self.streams
                and stream is self.streams[0]
                and stream.fed_blocks >= 3
            )
            if endpoint:
                self.endpoint_blocks = [block.copy() for block in stream.blocks]
            return endpoint

    class _ThreeBlockInput:
        def __init__(self, engine, blocks):
            self.engine = engine
            self.blocks = blocks
            self.index = 0

        def read(self, n):
            block = self.blocks[self.index]
            self.index += 1
            if self.index == len(self.blocks):
                # The first listening block is endpoint silence. The current
                # iteration still completes after stopping the finite loop.
                self.engine._vad.set_speech(False)
                self.engine._running.clear()
            return block.copy(), False

    eng = _engine(vad_speech=True)
    recognizer = _LifecycleRecognizer()
    first = np.full(1600, 0.11, dtype="float32")
    second = np.full(1600, 0.22, dtype="float32")
    endpoint_silence = np.zeros(1600, dtype="float32")
    finalized: list[tuple[np.ndarray, str, object, object, object]] = []

    def on_barge_in():
        # Real coordination stops playback after the cut; make that boundary
        # deterministic without involving a playback device/thread.
        eng._speaking.clear()

    def finalize(primary, raw, speech_end, alternate, speech_sec):
        finalized.append(
            (
                np.asarray(primary).copy(),
                raw,
                speech_end,
                None if alternate is None else np.asarray(alternate).copy(),
                speech_sec,
            )
        )

    eng._recognizer = recognizer
    eng._cb = EngineCallbacks(on_barge_in=on_barge_in)
    eng._capture_sr = eng.config.sample_rate
    eng._stream_in = _ThreeBlockInput(
        eng, [first, second, endpoint_silence]
    )
    eng._finalize_and_dispatch = finalize
    eng._speaking.set()
    eng._barge_sustain_reset_pending = True
    eng._first_audio_pending = True

    eng._running.set()
    thread = threading.Thread(target=eng._capture_loop)
    thread.start()
    thread.join(timeout=3.0)
    assert not thread.is_alive()

    assert len(finalized) == 1
    primary, raw, speech_end, alternate, speech_sec = finalized[0]
    expected = np.concatenate([first, second, endpoint_silence])
    np.testing.assert_array_equal(primary, expected)
    assert raw == "please wait for me"
    assert speech_end is not None
    assert alternate is None  # OS echo-cancel word-cut has one capture domain
    assert np.isclose(speech_sec, 2 * eng.config.block_sec)
    assert eng._word_cut_pending_samples == 0

    # The normal streaming recognizer sees the same two candidate blocks once,
    # followed by the one endpoint-silence block -- no lost head or duplicate.
    assert len(recognizer.endpoint_blocks) == 3
    np.testing.assert_array_equal(recognizer.endpoint_blocks[0], first)
    np.testing.assert_array_equal(recognizer.endpoint_blocks[1], second)
    np.testing.assert_array_equal(recognizer.endpoint_blocks[2], endpoint_silence)


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
    assert "handoffs=1" in text
    assert "tail_handoffs=0 tail_drops=0" in text
    assert "preroll_samples=3200" in text
    assert "replay_errors=0" in text
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
