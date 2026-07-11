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
import pytest

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
    # Most fixtures pin the historical identity-free state machine. Production
    # defaults to enrolled authority; dedicated tests below exercise that gate.
    cfg.setdefault("barge_word_cut_require_speaker", False)
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
    assert c.barge_word_cut_require_speaker is True
    assert c.barge_word_cut_speaker_min_words == 0
    assert c.barge_word_cut_speaker_min_sec == 0.35
    assert c.barge_word_cut_stop_speaker_min_sec == 0.10
    assert c.barge_word_cut_speaker_window_sec == 2.0
    assert c.barge_word_cut_speaker_threshold == 0.30
    assert c.barge_word_cut_speaker_reject_threshold == 0.22
    assert c.barge_word_cut_speaker_retry_sec == 0.25


def test_config_from_dict_roundtrip():
    c = SherpaConfig.from_dict(
        {
            "barge_word_cut_enabled": True,
            "barge_word_cut_min_words": 3,
            "barge_word_cut_reset_quiet_blocks": 5,
            "barge_word_cut_vad_preroll_sec": 0.4,
            "barge_word_cut_speaker_min_words": 1,
            "barge_word_cut_stop_speaker_min_sec": 0.15,
            "barge_word_cut_speaker_threshold": 0.4,
            "barge_word_cut_speaker_reject_threshold": 0.2,
            "barge_word_cut_speaker_retry_sec": 0.3,
        }
    )
    assert c.barge_word_cut_enabled is True
    assert c.barge_word_cut_min_words == 3
    assert c.barge_word_cut_reset_quiet_blocks == 5
    assert c.barge_word_cut_vad_preroll_sec == 0.4
    assert c.barge_word_cut_speaker_min_words == 1
    assert c.barge_word_cut_stop_speaker_min_sec == 0.15
    assert c.barge_word_cut_speaker_threshold == 0.4
    assert c.barge_word_cut_speaker_reject_threshold == 0.2
    assert c.barge_word_cut_speaker_retry_sec == 0.3


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


class _FakeSpeakerGate:
    def __init__(
        self, scores, *, enrolled=True, threshold=0.5, embedding=(1.0, 0.0)
    ):
        self._scores = list(scores)
        self.is_enrolled = bool(enrolled)
        self.threshold = float(threshold)
        self.embedding = embedding
        self.calls: list[np.ndarray] = []
        self.embed_calls = 0

    def similarity(self, samples, _sample_rate):
        self.calls.append(np.asarray(samples, dtype="float32").copy())
        return self._scores.pop(0)

    def embed(self, _samples, _sample_rate):
        self.embed_calls += 1
        return self.embedding


def test_audio_first_owner_authority_cuts_when_double_talk_asr_stays_empty():
    """Regression for live run 20260711-130601.

    The OS-cancelled waveform carried the owner, but streaming double-talk ASR
    produced zero words for two replies and only 1-3 unstable words in the story.
    Sustained compatible owner identity must therefore be sufficient authority.
    """
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.35,
        barge_word_cut_speaker_threshold=0.30,
        speaker_gate_input=False,
    )
    gate = _FakeSpeakerGate([0.387])
    eng._speaker_gate = gate
    eng._speaker_gate_warmed = True
    detect, normal = _FakeStream(), _FakeStream()
    recognizer = _FakeRecognizer([""] * 4)
    block = np.full(1600, 0.2, dtype="float32")
    now = time.monotonic()

    for index in range(3):
        assert not eng._barge_word_cut_step(
            recognizer,
            detect,
            block,
            now + index * 0.1,
            normal_stream=normal,
        )
    assert eng._barge_word_cut_step(
        recognizer, detect, block, now + 0.3, normal_stream=normal
    )

    assert rec.barges == 1
    assert len(gate.calls) == 1
    assert eng._wc_stats["speaker_accepts"] == 1
    assert normal.fed_blocks == 1  # accepted PCM is voiced/model-window bounded


def test_audio_first_own_tts_is_rejected_below_double_talk_threshold():
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.35,
        barge_word_cut_speaker_threshold=0.30,
        speaker_gate_input=False,
    )
    eng._speaker_gate = _FakeSpeakerGate([0.179])
    eng._speaker_gate_warmed = True
    recognizer = _FakeRecognizer([""] * 4)
    stream = _FakeStream()
    now = time.monotonic()

    for index in range(4):
        assert not eng._barge_word_cut_step(
            recognizer,
            stream,
            np.full(1600, 0.2, dtype="float32"),
            now + index * 0.1,
        )

    assert rec.barges == 0
    assert eng._wc_stats["speaker_rejects"] == 1
    assert eng._wc_stats["speaker_resets"] == 1
    assert eng._word_cut_candidate_samples == 0


def test_audio_first_threshold_never_undercuts_enabled_final_speaker_gate():
    eng = _engine(
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.0,
        barge_word_cut_speaker_threshold=0.30,
        speaker_gate_input=True,
    )
    eng._speaker_gate = _FakeSpeakerGate([0.49], threshold=0.50)
    eng._speaker_gate_warmed = True
    eng._append_word_cut_candidate(np.full(1600, 0.2, dtype="float32"))

    assert eng._word_cut_speaker_decision() == "defer"
    assert eng._wc_stats.get("speaker_rejects", 0) == 0
    assert eng._wc_stats["speaker_ambiguous"] == 1


def test_audio_first_ambiguous_score_retries_after_more_pcm_then_cuts():
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.35,
        barge_word_cut_speaker_threshold=0.30,
        barge_word_cut_speaker_reject_threshold=0.22,
        barge_word_cut_speaker_retry_sec=0.25,
        speaker_gate_input=False,
    )
    gate = _FakeSpeakerGate([0.25, 0.387])
    eng._speaker_gate = gate
    eng._speaker_gate_warmed = True
    recognizer = _FakeRecognizer([""] * 8)
    detect, normal = _FakeStream(), _FakeStream()
    block = np.full(1600, 0.2, dtype="float32")
    now = time.monotonic()

    for index in range(6):
        assert not eng._barge_word_cut_step(
            recognizer,
            detect,
            block,
            now + index * 0.1,
            normal_stream=normal,
        )
    assert len(gate.calls) == 1  # retry cadence avoids per-block ONNX work
    assert not eng._barge_word_cut_step(
        recognizer, detect, block, now + 0.6, normal_stream=normal
    )
    assert eng._barge_word_cut_step(
        recognizer, detect, block, now + 0.7, normal_stream=normal
    )
    assert len(gate.calls) == 2
    assert rec.barges == 1


def test_ambiguous_prefix_cannot_gain_owner_lineage_from_later_suffix():
    """A clip-level retry must not upgrade already-ambiguous PCM to owner PCM.

    The first block is explicitly not authoritative.  A later owner-like suffix
    may safely cause another retry (or remain deferred), but if it cuts, the
    promoted normal-ASR/finalizer PCM must not contain the ambiguous prefix.
    Otherwise text spoken by a different source can inherit the owner's later
    aggregate embedding score.
    """
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.0,
        barge_word_cut_speaker_threshold=0.30,
        barge_word_cut_speaker_reject_threshold=0.22,
        barge_word_cut_speaker_retry_sec=0.0,
        speaker_gate_input=False,
    )
    gate = _FakeSpeakerGate([0.25, 0.387])
    eng._speaker_gate = gate
    eng._speaker_gate_warmed = True
    recognizer = _FakeRecognizer(["", ""])
    detect, normal = _FakeStream(), _FakeStream()
    ambiguous_prefix = np.full(1600, 0.20, dtype="float32")
    owner_suffix = np.full(1600, 0.70, dtype="float32")
    now = time.monotonic()

    assert not eng._barge_word_cut_step(
        recognizer,
        detect,
        ambiguous_prefix,
        now,
        normal_stream=normal,
    )
    fired = eng._barge_word_cut_step(
        recognizer,
        detect,
        owner_suffix,
        now + 0.1,
        normal_stream=normal,
    )

    # Failing closed is valid.  If the later suffix does earn a cut, only PCM
    # whose owner lineage is known may be replayed/finalized.
    if fired:
        replay = np.concatenate(normal.blocks)
        assert np.all(replay == np.float32(0.70))


def test_audio_first_minimum_counts_voiced_frames_not_preroll_envelope():
    class _ScriptedVad(_FakeVad):
        def __init__(self):
            super().__init__(False)
            self._states = iter((True, False, False, True))

        def accept_waveform(self, samples):
            self.accepted += 1
            self._speech = next(self._states)

    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.35,
        speaker_gate_input=False,
    )
    eng._vad = _ScriptedVad()
    gate = _FakeSpeakerGate([0.8])
    eng._speaker_gate = gate
    eng._speaker_gate_warmed = True
    recognizer = _FakeRecognizer([""] * 4)
    stream = _FakeStream()
    now = time.monotonic()
    blocks = (
        np.full(1600, 0.2, dtype="float32"),
        np.zeros(1600, dtype="float32"),
        np.zeros(1600, dtype="float32"),
        np.full(1600, 0.2, dtype="float32"),
    )

    for index, block in enumerate(blocks):
        assert not eng._barge_word_cut_step(
            recognizer, stream, block, now + index * 0.1
        )

    assert eng._word_cut_candidate_samples == 6400  # 400 ms envelope retained
    assert gate.calls == []  # only 200 ms is actually energy-voiced
    assert rec.barges == 0


def test_audio_first_guarded_echo_cannot_contaminate_later_owner_handoff():
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.35,
        barge_word_cut_speaker_threshold=0.30,
        speaker_gate_input=False,
        barge_in_playback_onset_grace_sec=0.4,
    )
    eng._speaker_gate = _FakeSpeakerGate([0.387])
    eng._speaker_gate_warmed = True
    recognizer = _FakeRecognizer([""] * 8)
    detect, normal = _FakeStream(), _FakeStream()
    now = time.monotonic()
    eng._playback_onset_at = now

    for index in range(4):
        assert not eng._barge_word_cut_step(
            recognizer,
            detect,
            np.full(1600, 0.1, dtype="float32"),
            now + index * 0.1,
            normal_stream=normal,
        )
    for index in range(3):
        assert not eng._barge_word_cut_step(
            recognizer,
            detect,
            np.full(1600, 0.7, dtype="float32"),
            now + 0.5 + index * 0.1,
            normal_stream=normal,
        )
    assert eng._barge_word_cut_step(
        recognizer,
        detect,
        np.full(1600, 0.7, dtype="float32"),
        now + 0.8,
        normal_stream=normal,
    )

    replay = np.concatenate(normal.blocks)
    assert replay.size == 6400
    assert np.all(replay == np.float32(0.7))
    assert eng._wc_stats["guard_candidate_clears"] == 4


def test_audio_first_empty_tail_promotes_after_fresh_voiced_continuation():
    eng = _engine(
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.35,
        barge_word_cut_speaker_threshold=0.30,
        speaker_gate_input=False,
    )
    eng._speaker_gate = _FakeSpeakerGate([0.387])
    eng._speaker_gate_warmed = True
    recognizer = _FakeRecognizer([""] * 5)
    detect, normal = _FakeStream(), _FakeStream()
    block = np.full(1600, 0.2, dtype="float32")
    now = time.monotonic()

    for index in range(3):
        assert not eng._barge_word_cut_step(
            recognizer, detect, block, now + index * 0.1
        )
    assert not eng._finish_word_cut_reply(
        recognizer, detect, normal, now=now + 0.3
    )
    assert eng._word_cut_tail_staged
    assert eng._word_cut_tail_probation_step(
        recognizer, detect, normal, block, now + 0.4
    ) == "promoted"
    assert normal.fed_blocks == 1
    assert eng._wc_stats["speaker_accepts"] == 1


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_audio_first_nonfinite_similarity_fails_closed(score):
    eng = _engine(
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_sec=0.0,
        speaker_gate_input=False,
    )
    eng._speaker_gate = _FakeSpeakerGate([score])
    eng._speaker_gate_warmed = True
    eng._append_word_cut_candidate(np.full(1600, 0.2, dtype="float32"))

    assert eng._word_cut_speaker_decision() == "defer"
    assert eng._wc_stats["speaker_errors"] == 1
    assert eng._wc_stats.get("speaker_accepts", 0) == 0


def test_audio_first_busy_speaker_extractor_defers_without_blocking_capture():
    from core.engines.speaker_gate import SpeakerGate

    eng = _engine(
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_sec=0.0,
        speaker_gate_input=False,
    )
    gate = SpeakerGate(threshold=0.5, embed_fn=lambda _samples, _sr: [1.0, 0.0])
    gate.enroll_embedding([1.0, 0.0])
    eng._speaker_gate = gate
    eng._speaker_gate_warmed = True
    eng._append_word_cut_candidate(np.full(1600, 0.2, dtype="float32"))

    assert gate._inference_lock.acquire(blocking=False)
    try:
        assert eng._word_cut_speaker_decision() == "defer"
    finally:
        gate._inference_lock.release()
    assert eng._wc_stats["speaker_busy_deferred"] == 1


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.22, "reject"), (0.221, "defer"), (0.30, "accept")],
)
def test_audio_first_reject_defer_accept_boundaries(score, expected):
    eng = _engine(
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_sec=0.0,
        barge_word_cut_speaker_threshold=0.30,
        barge_word_cut_speaker_reject_threshold=0.22,
        speaker_gate_input=False,
    )
    eng._speaker_gate = _FakeSpeakerGate([score])
    eng._speaker_gate_warmed = True
    eng._append_word_cut_candidate(np.full(1600, 0.2, dtype="float32"))

    assert eng._word_cut_speaker_decision() == expected


def test_audio_first_ambiguous_retry_uses_monotonic_observed_pcm_after_trim():
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.0,
        barge_word_cut_speaker_retry_sec=0.25,
        speaker_gate_input=False,
    )
    eng._asr_utterance_limit_samples = lambda: 3200
    gate = _FakeSpeakerGate([0.25, 0.387])
    eng._speaker_gate = gate
    eng._speaker_gate_warmed = True
    recognizer = _FakeRecognizer([""] * 4)
    detect, normal = _FakeStream(), _FakeStream()
    block = np.full(1600, 0.2, dtype="float32")
    now = time.monotonic()

    for index in range(3):
        assert not eng._barge_word_cut_step(
            recognizer,
            detect,
            block,
            now + index * 0.1,
            normal_stream=normal,
        )
    assert eng._word_cut_candidate_samples == 3200
    assert eng._word_cut_candidate_observed_samples == 4800
    assert eng._barge_word_cut_step(
        recognizer, detect, block, now + 0.3, normal_stream=normal
    )
    assert len(gate.calls) == 2


def test_standalone_own_tts_stop_requires_speaker_authority():
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        speaker_gate_input=False,
    )
    eng._now_playing = "Stop."
    eng._speaker_gate = _FakeSpeakerGate([0.15])
    eng._speaker_gate_warmed = True

    assert not eng._barge_word_cut_step(
        _FakeRecognizer(["stop"]),
        _FakeStream(),
        np.full(1600, 0.2, dtype="float32"),
        time.monotonic(),
    )
    assert rec.barges == 0
    assert eng._wc_stats["speaker_rejects"] == 1


def test_short_owner_stop_cuts_when_recent_tts_makes_text_ambiguous():
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        speaker_gate_input=False,
    )
    eng._now_playing = "Stop."
    eng._speaker_gate = _FakeSpeakerGate([0.387])
    eng._speaker_gate_warmed = True

    assert eng._barge_word_cut_step(
        _FakeRecognizer(["stop"]),
        _FakeStream(),
        np.full(1600, 0.2, dtype="float32"),
        time.monotonic(),
    )
    assert rec.barges == 1
    assert eng._wc_stats["speaker_accepts"] == 1


def test_live_noisy_prefix_then_owner_stop_cuts_before_quiet_reset():
    """Headless replay of live run 20260711-154451's failed STOP burst.

    Playback-time ASR retained two residual words before the owner's canonical
    command.  The three-word ``OF HE STOP`` trace must cut on its last active
    block instead of falling through to the following quiet-run reset.
    """
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        speaker_gate_input=False,
    )
    gate = _FakeSpeakerGate([])
    eng._speaker_gate = gate
    eng._speaker_gate_warmed = True
    eng._now_playing = (
        "Silas could feel the tower swaying slightly with each monstrous wave."
    )
    recognizer = _FakeRecognizer(["OF", "OF HE", "OF HE STOP"])
    stream = _FakeStream()
    now = time.monotonic()

    assert not eng._barge_word_cut_step(
        recognizer, stream, np.full(1600, 0.4, dtype="float32"), now
    )
    assert not eng._barge_word_cut_step(
        recognizer, stream, np.full(1600, 0.5, dtype="float32"), now + 0.1
    )
    assert eng._barge_word_cut_step(
        recognizer, stream, np.full(1600, 0.6, dtype="float32"), now + 0.2
    )

    assert rec.barges == 1
    assert gate.calls == []  # canonical STOP cut is not general identity trust
    assert eng._wc_stats["cuts"] == 1


def test_own_tts_attested_stop_repair_requires_speaker_and_does_not_cut():
    """Corrupted TTS containing STOP cannot use the attested repair to self-cut."""
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        speaker_gate_input=False,
    )
    gate = _FakeSpeakerGate([0.16])
    eng._speaker_gate = gate
    eng._speaker_gate_warmed = True
    eng._now_playing = "The app may stop responding while it reconnects."
    recognizer = _FakeRecognizer(["OF", "OF HE", "OF HE STOP"])
    stream = _FakeStream()
    now = time.monotonic()

    assert not eng._barge_word_cut_step(
        recognizer, stream, np.full(1600, 0.2, dtype="float32"), now
    )
    assert not eng._barge_word_cut_step(
        recognizer, stream, np.full(1600, 0.2, dtype="float32"), now + 0.1
    )
    assert not eng._barge_word_cut_step(
        recognizer, stream, np.full(1600, 0.2, dtype="float32"), now + 0.2
    )

    assert rec.barges == 0
    assert len(gate.calls) == 1
    assert eng._wc_stats["speaker_rejects"] == 1
    assert eng._wc_stats.get("cuts", 0) == 0


@pytest.mark.parametrize(
    "text",
    [
        "BUS STOP",
        "NEXT STOP",
        "FULL STOP",
        "IT WILL STOP",
        "DO NOT STOP",
        "DON'T STOP",
        "NEVER STOP",
        "NO STOP",
        "NU STOP",
        "CAN'T STOP",
        "CANNOT STOP",
        "WON'T STOP",
    ],
)
def test_non_attested_stop_phrase_is_not_a_canonical_cut(text):
    """Ordinary and negated phrases ending in STOP remain non-controls."""
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.0,
        speaker_gate_input=False,
    )
    eng._speaker_gate = None

    assert not eng._barge_word_cut_step(
        _FakeRecognizer([text]),
        _FakeStream(),
        np.full(1600, 0.4, dtype="float32"),
        time.monotonic(),
    )

    assert rec.barges == 0
    assert eng._wc_stats["speaker_unavailable"] == 1
    assert eng._wc_stats.get("cuts", 0) == 0


def test_attested_stop_repair_is_shared_at_reply_tail():
    eng = _engine(
        vad_speech=True,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        speaker_gate_input=False,
    )
    detect, normal = _FakeStream(), _FakeStream()
    recognizer = _FakeRecognizer(["OF HE STOP"])
    block = np.full(1600, 0.4, dtype="float32")
    eng._word_cut_fed_stream = True
    eng._append_word_cut_candidate(block)

    assert eng._finish_word_cut_reply(
        recognizer, detect, normal, now=time.monotonic()
    )
    assert normal.fed_blocks == 1
    assert eng._wc_stats["tail_handoffs"] == 1


def test_attested_stop_repair_is_shared_at_tail_continuation():
    eng = _engine(
        vad_speech=True,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        speaker_gate_input=False,
    )
    detect, normal = _FakeStream(), _FakeStream()
    recognizer = _FakeRecognizer(["OF HE STOP"])
    block = np.full(1600, 0.4, dtype="float32")
    now = time.monotonic()
    eng._word_cut_tail_staged = True
    eng._word_cut_tail_words = 2
    eng._word_cut_tail_text = "OF HE"
    eng._word_cut_tail_deadline = now + 1.0
    eng._word_cut_tail_vad_reset_ok = True

    assert eng._word_cut_tail_probation_step(
        recognizer, detect, normal, block, now
    ) == "promoted"
    assert normal.fed_blocks == 1
    assert eng._wc_stats["tail_continuations"] == 1


def test_live_saturn_echo_rejected_by_enrolled_speaker_then_owner_cuts():
    """Regression for live run 20260711-115512 at 11:56:19.

    The word-cut stream emitted two words then four garbled words and promoted
    1.5 s that normal ASR proved was own TTS. Speaker authority rejects that PCM,
    resets the unstable detector stream, and admits a later owner command with
    only owner PCM replayed into normal ASR.
    """
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_sec=0.15,
    )
    gate = _FakeSpeakerGate([0.18, 0.54])
    eng._speaker_gate = gate
    eng._speaker_gate_warmed = True
    detect, normal = _FakeStream(), _FakeStream()
    echo = _FakeRecognizer(["THE HARRING", "THE HARRING OF SA"])
    now = time.monotonic()

    assert not eng._barge_word_cut_step(
        echo, detect, np.full(1600, 0.2, dtype="float32"), now,
        normal_stream=normal,
    )
    assert not eng._barge_word_cut_step(
        echo, detect, np.full(1600, 0.3, dtype="float32"), now + 0.1,
        normal_stream=normal,
    )
    assert rec.barges == 0
    assert eng._wc_stats["speaker_rejects"] == 1
    assert eng._wc_stats["speaker_resets"] == 1
    assert eng._word_cut_candidate_samples == 0
    assert detect in echo.reset_streams

    owner = _FakeRecognizer(
        [
            "switch to ancient Roman architecture",
            "switch to ancient Roman architecture",
        ]
    )
    user_blocks = [
        np.full(1600, 0.7, dtype="float32"),
        np.full(1600, 0.8, dtype="float32"),
    ]
    assert not eng._barge_word_cut_step(
        owner, detect, user_blocks[0], now + 0.2, normal_stream=normal
    )
    assert eng._barge_word_cut_step(
        owner, detect, user_blocks[1], now + 0.3, normal_stream=normal
    )
    assert rec.barges == 1
    assert eng._wc_stats["speaker_accepts"] == 1
    np.testing.assert_array_equal(
        np.concatenate(normal.blocks), np.concatenate(user_blocks)
    )


def test_enrolled_speaker_authority_overrides_text_overlap():
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_sec=0.35,
    )
    eng._speaker_gate = _FakeSpeakerGate([0.8])
    eng._speaker_gate_warmed = True
    eng._now_playing = "There are rings of Saturn."
    recognizer = _FakeRecognizer(["there are rings of Saturn"] * 4)
    stream = _FakeStream()
    voice_block = np.full(1600, 0.2, dtype="float32")
    now = time.monotonic()

    for index in range(3):
        assert not eng._barge_word_cut_step(
            recognizer, stream, voice_block, now + index * 0.1
        )
        assert eng._word_cut_base == ""  # lexical echo match cannot preempt ID
    assert eng._barge_word_cut_step(
        recognizer, stream, voice_block, now + 0.3
    )
    assert rec.barges == 1


def test_required_speaker_authority_fails_closed_but_bare_stop_still_cuts():
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_sec=0.0,
    )
    eng._speaker_gate = None
    assert not eng._barge_word_cut_step(
        _FakeRecognizer(["switch to ancient Roman architecture"]),
        _FakeStream(),
        _BLOCK,
        time.monotonic(),
    )
    assert rec.barges == 0

    stop = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_sec=0.0,
    )
    stop._speaker_gate = None
    assert stop._barge_word_cut_step(
        _FakeRecognizer(["stop"]), _FakeStream(), _BLOCK, time.monotonic()
    )
    assert rec.barges == 1


def test_cold_speaker_authority_defers_without_blocking_capture_inference():
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_sec=0.0,
    )
    gate = _FakeSpeakerGate([0.8])
    eng._speaker_gate = gate
    recognizer = _FakeRecognizer(
        ["switch to ancient Roman architecture"] * 2
    )
    stream = _FakeStream()
    now = time.monotonic()

    assert not eng._barge_word_cut_step(recognizer, stream, _BLOCK, now)
    assert gate.calls == []
    assert eng._wc_stats["speaker_cold_deferred"] == 1

    eng._speaker_gate_warmed = True
    assert eng._barge_word_cut_step(recognizer, stream, _BLOCK, now + 0.1)
    assert len(gate.calls) == 1
    assert rec.barges == 1


def test_required_speaker_authority_warms_idempotently_off_capture_thread():
    eng = _engine(barge_word_cut_require_speaker=True)
    gate = _FakeSpeakerGate([])
    eng._speaker_gate = gate

    assert eng._warm_speaker_gate()
    assert eng._speaker_gate_warmed
    assert gate.embed_calls == 1
    assert eng._warm_speaker_gate()
    assert gate.embed_calls == 1


def test_speaker_authority_embedding_window_is_bounded_off_capture_hot_path():
    eng = _engine(
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_sec=0.0,
        barge_word_cut_speaker_window_sec=2.0,
    )
    gate = _FakeSpeakerGate([0.8])
    eng._speaker_gate = gate
    eng._speaker_gate_warmed = True
    eng._append_word_cut_candidate(np.ones(16000 * 8, dtype="float32"))

    assert eng._word_cut_speaker_decision() == "accept"
    assert len(gate.calls) == 1
    assert gate.calls[0].size <= 2 * eng.config.sample_rate
    assert eng._word_cut_candidate_samples <= 2 * eng.config.sample_rate

    detect, normal = _FakeStream(), _FakeStream()
    recognizer = _FakeRecognizer(["switch to ancient Roman architecture"])
    assert eng._promote_word_cut_candidate(
        recognizer,
        detect,
        normal,
        reason="cut",
        text="switch to ancient Roman architecture",
    )
    assert eng._word_cut_pending_samples <= 2 * eng.config.sample_rate
    assert sum(block.size for block in normal.blocks) <= 2 * eng.config.sample_rate


def test_own_tts_sentence_ending_stop_does_not_bypass_speaker_authority():
    rec = _Rec()
    eng = _engine(
        rec,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_sec=0.0,
    )
    eng._speaker_gate = _FakeSpeakerGate([0.1])
    eng._speaker_gate_warmed = True
    recognizer = _FakeRecognizer(["if you want me to stop"])

    assert not eng._barge_word_cut_step(
        recognizer, _FakeStream(), _BLOCK, time.monotonic()
    )
    assert rec.barges == 0
    assert eng._wc_stats["speaker_rejects"] == 1


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


def test_post_cut_async_stop_boundary_cannot_overwrite_pending_owner_pcm():
    eng = _engine(_Rec())
    detect, normal = _FakeStream(), _FakeStream()
    first = np.full(1600, 0.4, dtype="float32")
    residue = np.full(1600, 0.8, dtype="float32")
    now = time.monotonic()

    assert eng._barge_word_cut_step(
        _FakeRecognizer(["what are you doing"]),
        detect,
        first,
        now,
        normal_stream=normal,
    )
    assert eng._word_cut_pending_samples == first.size
    np.testing.assert_array_equal(eng._word_cut_pending_pcm[0], first)

    # Playback/control cancellation has not cleared `_speaking` yet, so one more
    # capture block reaches the word-cut seam. It must be inert.
    assert not eng._barge_word_cut_step(
        _FakeRecognizer(["entirely different residue words"]),
        detect,
        residue,
        now + 0.1,
        normal_stream=normal,
    )
    assert not eng._finish_word_cut_reply(
        _FakeRecognizer(["entirely different residue words"]),
        detect,
        normal,
        now=now + 0.2,
    )
    assert eng._wc_stats.get("pending_overwrites", 0) == 0
    assert eng._word_cut_pending_samples == first.size
    np.testing.assert_array_equal(eng._word_cut_pending_pcm[0], first)


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


def test_guarded_pcm_cannot_become_authority_at_reply_tail():
    eng = _engine(vad_speech=True)
    detect, normal = _FakeStream(), _FakeStream()
    r = _FakeRecognizer(["what are you doing", "what are you doing"])
    now = time.monotonic()
    eng._barge_in_suppressed_until = now + 10.0

    # Suppression is an authority boundary: playback/echo PCM observed while the
    # guard is active cannot be promoted later merely because playback ends.
    assert eng._barge_word_cut_step(r, detect, _BLOCK, now) is False
    assert eng._finish_word_cut_reply(r, detect, normal, now=now + 0.1) is False
    assert eng._word_cut_tail_staged is False
    assert normal.fed_blocks == 0
    assert eng._word_cut_candidate_samples == 0
    assert eng._wc_stats["guard_candidate_clears"] == 1


def test_required_speaker_authority_cannot_be_bypassed_at_reply_tail():
    now = time.monotonic()

    missing = _engine(
        vad_speech=True,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=3,
        barge_word_cut_speaker_min_sec=0.0,
    )
    missing._speaker_gate = None
    detect_missing, normal_missing = _FakeStream(), _FakeStream()
    r_missing = _FakeRecognizer(["what are you doing", "what are you doing"])
    assert not missing._barge_word_cut_step(
        r_missing, detect_missing, _BLOCK, now
    )
    assert not missing._finish_word_cut_reply(
        r_missing, detect_missing, normal_missing, now=now + 0.1
    )
    assert missing._word_cut_tail_staged  # bounded wait, never immediate bypass
    assert normal_missing.fed_blocks == 0

    rejected = _engine(
        vad_speech=True,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=3,
        barge_word_cut_speaker_min_sec=0.0,
    )
    rejected._speaker_gate = _FakeSpeakerGate([0.1])
    rejected._speaker_gate_warmed = True
    detect_rejected, normal_rejected = _FakeStream(), _FakeStream()
    r_rejected = _FakeRecognizer(["what are you doing", "what are you doing"])
    assert not rejected._barge_word_cut_step(
        r_rejected, detect_rejected, _BLOCK, now
    )
    assert not rejected._finish_word_cut_reply(
        r_rejected, detect_rejected, normal_rejected, now=now + 0.1
    )
    assert not rejected._word_cut_tail_staged
    assert rejected._wc_stats["speaker_rejects"] == 1
    assert normal_rejected.fed_blocks == 0


def test_required_speaker_authority_gates_short_tail_continuation():
    now = time.monotonic()

    accepted = _engine(
        vad_speech=True,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=3,
        barge_word_cut_speaker_min_sec=0.0,
    )
    accepted._speaker_gate = _FakeSpeakerGate([0.8])
    accepted._speaker_gate_warmed = True
    detect_ok, normal_ok = _FakeStream(), _FakeStream()
    r_ok = _FakeRecognizer(["please wait", "please wait", "please wait now"])
    assert not accepted._barge_word_cut_step(r_ok, detect_ok, _BLOCK, now)
    assert not accepted._finish_word_cut_reply(
        r_ok, detect_ok, normal_ok, now=now
    )
    assert accepted._word_cut_tail_probation_step(
        r_ok, detect_ok, normal_ok, _BLOCK, now + 0.1
    ) == "promoted"
    np.testing.assert_array_equal(
        np.concatenate(normal_ok.blocks), np.concatenate([_BLOCK, _BLOCK])
    )

    rejected = _engine(
        vad_speech=True,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=3,
        barge_word_cut_speaker_min_sec=0.0,
    )
    rejected._speaker_gate = _FakeSpeakerGate([0.1])
    rejected._speaker_gate_warmed = True
    detect_no, normal_no = _FakeStream(), _FakeStream()
    r_no = _FakeRecognizer(["please wait", "please wait", "please wait now"])
    assert not rejected._barge_word_cut_step(r_no, detect_no, _BLOCK, now)
    assert not rejected._finish_word_cut_reply(
        r_no, detect_no, normal_no, now=now
    )
    assert rejected._word_cut_tail_probation_step(
        r_no, detect_no, normal_no, _BLOCK, now + 0.1
    ) == "dropped"
    assert normal_no.fed_blocks == 0
    assert rejected._wc_stats["tail_drop_speaker"] == 1


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
        return original_speaker_decision(samples)

    eng._final_transcribe = final_transcribe
    eng._final_above_floor = floor
    original_speaker_decision = eng._speaker_decision_for_final
    eng._speaker_decision_for_final = speaker
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


def test_capture_zero_word_speaker_cut_splices_and_dispatches_async_once():
    """Production authority survives capture -> splice -> async final dispatch."""
    import queue

    final_ready = threading.Event()
    finals: list[str] = []

    class _AsyncRecognizer:
        def __init__(self):
            self.streams: list[_FakeStream] = []
            self.endpoint_blocks: list[np.ndarray] = []

        def create_stream(self, **_kwargs):
            stream = _FakeStream()
            self.streams.append(stream)
            return stream

        def is_ready(self, _stream):
            return False

        def decode_stream(self, _stream):  # pragma: no cover - never ready
            pass

        def get_result(self, stream):
            if len(self.streams) > 1 and stream is self.streams[1]:
                return ""  # double-talk ASR remains empty during playback
            return "owner command" if stream.blocks else ""

        def reset(self, stream):
            stream.blocks.clear()
            stream.fed_blocks = 0

        def is_endpoint(self, stream):
            endpoint = bool(
                self.streams
                and stream is self.streams[0]
                and stream.fed_blocks >= 2
            )
            if endpoint:
                self.endpoint_blocks = [block.copy() for block in stream.blocks]
            return endpoint

    class _FiniteAsyncInput:
        def __init__(self, engine, blocks):
            self.engine = engine
            self.blocks = list(blocks)
            self.index = 0

        def read(self, _n):
            if self.index < len(self.blocks):
                block = self.blocks[self.index]
                self.index += 1
                if self.index == len(self.blocks):
                    self.engine._vad.set_speech(False)
                return block.copy(), False
            assert final_ready.wait(2.0)
            self.engine._running.clear()
            return np.zeros(1600, dtype="float32"), False

    eng = _engine(
        vad_speech=True,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.35,
        barge_word_cut_speaker_threshold=0.30,
        speaker_gate_input=False,
    )
    eng._speaker_gate = _FakeSpeakerGate([0.387])
    eng._speaker_gate_warmed = True
    recognizer = _AsyncRecognizer()
    voice_blocks = [
        np.full(1600, value, dtype="float32")
        for value in (0.11, 0.12, 0.13, 0.14)
    ]
    endpoint_silence = np.zeros(1600, dtype="float32")

    def on_barge_in():
        eng._speaking.clear()

    def on_final(text):
        finals.append(text)
        final_ready.set()

    eng._recognizer = recognizer
    eng._final_recognizer = None
    eng._final_q = queue.Queue(maxsize=8)
    eng._cb = EngineCallbacks(on_barge_in=on_barge_in, on_final=on_final)
    eng._capture_sr = eng.config.sample_rate
    eng._stream_in = _FiniteAsyncInput(
        eng, [*voice_blocks, endpoint_silence]
    )
    eng._final_above_floor = lambda _seg: True
    eng._speaking.set()
    eng._barge_sustain_reset_pending = True
    eng._first_audio_pending = True

    eng._running.set()
    final_worker = threading.Thread(target=eng._final_worker)
    capture_worker = threading.Thread(target=eng._capture_loop)
    final_worker.start()
    capture_worker.start()
    capture_worker.join(timeout=3.0)
    final_worker.join(timeout=3.0)

    assert not capture_worker.is_alive()
    assert not final_worker.is_alive()
    assert final_ready.is_set()
    assert len(finals) == 1
    assert "owner command" in finals[0].lower()
    assert eng._word_cut_pending_samples == 0
    expected = np.concatenate([*voice_blocks, endpoint_silence])
    np.testing.assert_array_equal(
        np.concatenate(recognizer.endpoint_blocks), expected
    )


def test_zero_word_handoff_uses_offline_final_when_streaming_stays_empty():
    """Accepted PCM must reach the second pass even if streaming stays empty.

    Audio-first authority exists precisely because the playback-time streaming
    recognizer can emit zero words.  Replaying the same degraded PCM into a fresh
    streaming state is not proof that it will suddenly produce a raw final, so a
    speaker-authorized segment must let the offline recognizer recover its text.
    """

    class _AlwaysEmptyRecognizer:
        def __init__(self):
            self.streams: list[_FakeStream] = []

        def create_stream(self, **_kwargs):
            stream = _FakeStream()
            self.streams.append(stream)
            return stream

        def is_ready(self, _stream):
            return False

        def decode_stream(self, _stream):  # pragma: no cover - never ready
            pass

        def get_result(self, _stream):
            return ""

        def reset(self, stream):
            stream.blocks.clear()
            stream.fed_blocks = 0

        def is_endpoint(self, stream):
            return bool(
                self.streams
                and stream is self.streams[0]
                and sum(block.size for block in stream.blocks) >= 7 * 1600
            )

    class _OfflineStream(_FakeStream):
        def __init__(self):
            super().__init__()
            self.result = type("Result", (), {"text": "owner command"})()

    class _OfflineRecognizer:
        def __init__(self):
            self.calls = 0

        def create_stream(self):
            self.calls += 1
            return _OfflineStream()

        def decode_stream(self, _stream):
            pass

    class _FiniteInput:
        def __init__(self, engine, blocks):
            self.engine = engine
            self.blocks = list(blocks)
            self.index = 0

        def read(self, _n):
            block = self.blocks[self.index]
            self.index += 1
            if self.index == len(self.blocks):
                self.engine._vad.set_speech(False)
                self.engine._running.clear()
            return block.copy(), False

    eng = _engine(
        vad_speech=True,
        barge_word_cut_require_speaker=True,
        barge_word_cut_speaker_min_words=0,
        barge_word_cut_speaker_min_sec=0.35,
        barge_word_cut_speaker_threshold=0.30,
        speaker_gate_input=False,
        asr_final_min_sec=0.5,
    )
    eng._speaker_gate = _FakeSpeakerGate([0.387])
    eng._speaker_gate_warmed = True
    streaming = _AlwaysEmptyRecognizer()
    offline = _OfflineRecognizer()
    finals: list[str] = []
    voice_blocks = [
        np.full(1600, value, dtype="float32")
        for value in (0.20, 0.21, 0.22, 0.23, 0.24, 0.25)
    ]
    endpoint_silence = np.zeros(1600, dtype="float32")

    def on_barge_in():
        eng._speaking.clear()

    eng._recognizer = streaming
    eng._final_recognizer = offline
    eng._final_q = None
    eng._cb = EngineCallbacks(on_barge_in=on_barge_in, on_final=finals.append)
    eng._capture_sr = eng.config.sample_rate
    eng._stream_in = _FiniteInput(
        eng, [*voice_blocks, endpoint_silence]
    )
    eng._final_above_floor = lambda _seg: True
    eng._speaking.set()
    eng._barge_sustain_reset_pending = True
    eng._first_audio_pending = True

    eng._running.set()
    capture_worker = threading.Thread(target=eng._capture_loop)
    capture_worker.start()
    capture_worker.join(timeout=3.0)

    assert not capture_worker.is_alive()
    assert offline.calls == 1
    assert finals == ["owner command"]


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
    assert "floor=0.0040" in caplog.text
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
