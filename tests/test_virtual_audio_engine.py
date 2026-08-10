from __future__ import annotations

from dataclasses import replace
import threading
import time
import types

import numpy as np
import pytest

from core.engine import EngineCallbacks
from core.engines._acoustic_turn import AcousticTurnTracker
from core.engines.sherpa import (
    SherpaConfig,
    SherpaOnnxEngine,
    _CaptureBlockRejected,
    _KwsControlAuthority,
)
from core.kws_contract import KWS_STOP_PHRASES, KwsPhraseBinding
from core.media_session import CaptureBlockContext, CapturedBlock


class _Binder:
    def __init__(self):
        self.capture_bound = False
        self.playback_bound = False
        self.contract = types.SimpleNamespace(provenance="autotest-virtual-delay:test")
        self.fail_duplex = False

    @property
    def fully_bound(self):
        return self.capture_bound and self.playback_bound

    def bind_capture(self, *, timeout_sec):
        assert timeout_sec > 0
        self.capture_bound = True

    def bind_playback(self, *, timeout_sec):
        assert timeout_sec > 0
        self.playback_bound = True

    def verify(self, *, require_playback):
        if require_playback and (self.fail_duplex or not self.fully_bound):
            return False, "playback wrong"
        return self.capture_bound, "exact virtual route"


def _engine(binder):
    return SherpaOnnxEngine(
        SherpaConfig(
            barge_in_enabled=True,
            barge_word_cut_enabled=True,
            aec_enabled=False,
        ),
        virtual_audio_binder=binder,
    )


def test_virtual_word_cut_authority_is_two_phase():
    binder = _Binder()
    engine = _engine(binder)

    engine._bind_virtual_capture_route()
    assert binder.capture_bound is True
    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False

    engine._prepare_virtual_playback_route()
    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False
    # Capture may have consumed the reply reset while playback was still
    # unbound. Successful authorization must re-arm this same reply.
    engine._barge_sustain_reset_pending = False
    engine._authorize_virtual_playback_route()
    assert binder.fully_bound is True
    assert engine._word_cut_route_verified is True
    assert engine._os_echo_route_verified is True
    assert engine._barge_sustain_reset_pending is True


def test_loss_started_before_duplex_grant_cannot_reauthorize_route():
    binder = _Binder()
    binder.capture_bound = True
    binder.playback_bound = True
    engine = _engine(binder)
    engine._virtual_route_failure_in_progress = True

    with pytest.raises(RuntimeError, match="revocation already started"):
        engine._authorize_virtual_playback_route()

    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False
    assert engine._barge_sustain_reset_pending is False


def test_rearmed_duplex_grant_initializes_word_cut_in_same_reply():
    class _Stream:
        def __init__(self):
            self.blocks = []

        def accept_waveform(self, _sample_rate, samples):
            self.blocks.append(np.asarray(samples).copy())

    class _Recognizer:
        def __init__(self):
            self.streams = []

        def create_stream(self, **_kwargs):
            stream = _Stream()
            self.streams.append(stream)
            return stream

        def is_ready(self, _stream):
            return False

        def get_result(self, _stream):
            return ""

        def reset(self, _stream):
            pass

        def is_endpoint(self, _stream):
            return False

    class _Vad:
        def accept_waveform(self, _samples):
            pass

        def is_speech_detected(self):
            return True

    class _OneBlockInput:
        generation = 0
        actual_device = None

        def __init__(self, engine):
            self.engine = engine

        def read(self, frames):
            self.engine._running.clear()
            return np.zeros(frames, dtype="float32"), False

    binder = _Binder()
    engine = _engine(binder)
    recognizer = _Recognizer()
    engine._recognizer = recognizer
    engine._vad = _Vad()
    engine._capture_sr = engine.config.sample_rate
    engine._stream_in = _OneBlockInput(engine)
    # This fixture bypasses the normal capture-domain resolution that binds the
    # live source identity before virtual-route authorization.
    engine._capture_authority_source_generation = 0
    engine._capture_authority_source_device = None
    engine._speaking.set()
    # Model the race: the reply reset was already consumed with capture-only
    # authority, so no detector stream exists yet.
    engine._barge_sustain_reset_pending = False
    engine._bind_virtual_capture_route()
    engine._prepare_virtual_playback_route()
    engine._authorize_virtual_playback_route()

    engine._running.set()
    engine._capture_loop()

    assert len(recognizer.streams) == 2
    assert len(recognizer.streams[0].blocks) == 0
    assert len(recognizer.streams[1].blocks) == 1


def test_private_kws_rechecks_route_at_callback_seam():
    class _KwsStream:
        def accept_waveform(self, _sample_rate, _samples):
            pass

    class _Kws:
        def __init__(self, engine):
            self.engine = engine
            self.resets = 0

        def is_ready(self, _stream):
            return False

        def get_result(self, _stream):
            # Revocation begins after the outer eligibility read and decode but
            # before the callback seam protected by the route lock.
            self.engine._virtual_route_failure_in_progress = True
            self.engine._speaking.clear()
            return "stop"

        def reset_stream(self, _stream):
            self.resets += 1

    binder = _Binder()
    engine = _engine(binder)
    commands = []
    engine._kws_stream = _KwsStream()
    engine._kws = _Kws(engine)
    engine._cb = EngineCallbacks(on_command=commands.append)
    engine._os_echo_route_verified = True

    consumed = engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        guard_private_route=True,
        require_os_echo_route=True,
    )

    assert engine._kws.resets == 1
    assert commands == []
    assert consumed is False


class _FixedKws:
    """Minimal spotter that reports one fixed keyword hit per poll."""

    def __init__(self, keyword):
        self.keyword = keyword
        self.resets = 0

    def is_ready(self, _stream):
        return False

    def get_result(self, _stream):
        return self.keyword

    def reset_stream(self, _stream):
        self.resets += 1


_PLAYBACK_KWS_TOKENS = tuple(phrase.tokens for phrase in KWS_STOP_PHRASES)
_PLAYBACK_KWS_BINDINGS = tuple(
    KwsPhraseBinding(
        tokens=tokens,
        surface=phrase.surface,
        result_label=phrase.result_label,
    )
    for tokens, phrase in zip(
        _PLAYBACK_KWS_TOKENS,
        KWS_STOP_PHRASES,
        strict=True,
    )
)


class _TokenKws(_FixedKws):
    def __init__(
        self,
        keyword,
        tokens,
        *,
        before_result=None,
        before_tokens=None,
    ):
        super().__init__(keyword)
        self._tokens = tokens
        self._before_result = before_result
        self._before_tokens = before_tokens
        self.token_calls = 0

    def get_result(self, stream):
        if self._before_result is not None:
            self._before_result()
        return super().get_result(stream)

    def tokens(self, _stream):
        self.token_calls += 1
        if self._before_tokens is not None:
            self._before_tokens()
        return self._tokens


class _NullKwsStream:
    def accept_waveform(self, _sample_rate, _samples):
        pass


class _BoundedKwsStream:
    def __init__(self, spotter: "_BoundedKws", sequence: int) -> None:
        self.spotter = spotter
        self.sequence = sequence
        self.decode_calls = 0

    def accept_waveform(self, _sample_rate, samples):
        self.spotter.accept_calls += 1
        self.spotter.accepted.append(np.asarray(samples).copy())
        if self.spotter.failure_phase == "accept":
            raise RuntimeError("injected KWS accept failure")


class _BoundedKws:
    """Keyword fake with exact native-call accounting and stream recreation."""

    def __init__(
        self,
        *,
        ready_decodes: int | None,
        keyword: str = "",
        failure_phase: str | None = None,
    ) -> None:
        self.ready_decodes = ready_decodes
        self.keyword = keyword
        self.failure_phase = failure_phase
        self.create_calls = 0
        self.accept_calls = 0
        self.ready_calls = 0
        self.decode_calls = 0
        self.result_calls = 0
        self.reset_calls = 0
        self.accepted: list[np.ndarray] = []
        self.streams: list[_BoundedKwsStream] = []

    def create_stream(self):
        self.create_calls += 1
        stream = _BoundedKwsStream(self, self.create_calls)
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        self.ready_calls += 1
        if self.failure_phase == "is_ready":
            raise RuntimeError("injected KWS readiness failure")
        return self.ready_decodes is None or stream.decode_calls < self.ready_decodes

    def decode_stream(self, stream):
        self.decode_calls += 1
        stream.decode_calls += 1
        if self.failure_phase == "decode":
            raise RuntimeError("injected KWS decode failure")

    def get_result(self, _stream):
        self.result_calls += 1
        if self.failure_phase == "result":
            raise RuntimeError("injected KWS result failure")
        return self.keyword

    def reset_stream(self, _stream):
        self.reset_calls += 1


def _kws_engine(keyword):
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._kws = _FixedKws(keyword)
    engine._kws_stream = _NullKwsStream()
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)
    return engine, commands


def _authority_kws_engine(
    keyword: str,
    *,
    phrase_index: int = 0,
    speaking: bool = True,
    os_echo_route_verified: bool = False,
    before_result=None,
    before_tokens=None,
):
    engine = SherpaOnnxEngine(SherpaConfig())
    spotter = _TokenKws(
        keyword,
        _PLAYBACK_KWS_TOKENS[phrase_index],
        before_result=before_result,
        before_tokens=before_tokens,
    )
    engine._kws = spotter
    engine._kws_stream = _NullKwsStream()
    engine._kws_phrase_bindings = _PLAYBACK_KWS_BINDINGS
    engine._stream_in = types.SimpleNamespace(
        generation=3,
        actual_device="mic-a",
    )
    engine._capture_epoch = 2
    engine._capture_callback_context.epoch = 2
    engine._capture_callback_context.capture_generation = 5
    engine._capture_media_session = types.SimpleNamespace(generation=5)
    engine._capture_authority_source_generation = 3
    engine._capture_authority_source_device = "mic-a"
    engine._os_echo_route_verified = os_echo_route_verified
    engine._speak_gen = 7
    engine._playback_generation = 11
    if speaking:
        engine._speaking.set()
    else:
        engine._speaking.clear()
    authority = _KwsControlAuthority(
        capture_epoch=2,
        capture_generation=5,
        source_generation=3,
        source_device="mic-a",
        authority_source_generation=3,
        authority_source_device="mic-a",
        os_echo_route_verified=os_echo_route_verified,
        speaking=speaking,
        speak_generation=7,
        playback_generation=11,
    )
    return engine, spotter, authority


def test_kws_own_echo_suppressed_when_now_playing_contains_keyword():
    # The assistant's own TTS ("I'll stop now") leaks back through imperfect AEC
    # and the spotter fires on it -- the own-echo guard must drop the hit so it
    # cannot self-interrupt.
    engine, commands = _kws_engine("stop")
    engine._speaking.set()
    engine._now_playing = "Okay, I will stop now."
    stop_results = []
    engine._cb = EngineCallbacks(
        on_command=commands.append,
        on_control_stop_result=stop_results.append,
    )

    consumed = engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert commands == []
    assert stop_results == []
    assert consumed is False
    assert engine._kws_own_echo_suppressions == 1
    assert engine._kws.resets == 1  # stream is still reset after the hit


def test_kws_hit_passes_when_not_speaking_even_if_now_playing_matches():
    # Not speaking -> the guard is inert; a genuine user "stop" must reach the
    # command callback even if the last-played line contained the word.
    engine, commands = _kws_engine("stop")
    engine._now_playing = "Okay, I will stop now."

    consumed = engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert commands == ["stop"]
    assert consumed is True
    assert engine._kws_own_echo_suppressions == 0


def test_kws_hit_passes_when_now_playing_does_not_contain_keyword():
    # Speaking, but the playing sentence does not contain the spotted word ->
    # this is the user talking over the assistant, not its own echo. Pass it.
    engine, commands = _kws_engine("stop")
    engine._speaking.set()
    engine._now_playing = "The weather is nice today."
    stop_commands = []
    engine._cb = EngineCallbacks(
        on_command=commands.append,
        on_control_stop_result=stop_commands.append,
    )

    consumed = engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert commands == []
    assert [result.text for result in stop_commands] == ["stop"]
    assert consumed is True
    assert engine._kws_own_echo_suppressions == 0


@pytest.mark.parametrize(
    "keyword,phrase_index",
    [("stop", 0), ("wait", 4)],
)
def test_playback_kws_uses_only_typed_stop_callback_and_canonicalizes_wait(
    keyword,
    phrase_index,
) -> None:
    engine, spotter, authority = _authority_kws_engine(
        keyword,
        phrase_index=phrase_index,
    )
    engine._now_playing = "The weather is pleasant today."
    generic_commands = []
    generic_results = []
    stop_results = []
    engine._cb = EngineCallbacks(
        on_command=generic_commands.append,
        on_command_result=generic_results.append,
        on_control_stop_result=stop_results.append,
    )

    consumed = engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    assert consumed is True
    assert generic_commands == []
    assert generic_results == []
    assert [result.text for result in stop_results] == ["stop"]
    assert spotter.token_calls == 1
    assert engine._playback_kws_claim is None


def test_playback_non_stop_abstains_while_idle_generic_label_is_unchanged() -> None:
    playback, playback_spotter, playback_authority = _authority_kws_engine(
        "command mode"
    )
    playback_generic = []
    playback_stop = []
    playback._cb = EngineCallbacks(
        on_command_result=playback_generic.append,
        on_control_stop_result=playback_stop.append,
    )

    assert not playback._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=playback_authority,
    )
    assert playback_generic == []
    assert playback_stop == []
    assert playback_spotter.token_calls == 1

    idle, idle_spotter, idle_authority = _authority_kws_engine(
        "command mode",
        speaking=False,
    )
    idle_generic = []
    idle_stop = []
    idle._cb = EngineCallbacks(
        on_command_result=idle_generic.append,
        on_control_stop_result=idle_stop.append,
    )

    assert idle._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=idle_authority,
    )
    assert [result.text for result in idle_generic] == ["command mode"]
    assert idle_stop == []
    assert idle_spotter.token_calls == 0


@pytest.mark.parametrize(
    "stop_callback", [None, object()], ids=("missing", "noncallable")
)
def test_playback_kws_without_callable_typed_stop_callback_abstains_before_native(
    stop_callback,
) -> None:
    engine, spotter, authority = _authority_kws_engine("stop")
    engine._now_playing = "A sentence with no control words."
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=5)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    generic = []
    engine._cb = EngineCallbacks(
        on_command_result=generic.append,
        on_control_stop_result=stop_callback,
    )

    def unexpected_claim(*_args, **_kwargs):
        raise AssertionError("missing STOP callback must not claim playback")

    engine._claim_playback_kws_if_current = unexpected_claim

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert generic == []
    assert spotter.resets == 0
    assert spotter.token_calls == 0
    assert engine._playback_kws_claim is None
    engine._enqueue_play("new speech remains admissible", None)
    assert engine._play_q.get_nowait()[0] == "new speech remains admissible"


@pytest.mark.parametrize(
    "phrase_index,own_speech",
    [
        (3, "Please be quiet while I explain."),
        (0, "Please be quiet while I explain."),
        (4, "I need you to hold on for a moment."),
    ],
    ids=("exact-be-quiet", "collapsed-stop-sibling", "collapsed-wait-sibling"),
)
def test_playback_kws_collapsed_alias_ambiguity_abstains(
    phrase_index,
    own_speech,
) -> None:
    keyword = KWS_STOP_PHRASES[phrase_index].result_label
    engine, _spotter, authority = _authority_kws_engine(
        keyword,
        phrase_index=phrase_index,
    )
    engine._now_playing = own_speech
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    assert stop_results == []
    assert engine._kws_own_echo_suppressions == 1


def test_playback_kws_token_label_mismatch_abstains() -> None:
    engine, _spotter, authority = _authority_kws_engine(
        "stop",
        phrase_index=4,
    )
    engine._now_playing = "A sentence with no control words."
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )
    assert stop_results == []


@pytest.mark.parametrize("native_label", ["STOP", "STOP!!!", " stop", "wait "])
def test_playback_kws_requires_raw_exact_native_label(native_label) -> None:
    phrase_index = 4 if native_label.startswith("wait") else 0
    engine, _spotter, authority = _authority_kws_engine(
        native_label,
        phrase_index=phrase_index,
    )
    engine._now_playing = "A sentence with no control words."
    generic = []
    stop_results = []
    engine._cb = EngineCallbacks(
        on_command_result=generic.append,
        on_control_stop_result=stop_results.append,
    )

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    assert generic == []
    assert stop_results == []
    assert engine._playback_kws_claim is None


def test_playback_kws_without_exact_phrase_binding_abstains() -> None:
    engine, spotter, authority = _authority_kws_engine("stop")
    engine._kws_phrase_bindings = ()
    engine._now_playing = "A sentence with no control words."
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    assert spotter.token_calls == 0
    assert stop_results == []
    assert engine._playback_kws_claim is None


@pytest.mark.parametrize(
    "stale_case",
    [
        "capture_epoch",
        "capture_generation",
        "media_generation",
        "source_generation",
        "source_device",
        "authority_source_generation",
        "authority_source_device",
        "os_echo_route_authority",
        "speak_generation",
        "playback_generation",
        "speaking_state",
    ],
)
def test_stale_kws_authority_cannot_callback_or_terminal_lineage(stale_case) -> None:
    holder = {}

    def make_stale() -> None:
        engine = holder["engine"]
        if stale_case == "capture_epoch":
            engine._capture_epoch += 1
        elif stale_case == "capture_generation":
            engine._capture_callback_context.capture_generation += 1
        elif stale_case == "media_generation":
            engine._capture_media_session.generation += 1
        elif stale_case == "source_generation":
            engine._stream_in.generation += 1
        elif stale_case == "source_device":
            engine._stream_in.actual_device = "mic-b"
        elif stale_case == "authority_source_generation":
            engine._capture_authority_source_generation += 1
        elif stale_case == "authority_source_device":
            engine._capture_authority_source_device = "mic-b"
        elif stale_case == "os_echo_route_authority":
            engine._os_echo_route_verified = True
        elif stale_case == "speak_generation":
            engine._speak_gen += 1
        elif stale_case == "playback_generation":
            engine._playback_generation += 1
        else:
            engine._speaking.clear()

    engine, spotter, authority = _authority_kws_engine(
        "stop",
        before_tokens=make_stale,
    )
    holder["engine"] = engine
    engine._now_playing = "A sentence with no control words."
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=5)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    generic = []
    stop_results = []
    engine._cb = EngineCallbacks(
        on_command_result=generic.append,
        on_control_stop_result=stop_results.append,
    )

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert generic == []
    assert stop_results == []
    assert spotter.resets == 1
    assert engine._playback_kws_claim is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("capture_epoch", True),
        ("capture_generation", np.int64(5)),
        ("source_generation", True),
        ("authority_source_generation", np.int64(3)),
        ("os_echo_route_verified", 0),
        ("speaking", 1),
        ("speak_generation", np.int64(7)),
        ("playback_generation", True),
    ],
)
def test_forged_coercible_kws_authority_abstains_without_terminal(
    field,
    value,
) -> None:
    engine, _spotter, authority = _authority_kws_engine("stop")
    authority = replace(authority, **{field: value})
    engine._now_playing = "A sentence with no control words."
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=5)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert stop_results == []
    assert engine._playback_kws_claim is None


def test_kws_authority_snapshot_does_not_coerce_forged_capture_types() -> None:
    engine, _spotter, _authority = _authority_kws_engine("stop")
    captured = CapturedBlock(
        pcm=np.zeros(1, dtype="float32"),
        sample_rate_hz=16_000,
        sequence=1,
        captured_at=1.0,
        capture_epoch=True,
        source_generation=np.int64(3),
        capture_generation=np.int64(5),
        source_device="mic-a",
    )
    context = CaptureBlockContext(
        authority_source_generation=np.int64(3),
        authority_source_device="mic-a",
        os_echo_route_verified=1,
        speaking=1,
        speak_generation=np.int64(7),
        playback_generation=np.int64(11),
    )

    authority = engine._kws_control_authority(captured, context)

    assert authority.capture_epoch is True
    assert type(authority.source_generation) is np.int64
    assert type(authority.capture_generation) is np.int64
    assert type(authority.authority_source_generation) is np.int64
    assert authority.os_echo_route_verified == 1
    assert type(authority.os_echo_route_verified) is int
    assert authority.speaking == 1
    assert type(authority.speaking) is int
    assert type(authority.speak_generation) is np.int64
    assert type(authority.playback_generation) is np.int64
    assert not engine._kws_control_authority_is_current(authority)


@pytest.mark.parametrize("route_case", ["virtual-route", "os-echo-route"])
def test_revoked_playback_route_cannot_callback_or_terminal_lineage(route_case) -> None:
    holder = {}

    def revoke_route() -> None:
        engine = holder["engine"]
        if route_case == "virtual-route":
            engine._virtual_route_failure_in_progress = True
        else:
            engine._os_echo_route_verified = False

    engine, _spotter, authority = _authority_kws_engine(
        "stop",
        os_echo_route_verified=True,
        before_tokens=revoke_route,
    )
    holder["engine"] = engine
    engine._now_playing = "A sentence with no control words."
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=5)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        guard_private_route=route_case == "virtual-route",
        require_os_echo_route=route_case == "os-echo-route",
        os_echo_route_verified=True,
        control_authority=authority,
    )

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert stop_results == []
    assert engine._playback_kws_claim is None


@pytest.mark.parametrize("initially_speaking", [False, True], ids=("idle", "active"))
def test_kws_authority_rejects_playback_state_aba(initially_speaking) -> None:
    holder = {}

    def cycle_playback_state() -> None:
        engine = holder["engine"]
        if initially_speaking:
            engine.stop_speaking()
            assert engine._begin_playback_run_if_current(engine._speak_gen) is False
        else:
            assert engine._begin_playback_run_if_current(engine._speak_gen) is False
            engine.stop_speaking()
        assert engine._speaking.is_set() is initially_speaking

    keyword = "stop" if initially_speaking else "command mode"
    engine, _spotter, authority = _authority_kws_engine(
        keyword,
        speaking=initially_speaking,
        before_result=cycle_playback_state,
    )
    holder["engine"] = engine
    engine._now_playing = "A sentence with no control words."
    generic = []
    stop_results = []
    engine._cb = EngineCallbacks(
        on_command_result=generic.append,
        on_control_stop_result=stop_results.append,
    )

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    assert generic == []
    assert stop_results == []
    assert engine._playback_kws_claim is None


def test_capture_callback_admission_failure_does_not_terminalize_or_leak_claim() -> (
    None
):
    engine, _spotter, authority = _authority_kws_engine("stop")
    engine._now_playing = "A sentence with no control words."
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=5)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    stop_results = []
    engine._cb = EngineCallbacks(on_control_stop_result=stop_results.append)
    claim_current = engine._claim_playback_kws_if_current

    def claim_then_stop_capture(control_authority, *, stop_only=None):
        claim = claim_current(control_authority, stop_only=stop_only)
        assert claim is not None
        engine._capture_epoch += 1
        return claim

    engine._claim_playback_kws_if_current = claim_then_stop_capture

    assert not engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        control_authority=authority,
    )

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert stop_results == []
    assert engine._playback_kws_claim is None


def test_kws_callback_exception_releases_playback_claim() -> None:
    class _CallbackFailure(RuntimeError):
        pass

    engine, _spotter, authority = _authority_kws_engine("stop")
    engine._now_playing = "A sentence with no control words."

    def fail_callback(_result):
        assert engine._playback_kws_claim is not None
        raise _CallbackFailure("typed STOP callback failed")

    engine._cb = EngineCallbacks(on_control_stop_result=fail_callback)

    with pytest.raises(_CallbackFailure, match="typed STOP callback failed"):
        engine._poll_keywords(
            np.zeros(1600, dtype="float32"),
            control_authority=authority,
        )

    assert engine._playback_kws_claim is None


def test_typed_kws_hit_closes_source_acoustic_turn() -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._kws = _FixedKws("stop")
    engine._kws_stream = _NullKwsStream()
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=3)
    engine._acoustic_turn_tracker = tracker
    commands = []
    engine._cb = EngineCallbacks(on_command_result=commands.append)

    consumed = engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert consumed is True
    assert len(commands) == 1
    assert commands[0].revision == 0
    assert commands[0].acoustic is not None
    assert not tracker.active


def test_legacy_kws_callback_still_closes_a_typed_partial_turn() -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._kws = _FixedKws("stop")
    engine._kws_stream = _NullKwsStream()
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=3)
    engine._acoustic_turn_tracker = tracker
    first_lineage, first_revision = tracker.partial(emitted_at=10.0)
    commands = []
    engine._cb = EngineCallbacks(
        on_command=commands.append,
        on_partial_result=lambda _result: None,
    )

    consumed = engine._poll_keywords(np.zeros(1600, dtype="float32"))
    next_lineage, next_revision = tracker.partial(emitted_at=11.0)

    assert consumed is True
    assert commands == ["stop"]
    assert first_revision == 0
    assert next_revision == 0
    assert first_lineage.spans[0].key != next_lineage.spans[0].key
    assert next_lineage.spans[0].turn_index == 2


def _bounded_kws_engine(spotter: _BoundedKws) -> SherpaOnnxEngine:
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._kws = spotter
    engine._kws_stream = spotter.create_stream()
    return engine


def test_kws_exactly_sixty_four_decodes_runs_final_probe_and_result_path() -> None:
    spotter = _BoundedKws(ready_decodes=64)
    engine = _bounded_kws_engine(spotter)

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert spotter.accept_calls == 1
    assert spotter.ready_calls == 65
    assert spotter.decode_calls == 64
    assert spotter.result_calls == 1
    assert spotter.reset_calls == 0
    assert spotter.create_calls == 1


def test_kws_decode_exhaustion_abstains_before_result_and_preserves_lineage() -> None:
    spotter = _BoundedKws(ready_decodes=None, keyword="stop")
    engine = _bounded_kws_engine(spotter)
    old_stream = engine._kws_stream
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=3)
    before, revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert revision == 0
    assert before.spans[0].key == after.spans[0].key
    assert tracker.active
    assert commands == []
    assert spotter.accept_calls == 1
    assert spotter.ready_calls == 65
    assert spotter.decode_calls == 64
    assert spotter.result_calls == 0
    assert spotter.reset_calls == 1
    assert spotter.create_calls == 2
    assert engine._kws_stream is spotter.streams[-1]
    assert engine._kws_stream is not old_stream


@pytest.mark.parametrize("native_result", [None, False, 0, (), []])
def test_kws_falsey_non_string_result_is_native_fault_and_preserves_lineage(
    native_result,
) -> None:
    spotter = _BoundedKws(ready_decodes=0, keyword=native_result)
    engine = _bounded_kws_engine(spotter)
    old_stream = engine._kws_stream
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=3)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    commands = []
    engine._cb = EngineCallbacks(on_command_result=commands.append)

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert after.spans[-1].key == before.spans[-1].key
    assert tracker.active
    assert commands == []
    assert spotter.result_calls == 1
    assert spotter.reset_calls == 1
    assert spotter.create_calls == 2
    assert engine._kws_stream is not old_stream


@pytest.mark.parametrize("failure_phase", ["accept", "is_ready", "decode", "result"])
def test_kws_native_fault_abstains_recreates_stream_and_preserves_lineage(
    failure_phase,
) -> None:
    spotter = _BoundedKws(
        ready_decodes=1,
        keyword="stop",
        failure_phase=failure_phase,
    )
    engine = _bounded_kws_engine(spotter)
    old_stream = engine._kws_stream
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=2, capture_generation=3)
    before, _revision = tracker.partial(emitted_at=10.0)
    engine._acoustic_turn_tracker = tracker
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    after = tracker.current(emitted_at=11.0)
    assert after is not None
    assert before.spans[0].key == after.spans[0].key
    assert tracker.active
    assert commands == []
    assert spotter.reset_calls == 1
    assert spotter.create_calls == 2
    assert engine._kws_stream is not old_stream


def test_kws_recovery_recreates_even_when_best_effort_reset_fails() -> None:
    class _ResetFailureKws(_BoundedKws):
        def reset_stream(self, stream):
            super().reset_stream(stream)
            raise RuntimeError("injected KWS reset failure")

    spotter = _ResetFailureKws(ready_decodes=None)
    engine = _bounded_kws_engine(spotter)
    old_stream = engine._kws_stream

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert spotter.reset_calls == 1
    assert spotter.create_calls == 2
    assert engine._kws_stream is spotter.streams[-1]
    assert engine._kws_stream is not old_stream


def test_kws_recovery_sets_stream_none_when_recreation_fails() -> None:
    class _RecreateFailureKws(_BoundedKws):
        def create_stream(self):
            if self.create_calls:
                self.create_calls += 1
                raise RuntimeError("injected KWS recreate failure")
            return super().create_stream()

    spotter = _RecreateFailureKws(ready_decodes=None)
    engine = _bounded_kws_engine(spotter)

    assert not engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert spotter.reset_calls == 1
    assert spotter.create_calls == 2
    assert engine._kws_stream is None


@pytest.mark.parametrize(
    "config_override, samples, detected_at",
    [
        ({}, np.zeros(0, dtype="float32"), None),
        (
            {
                "sample_rate": 100,
                "block_sec": 0.1,
                "media_pcm_queue_ms": 300.0,
            },
            np.zeros(31, dtype="float32"),
            None,
        ),
        ({"block_sec": float("nan")}, np.zeros(1, dtype="float32"), None),
        ({"sample_rate": True}, np.zeros(1, dtype="float32"), None),
        ({"block_sec": True}, np.zeros(1, dtype="float32"), None),
        ({"media_pcm_queue_ms": True}, np.zeros(1, dtype="float32"), None),
        ({}, np.zeros(1, dtype="float32"), -1.0),
        ({}, np.zeros(1, dtype="float32"), float("nan")),
        ({}, np.zeros(1, dtype="float32"), True),
        pytest.param(
            {},
            np.zeros(1, dtype="float32"),
            10**10_000,
            id="giant-detected-at",
        ),
    ],
)
def test_kws_rejects_empty_oversize_or_invalid_timing_before_native_work(
    config_override,
    samples,
    detected_at,
) -> None:
    spotter = _BoundedKws(ready_decodes=0, keyword="stop")
    engine = SherpaOnnxEngine(SherpaConfig(**config_override))
    engine._kws = spotter
    initial_stream = spotter.create_stream()
    engine._kws_stream = initial_stream
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)

    with pytest.raises(_CaptureBlockRejected):
        engine._poll_keywords(samples, detected_at=detected_at)

    assert commands == []
    assert spotter.accept_calls == 0
    assert spotter.ready_calls == 0
    assert spotter.decode_calls == 0
    assert spotter.result_calls == 0
    assert spotter.reset_calls == 0
    assert spotter.create_calls == 1
    assert engine._kws_stream is initial_stream


def test_kws_callback_exception_is_not_reclassified_as_native_failure() -> None:
    class _CallbackFailure(RuntimeError):
        pass

    spotter = _BoundedKws(ready_decodes=0, keyword="stop")
    engine = _bounded_kws_engine(spotter)

    def fail_callback(_keyword):
        raise _CallbackFailure("command callback failed")

    engine._cb = EngineCallbacks(on_command=fail_callback)

    with pytest.raises(_CallbackFailure, match="command callback failed"):
        engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert spotter.result_calls == 1
    assert spotter.reset_calls == 1  # the ordinary hit reset, not recovery
    assert spotter.create_calls == 1


def _one_block_kws_capture() -> tuple[SherpaOnnxEngine, object]:
    class _Stream:
        def __init__(self) -> None:
            self.blocks = []

        def accept_waveform(self, _sample_rate, samples) -> None:
            self.blocks.append(np.asarray(samples).copy())

    class _Recognizer:
        def __init__(self) -> None:
            self.stream = _Stream()
            self.reset_calls = 0

        def create_stream(self, **_kwargs):
            return self.stream

        def is_ready(self, _stream) -> bool:
            return False

        def get_result(self, _stream) -> str:
            return ""

        def reset(self, _stream) -> None:
            self.reset_calls += 1

        def is_endpoint(self, _stream) -> bool:
            return False

    class _OneBlockInput:
        generation = 0
        actual_device = None

        def __init__(self, engine) -> None:
            self.engine = engine

        def read(self, frames):
            self.engine._running.clear()
            return np.ones(frames, dtype="float32"), False

    engine = SherpaOnnxEngine(
        SherpaConfig(
            barge_in_enabled=False,
            aec_enabled=False,
        )
    )
    recognizer = _Recognizer()
    engine._recognizer = recognizer
    engine._capture_sr = engine.config.sample_rate
    engine._stream_in = _OneBlockInput(engine)
    # Inline fixture equivalent of production's capture-domain authority bind.
    engine._capture_authority_source_generation = 0
    engine._capture_authority_source_device = None
    return engine, recognizer


def test_capture_kws_consumed_frame_never_reaches_normal_asr() -> None:
    engine, recognizer = _one_block_kws_capture()
    engine._poll_keywords = lambda _samples, **_kwargs: True

    engine._running.set()
    engine._capture_loop()

    assert recognizer.reset_calls == 1
    assert recognizer.stream.blocks == []


def test_legacy_kws_exhaustion_recreates_stream_then_preserves_same_block_asr():
    engine, recognizer = _one_block_kws_capture()
    spotter = _BoundedKws(ready_decodes=None, keyword="stop")
    engine._kws = spotter
    engine._kws_stream = spotter.create_stream()
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)

    engine._running.set()
    engine._capture_loop()

    assert commands == []
    assert spotter.ready_calls == 65
    assert spotter.decode_calls == 64
    assert spotter.result_calls == 0
    assert spotter.reset_calls == 1
    assert spotter.create_calls == 2
    assert len(recognizer.stream.blocks) == 1
    assert np.array_equal(
        recognizer.stream.blocks[0],
        np.ones(1600, dtype="float32"),
    )


def test_oversize_kws_capture_block_is_rejected_without_normal_asr_laundering():
    engine, recognizer = _one_block_kws_capture()
    spotter = _BoundedKws(ready_decodes=0, keyword="stop")
    engine._kws = spotter
    engine._kws_stream = spotter.create_stream()

    class _OversizeInput:
        generation = 0

        def read(self, _frames):
            engine._running.clear()
            return np.ones(4801, dtype="float32"), False

    engine._stream_in = _OversizeInput()
    engine._running.set()
    engine._capture_loop()

    assert spotter.accept_calls == 0
    assert spotter.ready_calls == 0
    assert spotter.result_calls == 0
    assert recognizer.stream.blocks == []


def test_capture_kws_terminal_closes_open_confirm_window() -> None:
    engine, recognizer = _one_block_kws_capture()
    engine._capture_authority_source_generation = 0
    engine._capture_authority_source_device = None
    engine._os_echo_route_verified = True
    engine._kws = _TokenKws("stop", _PLAYBACK_KWS_TOKENS[0])
    engine._kws_stream = _NullKwsStream()
    engine._kws_phrase_bindings = _PLAYBACK_KWS_BINDINGS
    stop_results = []
    barges: list[str] = []
    engine._cb = EngineCallbacks(
        on_control_stop_result=stop_results.append,
        on_barge_in=lambda: barges.append("barge"),
    )
    engine._speaking.set()
    engine._begin_barge_confirm(recognizer, recognizer.stream, time.monotonic())
    engine._confirm_base_text = "old partial"
    engine._confirm_handoff_stream_live = True
    engine._confirm_handoff_pending = object()
    engine._confirm_primary_pcm.append(np.ones(3, dtype="float32"))
    engine._confirm_alternate_pcm.append(np.ones(3, dtype="float32"))
    engine._confirm_audio_started_at = 1.0
    engine._confirm_audio_ended_at = 2.0
    engine._confirm_echo_obs.append((1.0, 2.0, 0.0))
    assert engine._barge_confirm_active()
    assert engine._duck_gain < 1.0

    engine._running.set()
    engine._capture_loop()

    assert recognizer.reset_calls == 1
    assert recognizer.stream.blocks == []
    assert engine._kws.resets == 1
    assert [result.text for result in stop_results] == ["stop"]
    assert barges == []
    assert not engine._barge_confirm_active()
    assert engine._duck_gain == 1.0
    assert engine._confirm_base_text == ""
    assert engine._confirm_speak_generation is None
    assert engine._confirm_playback_generation is None
    assert not engine._confirm_handoff_stream_live
    assert engine._confirm_handoff_pending is None
    assert engine._confirm_primary_pcm == []
    assert engine._confirm_alternate_pcm == []
    assert engine._confirm_audio_started_at is None
    assert engine._confirm_audio_ended_at is None
    assert engine._confirm_echo_obs == []


def test_failed_virtual_playback_proof_never_grants_authority():
    binder = _Binder()
    binder.capture_bound = True
    binder.fail_duplex = True
    engine = _engine(binder)
    engine._word_cut_route_verified = True
    engine._os_echo_route_verified = True

    with pytest.raises(RuntimeError, match="duplex route"):
        engine._prepare_virtual_playback_route()

    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False


def test_production_engine_has_no_virtual_route_authority():
    engine = SherpaOnnxEngine(SherpaConfig())

    engine._bind_virtual_capture_route()
    engine._prepare_virtual_playback_route()
    engine._authorize_virtual_playback_route()

    assert engine._virtual_audio_binder is None
    assert engine._word_cut_route_verified is False


def test_route_loss_hard_flushes_and_closes_before_publishing_failure():
    engine = _engine(_Binder())
    engine._running.set()
    engine._word_cut_route_verified = True
    engine._os_echo_route_verified = True

    class _FIFO:
        def __init__(self):
            self.calls = []

        def interrupt_tags(self, status, fade_samples=0):
            self.calls.append(("interrupt", status, fade_samples))
            return ()

        def flush(self, fade_samples=0):
            self.calls.append(("flush", fade_samples))

    class _Output:
        def __init__(self):
            self.calls = []

        def abort(self):
            assert engine.virtual_route_failure == ""
            self.calls.append("abort")

        def close(self):
            assert engine.virtual_route_failure == ""
            self.calls.append("close")

    fifo = _FIFO()
    output = _Output()
    engine._fifo = fifo
    engine._out_stream = output

    engine._fail_virtual_route("target drift")

    assert [call[0] for call in fifo.calls] == ["interrupt", "flush"]
    assert all(call[-1] == 0 for call in fifo.calls)
    assert output.calls == ["abort", "close"]
    assert engine._fifo is None and engine._out_stream is None
    assert engine._running.is_set() is False
    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False
    assert engine.virtual_route_failure == "target drift"


def test_concurrent_route_failure_cannot_publish_before_blocked_abort_finishes():
    engine = _engine(_Binder())
    engine._running.set()
    abort_entered = threading.Event()
    release_abort = threading.Event()

    class _Output:
        def __init__(self):
            self.abort_calls = 0

        def abort(self):
            self.abort_calls += 1
            abort_entered.set()
            assert release_abort.wait(2.0)

        def close(self):
            pass

    output = _Output()
    engine._out_stream = output
    first = threading.Thread(
        target=engine._fail_virtual_route,
        args=("first failure",),
    )
    first.start()
    assert abort_entered.wait(1.0)

    engine._fail_virtual_route("second failure")
    assert engine.virtual_route_failure == ""
    assert output.abort_calls == 1

    release_abort.set()
    first.join(timeout=1.0)
    assert first.is_alive() is False
    assert engine.virtual_route_failure == "first failure"
    assert output.abort_calls == 1


def test_unexpected_private_capture_exit_publishes_fatal_but_stop_does_not():
    engine = _engine(_Binder())
    engine._running.set()

    assert engine._fail_virtual_capture_if_unexpected("capture failed") is True
    assert engine.virtual_route_failure == "capture failed"
    assert engine._running.is_set() is False

    stopping = _engine(_Binder())
    stopping._running.set()
    stopping._capture_stopping.set()
    assert stopping._fail_virtual_capture_if_unexpected("normal stop") is False
    assert stopping.virtual_route_failure == ""


def test_route_loss_fences_an_inflight_capture_block_before_callbacks():
    read_entered = threading.Event()
    release_read = threading.Event()
    commands = []

    class _Stream:
        def accept_waveform(self, _sample_rate, _samples):
            pass

    class _Recognizer:
        def create_stream(self, **_kwargs):
            return _Stream()

        def is_ready(self, _stream):
            return False

        def get_result(self, _stream):
            return ""

        def reset(self, _stream):
            pass

        def is_endpoint(self, _stream):
            return False

    class _BlockingInput:
        generation = 0

        def read(self, frames):
            read_entered.set()
            assert release_read.wait(2.0)
            return np.ones(frames, dtype="float32"), False

    engine = _engine(_Binder())
    engine._recognizer = _Recognizer()
    engine._capture_sr = engine.config.sample_rate
    engine._stream_in = _BlockingInput()
    engine._poll_keywords = lambda _samples, **_kwargs: commands.append("stop")
    engine._running.set()
    worker = threading.Thread(target=engine._capture_loop)
    worker.start()
    assert read_entered.wait(1.0)

    engine._fail_virtual_route("proof lost during read")
    release_read.set()
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert commands == []
    assert engine._capture_stopping.is_set()
    assert engine.virtual_route_failure == "proof lost during read"


def test_private_capture_without_recognizer_fails_parent_instead_of_hanging():
    engine = _engine(_Binder())
    engine._recognizer = None
    engine._running.set()

    engine._capture_loop()

    assert engine._running.is_set() is False
    assert engine.virtual_route_failure == "virtual capture loop stopped unexpectedly"


def test_restart_refuses_retained_virtual_route_monitor_before_audio_open():
    engine = _engine(_Binder())
    engine._virtual_route_thread = types.SimpleNamespace(is_alive=lambda: True)

    with pytest.raises(RuntimeError, match="virtual-route monitor"):
        engine.start(EngineCallbacks())
