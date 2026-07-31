"""Headless LiveKit Agents adapter tests; no SDK, server, model, or device."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import threading
from types import SimpleNamespace

import pytest

from always_on_agent.acoustic import AcousticSource
from always_on_agent.origin import is_action_allowed
from core.engine import (
    EngineCallbacks,
    OwnerVerification,
    PlaybackOutcome,
    TrackedSpeech,
    TranscriptAbortReason,
)
from core.engines.livekit_agents import (
    LiveKitAgentsEngine,
    PublisherSessionPolicy,
)


class _Emitter:
    def __init__(self) -> None:
        self.handlers = defaultdict(list)
        self.calls: list[tuple[str, str, int]] = []

    def on(self, event, callback):
        self.calls.append(("on", event, threading.get_ident()))
        self.handlers[event].append(callback)

    def off(self, event, callback):
        self.calls.append(("off", event, threading.get_ident()))
        try:
            self.handlers[event].remove(callback)
        except ValueError:
            pass

    def emit(self, event, payload=None):
        for callback in tuple(self.handlers[event]):
            callback(payload)


class _Handle:
    def __init__(self) -> None:
        self.interrupted = False
        self.error = None
        self.callbacks = []
        self.done = False
        self.interrupt_calls = []
        self.interrupt_error = None

    def exception(self):
        if not self.done:
            raise asyncio.InvalidStateError
        return self.error

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def interrupt(self, *, force=False):
        self.interrupt_calls.append((force, threading.get_ident()))
        if self.interrupt_error is not None:
            raise self.interrupt_error
        self.interrupted = True
        return self

    def finish(self, *, interrupted=False, error=None):
        if self.done:
            return
        self.done = True
        self.interrupted = interrupted
        self.error = error
        for callback in tuple(self.callbacks):
            callback(self)


@dataclass(frozen=True)
class _Lease:
    owner: object
    epoch: int


class _Session(_Emitter):
    def __init__(
        self,
        participant_identity="publisher-7",
        *,
        policy=None,
    ) -> None:
        super().__init__()
        self.participant_identity = participant_identity
        self.policy = policy or PublisherSessionPolicy(
            publisher_scoped=True,
            recording_disabled=True,
            llm_disabled=True,
            agent_tools_disabled=True,
            session_tools_disabled=True,
            mcp_disabled=True,
            remote_inference_disabled=True,
            preemptive_generation_disabled=True,
            output_leases=True,
        )
        self.audio_available = True
        self.audio_enabled = True
        self.output_epoch = 0
        self.lease_owner = object()
        self.say_calls = []
        self.handles = []
        self.close_requests = []

    def say_tracked(
        self,
        text,
        *,
        allow_interruptions,
        add_to_chat_ctx,
        on_done,
    ):
        if not self.audio_available or not self.audio_enabled:
            raise RuntimeError("audio output unavailable")
        self.say_calls.append(
            (
                text,
                allow_interruptions,
                add_to_chat_ctx,
                threading.get_ident(),
            )
        )
        handle = _Handle()
        handle.add_done_callback(on_done)
        self.handles.append(handle)
        return handle, _Lease(self.lease_owner, self.output_epoch)

    def output_lease_valid(self, lease):
        return bool(
            isinstance(lease, _Lease)
            and lease.owner is self.lease_owner
            and lease.epoch == self.output_epoch
            and self.audio_available
            and self.audio_enabled
        )

    def detach_audio(self):
        if self.audio_available:
            self.output_epoch += 1
            self.audio_available = False

    def set_audio_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled != self.audio_enabled:
            self.output_epoch += 1
            self.audio_enabled = enabled

    def replace_audio_tail(self):
        self.output_epoch += 1

    def close_on_playout_failure(self):
        self.close_requests.append(threading.get_ident())

class _LoopHarness:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.thread_id = self.call(lambda: threading.get_ident())

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, fn):
        async def invoke():
            return fn()

        return asyncio.run_coroutine_threadsafe(invoke(), self.loop).result(1.0)

    def flush(self) -> None:
        self.call(lambda: None)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(1.0)
        assert not self.thread.is_alive()
        self.loop.close()


@dataclass
class _TranscriptEvent:
    transcript: str
    is_final: bool
    item_id: str | None = None
    created_at: float = 0.0


@pytest.fixture
def harness():
    value = _LoopHarness()
    try:
        yield value
    finally:
        value.close()


def _engine(harness, session=None, *, clock=None):
    return LiveKitAgentsEngine(
        session or _Session(),
        participant_identity="publisher-7",
        loop=harness.loop,
        clock=clock or (lambda: 10.0),
    )


def test_requires_explicit_bounded_publisher_identity(harness):
    for identity in ("", "   ", "bad\nidentity", "x" * 129):
        with pytest.raises(ValueError):
            LiveKitAgentsEngine(
                _Session(identity),
                participant_identity=identity,
                loop=harness.loop,
            )
    with pytest.raises(ValueError, match="not bound"):
        LiveKitAgentsEngine(
            _Session("somebody-else"),
            participant_identity="publisher-7",
            loop=harness.loop,
        )


def test_session_operations_are_marshaled_to_job_loop(harness):
    session = _Session()
    engine = _engine(harness, session)
    engine.start(EngineCallbacks())
    engine.speak_tracked(
        TrackedSpeech("f-1", "hello"),
        on_terminal=lambda _receipt: None,
    )
    harness.flush()
    engine.stop_speaking()
    harness.flush()
    harness.call(lambda: session.handles[0].finish(interrupted=True))
    engine.stop()

    operation_threads = [thread_id for _op, _event, thread_id in session.calls]
    operation_threads += [call[3] for call in session.say_calls]
    operation_threads += [call[1] for call in session.handles[0].interrupt_calls]
    assert operation_threads
    assert set(operation_threads) == {harness.thread_id}


def test_interims_are_typed_but_segment_finals_wait_for_whole_turn(harness):
    session = _Session()
    ticks = iter((1.0, 2.0, 3.0))
    engine = _engine(harness, session, clock=lambda: next(ticks))
    partials, finals = [], []
    engine.start(
        EngineCallbacks(
            on_partial_result=partials.append,
            on_final_result=finals.append,
        )
    )

    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("find in my vault", is_final=False),
        )
    )
    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("find in my vault", is_final=True),
        )
    )
    assert [item.text for item in partials] == ["find in my vault"]
    assert finals == []

    harness.call(lambda: engine.commit_user_turn("find in my vault"))
    assert [item.text for item in finals] == ["find in my vault"]
    assert partials[0].revision == 0
    assert finals[0].revision == 1
    assert partials[0].acoustic.spans[0].key == finals[0].acoustic.spans[0].key
    assert finals[0].acoustic.spans[0].source is AcousticSource.REMOTE_TRACK
    assert "publisher-7" not in finals[0].acoustic.spans[0].stream_id
    assert finals[0].origin == "remote_audio"
    assert finals[0].owner_verification is OwnerVerification.UNKNOWN
    assert finals[0].acoustic.spans[0].speech_end_at is None
    assert not is_action_allowed(finals[0].origin, owner_verified=True)


def test_whole_turn_confirms_one_barge_before_final_while_playing(harness):
    session = _Session()
    events = []
    engine = _engine(harness, session)
    engine.start(
        EngineCallbacks(
            on_barge_in_result=lambda result: events.append(("barge", result)),
            on_final_result=lambda result: events.append(("final", result)),
        )
    )
    engine.speak_tracked(
        TrackedSpeech("active", "assistant answer"),
        on_terminal=lambda _receipt: None,
    )
    harness.flush()

    # VAD onset is provisional. A later STT segment final confirms it once, but
    # is still not the complete turn.
    harness.call(
        lambda: session.emit(
            "agent_state_changed",
            SimpleNamespace(new_state="speaking"),
        )
    )
    harness.call(
        lambda: session.emit(
            "user_state_changed",
            SimpleNamespace(new_state="speaking"),
        )
    )
    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("wait a moment", is_final=True),
        )
    )
    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("wait a moment please", is_final=True),
        )
    )
    harness.call(lambda: engine.commit_user_turn("wait a moment please"))

    assert [kind for kind, _item in events] == ["barge", "final"]
    assert events[0][1].revision == 0
    assert events[1][1].revision == 1


def test_completed_turn_id_is_bounded_and_idempotent(harness):
    session = _Session()
    engine = _engine(harness, session)
    finals = []
    engine.start(EngineCallbacks(on_final_result=finals.append))

    assert harness.call(
        lambda: engine.commit_user_turn("one turn", turn_id="item-1")
    )
    assert not harness.call(
        lambda: engine.commit_user_turn("one turn", turn_id="item-1")
    )
    assert [item.text for item in finals] == ["one turn"]
    with pytest.raises(ValueError):
        harness.call(lambda: engine.commit_user_turn("bad", turn_id="bad\nitem"))


def test_tracked_playback_completion_attests_full_text(harness):
    session = _Session()
    engine = _engine(harness, session)
    started, receipts = [], []
    engine.start(EngineCallbacks())
    caps = engine.playback_capabilities
    assert caps.tracked_terminal and not caps.exact_started
    assert not caps.sample_counts and not caps.speech_style_hints

    engine.speak_tracked(
        TrackedSpeech("f-1", "complete answer"),
        on_started=started.append,
        on_terminal=receipts.append,
    )
    harness.flush()
    harness.call(lambda: session.handles[0].finish())

    assert started == []
    assert len(receipts) == 1
    assert receipts[0].outcome is PlaybackOutcome.COMPLETED
    assert receipts[0].safe_text_prefix == "complete answer"
    assert receipts[0].played_samples is None


@pytest.mark.parametrize(
    ("interrupted", "error", "expected"),
    [
        (True, None, PlaybackOutcome.INTERRUPTED),
        (False, RuntimeError("tts"), PlaybackOutcome.FAILED),
    ],
)
def test_tracked_playback_noncompletion_attests_no_text(
    harness, interrupted, error, expected
):
    session = _Session()
    engine = _engine(harness, session)
    receipts = []
    engine.start(EngineCallbacks())
    engine.speak_tracked(
        TrackedSpeech("f", "must not be remembered"),
        on_terminal=receipts.append,
    )
    harness.flush()
    harness.call(
        lambda: session.emit(
            "agent_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(
        lambda: session.handles[0].finish(
            interrupted=interrupted,
            error=error,
        )
    )
    assert len(receipts) == 1
    assert receipts[0].outcome is expected
    assert receipts[0].safe_text_prefix == ""


def test_stop_waits_for_old_handles_and_cannot_cut_new_generation(harness):
    session = _Session()
    engine = _engine(harness, session)
    receipts = []
    engine.start(EngineCallbacks())
    for fragment in ("first", "second"):
        engine.speak_tracked(
            TrackedSpeech(fragment, fragment),
            on_terminal=receipts.append,
        )
    harness.flush()

    engine.stop_speaking()
    harness.flush()
    assert receipts == []  # receipt waits for actual handle terminality
    assert [h.interrupt_calls[-1][0] for h in session.handles] == [True, True]

    engine.speak_tracked(
        TrackedSpeech("new", "new"),
        on_terminal=receipts.append,
    )
    harness.flush()
    assert len(session.handles) == 3
    assert session.handles[2].interrupt_calls == []

    harness.call(lambda: session.handles[0].finish(interrupted=True))
    harness.call(lambda: session.handles[1].finish(interrupted=True))
    harness.call(lambda: session.handles[2].finish())
    assert [(r.fragment_id, r.outcome) for r in receipts] == [
        ("first", PlaybackOutcome.DROPPED),
        ("second", PlaybackOutcome.DROPPED),
        ("new", PlaybackOutcome.COMPLETED),
    ]

    # Duplicate/late SDK callbacks cannot manufacture another receipt.
    harness.call(lambda: session.handles[0].finish(interrupted=True))
    assert len(receipts) == 3


def test_stop_unbinds_input_and_terminals_future_admission(harness):
    session = _Session()
    engine = _engine(harness, session)
    partials, receipts, aborts = [], [], []
    engine.start(
        EngineCallbacks(
            on_partial_result=partials.append,
            on_transcript_abort=aborts.append,
        )
    )
    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("unfinished", is_final=False),
        )
    )
    engine.abort_user_turn(TranscriptAbortReason.SHUTDOWN)
    assert len(aborts) == 1
    assert aborts[0].reason is TranscriptAbortReason.SHUTDOWN

    engine.stop()
    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("late", is_final=False),
        )
    )
    engine.speak_tracked(
        TrackedSpeech("late", "late"),
        on_terminal=receipts.append,
    )
    assert [partial.text for partial in partials] == ["unfinished"]
    assert len(receipts) == 1
    assert receipts[0].outcome is PlaybackOutcome.DROPPED


@pytest.mark.parametrize("missing", ["sink", "enabled"])
def test_text_only_or_disabled_output_cannot_attest_audio(harness, missing):
    session = _Session()
    if missing == "sink":
        session.detach_audio()
    else:
        session.set_audio_enabled(False)
    engine = _engine(harness, session)
    receipts = []
    engine.start(EngineCallbacks())
    engine.speak_tracked(
        TrackedSpeech("no-audio", "not heard"),
        on_terminal=receipts.append,
    )
    harness.flush()
    assert session.say_calls == []
    assert receipts[0].outcome is PlaybackOutcome.FAILED
    assert receipts[0].safe_text_prefix == ""


@pytest.mark.parametrize("mutation", ["tail", "disable_reenable", "detach"])
def test_any_output_route_epoch_change_fails_closed_at_handle_terminal(
    harness, mutation
):
    session = _Session()
    engine = _engine(harness, session)
    receipts = []
    engine.start(EngineCallbacks())
    engine.speak_tracked(
        TrackedSpeech("swap", "uncertain"),
        on_terminal=receipts.append,
    )
    harness.flush()
    if mutation == "tail":
        session.replace_audio_tail()
    elif mutation == "disable_reenable":
        session.set_audio_enabled(False)
        session.set_audio_enabled(True)
    else:
        session.detach_audio()
    harness.call(lambda: session.handles[0].finish())
    assert receipts[0].outcome is PlaybackOutcome.FAILED
    assert receipts[0].safe_text_prefix == ""


def test_false_interruption_never_cancels_controller(harness):
    session = _Session()
    engine = _engine(harness, session)
    events = []
    engine.start(
        EngineCallbacks(
            on_barge_in_result=lambda result: events.append(("barge", result)),
            on_final_result=lambda result: events.append(("final", result)),
        )
    )
    engine.speak_tracked(
        TrackedSpeech("active", "answer"),
        on_terminal=lambda _receipt: None,
    )
    harness.flush()
    harness.call(
        lambda: session.emit(
            "agent_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(
        lambda: session.emit(
            "user_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(
        lambda: session.emit(
            "agent_false_interruption", SimpleNamespace(resumed=True)
        )
    )
    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("okay", is_final=True),
        )
    )
    harness.call(lambda: engine.commit_user_turn("okay", turn_id="false-1"))
    assert [kind for kind, _item in events] == ["final"]


def test_sdk_cut_before_transcript_retains_confirmed_barge_candidate(harness):
    session = _Session()
    engine = _engine(harness, session)
    events = []
    engine.start(
        EngineCallbacks(
            on_barge_in_result=lambda result: events.append(("barge", result)),
            on_final_result=lambda result: events.append(("final", result)),
        )
    )
    engine.speak_tracked(
        TrackedSpeech("active", "answer"),
        on_terminal=lambda _receipt: None,
    )
    harness.flush()
    harness.call(
        lambda: session.emit(
            "agent_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(lambda: session.handles[0].finish(interrupted=True))
    assert not engine.is_speaking
    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("new question", is_final=True),
        )
    )
    harness.call(
        lambda: engine.commit_user_turn("new question", turn_id="new-1")
    )
    assert [kind for kind, _item in events] == ["barge", "final"]


def test_queued_output_that_never_started_is_not_a_barge(harness):
    session = _Session()
    engine = _engine(harness, session)
    events = []
    engine.start(
        EngineCallbacks(
            on_barge_in_result=lambda result: events.append(("barge", result)),
            on_final_result=lambda result: events.append(("final", result)),
        )
    )
    engine.speak_tracked(
        TrackedSpeech("queued", "not yet active"),
        on_terminal=lambda _receipt: None,
    )
    harness.flush()
    harness.call(
        lambda: session.emit(
            "user_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("replacement", is_final=True),
        )
    )
    harness.call(
        lambda: engine.commit_user_turn("replacement", turn_id="queued-1")
    )
    assert [kind for kind, _item in events] == ["final"]


def test_old_turn_committer_and_event_handler_cannot_cross_restart(harness):
    session = _Session()
    engine = _engine(harness, session)
    old_finals, new_finals, partials = [], [], []
    engine.start(EngineCallbacks(on_final_result=old_finals.append))
    old_commit = engine.user_turn_committer()
    old_handler = session.handlers["user_input_transcribed"][0]
    engine.stop()
    engine.start(
        EngineCallbacks(
            on_final_result=new_finals.append,
            on_partial_result=partials.append,
        )
    )

    assert not harness.call(
        lambda: old_commit("stale", turn_id="old-item")
    )
    harness.call(
        lambda: old_handler(_TranscriptEvent("stale partial", is_final=False))
    )
    assert old_finals == []
    assert new_finals == []
    assert partials == []


def test_rejects_self_reported_unsafe_session_policy(harness):
    unsafe = PublisherSessionPolicy(
        publisher_scoped=True,
        recording_disabled=False,
        llm_disabled=True,
        agent_tools_disabled=True,
        session_tools_disabled=True,
        mcp_disabled=True,
        remote_inference_disabled=True,
        preemptive_generation_disabled=True,
        output_leases=True,
    )
    with pytest.raises(ValueError, match="safe publisher policy"):
        _engine(harness, _Session(policy=unsafe))


def test_user_state_end_is_not_assistant_speech_end(harness):
    session = _Session()
    lifecycle = []
    engine = _engine(harness, session)
    engine.start(
        EngineCallbacks(
            on_speech_start=lambda: lifecycle.append("assistant-start"),
            on_speech_end=lambda: lifecycle.append("assistant-end"),
        )
    )
    harness.call(
        lambda: session.emit(
            "user_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(
        lambda: session.emit(
            "user_state_changed", SimpleNamespace(new_state="listening")
        )
    )
    assert lifecycle == []

    harness.call(
        lambda: session.emit(
            "agent_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(
        lambda: session.emit(
            "agent_state_changed", SimpleNamespace(new_state="listening")
        )
    )
    assert lifecycle == ["assistant-start", "assistant-end"]


def test_interrupt_failure_retains_ownership_until_post_drain_close(harness):
    session = _Session()
    receipts, health, lifecycle = [], [], []
    engine = _engine(harness, session)
    engine.start(
        EngineCallbacks(
            on_capture_state=lambda state, detail: health.append((state, detail)),
            on_speech_start=lambda: lifecycle.append("start"),
            on_speech_end=lambda: lifecycle.append("end"),
        )
    )
    engine.speak_tracked(
        TrackedSpeech("active", "still audible"),
        on_terminal=receipts.append,
    )
    harness.flush()
    harness.call(
        lambda: session.emit(
            "agent_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    session.handles[0].interrupt_error = RuntimeError("request failed")

    engine.stop_speaking()
    harness.flush()
    assert receipts == []
    assert engine.is_speaking
    assert session.close_requests == [harness.thread_id]
    assert lifecycle == ["start"]
    rejected = []
    engine.speak_tracked(
        TrackedSpeech("new", "must not overlap"),
        on_terminal=rejected.append,
    )
    assert rejected[0].outcome is PlaybackOutcome.DROPPED

    session.detach_audio()
    harness.call(
        lambda: session.emit(
            "close",
            SimpleNamespace(reason=SimpleNamespace(value="error")),
        )
    )
    assert [(item.outcome, item.safe_text_prefix) for item in receipts] == [
        (PlaybackOutcome.INTERRUPTED, "")
    ]
    assert health == [
        ("recovering", "livekit_playout_close_requested"),
        ("fatal", "livekit_session_closed:error"),
    ]
    assert lifecycle == ["start", "end"]
    assert not engine.is_speaking


def test_session_close_aborts_open_turn_and_blocks_late_events(harness):
    session = _Session()
    partials, finals, aborts = [], [], []
    engine = _engine(harness, session)
    engine.start(
        EngineCallbacks(
            on_partial_result=partials.append,
            on_final_result=finals.append,
            on_transcript_abort=aborts.append,
        )
    )
    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("open turn", is_final=False),
        )
    )
    session.detach_audio()
    harness.call(
        lambda: session.emit(
            "close",
            SimpleNamespace(
                reason=SimpleNamespace(value="participant_disconnected")
            ),
        )
    )
    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("late", is_final=False),
        )
    )
    assert [item.text for item in partials] == ["open turn"]
    assert finals == []
    assert len(aborts) == 1
    assert aborts[0].reason is TranscriptAbortReason.SHUTDOWN
    with pytest.raises(RuntimeError, match="closed"):
        engine.start(EngineCallbacks())


def test_clean_output_end_clears_unconfirmed_old_barge_episode(harness):
    session = _Session()
    events = []
    engine = _engine(harness, session)
    engine.start(
        EngineCallbacks(
            on_barge_in_result=lambda item: events.append(("barge", item)),
            on_final_result=lambda item: events.append(("final", item)),
        )
    )
    engine.speak_tracked(
        TrackedSpeech("active", "answer"),
        on_terminal=lambda _receipt: None,
    )
    harness.flush()
    harness.call(
        lambda: session.emit(
            "agent_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(
        lambda: session.emit(
            "user_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(lambda: session.handles[0].finish())
    harness.call(
        lambda: session.emit(
            "user_state_changed", SimpleNamespace(new_state="listening")
        )
    )
    harness.call(
        lambda: session.emit(
            "user_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("later turn", is_final=True),
        )
    )
    harness.call(lambda: engine.commit_user_turn("later turn", turn_id="later"))
    assert [kind for kind, _item in events] == ["final"]


def test_stale_partial_cannot_advance_new_lifecycle_turn(harness):
    entered, release = threading.Event(), threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def clock():
        nonlocal calls
        with calls_lock:
            calls += 1
            ordinal = calls
        if ordinal == 1:
            entered.set()
            assert release.wait(1.0)
        return float(ordinal)

    session = _Session()
    engine = _engine(harness, session, clock=clock)
    old_partials, new_partials = [], []
    engine.start(EngineCallbacks(on_partial_result=old_partials.append))
    old_handler = session.handlers["user_input_transcribed"][0]
    worker = threading.Thread(
        target=lambda: old_handler(_TranscriptEvent("stale", is_final=False))
    )
    worker.start()
    assert entered.wait(1.0)
    engine.stop()
    engine.start(EngineCallbacks(on_partial_result=new_partials.append))
    release.set()
    worker.join(1.0)
    assert not worker.is_alive()

    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("current", is_final=False),
        )
    )
    assert old_partials == []
    assert len(new_partials) == 1
    assert new_partials[0].revision == 0
    assert new_partials[0].acoustic.spans[0].turn_index == 1


def test_stale_committer_cannot_close_new_lifecycle_turn(harness):
    entered, release = threading.Event(), threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def clock():
        nonlocal calls
        with calls_lock:
            calls += 1
            ordinal = calls
        if ordinal == 1:
            entered.set()
            assert release.wait(1.0)
        return float(ordinal)

    session = _Session()
    engine = _engine(harness, session, clock=clock)
    engine.start(EngineCallbacks())
    old_commit = engine.user_turn_committer()
    result = []
    worker = threading.Thread(
        target=lambda: result.append(old_commit("stale", turn_id="old"))
    )
    worker.start()
    assert entered.wait(1.0)
    engine.stop()
    partials = []
    engine.start(EngineCallbacks(on_partial_result=partials.append))
    release.set()
    worker.join(1.0)
    assert result == [False]

    harness.call(
        lambda: session.emit(
            "user_input_transcribed",
            _TranscriptEvent("current", is_final=False),
        )
    )
    assert len(partials) == 1
    assert partials[0].revision == 0
    assert partials[0].acoustic.spans[0].turn_index == 1


def test_playback_generation_bump_invalidates_blocked_barge(harness):
    entered, release = threading.Event(), threading.Event()

    def clock():
        entered.set()
        assert release.wait(1.0)
        return 1.0

    session = _Session()
    engine = _engine(harness, session, clock=clock)
    events = []
    engine.start(
        EngineCallbacks(
            on_barge_in_result=lambda item: events.append(("barge", item)),
            on_final_result=lambda item: events.append(("final", item)),
        )
    )
    engine.speak_tracked(
        TrackedSpeech("active", "answer"),
        on_terminal=lambda _receipt: None,
    )
    harness.flush()
    harness.call(
        lambda: session.emit(
            "agent_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(
        lambda: session.emit(
            "user_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            engine.commit_user_turn("replacement", turn_id="replacement")
        )
    )
    worker.start()
    assert entered.wait(1.0)
    engine.stop_speaking()
    release.set()
    worker.join(1.0)
    assert result == [True]
    assert [kind for kind, _item in events] == ["final"]


def test_late_stopped_handle_callbacks_cannot_rearm_barge(harness):
    session = _Session()
    events = []
    engine = _engine(harness, session)
    engine.start(
        EngineCallbacks(
            on_barge_in_result=lambda item: events.append(("barge", item)),
            on_final_result=lambda item: events.append(("final", item)),
        )
    )
    engine.speak_tracked(
        TrackedSpeech("old", "old"), on_terminal=lambda _receipt: None
    )
    harness.flush()
    engine.stop_speaking()
    harness.flush()
    harness.call(
        lambda: session.emit(
            "agent_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(
        lambda: session.emit(
            "user_state_changed", SimpleNamespace(new_state="speaking")
        )
    )
    harness.call(lambda: session.handles[0].finish(interrupted=True))
    harness.call(lambda: engine.commit_user_turn("later", turn_id="later"))
    assert [kind for kind, _item in events] == ["final"]
