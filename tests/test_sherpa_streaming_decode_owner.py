from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread, current_thread

import numpy as np
import pytest

from core.engines._sherpa_streaming_decode import SherpaStreamingDecodeSession
from core.engines import _sherpa_streaming_decode_owner as owner_module
from core.engines._sherpa_streaming_decode_owner import (
    DecodeLossReason,
    DecodeOwnerState,
    DecodeStreamRole,
    SherpaStreamingDecodeOwner,
)
from core.realtime_media_stage import CaptureScope


@dataclass(frozen=True, slots=True)
class _Call:
    name: str
    thread: Thread


class _NativeStream:
    def __init__(self, recognizer: "_Recognizer") -> None:
        self._recognizer = recognizer

    def accept_waveform(self, _sample_rate: int, _samples: object) -> None:
        self._recognizer.calls.append(_Call("accept_waveform", current_thread()))


class _Recognizer:
    def __init__(self) -> None:
        self.calls: list[_Call] = []

    def _call(self, name: str) -> None:
        self.calls.append(_Call(name, current_thread()))

    def create_stream(self, **_kwargs) -> _NativeStream:
        self._call("create_stream")
        return _NativeStream(self)

    def is_ready(self, _stream: _NativeStream) -> bool:
        self._call("is_ready")
        return False

    def decode_stream(self, _stream: _NativeStream) -> None:
        self._call("decode_stream")

    def get_result(self, _stream: _NativeStream) -> str:
        self._call("get_result")
        return "private transcript"

    def is_endpoint(self, _stream: _NativeStream) -> bool:
        self._call("is_endpoint")
        return False

    def reset(self, _stream: _NativeStream) -> None:
        self._call("reset")


def test_complete_owner_tracks_three_distinct_identities_on_one_thread() -> None:
    native = _Recognizer()
    session = SherpaStreamingDecodeSession(native)
    initial_scope = CaptureScope(capture_epoch=3, capture_generation=4)
    next_scope = CaptureScope(capture_epoch=3, capture_generation=5)
    ready = Event()
    release = Event()
    observations: dict[str, object] = {}

    def process(owner: SherpaStreamingDecodeOwner) -> None:
        primary = owner.create_stream(role=DecodeStreamRole.PRIMARY)
        primary_key = owner.stream_key(primary)
        primary.accept_waveform(16_000, np.ones(160, dtype="float32"))
        assert not owner.is_ready(primary)
        assert owner.get_result(primary) == "private transcript"
        assert not owner.is_endpoint(primary)
        owner.reset(primary)
        reset_key = owner.stream_key(primary)
        word_cut = owner.create_stream(role=DecodeStreamRole.WORD_CUT)
        word_cut_key = owner.stream_key(word_cut)

        decode_identity = owner.rotate_decode_continuity(
            DecodeLossReason.NATIVE_ERROR
        )
        replacement = owner.create_stream(role=DecodeStreamRole.PRIMARY)
        replacement_key = owner.stream_key(replacement)
        capture_identity = owner.rotate_capture_scope(next_scope)
        final_primary = owner.create_stream(role=DecodeStreamRole.PRIMARY)

        observations.update(
            primary_key=primary_key,
            reset_key=reset_key,
            word_cut_key=word_cut_key,
            decode_identity=decode_identity,
            replacement_key=replacement_key,
            capture_identity=capture_identity,
            final_key=owner.stream_key(final_primary),
        )
        owner.publish_processor_ready()
        ready.set()
        assert release.wait(2.0)

    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=9,
        capture_scope=initial_scope,
        processor=process,
    )
    owner.start()
    assert ready.wait(1.0)

    snapshot = owner.snapshot()
    assert snapshot.state is DecodeOwnerState.RUNNING
    assert snapshot.identity.capture_scope == next_scope
    assert snapshot.identity.continuity_generation == 0
    assert snapshot.primary_stream == observations["final_key"]
    assert snapshot.word_cut_stream is None
    assert observations["primary_key"] != observations["reset_key"]
    assert observations["word_cut_key"].role is DecodeStreamRole.WORD_CUT
    assert observations["decode_identity"].capture_scope == initial_scope
    assert observations["decode_identity"].continuity_generation == 1
    assert observations["capture_identity"].capture_scope == next_scope
    assert observations["capture_identity"].continuity_generation == 0
    assert not owner.close(timeout=0.0)

    release.set()
    assert owner.close(timeout=1.0)
    terminal = owner.snapshot()
    assert terminal.state is DecodeOwnerState.CLOSED
    assert terminal.session_closed
    assert terminal.streams_created == 4
    assert terminal.streams_reset == 1
    assert terminal.streams_retired == 4
    assert terminal.capture_rotations == 1
    assert terminal.decode_rotations == 1
    assert native.calls
    assert {call.thread.name for call in native.calls} == {
        "speaker-streaming-decode-owner"
    }
    assert "private transcript" not in repr(terminal)


def test_owner_prestart_close_is_construction_thread_only_and_exact() -> None:
    native = _Recognizer()
    session = SherpaStreamingDecodeSession(native)
    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=1,
        capture_scope=CaptureScope(capture_epoch=1, capture_generation=1),
        processor=lambda _owner: None,
    )

    assert owner.close(timeout=0.0)
    assert session.closed
    assert owner.snapshot().state is DecodeOwnerState.CLOSED


def test_zero_timeout_close_never_demotes_a_terminal_owner() -> None:
    native = _Recognizer()
    session = SherpaStreamingDecodeSession(native)
    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=1,
        capture_scope=CaptureScope(capture_epoch=1, capture_generation=1),
        processor=lambda _owner: None,
    )

    assert owner.close(timeout=0.0)
    # Force the formerly racy observation deterministically: CLOSED was once
    # published immediately before terminal. close() must keep terminal states
    # monotonic even if that impossible-after-fix snapshot is recreated.
    owner._terminal.clear()
    try:
        assert not owner.close(timeout=0.0)
        assert owner.snapshot().state is DecodeOwnerState.CLOSED
    finally:
        owner._terminal.set()


def test_startup_timeout_never_demotes_an_owner_that_finished_cleanly() -> None:
    native = _Recognizer()
    session = SherpaStreamingDecodeSession(native)
    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=1,
        capture_scope=CaptureScope(capture_epoch=1, capture_generation=1),
        processor=lambda _owner: None,
    )

    class _FalseAfterTerminal:
        def set(self) -> None:
            return None

        def wait(self, _timeout: float) -> bool:
            assert owner._terminal.wait(1.0)
            return False

    owner._startup_complete = _FalseAfterTerminal()

    with pytest.raises(TimeoutError, match="did not publish readiness"):
        owner.start(timeout=1.0)

    snapshot = owner.snapshot()
    assert snapshot.state is DecodeOwnerState.CLOSED
    assert snapshot.terminal
    assert snapshot.session_closed
    with owner_module._LIVE_OWNER_LOCK:
        assert owner_module._LIVE_OWNERS.get(id(owner)) is not owner


def test_registry_updates_are_published_while_owner_state_is_locked(
    monkeypatch,
) -> None:
    native = _Recognizer()
    session = SherpaStreamingDecodeSession(native)
    ready = Event()
    release = Event()

    def process(owner: SherpaStreamingDecodeOwner) -> None:
        owner.publish_processor_ready()
        ready.set()
        assert release.wait(2.0)

    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=1,
        capture_scope=CaptureScope(capture_epoch=1, capture_generation=1),
        processor=process,
    )
    retain_locked: list[bool] = []
    release_locked: list[bool] = []
    real_retain = owner_module._retain_live_owner
    real_release = owner_module._release_closed_owner

    def observe_retain(value: object) -> None:
        acquired = owner._lock.acquire(blocking=False)
        retain_locked.append(not acquired)
        if acquired:
            owner._lock.release()
        real_retain(value)

    def observe_release(value: object) -> None:
        acquired = owner._lock.acquire(blocking=False)
        release_locked.append(not acquired)
        if acquired:
            owner._lock.release()
        real_release(value)

    monkeypatch.setattr(owner_module, "_retain_live_owner", observe_retain)
    monkeypatch.setattr(owner_module, "_release_closed_owner", observe_release)

    owner.start(timeout=1.0)
    assert ready.wait(1.0)
    assert not owner.close(timeout=0.0)
    release.set()
    assert owner.close(timeout=1.0)

    assert retain_locked == [True]
    assert release_locked == [True]
    with owner_module._LIVE_OWNER_LOCK:
        assert owner_module._LIVE_OWNERS.get(id(owner)) is not owner


def test_prestart_close_allows_native_destructor_to_reenter_snapshot() -> None:
    observed_states = []
    outcomes = []

    def exercise() -> None:
        holder = {}

        class _ReentrantRecognizer:
            def __del__(self) -> None:
                observed_states.append(holder["owner"].snapshot().state)

        session = SherpaStreamingDecodeSession(_ReentrantRecognizer())
        owner = SherpaStreamingDecodeOwner(
            session,
            run_id=1,
            capture_scope=CaptureScope(capture_epoch=1, capture_generation=1),
            processor=lambda _owner: None,
        )
        holder["owner"] = owner
        outcomes.append(owner.close(timeout=0.0))

    thread = Thread(target=exercise, daemon=True)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive(), "pre-start native cleanup deadlocked"
    assert outcomes == [True]
    assert observed_states == [DecodeOwnerState.CLOSING]


def test_capture_scope_cannot_be_reused_within_run() -> None:
    native = _Recognizer()
    session = SherpaStreamingDecodeSession(native)
    first = CaptureScope(capture_epoch=1, capture_generation=1)
    second = CaptureScope(capture_epoch=1, capture_generation=2)
    ready = Event()
    release = Event()
    identities = []

    def process(owner: SherpaStreamingDecodeOwner) -> None:
        identities.append(owner.identity)
        identities.append(
            owner.rotate_decode_continuity(
                DecodeLossReason.EXPLICIT_DISCONTINUITY
            )
        )
        identities.append(owner.rotate_capture_scope(second))
        with pytest.raises(ValueError, match="cannot be reused"):
            owner.rotate_capture_scope(first)
        owner.publish_processor_ready()
        ready.set()
        assert release.wait(2.0)

    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=7,
        capture_scope=first,
        processor=process,
    )
    owner.start()
    assert ready.wait(1.0)
    release.set()
    assert owner.close(timeout=1.0)

    assert [identity.continuity_generation for identity in identities] == [
        0,
        1,
        0,
    ]
    assert identities[0].capture_scope != identities[-1].capture_scope
    snapshot = owner.snapshot()
    assert snapshot.capture_rotations == 1
    assert snapshot.streams_created == 0


def test_owner_start_rejects_processor_exit_before_readiness() -> None:
    native = _Recognizer()
    session = SherpaStreamingDecodeSession(native)
    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=2,
        capture_scope=CaptureScope(capture_epoch=1, capture_generation=1),
        processor=lambda _owner: None,
    )

    with pytest.raises(RuntimeError, match="ProcessorExited"):
        owner.start(timeout=1.0)

    assert owner.close(timeout=1.0)
    snapshot = owner.snapshot()
    assert snapshot.state is DecodeOwnerState.CLOSED
    assert not snapshot.ready
    assert snapshot.terminal


def test_owner_start_rejects_ready_processor_already_in_native_teardown() -> None:
    allow_processor_return = Event()
    teardown_entered = Event()
    release_teardown = Event()

    class _BlockingDestructorRecognizer:
        def __del__(self) -> None:
            teardown_entered.set()
            assert release_teardown.wait(2.0)

    session = SherpaStreamingDecodeSession(_BlockingDestructorRecognizer())

    def process(owner: SherpaStreamingDecodeOwner) -> None:
        owner.publish_processor_ready()
        assert allow_processor_return.wait(2.0)

    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=2,
        capture_scope=CaptureScope(capture_epoch=1, capture_generation=1),
        processor=process,
    )
    real_snapshot = owner.snapshot

    def snapshot_after_processor_exit():
        allow_processor_return.set()
        assert teardown_entered.wait(1.0)
        return real_snapshot()

    owner.snapshot = snapshot_after_processor_exit
    try:
        with pytest.raises(RuntimeError, match="ProcessorExited"):
            owner.start(timeout=1.0)
        closing = real_snapshot()
        assert closing.state is DecodeOwnerState.CLOSING
        assert closing.ready
        assert closing.session_closed
    finally:
        release_teardown.set()

    assert owner.close(timeout=1.0)
    assert real_snapshot().state is DecodeOwnerState.CLOSED


def test_duplicate_live_role_is_rejected_and_scope_fences_before_retire() -> None:
    native = _Recognizer()
    session = SherpaStreamingDecodeSession(native)
    first = CaptureScope(capture_epoch=4, capture_generation=1)
    second = CaptureScope(capture_epoch=4, capture_generation=2)
    ready = Event()
    release = Event()
    identity_seen_at_retire = []

    def process(owner: SherpaStreamingDecodeOwner) -> None:
        owner.create_stream(role=DecodeStreamRole.PRIMARY)
        with pytest.raises(RuntimeError, match="already live"):
            owner.create_stream(role=DecodeStreamRole.PRIMARY)
        retire = owner._retire_all_streams

        def observe_then_retire() -> None:
            identity_seen_at_retire.append(owner.identity)
            retire()

        owner._retire_all_streams = observe_then_retire
        rotated = owner.rotate_capture_scope(second)
        assert rotated == identity_seen_at_retire[0]
        owner.publish_processor_ready()
        ready.set()
        assert release.wait(2.0)

    owner = SherpaStreamingDecodeOwner(
        session,
        run_id=3,
        capture_scope=first,
        processor=process,
    )
    owner.start()
    assert ready.wait(1.0)
    assert identity_seen_at_retire[0].capture_scope == second
    release.set()
    assert owner.close(timeout=1.0)
    snapshot = owner.snapshot()
    assert snapshot.streams_created == 1
    assert snapshot.streams_retired == 1
