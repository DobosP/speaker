from __future__ import annotations

from dataclasses import dataclass, field
import gc
import sys
from threading import Barrier, Event, Lock, Thread, current_thread
import time
from types import SimpleNamespace
from typing import Callable
import weakref

import numpy as np
import pytest

import core.engines._recovering_input as recovering_input_module
import core.engines._sherpa_streaming_decode_owner as decode_owner_module
import core.engines.sherpa as sherpa_module
from core.engine import (
    EngineCallbacks,
    PlaybackOutcome,
    PlaybackReceipt,
    TrackedSpeech,
)
from core.engines._sherpa_streaming_decode import (
    SherpaStreamingDecodeClosedError,
    SherpaStreamingDecodeOwnerError,
    SherpaStreamingDecodeReentryError,
    SherpaStreamingDecodeSession,
    SherpaStreamingDecodeStream,
    SherpaStreamingDecodeStreamError,
)
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


_WAIT_SECONDS = 5.0
_HOTWORDS_OMITTED = object()


@dataclass(frozen=True, slots=True)
class _NativeCall:
    name: str
    thread: Thread
    stream: object | None = field(repr=False)
    detail: object = field(repr=False)


class _NativeStream:
    def __init__(self, recognizer: _Recognizer) -> None:
        self._recognizer = recognizer

    def accept_waveform(self, sample_rate: int, samples: object) -> str:
        return self._recognizer._invoke(
            "accept_waveform",
            self,
            detail=(sample_rate, samples),
            result="accepted",
        )

    def __del__(self) -> None:
        self._recognizer.destroyed_on.append(current_thread())

    def __repr__(self) -> str:
        raise AssertionError("native stream repr must never be called")


class _Recognizer:
    def __init__(self) -> None:
        self.calls: list[_NativeCall] = []
        self.destroyed_on: list[Thread] = []
        self._state_lock = Lock()
        self.active = 0
        self.max_active = 0
        self.fail_next: set[str] = set()
        self.decode_entered: Event | None = None
        self.decode_release: Event | None = None
        self.during_decode: Callable[[], None] | None = None
        self.result_value: object = "heard words"
        self.last_stream_ref: weakref.ReferenceType[_NativeStream] | None = None

    def create_stream(self, **kwargs: object) -> _NativeStream:
        detail = kwargs.get("hotwords", _HOTWORDS_OMITTED)

        def create() -> _NativeStream:
            stream = _NativeStream(self)
            self.last_stream_ref = weakref.ref(stream)
            return stream

        return self._invoke(
            "create_stream",
            None,
            detail=detail,
            action=create,
        )

    def is_ready(self, stream: _NativeStream) -> bool:
        return self._invoke("is_ready", stream, result=True)

    def decode_stream(self, stream: _NativeStream) -> str:
        def decode() -> str:
            if self.decode_entered is not None:
                self.decode_entered.set()
            if self.decode_release is not None:
                assert self.decode_release.wait(_WAIT_SECONDS)
            if self.during_decode is not None:
                self.during_decode()
            return "decoded"

        return self._invoke("decode_stream", stream, action=decode)

    def get_result(self, stream: _NativeStream) -> str:
        return self._invoke("get_result", stream, result=self.result_value)

    def is_endpoint(self, stream: _NativeStream) -> bool:
        return self._invoke("is_endpoint", stream, result=False)

    def reset(self, stream: _NativeStream) -> str:
        return self._invoke("reset", stream, result="reset")

    def _invoke(
        self,
        name: str,
        stream: object | None,
        *,
        detail: object = None,
        result: object = None,
        action: Callable[[], object] | None = None,
    ):
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(
                _NativeCall(
                    name=name,
                    thread=current_thread(),
                    stream=stream,
                    detail=detail,
                )
            )
            should_fail = name in self.fail_next
            self.fail_next.discard(name)
        try:
            if should_fail:
                raise ValueError(f"synthetic {name} failure")
            return action() if action is not None else result
        finally:
            with self._state_lock:
                self.active -= 1

    def __repr__(self) -> str:
        raise AssertionError("native recognizer repr must never be called")


def _bound_session(
    recognizer: _Recognizer | None = None,
) -> tuple[_Recognizer, SherpaStreamingDecodeSession]:
    native = recognizer or _Recognizer()
    session = SherpaStreamingDecodeSession(native)
    session.bind_current_thread()
    return native, session


def _thread_outcome(target: Callable[[], object]) -> tuple[Thread, list[object]]:
    outcome: list[object] = []

    def run() -> None:
        try:
            outcome.append(target())
        except BaseException as exc:
            outcome.append(exc)

    thread = Thread(target=run)
    thread.start()
    return thread, outcome


def test_all_online_operations_use_the_exact_owner_and_native_stream() -> None:
    recognizer = _Recognizer()
    session = SherpaStreamingDecodeSession(recognizer)

    with pytest.raises(SherpaStreamingDecodeOwnerError, match="bind_current_thread"):
        session.create_stream()
    assert recognizer.calls == []

    owner = current_thread()
    session.bind_current_thread()
    session.bind_current_thread()
    stream = session.create_stream(hotwords="search vault\nfind note")
    samples = object()

    assert stream.accept_waveform(16_000, samples) is None
    assert session.is_ready(stream) is True
    assert session.decode_stream(stream) is None
    assert session.get_result(stream) == "heard words"
    assert session.is_endpoint(stream) is False
    assert session.reset(stream) is None

    assert [call.name for call in recognizer.calls] == [
        "create_stream",
        "accept_waveform",
        "is_ready",
        "decode_stream",
        "get_result",
        "is_endpoint",
        "reset",
    ]
    assert all(call.thread is owner for call in recognizer.calls)
    native_stream = recognizer.calls[1].stream
    assert native_stream is not None
    assert all(call.stream is native_stream for call in recognizer.calls[1:])
    assert recognizer.calls[0].detail == "search vault\nfind note"
    assert recognizer.calls[1].detail == (16_000, samples)
    assert recognizer.max_active == 1
    session.close()


def test_create_stream_preserves_plain_and_hotword_calls() -> None:
    recognizer, session = _bound_session()

    first = session.create_stream()
    second = session.create_stream(hotwords="open vault")

    assert first is not second
    assert recognizer.calls[0].detail is _HOTWORDS_OMITTED
    assert recognizer.calls[1].detail == "open vault"
    session.close()


def test_foreign_operations_fail_before_stream_or_native_state() -> None:
    recognizer, session = _bound_session()
    stream = session.create_stream()

    def foreign() -> tuple[BaseException, BaseException, BaseException]:
        errors: list[BaseException] = []
        for operation in (
            lambda: session.is_ready(stream),
            lambda: session.is_ready(object()),
            session.close,
        ):
            try:
                operation()
            except BaseException as exc:
                errors.append(exc)
        return errors[0], errors[1], errors[2]

    thread, outcome = _thread_outcome(foreign)
    thread.join(_WAIT_SECONDS)

    assert not thread.is_alive()
    assert len(outcome) == 1
    assert all(
        isinstance(error, SherpaStreamingDecodeOwnerError) for error in outcome[0]
    )
    assert [call.name for call in recognizer.calls] == ["create_stream"]
    assert session.get_result(stream) == "heard words"
    session.close()


def test_blocked_decode_rejects_foreign_contenders_without_overlap() -> None:
    recognizer = _Recognizer()
    recognizer.decode_entered = Event()
    recognizer.decode_release = Event()
    session = SherpaStreamingDecodeSession(recognizer)
    owner_ready = Barrier(2)
    holder: dict[str, object] = {}

    def owner() -> None:
        session.bind_current_thread()
        stream = session.create_stream()
        holder["stream"] = stream
        owner_ready.wait()
        holder["result"] = session.decode_stream(stream)
        session.close()

    owner_thread = Thread(target=owner)
    owner_thread.start()
    owner_ready.wait()
    assert recognizer.decode_entered.wait(_WAIT_SECONDS)
    stream = holder["stream"]

    operations = (
        lambda: session.is_ready(stream),
        lambda: session.decode_stream(stream),
        session.close,
    )
    contenders = [_thread_outcome(operation) for operation in operations]
    for thread, _ in contenders:
        thread.join(_WAIT_SECONDS)

    assert all(not thread.is_alive() for thread, _ in contenders)
    assert all(
        isinstance(outcome[0], SherpaStreamingDecodeOwnerError)
        for _, outcome in contenders
    )
    assert recognizer.active == 1
    assert recognizer.max_active == 1
    recognizer.decode_release.set()
    owner_thread.join(_WAIT_SECONDS)
    assert not owner_thread.is_alive()
    assert holder["result"] is None
    assert session.closed is True


def test_same_owner_reentry_is_rejected_without_poisoning() -> None:
    recognizer, session = _bound_session()
    stream = session.create_stream()
    errors: list[BaseException] = []

    def reenter() -> None:
        for operation in (lambda: session.get_result(stream), session.close):
            try:
                operation()
            except BaseException as exc:
                errors.append(exc)

    recognizer.during_decode = reenter
    assert session.decode_stream(stream) is None

    assert len(errors) == 2
    assert all(isinstance(error, SherpaStreamingDecodeReentryError) for error in errors)
    assert session.snapshot().native_call_active is False
    assert session.get_result(stream) == "heard words"
    session.close()


def test_native_failure_releases_lease_and_preserves_the_handle() -> None:
    recognizer, session = _bound_session()
    stream = session.create_stream()
    recognizer.fail_next.add("decode_stream")

    with pytest.raises(ValueError, match="synthetic decode_stream failure"):
        session.decode_stream(stream)

    snapshot = session.snapshot()
    assert snapshot.native_call_active is False
    assert snapshot.native_calls_failed == 1
    assert snapshot.live_streams == 1
    assert session.get_result(stream) == "heard words"
    session.close()


def test_result_normalization_is_inside_reentry_and_failure_fence() -> None:
    recognizer, session = _bound_session()
    stream = session.create_stream()
    error_types: list[type[BaseException]] = []

    class _ReenteringResult:
        def __str__(self) -> str:
            recognizer.result_value = None
            try:
                session.is_ready(stream)
            except BaseException as exc:
                # Retaining ``exc`` also retains this frame and ``self`` via its
                # traceback, postponing __del__ until after the native fence.
                error_types.append(type(exc))
            return "normalized"

        def __del__(self) -> None:
            try:
                session.is_ready(stream)
            except BaseException as exc:
                error_types.append(type(exc))

    recognizer.result_value = _ReenteringResult()
    assert session.get_result(stream) == "normalized"
    assert error_types == [
        SherpaStreamingDecodeReentryError,
        SherpaStreamingDecodeReentryError,
    ]

    class _BrokenResult:
        def __str__(self) -> str:
            raise ValueError("synthetic result conversion failure")

    recognizer.result_value = _BrokenResult()
    with pytest.raises(ValueError, match="synthetic result conversion failure"):
        session.get_result(stream)
    snapshot = session.snapshot()
    assert snapshot.native_call_active is False
    assert snapshot.native_calls_failed == 1
    session.close()


def test_native_object_return_values_cannot_escape_the_session() -> None:
    class _ReturningStream:
        def accept_waveform(self, _sample_rate, _samples):
            return self

    class _ReturningRecognizer:
        def __init__(self) -> None:
            self.stream = _ReturningStream()

        def create_stream(self):
            return self.stream

        def is_ready(self, _stream) -> bool:
            return False

        def decode_stream(self, _stream):
            return self

        def get_result(self, _stream) -> str:
            return ""

        def is_endpoint(self, _stream) -> bool:
            return False

        def reset(self, _stream):
            return self.stream

    recognizer = _ReturningRecognizer()
    session = SherpaStreamingDecodeSession(recognizer)
    session.bind_current_thread()
    stream = session.create_stream()

    assert stream.accept_waveform(16_000, object()) is None
    assert session.decode_stream(stream) is None
    assert session.reset(stream) is None
    session.close()


def test_cross_session_forged_retired_and_closed_handles_fail_closed() -> None:
    first_recognizer, first = _bound_session()
    second_recognizer, second = _bound_session()
    first_stream = first.create_stream()
    second_stream = second.create_stream()

    with pytest.raises(SherpaStreamingDecodeStreamError, match="different"):
        second.is_ready(first_stream)
    with pytest.raises(SherpaStreamingDecodeStreamError, match="different"):
        first.reset(second_stream)
    with pytest.raises(SherpaStreamingDecodeStreamError, match="exact"):
        first.is_ready(object())
    with pytest.raises(TypeError, match="created by their decode session"):
        SherpaStreamingDecodeStream(object(), first, object(), 1)

    first.retire_stream(first_stream)
    assert "fenced" in repr(first_stream)
    with pytest.raises(SherpaStreamingDecodeStreamError, match="stale"):
        first.get_result(first_stream)
    first.close()
    with pytest.raises(SherpaStreamingDecodeClosedError):
        first_stream.accept_waveform(16_000, object())
    assert [call.name for call in first_recognizer.calls] == ["create_stream"]
    assert [call.name for call in second_recognizer.calls] == ["create_stream"]
    second.close()


def test_foreign_handle_drop_and_snapshot_never_destroy_native_stream() -> None:
    recognizer, session = _bound_session()
    owner = current_thread()
    stream = session.create_stream()
    handle_ref = weakref.ref(stream)
    assert recognizer.last_stream_ref is not None
    native_ref = recognizer.last_stream_ref
    holder = [stream]
    del stream

    def drop_last_handle() -> None:
        handle = holder.pop()
        del handle
        gc.collect()

    thread = Thread(target=drop_last_handle)
    thread.start()
    thread.join(_WAIT_SECONDS)

    assert not thread.is_alive()
    assert handle_ref() is None
    assert native_ref() is not None
    assert recognizer.destroyed_on == []
    snapshot = session.snapshot()
    assert snapshot.live_streams == 0
    assert snapshot.retained_streams == 1
    assert snapshot.orphaned_streams == 1
    assert native_ref() is not None

    replacement = session.create_stream()
    assert replacement is not None
    assert native_ref() is None
    assert recognizer.destroyed_on == [owner]
    assert session.snapshot().streams_retired == 1
    session.close()


def test_foreign_drop_of_session_and_handle_leaks_safe_until_owner_close() -> None:
    recognizer, session = _bound_session()
    owner = current_thread()
    stream = session.create_stream()
    session_ref = weakref.ref(session)
    assert recognizer.last_stream_ref is not None
    native_ref = recognizer.last_stream_ref
    holder = [session, stream]
    del session
    del stream

    def drop_external_references() -> None:
        values = list(holder)
        holder.clear()
        del values
        gc.collect()

    thread = Thread(target=drop_external_references, name="foreign-drop")
    thread.start()
    thread.join(_WAIT_SECONDS)

    assert not thread.is_alive()
    retained = session_ref()
    assert retained is not None
    assert native_ref() is not None
    assert recognizer.destroyed_on == []
    retained.close()
    assert native_ref() is None
    assert recognizer.destroyed_on == [owner]


def test_explicit_retirement_fences_native_destructor_reentry() -> None:
    errors: list[BaseException] = []
    session_ref: weakref.ReferenceType[SherpaStreamingDecodeSession] | None = None

    class _ReenteringStream:
        def __del__(self) -> None:
            assert session_ref is not None
            session = session_ref()
            assert session is not None
            try:
                session.create_stream()
            except BaseException as exc:
                errors.append(exc)

    class _ReenteringRecognizer:
        def create_stream(self):
            return _ReenteringStream()

    session = SherpaStreamingDecodeSession(_ReenteringRecognizer())
    session_ref = weakref.ref(session)
    session.bind_current_thread()
    stream = session.create_stream()

    session.retire_stream(stream)

    assert len(errors) == 1
    assert isinstance(errors[0], SherpaStreamingDecodeReentryError)
    assert session.snapshot().native_call_active is False
    session.close()


def test_close_destroys_native_streams_before_recognizer_on_owner() -> None:
    events: list[tuple[str, Thread]] = []

    class _OrderedStream:
        def __del__(self) -> None:
            events.append(("stream", current_thread()))

    class _OrderedRecognizer:
        def create_stream(self):
            return _OrderedStream()

        def __del__(self) -> None:
            events.append(("recognizer", current_thread()))

    owner = current_thread()
    recognizer = _OrderedRecognizer()
    recognizer_ref = weakref.ref(recognizer)
    session = SherpaStreamingDecodeSession(recognizer)
    session.bind_current_thread()
    session.create_stream()
    del recognizer

    session.close()

    assert recognizer_ref() is None
    assert events == [("stream", owner), ("recognizer", owner)]


def test_close_is_owner_only_idempotent_and_releases_streams_before_restart() -> None:
    recognizer, first = _bound_session()
    old_stream = first.create_stream()
    owner = current_thread()

    first.close()
    first.close()

    assert first.closed is True
    assert recognizer.destroyed_on == [owner]
    assert "fenced" in repr(old_stream)
    second = SherpaStreamingDecodeSession(recognizer)
    second.bind_current_thread()
    new_stream = second.create_stream()
    assert second.get_result(new_stream) == "heard words"
    with pytest.raises(SherpaStreamingDecodeStreamError, match="different"):
        second.get_result(old_stream)
    second.close()


def test_unbound_startup_close_claims_the_cleanup_thread() -> None:
    recognizer = _Recognizer()
    session = SherpaStreamingDecodeSession(recognizer)

    thread, outcome = _thread_outcome(session.close)
    thread.join(_WAIT_SECONDS)
    assert not thread.is_alive()
    assert isinstance(outcome[0], SherpaStreamingDecodeOwnerError)
    assert session.closed is False

    session.close()
    session.close()

    assert session.closed is True
    thread, outcome = _thread_outcome(session.close)
    thread.join(_WAIT_SECONDS)
    assert not thread.is_alive()
    assert isinstance(outcome[0], SherpaStreamingDecodeOwnerError)
    assert recognizer.calls == []


def test_representations_never_touch_native_or_private_values() -> None:
    _, session = _bound_session()
    stream = session.create_stream(hotwords="private phrase")

    representations = (repr(session), repr(stream), repr(session.snapshot()))

    assert all("private phrase" not in value for value in representations)
    assert all("_Recognizer" not in value for value in representations)
    assert all("_NativeStream" not in value for value in representations)
    assert not hasattr(session.snapshot(), "recognizer")
    assert not hasattr(session.snapshot(), "owner_thread")
    session.close()


def test_engine_claim_transfers_raw_recognizer_once_and_rejects_aliasing() -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    recognizer = _Recognizer()
    engine._recognizer = recognizer

    session = engine._claim_streaming_decode_session()

    assert session is not None
    assert engine._recognizer is None
    assert engine._claim_streaming_decode_session() is session
    session.bind_current_thread()
    stream = engine._new_asr_stream(session)
    assert session.get_result(stream) == "heard words"

    engine._recognizer = _Recognizer()
    with pytest.raises(RuntimeError, match="cannot both be live"):
        engine._claim_streaming_decode_session()
    engine._recognizer = None
    session.close()

    replacement_recognizer = _Recognizer()
    engine._recognizer = replacement_recognizer
    replacement = engine._claim_streaming_decode_session()
    assert replacement is not None
    assert replacement is not session
    assert engine._recognizer is None
    replacement.bind_current_thread()
    replacement.close()


@pytest.mark.parametrize(
    "failure_phase",
    (
        "construct",
        "start",
        "transferred",
        "delayed_transfer",
        "late_after_timeout",
        "ambiguous_timeout",
    ),
)
def test_capture_thread_launch_failure_reclaims_owner_and_fences_playback(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    config = SherpaConfig(
        media_pcm_queue_ms=0.0,
        input_calibrate=False,
        endpoint_enabled=False,
        final_speech_evidence_enabled=False,
        aec_enabled=False,
        barge_in_enabled=False,
        coherence_barge_in_enabled=False,
    )
    engine = SherpaOnnxEngine(config)
    recognizer = _Recognizer()
    input_events: list[str] = []
    instances: list[object] = []
    read_entered = Event()
    close_requested = Event()

    class _FakeRecoveringInput:
        actual_samplerate = 16_000
        actual_device = "fake-input"
        generation = 1

        def __init__(self, *_args, **_kwargs) -> None:
            instances.append(self)

        def open(self) -> None:
            input_events.append("open")

        def request_close(self) -> None:
            input_events.append("request_close")
            close_requested.set()

        def read(self, frames: int):
            read_entered.set()
            assert close_requested.wait(_WAIT_SECONDS)
            return np.zeros(frames, dtype="float32"), False

        def abort_read(self, *, timeout=None) -> bool:
            input_events.append("abort_read")
            close_requested.set()
            return True

        def close(self, *, teardown_timeout=None) -> bool:
            assert teardown_timeout is not None
            input_events.append("close")
            close_requested.set()
            return True

    sounddevice = SimpleNamespace(
        query_devices=lambda *_args, **_kwargs: {
            "name": "fake-input",
            "default_samplerate": 16_000,
        },
        check_input_settings=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    monkeypatch.setattr(
        recovering_input_module,
        "_RecoveringInputStream",
        _FakeRecoveringInput,
    )

    def build() -> None:
        engine._recognizer = recognizer
        engine._tts = object()

    monkeypatch.setattr(engine, "_build", build)
    monkeypatch.setattr(engine, "_resolve_capture_domain", lambda *_a, **_k: True)
    monkeypatch.setattr(engine, "_rebind_speech_evidence_domain", lambda: None)

    receipts: list[PlaybackReceipt] = []
    receipt_fence_states: list[tuple[bool, bool, bool]] = []
    raced_enqueued: list[bool] = []
    launched_capture_threads: list[Thread] = []
    delayed_launchers: list[Thread] = []
    ambiguous_initial_states: list[tuple[int | None, bool]] = []
    callback_stop_states: list[tuple[bool, bool]] = []
    allow_late_launch = Event()
    bundle_close_calls: list[bool] = []

    if failure_phase == "delayed_transfer":
        class _ManifestStatus:
            value = "incomplete"

        class _FakeBundle:
            manifest_status = _ManifestStatus()
            failure_codes = ("startup_rollback",)

            def close(self, *, clean_shutdown: bool):
                bundle_close_calls.append(clean_shutdown)
                return "fake-manifest.json"

        engine.set_record_path("delayed-startup-bundle.wav")

        def open_bundle() -> None:
            engine._diagnostic_bundle = _FakeBundle()

        monkeypatch.setattr(engine, "_open_recorders", open_bundle)

    def receive(receipt: PlaybackReceipt) -> None:
        receipt_fence_states.append(
            (
                not engine._running.is_set(),
                engine._capture_stopping.is_set(),
                engine._capture_thread is None,
            )
        )
        receipts.append(receipt)
        if failure_phase == "ambiguous_timeout":
            engine.stop()
            callback_stop_states.append(
                (
                    engine._capture_thread is not None,
                    engine._capture_resource_hold.is_set(),
                )
            )

    def enqueue_raced_speech() -> None:
        engine.speak_tracked(
            TrackedSpeech("startup-race", "raced before capture launch"),
            on_terminal=receive,
        )
        assert engine._play_q.qsize() == 1
        raced_enqueued.append(True)

    if failure_phase == "construct":
        original_thread_constructor = decode_owner_module.threading.Thread

        def fail_capture_constructor(*args, **kwargs):
            if kwargs.get("name") == "speaker-streaming-decode-owner":
                enqueue_raced_speech()
                raise RuntimeError("synthetic capture launch failure")
            return original_thread_constructor(*args, **kwargs)

        monkeypatch.setattr(
            decode_owner_module.threading,
            "Thread",
            fail_capture_constructor,
        )
    elif failure_phase in {
        "delayed_transfer",
        "late_after_timeout",
        "ambiguous_timeout",
    }:
        original_thread_constructor = decode_owner_module.threading.Thread
        original_thread_start = Thread.start

        class _AmbiguousCaptureThread(Thread):
            def start(self) -> None:
                enqueue_raced_speech()
                ambiguous_initial_states.append(
                    (self.ident, self._started.is_set())
                )
                if failure_phase == "delayed_transfer":
                    def launch_later() -> None:
                        time.sleep(0.02)
                        launched_capture_threads.append(self)
                        original_thread_start(self)

                    launcher = Thread(
                        target=launch_later,
                        name="delayed-capture-launcher",
                    )
                    delayed_launchers.append(launcher)
                    launcher.start()
                elif failure_phase == "late_after_timeout":
                    def launch_after_rollback() -> None:
                        assert allow_late_launch.wait(_WAIT_SECONDS)
                        launched_capture_threads.append(self)
                        original_thread_start(self)

                    launcher = Thread(
                        target=launch_after_rollback,
                        name="post-timeout-capture-launcher",
                    )
                    delayed_launchers.append(launcher)
                    launcher.start()
                if failure_phase == "ambiguous_timeout":
                    raise KeyboardInterrupt(
                        "synthetic capture launch interruption"
                    )
                raise RuntimeError("synthetic capture launch failure")

        def ambiguous_thread_constructor(*args, **kwargs):
            if kwargs.get("name") == "speaker-streaming-decode-owner":
                return _AmbiguousCaptureThread(*args, **kwargs)
            return original_thread_constructor(*args, **kwargs)

        monkeypatch.setattr(
            decode_owner_module.threading,
            "Thread",
            ambiguous_thread_constructor,
        )
    else:
        original_thread_start = Thread.start

        def fail_capture_start(thread: Thread) -> None:
            if thread.name == "speaker-streaming-decode-owner":
                if failure_phase == "transferred":
                    original_thread_start(thread)
                    launched_capture_threads.append(thread)
                    assert read_entered.wait(_WAIT_SECONDS)
                enqueue_raced_speech()
                if failure_phase == "transferred":
                    raise KeyboardInterrupt(
                        "synthetic capture launch interruption"
                    )
                raise RuntimeError("synthetic capture launch failure")
            original_thread_start(thread)

        monkeypatch.setattr(Thread, "start", fail_capture_start)

    expected_error = (
        KeyboardInterrupt
        if failure_phase in {"transferred", "ambiguous_timeout"}
        else RuntimeError
    )
    with pytest.raises(expected_error, match="synthetic capture launch"):
        engine.start(EngineCallbacks())

    if failure_phase == "late_after_timeout":
        late_session = engine._streaming_decode_session
        late_thread = engine._capture_thread
        assert late_session is not None and not late_session.closed
        assert late_thread is not None
        assert not engine._thread_start_was_observed(late_thread)
        assert engine._capture_resource_hold.is_set()
        assert engine._stream_in is not None
        allow_late_launch.set()
    for launcher in delayed_launchers:
        launcher.join(_WAIT_SECONDS)
        assert not launcher.is_alive()
    if failure_phase == "late_after_timeout":
        assert len(launched_capture_threads) == 1
        launched_capture_threads[0].join(_WAIT_SECONDS)
        assert not launched_capture_threads[0].is_alive()
        assert late_session.closed
        engine.stop()
        assert engine._play_q.empty()
        assert not engine._capture_resource_hold.is_set()
        assert engine._stream_in is None
        assert not engine._thread_may_start_or_is_alive(engine._capture_thread)
        engine._capture_thread = None

    assert len(instances) == 1
    if failure_phase == "ambiguous_timeout":
        assert input_events == [
            "open",
            "request_close",
            "abort_read",
            "request_close",
            "abort_read",
        ]
        assert callback_stop_states == [(True, True)]
    elif failure_phase == "start":
        assert input_events == ["open", "request_close", "abort_read"]
        assert callback_stop_states == []
    elif failure_phase == "late_after_timeout":
        assert input_events == [
            "open",
            "request_close",
            "abort_read",
            "close",
        ]
        assert callback_stop_states == []
    elif failure_phase == "delayed_transfer":
        # The late owner observes the startup fence before its first read,
        # then its owner-loop terminal path collects the retained input.
        assert input_events == ["open", "close"]
        assert bundle_close_calls == [False]
        assert callback_stop_states == []
    else:
        assert input_events == ["open", "request_close", "close"]
        assert callback_stop_states == []
    assert raced_enqueued == [True]
    assert len(receipts) == 1
    assert receipts[0].fragment_id == "startup-race"
    assert receipts[0].outcome is PlaybackOutcome.DROPPED
    assert len(receipt_fence_states) == 1
    assert receipt_fence_states[0][:2] == (True, True)
    if failure_phase == "construct":
        assert receipt_fence_states[0][2] is True
    elif failure_phase in {
        "start",
        "transferred",
        "late_after_timeout",
        "ambiguous_timeout",
    }:
        assert receipt_fence_states[0][2] is False
    assert not engine._receipt_running.is_set()
    assert engine._receipt_thread is None
    assert not engine._receipt_pending
    assert not engine._receipt_events
    assert engine._play_q.empty()
    assert engine._playback_stopping.is_set()
    assert engine._stop_speaking.is_set()
    assert engine._speak_gen == 1
    assert engine._playback_generation == 0
    assert engine._capture_stopping.is_set()
    assert not engine._running.is_set()
    assert engine._capture_control_lane is None
    assert engine._play_thread is None
    assert engine._recognizer is None
    session = engine._streaming_decode_session
    assert session is not None
    if failure_phase in {"start", "ambiguous_timeout"}:
        retained_thread = engine._capture_thread
        assert retained_thread is not None
        assert not engine._thread_start_was_observed(retained_thread)
        assert engine._capture_resource_hold.is_set()
        assert engine._stream_in is not None
        assert engine._capture_media_session is None
        assert session.closed is False
    else:
        assert not engine._capture_resource_hold.is_set()
        assert engine._stream_in is None
        assert engine._capture_media_session is None
        assert engine._capture_thread is None
        assert session.closed is True
    if failure_phase in {
        "transferred",
        "delayed_transfer",
        "late_after_timeout",
    }:
        assert len(launched_capture_threads) == 1
        assert not launched_capture_threads[0].is_alive()
    if failure_phase in {
        "delayed_transfer",
        "late_after_timeout",
        "ambiguous_timeout",
    }:
        assert ambiguous_initial_states == [(None, False)]

    rejected_done: list[bool] = []
    engine.speak("must be rejected", lambda: rejected_done.append(True))
    assert rejected_done == [True]
    assert engine._play_q.empty()

    if failure_phase in {"start", "ambiguous_timeout"}:
        # An ordinary pre-publication exception is deliberately retained too:
        # exception hierarchy cannot prove that OS-thread transfer did not win.
        assert engine._capture_thread is retained_thread
        assert engine._capture_resource_hold.is_set()
        assert session.closed is False
        if failure_phase == "ambiguous_timeout":
            # The receipt callback's re-entrant public stop also retained,
            # rather than joining, the unpublished capture owner.
            assert callback_stop_states == [(True, True)]

        # The test controls this fake and knows it can no longer launch. Release
        # the intentional fail-closed retention without leaking into other tests.
        session.close()
        engine._capture_thread = None
        assert engine._close_capture_input()
        engine._capture_resource_hold.clear()


def test_post_input_playback_launch_interruption_rolls_back_all_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tail-worker start escape must unwind the already-live capture graph."""

    config = SherpaConfig(
        media_pcm_queue_ms=300.0,
        input_calibrate=False,
        endpoint_enabled=False,
        final_speech_evidence_enabled=False,
        aec_enabled=False,
        barge_in_enabled=False,
        coherence_barge_in_enabled=False,
    )
    engine = SherpaOnnxEngine(config)
    recognizer = _Recognizer()
    input_events: list[str] = []
    read_entered = Event()
    close_requested = Event()
    playback_started = Event()

    class _FakeRecoveringInput:
        actual_samplerate = 16_000
        actual_device = "fake-input"
        generation = 1

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def open(self) -> None:
            input_events.append("open")

        def request_close(self) -> None:
            input_events.append("request_close")
            close_requested.set()

        def read(self, frames: int):
            read_entered.set()
            assert close_requested.wait(_WAIT_SECONDS)
            return np.zeros(frames, dtype="float32"), False

        def abort_read(self, *, timeout=None) -> bool:
            input_events.append("abort_read")
            close_requested.set()
            return True

        def close(self, *, teardown_timeout=None) -> bool:
            assert teardown_timeout is not None
            input_events.append("close")
            close_requested.set()
            return True

    sounddevice = SimpleNamespace(
        query_devices=lambda *_args, **_kwargs: {
            "name": "fake-input",
            "default_samplerate": 16_000,
        },
        check_input_settings=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    monkeypatch.setattr(
        recovering_input_module,
        "_RecoveringInputStream",
        _FakeRecoveringInput,
    )

    def build() -> None:
        engine._recognizer = recognizer
        engine._tts = object()

    monkeypatch.setattr(engine, "_build", build)
    monkeypatch.setattr(engine, "_resolve_capture_domain", lambda *_a, **_k: True)
    monkeypatch.setattr(engine, "_rebind_speech_evidence_domain", lambda: None)

    original_thread_start = Thread.start

    def interrupt_playback_start(thread: Thread) -> None:
        target = getattr(thread, "_target", None)
        is_playback_worker = bool(
            getattr(target, "__self__", None) is engine
            and getattr(target, "__func__", None)
            is SherpaOnnxEngine._playback_loop
        )
        if not is_playback_worker:
            original_thread_start(thread)
            return
        # Prove this is a post-input tail failure: both the decode owner and the
        # native capture reader have already transferred, and the receipt worker
        # is live, before playback launch.
        assert read_entered.wait(_WAIT_SECONDS)
        owner = engine._streaming_decode_owner
        assert owner is not None
        owner_snapshot = owner.snapshot()
        assert owner_snapshot.ready
        assert owner_snapshot.thread_alive
        assert not owner_snapshot.terminal
        assert engine._receipt_thread is not None
        assert engine._receipt_thread.is_alive()
        original_thread_start(thread)
        playback_started.set()
        raise KeyboardInterrupt("synthetic playback launch interruption")

    monkeypatch.setattr(Thread, "start", interrupt_playback_start)

    with pytest.raises(
        KeyboardInterrupt,
        match="synthetic playback launch interruption",
    ):
        engine.start(EngineCallbacks())

    assert playback_started.is_set()
    owner = engine._streaming_decode_owner
    session = engine._streaming_decode_session
    assert owner is not None
    assert session is not None and session.closed
    assert owner.snapshot().session_closed
    assert input_events == ["open", "request_close", "close"]
    assert not engine._running.is_set()
    assert engine._capture_stopping.is_set()
    assert engine._playback_stopping.is_set()
    assert engine._capture_thread is None
    assert engine._capture_media_session is None
    assert engine._stream_in is None
    assert engine._play_thread is None
    assert engine._receipt_thread is None
    assert not engine._receipt_running.is_set()
    assert not engine._capture_resource_hold.is_set()
    assert engine._play_q.empty()

    # Rollback leaves ordinary public cleanup idempotent and bounded.
    engine.stop()
    assert engine._capture_thread is None
    assert engine._capture_media_session is None
    assert engine._stream_in is None
    assert engine._play_thread is None
    assert engine._receipt_thread is None
    assert not engine._capture_resource_hold.is_set()
    assert engine._play_q.empty()


def test_rollback_closes_new_session_with_stale_closed_owner_metadata() -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    old_session = SherpaStreamingDecodeSession(_Recognizer())
    old_owner = decode_owner_module.SherpaStreamingDecodeOwner(
        old_session,
        run_id=1,
        capture_scope=sherpa_module.CaptureScope(
            capture_epoch=0,
            capture_generation=1,
        ),
        processor=lambda _owner: None,
    )
    assert old_owner.close(timeout=0.0)
    engine._streaming_decode_owner = old_owner
    engine._capture_thread = old_owner.thread

    new_session = SherpaStreamingDecodeSession(_Recognizer())
    engine._streaming_decode_session = new_session

    engine._rollback_failed_capture_processor_launch(
        new_session,
        decode_owner=None,
        capture_start_ambiguous=False,
    )

    assert new_session.closed
    assert old_owner.snapshot().state is decode_owner_module.DecodeOwnerState.CLOSED
    assert engine._capture_thread is None
    assert not engine._capture_resource_hold.is_set()


def test_unobserved_playback_launch_is_retained_until_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later stop must not join a tail worker whose start never published."""

    engine = SherpaOnnxEngine(
        SherpaConfig(
            media_pcm_queue_ms=300.0,
            input_calibrate=False,
            endpoint_enabled=False,
            final_speech_evidence_enabled=False,
            aec_enabled=False,
            barge_in_enabled=False,
            coherence_barge_in_enabled=False,
        )
    )
    recognizer = _Recognizer()
    read_entered = Event()
    close_requested = Event()
    input_events: list[str] = []
    pending_playback: list[Thread] = []

    class _FakeRecoveringInput:
        actual_samplerate = 16_000
        actual_device = "fake-input"
        generation = 1

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def open(self) -> None:
            input_events.append("open")

        def request_close(self) -> None:
            input_events.append("request_close")
            close_requested.set()

        def read(self, frames: int):
            read_entered.set()
            assert close_requested.wait(_WAIT_SECONDS)
            return np.zeros(frames, dtype="float32"), False

        def abort_read(self, *, timeout=None) -> bool:
            input_events.append("abort_read")
            close_requested.set()
            return True

        def close(self, *, teardown_timeout=None) -> bool:
            assert teardown_timeout is not None
            input_events.append("close")
            return True

    sounddevice = SimpleNamespace(
        query_devices=lambda *_args, **_kwargs: {
            "name": "fake-input",
            "default_samplerate": 16_000,
        },
        check_input_settings=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    monkeypatch.setattr(
        recovering_input_module,
        "_RecoveringInputStream",
        _FakeRecoveringInput,
    )

    def build() -> None:
        engine._recognizer = recognizer
        engine._tts = object()

    monkeypatch.setattr(engine, "_build", build)
    monkeypatch.setattr(engine, "_resolve_capture_domain", lambda *_a, **_k: True)
    monkeypatch.setattr(engine, "_rebind_speech_evidence_domain", lambda: None)

    original_thread_start = Thread.start

    def interrupt_before_playback_publication(thread: Thread) -> None:
        target = getattr(thread, "_target", None)
        is_playback_worker = bool(
            getattr(target, "__self__", None) is engine
            and getattr(target, "__func__", None)
            is SherpaOnnxEngine._playback_loop
        )
        if not is_playback_worker:
            original_thread_start(thread)
            return
        assert read_entered.wait(_WAIT_SECONDS)
        pending_playback.append(thread)
        raise KeyboardInterrupt("synthetic unpublished playback launch")

    monkeypatch.setattr(Thread, "start", interrupt_before_playback_publication)

    with pytest.raises(
        KeyboardInterrupt,
        match="synthetic unpublished playback launch",
    ):
        engine.start(EngineCallbacks())

    assert len(pending_playback) == 1
    worker = pending_playback[0]
    assert not worker._started.is_set()
    assert engine._play_thread is worker
    assert engine._capture_resource_hold.is_set()
    assert engine._stream_in is not None

    # This used to raise ``cannot join thread before it is started``. The
    # unpublished tail worker remains fail-closed; capture can still close once
    # its own reader and owner are independently quiescent.
    engine.stop()
    assert engine._play_thread is worker
    assert engine._capture_resource_hold.is_set()
    assert engine._stream_in is None

    # Resolve the test-controlled launch ambiguity exactly as a late OS-thread
    # transfer would. It observes the shutdown fence and exits without audio.
    original_thread_start(worker)
    worker.join(_WAIT_SECONDS)
    assert not worker.is_alive()

    engine.stop()
    assert engine._stream_in is None
    assert engine._capture_media_session is None
    assert not engine._capture_resource_hold.is_set()
    assert input_events.count("close") == 1


@pytest.mark.parametrize(
    "worker_attribute",
    ["_final_thread", "_virtual_route_thread"],
)
def test_stop_does_not_join_other_unobserved_tail_workers(
    worker_attribute: str,
) -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    worker = Thread(target=lambda: None)
    setattr(engine, worker_attribute, worker)

    engine.stop()

    assert getattr(engine, worker_attribute) is worker
    assert not worker._started.is_set()
    worker.start()
    worker.join(_WAIT_SECONDS)
    assert not worker.is_alive()

    engine.stop()
    assert not engine._thread_may_start_or_is_alive(worker)


def test_stop_cannot_upgrade_rollback_bundle_with_unobserved_final_worker() -> None:
    close_calls: list[bool] = []

    class _ManifestStatus:
        value = "incomplete"

    class _FakeBundle:
        manifest_status = _ManifestStatus()
        failure_codes = ("startup_rollback",)

        def close(self, *, clean_shutdown: bool):
            close_calls.append(clean_shutdown)
            return "fake-manifest.json"

    engine = SherpaOnnxEngine(SherpaConfig())
    worker = Thread(target=lambda: None)
    engine._final_thread = worker
    engine._diagnostic_bundle = _FakeBundle()
    engine._capture_startup_rollback.set()

    engine.stop()

    assert close_calls == [False]
    assert engine._diagnostic_bundle is None
    assert engine._final_thread is worker
    assert not worker._started.is_set()
    worker.start()
    worker.join(_WAIT_SECONDS)
    engine.stop()


def test_late_tracked_speech_cannot_resurrect_receipt_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._running.set()
    engine._playback_stopping.clear()
    dispatcher_call_entered = Event()
    resume_dispatcher_call = Event()
    receipts: list[PlaybackReceipt] = []
    real_start_dispatcher = engine._start_receipt_dispatcher

    def delayed_start_dispatcher() -> None:
        dispatcher_call_entered.set()
        assert resume_dispatcher_call.wait(_WAIT_SECONDS)
        real_start_dispatcher()

    monkeypatch.setattr(engine, "_start_receipt_dispatcher", delayed_start_dispatcher)
    thread, outcome = _thread_outcome(
        lambda: engine.speak_tracked(
            TrackedSpeech("late-startup-race", "late tracked speech"),
            on_terminal=receipts.append,
        )
    )
    assert dispatcher_call_entered.wait(_WAIT_SECONDS)

    try:
        engine._rollback_failed_capture_processor_launch(None)
    finally:
        resume_dispatcher_call.set()
    thread.join(_WAIT_SECONDS)

    assert not thread.is_alive()
    assert outcome == [None]
    assert len(receipts) == 1
    assert receipts[0].fragment_id == "late-startup-race"
    assert receipts[0].outcome is PlaybackOutcome.DROPPED
    assert not engine._receipt_running.is_set()
    assert engine._receipt_thread is None
    assert not engine._receipt_pending
    assert not engine._receipt_events


def test_live_decode_owner_blocks_restart_before_shared_state_changes() -> None:
    engine = SherpaOnnxEngine(SherpaConfig())
    _, session = _bound_session()
    engine._streaming_decode_session = session
    engine._capture_stopping.set()
    engine._playback_stopping.set()

    with pytest.raises(RuntimeError, match="previous streaming decode owner"):
        engine.start(EngineCallbacks())

    assert engine._streaming_decode_session is session
    assert engine._capture_stopping.is_set()
    assert engine._playback_stopping.is_set()
    assert not engine._running.is_set()
    session.close()
