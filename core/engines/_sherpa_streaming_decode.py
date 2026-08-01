"""Exact-thread ownership for one Sherpa online recognizer.

Sherpa's online recognizer and streams are mutable native objects.  A local
capture run therefore places all of them behind one synchronous session before
the capture processor begins using them.  Only opaque stream handles can leave
this module; the recognizer and native streams are never represented or exposed.

This is deliberately not an asynchronous PCM stage.  Keeping the complete
owner synchronous first makes a later worker handoff possible without moving a
subset of native calls onto another thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock, Thread, current_thread
import weakref


class SherpaStreamingDecodeSessionError(RuntimeError):
    """Base error raised by the streaming decode ownership boundary."""


class SherpaStreamingDecodeOwnerError(SherpaStreamingDecodeSessionError):
    """A caller is not the exact thread that owns this session."""


class SherpaStreamingDecodeReentryError(SherpaStreamingDecodeSessionError):
    """A native operation tried to re-enter the synchronous session."""


class SherpaStreamingDecodeClosedError(SherpaStreamingDecodeSessionError):
    """An operation was attempted after the session was closed."""


class SherpaStreamingDecodeStreamError(SherpaStreamingDecodeSessionError):
    """A handle is forged, stale, or belongs to another session."""


@dataclass(frozen=True, slots=True)
class SherpaStreamingDecodeSnapshot:
    """Bounded metadata that retains no native or thread object."""

    bound: bool
    closed: bool
    native_call_active: bool
    live_streams: int
    retained_streams: int
    orphaned_streams: int
    streams_created: int
    streams_retired: int
    native_calls_started: int
    native_calls_failed: int


_STREAM_CONSTRUCTION_TOKEN = object()
_LIVE_SESSION_LOCK = Lock()
_LIVE_SESSIONS: dict[int, object] = {}


def _retain_live_session(session: object) -> None:
    """Fail closed by keeping an unclosed native owner alive process-wide."""

    with _LIVE_SESSION_LOCK:
        _LIVE_SESSIONS[id(session)] = session


def _release_closed_session(session: object) -> None:
    with _LIVE_SESSION_LOCK:
        if _LIVE_SESSIONS.get(id(session)) is session:
            del _LIVE_SESSIONS[id(session)]


class SherpaStreamingDecodeStream:
    """Opaque handle for one stream owned by a decode session."""

    __slots__ = (
        "_session",
        "_session_token",
        "_sequence",
        "_fenced",
        "__weakref__",
    )

    def __init__(
        self,
        construction_token: object,
        session: SherpaStreamingDecodeSession,
        session_token: object,
        sequence: int,
    ) -> None:
        if construction_token is not _STREAM_CONSTRUCTION_TOKEN:
            raise TypeError("stream handles are created by their decode session")
        self._session = session
        self._session_token = session_token
        self._sequence = sequence
        self._fenced = False

    def accept_waveform(self, sample_rate: int, samples: object) -> None:
        """Synchronously admit PCM through the exact owning session."""

        self._session._accept_waveform(self, sample_rate, samples)

    def __repr__(self) -> str:
        state = "fenced" if self._fenced else "live"
        return (
            f"{type(self).__name__}(sequence={self._sequence}, "
            f"state={state!r})"
        )


@dataclass(slots=True)
class _NativeStreamRecord:
    """Private association between an opaque handle and one native stream."""

    handle_ref: weakref.ReferenceType[SherpaStreamingDecodeStream] = field(
        repr=False
    )
    native_stream: object = field(repr=False)


class SherpaStreamingDecodeSession:
    """Own one online recognizer and all of its streams on one exact thread.

    Construction is intentionally unbound so ``start()`` may build the native
    model before launching capture.  The capture processor must explicitly bind
    before the first stream operation.  A close on an unbound session lets the
    startup caller reclaim a session whose capture thread never launched.

    A dead handle never destroys its native stream in the garbage-collecting
    thread.  Its weak callback records metadata only; the next owner operation
    or owner close performs the actual retirement. Unclosed sessions are retained
    in a process-private registry, preferring a visible native-memory leak over
    foreign-thread finalization.
    """

    def __init__(self, recognizer: object) -> None:
        if recognizer is None:
            raise TypeError("recognizer must be a native recognizer object")
        self._lock = Lock()
        self._recognizer: object | None = recognizer
        self._construction_thread = current_thread()
        self._owner_thread: Thread | None = None
        self._session_token = object()
        self._streams: dict[int, _NativeStreamRecord] = {}
        self._orphaned_stream_ids: set[int] = set()
        self._next_sequence = 1
        self._closed = False
        self._native_call_active = False
        self._streams_created = 0
        self._streams_retired = 0
        self._native_calls_started = 0
        self._native_calls_failed = 0
        # A forgotten live session leaks safely instead of letting arbitrary
        # garbage-collector activity destroy native state on a foreign thread.
        _retain_live_session(self)

    def bind_current_thread(self) -> None:
        """Bind once to the exact current ``Thread`` object."""

        caller = current_thread()
        with self._lock:
            if self._closed:
                raise SherpaStreamingDecodeClosedError("decode session is closed")
            owner = self._owner_thread
            if owner is None:
                self._owner_thread = caller
            elif caller is not owner:
                raise SherpaStreamingDecodeOwnerError(
                    "decode session belongs to a different thread"
                )

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def create_stream(
        self,
        *,
        hotwords: str | None = None,
    ) -> SherpaStreamingDecodeStream:
        """Create an opaque stream and preserve older-Sherpa fallback semantics."""

        recognizer, _, orphaned = self._begin_native_call()
        orphaned.clear()  # native stream destruction stays on the owner thread
        try:
            if hotwords is None:
                native_stream = recognizer.create_stream()
            else:
                native_stream = recognizer.create_stream(hotwords=hotwords)
            handle = self._register_native_stream(native_stream)
        except BaseException:
            self._finish_native_call(failed=True)
            raise
        self._finish_native_call(failed=False)
        return handle

    def is_ready(self, stream: SherpaStreamingDecodeStream) -> bool:
        recognizer, native_stream, orphaned = self._begin_native_call(stream)
        orphaned.clear()
        try:
            result = bool(recognizer.is_ready(native_stream))
        except BaseException:
            self._finish_native_call(failed=True)
            raise
        self._finish_native_call(failed=False)
        return result

    def decode_stream(self, stream: SherpaStreamingDecodeStream) -> None:
        recognizer, native_stream, orphaned = self._begin_native_call(stream)
        orphaned.clear()
        try:
            recognizer.decode_stream(native_stream)
        except BaseException:
            self._finish_native_call(failed=True)
            raise
        self._finish_native_call(failed=False)

    def get_result(self, stream: SherpaStreamingDecodeStream) -> str:
        recognizer, native_stream, orphaned = self._begin_native_call(stream)
        orphaned.clear()
        try:
            result = recognizer.get_result(native_stream)
            try:
                text = "" if result is None else str(result)
            finally:
                # A native result wrapper must also be released while reentry
                # remains fenced, including when text conversion fails.
                del result
        except BaseException:
            self._finish_native_call(failed=True)
            raise
        self._finish_native_call(failed=False)
        return text

    def is_endpoint(self, stream: SherpaStreamingDecodeStream) -> bool:
        recognizer, native_stream, orphaned = self._begin_native_call(stream)
        orphaned.clear()
        try:
            result = bool(recognizer.is_endpoint(native_stream))
        except BaseException:
            self._finish_native_call(failed=True)
            raise
        self._finish_native_call(failed=False)
        return result

    def reset(self, stream: SherpaStreamingDecodeStream) -> None:
        recognizer, native_stream, orphaned = self._begin_native_call(stream)
        orphaned.clear()
        try:
            recognizer.reset(native_stream)
        except BaseException:
            self._finish_native_call(failed=True)
            raise
        self._finish_native_call(failed=False)

    def retire_stream(self, stream: SherpaStreamingDecodeStream) -> None:
        """Fence one handle and destroy its native stream on the owner thread."""

        caller = current_thread()
        record: _NativeStreamRecord
        with self._lock:
            self._require_owner_locked(caller)
            if self._closed:
                raise SherpaStreamingDecodeClosedError("decode session is closed")
            if self._native_call_active:
                raise SherpaStreamingDecodeReentryError(
                    "cannot retire a stream during a native operation"
                )
            self._resolve_native_stream_locked(stream)
            record = self._streams.pop(id(stream))
            self._orphaned_stream_ids.discard(id(stream))
            self._streams_retired += 1
            self._native_call_active = True
            stream._fenced = True
        try:
            del record
        finally:
            with self._lock:
                self._native_call_active = False

    def close(self) -> None:
        """Fence handles and release all native objects on the exact owner."""

        caller = current_thread()
        recognizer: object | None
        records: list[_NativeStreamRecord]
        handles: list[SherpaStreamingDecodeStream]
        with self._lock:
            owner = self._owner_thread
            if owner is None:
                if caller is not self._construction_thread:
                    raise SherpaStreamingDecodeOwnerError(
                        "unbound decode session belongs to its construction thread"
                    )
                self._owner_thread = caller
            elif caller is not owner:
                raise SherpaStreamingDecodeOwnerError(
                    "decode session belongs to a different thread"
                )
            if self._closed:
                return
            if self._native_call_active:
                raise SherpaStreamingDecodeReentryError(
                    "cannot close during an active native operation"
                )
            self._closed = True
            recognizer = self._recognizer
            self._recognizer = None
            records = list(self._streams.values())
            handles = [
                handle
                for record in records
                if (handle := record.handle_ref()) is not None
            ]
            self._streams_retired += len(records)
            self._streams.clear()
            self._orphaned_stream_ids.clear()
        for handle in handles:
            handle._fenced = True
        # CPython extension destructors are outside the state lock but remain on
        # the owner thread. Streams go before their recognizer.
        records.clear()
        del recognizer
        _release_closed_session(self)

    def snapshot(self) -> SherpaStreamingDecodeSnapshot:
        """Return metadata only; inspection never retires native objects."""

        with self._lock:
            # Never dereference a possibly-last weak handle while holding this
            # non-reentrant lock: releasing that temporary could synchronously
            # invoke ``_note_collected_stream`` and deadlock on the same lock.
            orphaned_streams = len(self._orphaned_stream_ids)
            live_streams = len(self._streams) - orphaned_streams
            return SherpaStreamingDecodeSnapshot(
                bound=self._owner_thread is not None,
                closed=self._closed,
                native_call_active=self._native_call_active,
                live_streams=live_streams,
                retained_streams=len(self._streams),
                orphaned_streams=orphaned_streams,
                streams_created=self._streams_created,
                streams_retired=self._streams_retired,
                native_calls_started=self._native_calls_started,
                native_calls_failed=self._native_calls_failed,
            )

    def __repr__(self) -> str:
        snapshot = self.snapshot()
        return (
            f"{type(self).__name__}(bound={snapshot.bound}, "
            f"closed={snapshot.closed}, "
            f"native_call_active={snapshot.native_call_active}, "
            f"live_streams={snapshot.live_streams}, "
            f"retained_streams={snapshot.retained_streams})"
        )

    def _accept_waveform(
        self,
        stream: SherpaStreamingDecodeStream,
        sample_rate: int,
        samples: object,
    ) -> None:
        _, native_stream, orphaned = self._begin_native_call(stream)
        orphaned.clear()
        try:
            native_stream.accept_waveform(sample_rate, samples)
        except BaseException:
            self._finish_native_call(failed=True)
            raise
        self._finish_native_call(failed=False)

    def _begin_native_call(
        self,
        stream: SherpaStreamingDecodeStream | None = None,
    ) -> tuple[object, object | None, list[_NativeStreamRecord]]:
        caller = current_thread()
        with self._lock:
            self._require_owner_locked(caller)
            if self._closed:
                raise SherpaStreamingDecodeClosedError("decode session is closed")
            if self._native_call_active:
                raise SherpaStreamingDecodeReentryError(
                    "a native decode operation is already active"
                )
            recognizer = self._recognizer
            if recognizer is None:
                raise SherpaStreamingDecodeClosedError("decode session is closed")
            native_stream = (
                None
                if stream is None
                else self._resolve_native_stream_locked(stream)
            )
            orphaned = self._retire_orphaned_streams_locked()
            self._native_call_active = True
            self._native_calls_started += 1
            return recognizer, native_stream, orphaned

    def _finish_native_call(self, *, failed: bool) -> None:
        with self._lock:
            if failed:
                self._native_calls_failed += 1
            self._native_call_active = False

    def _require_owner_locked(self, caller: Thread) -> None:
        owner = self._owner_thread
        if owner is None:
            raise SherpaStreamingDecodeOwnerError(
                "bind_current_thread() is required before native operations"
            )
        if caller is not owner:
            raise SherpaStreamingDecodeOwnerError(
                "decode session belongs to a different thread"
            )

    def _register_native_stream(
        self,
        native_stream: object,
    ) -> SherpaStreamingDecodeStream:
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            handle = SherpaStreamingDecodeStream(
                _STREAM_CONSTRUCTION_TOKEN,
                self,
                self._session_token,
                sequence,
            )
            handle_id = id(handle)
            owner_ref = weakref.ref(self)

            def note_collected(
                handle_ref: weakref.ReferenceType[SherpaStreamingDecodeStream],
                *,
                expected_id: int = handle_id,
                session_ref: weakref.ReferenceType[
                    SherpaStreamingDecodeSession
                ] = owner_ref,
            ) -> None:
                session = session_ref()
                if session is not None:
                    session._note_collected_stream(expected_id, handle_ref)

            handle_ref = weakref.ref(handle, note_collected)
            self._streams[handle_id] = _NativeStreamRecord(
                handle_ref=handle_ref,
                native_stream=native_stream,
            )
            self._streams_created += 1
            return handle

    def _resolve_native_stream_locked(
        self,
        stream: SherpaStreamingDecodeStream,
    ) -> object:
        if type(stream) is not SherpaStreamingDecodeStream:
            raise SherpaStreamingDecodeStreamError(
                "stream handle is not an exact session-created handle"
            )
        if (
            getattr(stream, "_session", None) is not self
            or getattr(stream, "_session_token", None) is not self._session_token
        ):
            raise SherpaStreamingDecodeStreamError(
                "stream handle belongs to a different decode session"
            )
        record = self._streams.get(id(stream))
        if record is None or record.handle_ref() is not stream:
            raise SherpaStreamingDecodeStreamError(
                "stream handle is stale or was not created by this session"
            )
        return record.native_stream

    def _note_collected_stream(
        self,
        handle_id: int,
        handle_ref: weakref.ReferenceType[SherpaStreamingDecodeStream],
    ) -> None:
        # This callback may run on any thread. It only marks metadata and never
        # removes the record that owns the native stream.
        with self._lock:
            record = self._streams.get(handle_id)
            if record is not None and record.handle_ref is handle_ref:
                self._orphaned_stream_ids.add(handle_id)

    def _retire_orphaned_streams_locked(self) -> list[_NativeStreamRecord]:
        retired: list[_NativeStreamRecord] = []
        for handle_id in tuple(self._orphaned_stream_ids):
            record = self._streams.get(handle_id)
            if record is not None and record.handle_ref() is None:
                retired.append(self._streams.pop(handle_id))
                self._streams_retired += 1
            self._orphaned_stream_ids.discard(handle_id)
        return retired


__all__ = (
    "SherpaStreamingDecodeClosedError",
    "SherpaStreamingDecodeOwnerError",
    "SherpaStreamingDecodeReentryError",
    "SherpaStreamingDecodeSession",
    "SherpaStreamingDecodeSessionError",
    "SherpaStreamingDecodeSnapshot",
    "SherpaStreamingDecodeStream",
    "SherpaStreamingDecodeStreamError",
)
