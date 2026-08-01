"""Exact owner for the complete local Sherpa streaming state machine.

The native capture reader already publishes immutable PCM through a bounded
``CaptureMailbox``.  Adding another PCM queue would add latency without adding
isolation.  This owner therefore binds the synchronous native session to one
dedicated thread and runs the engine's complete ordered capture loop there:
DSP, VAD, normal ASR, confirm/word-cut replay, endpoint decisions, and callback
publication remain one state machine.

Three identities stay deliberately distinct:

* ``CaptureScope`` changes only when raw capture continuity changes.
* ``continuity_generation`` changes for decoder-local discontinuity.
* a role-specific stream incarnation changes on every create/reset/replace.

The owner retains itself process-wide whenever startup or cleanup is ambiguous,
preferring a visible native-resource leak and restart refusal over foreign-
thread destruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Callable, Optional

from ..realtime_media_stage import CaptureScope
from ._sherpa_streaming_decode import (
    SherpaStreamingDecodeSession,
    SherpaStreamingDecodeStream,
)


class DecodeLossReason(str, Enum):
    """Why decoder-local continuity was retired."""

    CAPTURE_SCOPE = "capture_scope"
    NATIVE_ERROR = "native_error"
    EXPLICIT_DISCONTINUITY = "explicit_discontinuity"


class DecodeStreamRole(str, Enum):
    """Logical native stream within the full online state machine."""

    PRIMARY = "primary"
    WORD_CUT = "word_cut"


class DecodeOwnerState(str, Enum):
    NEW = "new"
    STARTING = "starting"
    RUNNING = "running"
    CLOSING = "closing"
    RETAINED = "retained"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DecodeIdentity:
    run_id: int
    capture_scope: CaptureScope
    continuity_generation: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.run_id, bool)
            or not isinstance(self.run_id, int)
            or self.run_id <= 0
        ):
            raise ValueError("run_id must be a positive integer")
        if not isinstance(self.capture_scope, CaptureScope):
            raise TypeError("capture_scope must be a CaptureScope")
        if (
            isinstance(self.continuity_generation, bool)
            or not isinstance(self.continuity_generation, int)
            or self.continuity_generation < 0
        ):
            raise ValueError("continuity_generation must be non-negative")


@dataclass(frozen=True, slots=True)
class DecodeStreamKey:
    role: DecodeStreamRole
    incarnation: int

    def __post_init__(self) -> None:
        if not isinstance(self.role, DecodeStreamRole):
            raise TypeError("role must be a DecodeStreamRole")
        if (
            isinstance(self.incarnation, bool)
            or not isinstance(self.incarnation, int)
            or self.incarnation <= 0
        ):
            raise ValueError("incarnation must be a positive integer")


@dataclass(frozen=True, slots=True)
class DecodeOwnerSnapshot:
    identity: DecodeIdentity
    state: DecodeOwnerState
    ready: bool
    terminal: bool
    thread_alive: bool
    session_closed: bool
    primary_stream: Optional[DecodeStreamKey]
    word_cut_stream: Optional[DecodeStreamKey]
    streams_created: int
    streams_reset: int
    streams_retired: int
    capture_rotations: int
    decode_rotations: int
    processor_error_type: str
    cleanup_error_type: str


@dataclass(slots=True)
class _OwnedStream:
    handle: SherpaStreamingDecodeStream
    key: DecodeStreamKey


_LIVE_OWNER_LOCK = threading.Lock()
_LIVE_OWNERS: dict[int, object] = {}


def _retain_live_owner(owner: object) -> None:
    with _LIVE_OWNER_LOCK:
        _LIVE_OWNERS[id(owner)] = owner


def _release_closed_owner(owner: object) -> None:
    with _LIVE_OWNER_LOCK:
        if _LIVE_OWNERS.get(id(owner)) is owner:
            del _LIVE_OWNERS[id(owner)]


class SherpaStreamingDecodeOwner:
    """Recognizer facade and lifecycle owner for one complete capture run."""

    def __init__(
        self,
        session: SherpaStreamingDecodeSession,
        *,
        run_id: int,
        capture_scope: CaptureScope,
        processor: Callable[["SherpaStreamingDecodeOwner"], None],
        thread_name: str = "speaker-streaming-decode-owner",
        thread_factory: Optional[Callable[..., threading.Thread]] = None,
    ) -> None:
        if not isinstance(session, SherpaStreamingDecodeSession):
            raise TypeError("session must be a SherpaStreamingDecodeSession")
        if not callable(processor):
            raise TypeError("processor must be callable")
        self._session = session
        self._processor = processor
        self._construction_thread = threading.current_thread()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._startup_complete = threading.Event()
        self._terminal = threading.Event()
        self._state = DecodeOwnerState.NEW
        self._owner_thread: Optional[threading.Thread] = None
        self._identity = DecodeIdentity(run_id, capture_scope, 0)
        self._seen_capture_scopes = {capture_scope}
        self._next_incarnation = 1
        self._streams: dict[int, _OwnedStream] = {}
        self._active_by_role: dict[DecodeStreamRole, DecodeStreamKey] = {}
        self._streams_created = 0
        self._streams_reset = 0
        self._streams_retired = 0
        self._capture_rotations = 0
        self._decode_rotations = 0
        self._processor_error_type = ""
        self._cleanup_error_type = ""
        factory = threading.Thread if thread_factory is None else thread_factory
        self._thread = factory(
            target=self._run,
            name=thread_name,
            daemon=True,
        )

    @property
    def thread(self) -> threading.Thread:
        return self._thread

    @property
    def identity(self) -> DecodeIdentity:
        with self._lock:
            return self._identity

    @property
    def closed(self) -> bool:
        return self._session.closed

    def start(self, timeout: float = 1.0) -> None:
        deadline = self._deadline(timeout, positive=True)
        with self._lock:
            if self._state is not DecodeOwnerState.NEW:
                raise RuntimeError("decode owner already started or retained")
            self._state = DecodeOwnerState.STARTING
        try:
            self._thread.start()
        except BaseException:
            # Thread.start can escape after the OS child won transfer but
            # before ident/_started publication. Retain and refuse restart.
            with self._lock:
                if self._state not in (
                    DecodeOwnerState.CLOSED,
                    DecodeOwnerState.FAILED,
                ):
                    self._state = DecodeOwnerState.RETAINED
                    # Keep lifecycle state and the process-wide strong
                    # reference linearized under owner -> registry lock order.
                    _retain_live_owner(self)
            raise
        if not self._startup_complete.wait(
            max(0.0, deadline - time.monotonic())
        ):
            with self._lock:
                if self._state not in (
                    DecodeOwnerState.CLOSED,
                    DecodeOwnerState.FAILED,
                ):
                    self._state = DecodeOwnerState.RETAINED
                    _retain_live_owner(self)
            raise TimeoutError("streaming decode owner did not publish readiness")
        snapshot = self.snapshot()
        if (
            snapshot.state is not DecodeOwnerState.RUNNING
            or not snapshot.ready
            or snapshot.terminal
            or not snapshot.thread_alive
            or snapshot.session_closed
        ):
            failure = (
                snapshot.processor_error_type
                or snapshot.cleanup_error_type
                or "ProcessorExited"
            )
            raise RuntimeError(
                "streaming decode owner startup failed: "
                + failure
            )

    def publish_processor_ready(self) -> None:
        """Publish that the complete state machine is initialized and waiting."""

        self._require_owner_thread()
        with self._lock:
            if self._ready.is_set():
                raise RuntimeError("decode owner readiness was already published")
            if self._state is DecodeOwnerState.STARTING:
                self._state = DecodeOwnerState.RUNNING
            elif self._state is not DecodeOwnerState.RETAINED:
                raise RuntimeError("decode owner cannot publish readiness now")
            self._ready.set()
            self._startup_complete.set()

    def close(self, timeout: float = 1.0) -> bool:
        """Join boundedly; only the owner thread closes a transferred session."""

        deadline = self._deadline(timeout)
        close_unstarted = False
        retain_unstarted = False
        with self._lock:
            state = self._state
            if state is DecodeOwnerState.NEW:
                if threading.current_thread() is not self._construction_thread:
                    self._state = DecodeOwnerState.RETAINED
                    _retain_live_owner(self)
                    retain_unstarted = True
                else:
                    # Fence concurrent start/close before native destruction.
                    # Session.close() can release extension objects whose
                    # destructors re-enter snapshot(), so it must run outside
                    # this non-reentrant owner lock.
                    self._state = DecodeOwnerState.CLOSING
                    close_unstarted = True
        if retain_unstarted:
            return False
        if close_unstarted:
            try:
                self._session.close()
            except BaseException as exc:
                with self._lock:
                    self._cleanup_error_type = type(exc).__name__
                    self._state = DecodeOwnerState.RETAINED
                    _retain_live_owner(self)
                return False
            with self._lock:
                self._state = DecodeOwnerState.CLOSED
                self._terminal.set()
                _release_closed_owner(self)
            return True
        if self._thread is threading.current_thread():
            return False
        if self._thread.ident is None and not self._ready.is_set():
            with self._lock:
                if self._state in (
                    DecodeOwnerState.CLOSED,
                    DecodeOwnerState.FAILED,
                ):
                    return bool(
                        self._terminal.is_set()
                        and not self._cleanup_error_type
                    )
                self._state = DecodeOwnerState.RETAINED
                _retain_live_owner(self)
            return False
        if self._thread.is_alive():
            self._thread.join(max(0.0, deadline - time.monotonic()))
        clean = bool(
            not self._thread.is_alive()
            and self._terminal.is_set()
            and self._session.closed
            and not self._cleanup_error_type
        )
        if not clean:
            with self._lock:
                if self._state not in (
                    DecodeOwnerState.CLOSED,
                    DecodeOwnerState.FAILED,
                ):
                    self._state = DecodeOwnerState.RETAINED
                    _retain_live_owner(self)
        return clean

    # Recognizer facade. Every method below remains synchronous; the owner
    # thread runs the complete result-dependent state machine around it.
    def create_stream(
        self,
        *,
        hotwords: Optional[str] = None,
        role: DecodeStreamRole = DecodeStreamRole.PRIMARY,
    ) -> SherpaStreamingDecodeStream:
        if not isinstance(role, DecodeStreamRole):
            raise TypeError("role must be a DecodeStreamRole")
        with self._lock:
            if role in self._active_by_role:
                raise RuntimeError(f"decode stream role {role.value!r} is already live")
        handle = self._session.create_stream(hotwords=hotwords)
        with self._lock:
            key = self._new_key_locked(role)
            self._streams[id(handle)] = _OwnedStream(handle, key)
            self._active_by_role[role] = key
            self._streams_created += 1
        return handle

    def is_ready(self, stream: SherpaStreamingDecodeStream) -> bool:
        self._require_stream(stream)
        return self._session.is_ready(stream)

    def decode_stream(self, stream: SherpaStreamingDecodeStream) -> None:
        self._require_stream(stream)
        self._session.decode_stream(stream)

    def get_result(self, stream: SherpaStreamingDecodeStream) -> str:
        self._require_stream(stream)
        return self._session.get_result(stream)

    def is_endpoint(self, stream: SherpaStreamingDecodeStream) -> bool:
        self._require_stream(stream)
        return self._session.is_endpoint(stream)

    def reset(self, stream: SherpaStreamingDecodeStream) -> None:
        owned = self._require_stream(stream)
        self._session.reset(stream)
        with self._lock:
            current = self._streams.get(id(stream))
            if current is not owned:
                raise RuntimeError("stream changed during owner reset")
            key = self._new_key_locked(owned.key.role)
            owned.key = key
            self._active_by_role[owned.key.role] = key
            self._streams_reset += 1

    def retire_stream(self, stream: SherpaStreamingDecodeStream) -> None:
        owned = self._require_stream(stream)
        self._session.retire_stream(stream)
        with self._lock:
            current = self._streams.pop(id(stream), None)
            if current is not owned:
                raise RuntimeError("stream changed during owner retirement")
            if self._active_by_role.get(owned.key.role) == owned.key:
                del self._active_by_role[owned.key.role]
            self._streams_retired += 1

    def stream_key(self, stream: SherpaStreamingDecodeStream) -> DecodeStreamKey:
        return self._require_stream(stream).key

    def rotate_capture_scope(
        self,
        capture_scope: CaptureScope,
    ) -> DecodeIdentity:
        """Retire every role and adopt an exact new raw-capture scope."""

        if not isinstance(capture_scope, CaptureScope):
            raise TypeError("capture_scope must be a CaptureScope")
        self._require_owner_thread()
        with self._lock:
            if capture_scope == self._identity.capture_scope:
                raise ValueError(
                    "unchanged capture scope requires decoder-continuity rotation"
                )
            if capture_scope in self._seen_capture_scopes:
                raise ValueError("capture scope cannot be reused within one run")
            self._identity = DecodeIdentity(
                self._identity.run_id,
                capture_scope,
                0,
            )
            self._seen_capture_scopes.add(capture_scope)
            self._capture_rotations += 1
            identity = self._identity
        # Publish the callback fence before native retirement can block. No
        # further native operation can race because this is the exact owner.
        self._retire_all_streams()
        return identity

    def rotate_decode_continuity(
        self,
        reason: DecodeLossReason = DecodeLossReason.EXPLICIT_DISCONTINUITY,
    ) -> DecodeIdentity:
        """Retire every role without changing capture lineage."""

        if reason is DecodeLossReason.CAPTURE_SCOPE:
            raise ValueError("capture-scope rotation requires an exact new scope")
        if not isinstance(reason, DecodeLossReason):
            raise TypeError("reason must be a DecodeLossReason")
        self._require_owner_thread()
        with self._lock:
            prior = self._identity
            self._identity = DecodeIdentity(
                prior.run_id,
                prior.capture_scope,
                prior.continuity_generation + 1,
            )
            self._decode_rotations += 1
            identity = self._identity
        self._retire_all_streams()
        return identity

    def snapshot(self) -> DecodeOwnerSnapshot:
        with self._lock:
            fields = (
                self._identity,
                self._state,
                self._ready.is_set(),
                self._terminal.is_set(),
                self._thread.is_alive(),
                self._active_by_role.get(DecodeStreamRole.PRIMARY),
                self._active_by_role.get(DecodeStreamRole.WORD_CUT),
                self._streams_created,
                self._streams_reset,
                self._streams_retired,
                self._capture_rotations,
                self._decode_rotations,
                self._processor_error_type,
                self._cleanup_error_type,
            )
        # The native session has its own lock. Never hold the owner lock while
        # acquiring it: owner operations intentionally call session first and
        # then publish their metadata under the owner lock.
        session_closed = self._session.closed
        return DecodeOwnerSnapshot(
            identity=fields[0],
            state=fields[1],
            ready=fields[2],
            terminal=fields[3],
            thread_alive=fields[4],
            session_closed=session_closed,
            primary_stream=fields[5],
            word_cut_stream=fields[6],
            streams_created=fields[7],
            streams_reset=fields[8],
            streams_retired=fields[9],
            capture_rotations=fields[10],
            decode_rotations=fields[11],
            processor_error_type=fields[12],
            cleanup_error_type=fields[13],
        )

    def _run(self) -> None:
        bound = False
        try:
            self._session.bind_current_thread()
            bound = True
            with self._lock:
                self._owner_thread = threading.current_thread()
            self._processor(self)
        except BaseException as exc:
            with self._lock:
                self._processor_error_type = type(exc).__name__
        finally:
            self._startup_complete.set()
            with self._lock:
                # Publish that the processor is no longer active before native
                # stream/recognizer destruction can block or re-enter metadata.
                if self._state not in (
                    DecodeOwnerState.RETAINED,
                    DecodeOwnerState.CLOSED,
                    DecodeOwnerState.FAILED,
                ):
                    self._state = DecodeOwnerState.CLOSING
            if bound:
                try:
                    self._retire_all_streams()
                except BaseException as exc:
                    with self._lock:
                        self._cleanup_error_type = type(exc).__name__
                try:
                    self._session.close()
                except BaseException as exc:
                    with self._lock:
                        if not self._cleanup_error_type:
                            self._cleanup_error_type = type(exc).__name__
            else:
                with self._lock:
                    self._cleanup_error_type = "OwnershipTransferError"
            clean = bool(self._session.closed and not self._cleanup_error_type)
            with self._lock:
                if clean:
                    self._state = (
                        DecodeOwnerState.FAILED
                        if self._processor_error_type
                        else DecodeOwnerState.CLOSED
                    )
                    _release_closed_owner(self)
                else:
                    self._state = DecodeOwnerState.RETAINED
                    _retain_live_owner(self)
                # State and terminal publication are one monotonic observation.
                # A zero-timeout close must never see CLOSED without terminal
                # and demote an already-clean owner to RETAINED.
                self._terminal.set()

    def _require_stream(
        self,
        stream: SherpaStreamingDecodeStream,
    ) -> _OwnedStream:
        with self._lock:
            owned = self._streams.get(id(stream))
            if owned is None or owned.handle is not stream:
                raise RuntimeError("stream is not live in this decode owner")
            return owned

    def _require_owner_thread(self) -> None:
        caller = threading.current_thread()
        with self._lock:
            if self._owner_thread is not caller:
                raise RuntimeError("decode owner operation belongs to another thread")

    def _new_key_locked(self, role: DecodeStreamRole) -> DecodeStreamKey:
        key = DecodeStreamKey(role, self._next_incarnation)
        self._next_incarnation += 1
        return key

    def _retire_all_streams(self) -> None:
        with self._lock:
            handles = tuple(owned.handle for owned in self._streams.values())
        first_error: Optional[BaseException] = None
        for handle in handles:
            try:
                self.retire_stream(handle)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    @staticmethod
    def _deadline(timeout: float, *, positive: bool = False) -> float:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) < (1e-300 if positive else 0.0)
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"timeout must be finite and {qualifier}")
        return time.monotonic() + float(timeout)


__all__ = (
    "DecodeIdentity",
    "DecodeLossReason",
    "DecodeOwnerSnapshot",
    "DecodeOwnerState",
    "DecodeStreamKey",
    "DecodeStreamRole",
    "SherpaStreamingDecodeOwner",
)
