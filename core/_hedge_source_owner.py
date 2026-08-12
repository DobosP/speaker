"""Exact per-source ownership for synchronous Hedge LLM work.

Hedge workers run arbitrary Python, network SDK, or native model code.  Python
cannot safely kill that work when a turn is retired, so a worker that has not
proved its return must continue to own its source key.  This module supplies a
small, injectable registry used by :class:`core.llm.HedgeLLM`:

* admission is nonblocking and per exact source key;
* the owner is published before ``Thread.start`` is attempted;
* an ambiguous start remains owned until the exact worker proves return;
* request payload aliases are cleared before return is published; and
* release requires cleared references, a dead and joined exact thread, and an
  identity-checked registry removal.

The synchronous direct lease is for local-only Hedge calls.  It starts no
thread and may be released only after stream close, ContextVar restoration,
and call-payload alias clearing all succeeded.  A blocked or failed close
therefore keeps the source unavailable until process exit.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
import queue
import threading
from typing import Callable, Mapping, Optional


@dataclass(frozen=True, slots=True)
class HedgeSourceKey:
    """Stable exact source identity within one registry."""

    namespace: str
    source: str

    def __post_init__(self) -> None:
        if type(self.namespace) is not str or not self.namespace:
            raise TypeError("namespace must be a nonempty exact string")
        if type(self.source) is not str or not self.source:
            raise TypeError("source must be a nonempty exact string")


FACTORY_LOCAL_MAIN_SOURCE = HedgeSourceKey("factory", "LOCAL_MAIN")
FACTORY_SINGLE_CLOUD_SOURCE = HedgeSourceKey("factory", "SINGLE_CLOUD")


def factory_named_cloud_source(preset_name: str) -> HedgeSourceKey:
    """Return the shared factory key for one exact configured preset name."""

    if type(preset_name) is not str or not preset_name:
        raise TypeError("preset_name must be a nonempty exact string")
    return HedgeSourceKey("factory.named_cloud", preset_name)


class HedgeSourceOwnerState(str, Enum):
    NEW = "new"
    STARTING = "starting"
    RUNNING = "running"
    RETURNED = "returned"
    RETAINED = "retained"
    CLOSED = "closed"


class HedgeSourceLeaseState(str, Enum):
    ACTIVE = "active"
    RETAINED = "retained"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class HedgeSourceOwnerSnapshot:
    key: HedgeSourceKey
    state: HedgeSourceOwnerState
    start_attempted: bool
    start_returned: bool
    worker_returned: bool
    references_cleared: bool
    thread_alive: bool
    reaped: bool
    error_type: str


@dataclass(frozen=True, slots=True)
class HedgeSourceLeaseSnapshot:
    key: HedgeSourceKey
    state: HedgeSourceLeaseState
    references_cleared: bool
    released: bool
    error_type: str


@dataclass(slots=True)
class HedgeSourceWork:
    """One worker's complete and deliberately finite reference payload."""

    client: object
    prompt: str
    system: Optional[str]
    images: object
    history: object
    output: "queue.Queue[tuple[str, str, object]]"
    stop: threading.Event
    turn_context: Mapping[str, object]
    context_var: object
    tag: str

    def __post_init__(self) -> None:
        if type(self.prompt) is not str:
            raise TypeError("prompt must be an exact string")
        if self.system is not None and type(self.system) is not str:
            raise TypeError("system must be an exact string or None")
        if type(self.tag) is not str or not self.tag:
            raise TypeError("tag must be a nonempty exact string")
        # Do not inspect ``client`` here: even attribute lookup can execute a
        # custom descriptor before registry publication.  The worker performs
        # the first provider access after it owns the key.
        if type(self.output) is not queue.Queue:
            raise TypeError("output must be an exact Queue")
        if type(self.stop) is not threading.Event:
            raise TypeError("stop must be an exact Event")
        if type(self.turn_context) is not dict:
            raise TypeError("turn_context must be an exact dict")
        if type(self.context_var) is not ContextVar:
            raise TypeError("context_var must be an exact ContextVar")


class HedgeSourceRegistry:
    """Exact keyed owner registry; callers never wait for a busy source."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owners: dict[HedgeSourceKey, object] = {}
        self._reservations: dict[HedgeSourceKey, object] = {}

    @staticmethod
    def _require_key(key: HedgeSourceKey) -> None:
        if type(key) is not HedgeSourceKey:
            raise TypeError("key must be an exact HedgeSourceKey")

    def current(self, key: HedgeSourceKey) -> object | None:
        self._require_key(key)
        with self._lock:
            return self._owners.get(key)

    def try_reserve(self, key: HedgeSourceKey) -> object | None:
        self._require_key(key)
        reservation = object()
        with self._lock:
            if key in self._owners or key in self._reservations:
                return None
            self._reservations[key] = reservation
            return reservation

    def publish(
        self,
        key: HedgeSourceKey,
        reservation: object,
        owner: object,
    ) -> bool:
        self._require_key(key)
        if type(owner) not in (HedgeSourceOwner, HedgeSourceLease):
            return False
        if owner._registry is not self or owner.key != key:  # noqa: SLF001
            return False
        with self._lock:
            if self._reservations.get(key) is not reservation or key in self._owners:
                return False
            del self._reservations[key]
            self._owners[key] = owner
            return True

    def cancel_reservation(self, key: HedgeSourceKey, reservation: object) -> bool:
        self._require_key(key)
        with self._lock:
            if self._reservations.get(key) is not reservation:
                return False
            del self._reservations[key]
            return True

    def release(self, key: HedgeSourceKey, owner: object) -> bool:
        self._require_key(key)
        with self._lock:
            if self._owners.get(key) is not owner:
                return False
            del self._owners[key]
            return True


FACTORY_HEDGE_SOURCE_REGISTRY = HedgeSourceRegistry()


def _thread_alive(thread: object) -> bool:
    try:
        alive = getattr(thread, "is_alive", None)
    except BaseException:
        return True
    if not callable(alive):
        return True
    try:
        return bool(alive())
    except BaseException:
        return True


def _cross_thread_cancel_safe(stream: object) -> bool:
    try:
        return bool(
            getattr(stream, "_cross_thread_cancel_safe", False) is True
            and callable(getattr(stream, "cancel", None))
        )
    except BaseException:
        return False


def _run_hedge_source_owner(owner: "HedgeSourceOwner") -> None:
    """Static thread target: the thread argument retains only its exact owner."""

    owner._run_worker()


class HedgeSourceOwner:
    """One exact asynchronous provider invocation and its lifecycle proof."""

    def __init__(
        self,
        key: HedgeSourceKey,
        work: HedgeSourceWork,
        *,
        registry: HedgeSourceRegistry,
        thread_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        if type(key) is not HedgeSourceKey:
            raise TypeError("key must be an exact HedgeSourceKey")
        if type(work) is not HedgeSourceWork:
            raise TypeError("work must be an exact HedgeSourceWork")
        if type(registry) is not HedgeSourceRegistry:
            raise TypeError("registry must be an exact HedgeSourceRegistry")
        if thread_factory is not None and not callable(thread_factory):
            raise TypeError("thread_factory must be callable or None")
        self.key = key
        self._registry = registry
        self._condition = threading.Condition()
        self._state = HedgeSourceOwnerState.NEW
        self._work: HedgeSourceWork | None = work
        self._stop = work.stop
        self._active_stream: object | None = None
        self._start_attempted = False
        self._start_returned = False
        self._worker_returned = False
        self._references_cleared = False
        self._reaping = False
        self._reaped = False
        self._launch_error_type = ""
        self._cleanup_error_type = ""
        self._thread_factory = (
            threading.Thread if thread_factory is None else thread_factory
        )
        self._thread_name = f"speaker-hedge-source-{work.tag}"
        self._thread: object | None = None

    def _install_thread(self) -> None:
        """Construct the thread only after this owner is registry-published."""

        with self._condition:
            if self._thread is not None or self._start_attempted or self._reaped:
                raise RuntimeError("hedge source thread was already installed")
            factory = self._thread_factory
            thread_name = self._thread_name
        thread = factory(
            target=_run_hedge_source_owner,
            args=(self,),
            name=thread_name,
            daemon=True,
        )
        with self._condition:
            if self._thread is not None or self._start_attempted or self._reaped:
                raise RuntimeError("hedge source thread installation raced")
            self._thread = thread
            self._thread_factory = None

    def _retain_install_failure(self, exc: BaseException) -> None:
        with self._condition:
            self._launch_error_type = type(exc).__name__
            self._state = HedgeSourceOwnerState.RETAINED
        self.request_stop()

    def start(self) -> None:
        """Start once; an escaping failure remains process-owned and ambiguous."""

        with self._condition:
            if self._state is not HedgeSourceOwnerState.NEW:
                raise RuntimeError("hedge source owner was already started")
            if self._thread is None:
                raise RuntimeError("hedge source thread was not installed")
            self._start_attempted = True
            self._state = HedgeSourceOwnerState.STARTING
            thread = self._thread
        try:
            thread.start()
        except BaseException as exc:
            with self._condition:
                self._launch_error_type = type(exc).__name__
                self._state = HedgeSourceOwnerState.RETAINED
            self.request_stop()
            raise
        with self._condition:
            self._start_returned = True
            self._condition.notify_all()

    def request_stop(self) -> None:
        """Signal the exact worker and invoke only an advertised safe cancel."""

        self._stop.set()
        with self._condition:
            stream = self._active_stream
        if stream is not None and _cross_thread_cancel_safe(stream):
            try:
                cancel = getattr(stream, "cancel", None)
                if callable(cancel):
                    cancel()
            except BaseException:
                pass

    def try_reap(self) -> bool:
        """Release only after exact return, alias clearing, death, and join(0)."""

        with self._condition:
            if self._reaped:
                return True
            if (
                self._reaping
                or not self._worker_returned
                or not self._references_cleared
                or self._cleanup_error_type
            ):
                return False
            self._reaping = True
            thread = self._thread
        if _thread_alive(thread):
            with self._condition:
                self._reaping = False
            return False
        try:
            join = getattr(thread, "join", None)
        except BaseException as exc:
            with self._condition:
                self._reaping = False
                self._cleanup_error_type = type(exc).__name__
                self._state = HedgeSourceOwnerState.RETAINED
            return False
        if not callable(join):
            with self._condition:
                self._reaping = False
                self._cleanup_error_type = "WorkerJoinUnavailable"
                self._state = HedgeSourceOwnerState.RETAINED
            return False
        try:
            join(0.0)
        except BaseException as exc:
            with self._condition:
                self._reaping = False
                self._cleanup_error_type = type(exc).__name__
                self._state = HedgeSourceOwnerState.RETAINED
            return False
        if _thread_alive(thread):
            with self._condition:
                self._reaping = False
            return False
        try:
            released = self._registry.release(self.key, self)
        except BaseException as exc:
            released = False
            error_type = type(exc).__name__
        else:
            error_type = "RegistryReleaseError"
        with self._condition:
            self._reaping = False
            if not released:
                self._cleanup_error_type = error_type
                self._state = HedgeSourceOwnerState.RETAINED
                return False
            self._reaped = True
            self._state = HedgeSourceOwnerState.CLOSED
            self._condition.notify_all()
            return True

    def wait_and_reap(self, timeout: float) -> bool:
        """Request stop, join for one bounded interval, then prove exact reap."""

        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be numeric")
        timeout = max(0.0, float(timeout))
        self.request_stop()
        with self._condition:
            if self._reaped:
                return True
            thread = self._thread
        if _thread_alive(thread):
            try:
                join = getattr(thread, "join", None)
            except BaseException as exc:
                with self._condition:
                    self._cleanup_error_type = type(exc).__name__
                    self._state = HedgeSourceOwnerState.RETAINED
                return False
            if callable(join):
                try:
                    join(timeout)
                except BaseException as exc:
                    with self._condition:
                        self._cleanup_error_type = type(exc).__name__
                        self._state = HedgeSourceOwnerState.RETAINED
                    return False
        return self.try_reap()

    def snapshot(self) -> HedgeSourceOwnerSnapshot:
        with self._condition:
            state = self._state
            start_attempted = self._start_attempted
            start_returned = self._start_returned
            worker_returned = self._worker_returned
            references_cleared = self._references_cleared
            reaped = self._reaped
            error_type = self._cleanup_error_type or self._launch_error_type
            thread = self._thread
        return HedgeSourceOwnerSnapshot(
            key=self.key,
            state=state,
            start_attempted=start_attempted,
            start_returned=start_returned,
            worker_returned=worker_returned,
            references_cleared=references_cleared,
            thread_alive=_thread_alive(thread),
            reaped=reaped,
            error_type=error_type,
        )

    def _take_work(self) -> HedgeSourceWork | None:
        with self._condition:
            if self._state is HedgeSourceOwnerState.STARTING:
                self._state = HedgeSourceOwnerState.RUNNING
            work = self._work
            self._work = None
            return work

    def _publish_stream(self, stream: object) -> None:
        if not _cross_thread_cancel_safe(stream):
            return
        with self._condition:
            self._active_stream = stream

    def _publish_returned(self, error_type: str) -> None:
        with self._condition:
            self._work = None
            self._active_stream = None
            self._references_cleared = True
            self._worker_returned = True
            if error_type:
                self._cleanup_error_type = error_type
                self._state = HedgeSourceOwnerState.RETAINED
            else:
                # A Thread.start exception is ambiguous, not permanently
                # terminal.  The exact published target's positive return
                # proves that transfer occurred and permits normal reaping.
                self._state = HedgeSourceOwnerState.RETURNED
            self._condition.notify_all()

    def _run_worker(self) -> None:
        work = self._take_work()
        client = prompt = system = images = history = output = stop = None
        turn_context = context_var = context_token = stream = kwargs = token = None
        closer = None
        cleanup_error_type = ""
        try:
            if work is None:
                return
            client = work.client
            prompt = work.prompt
            system = work.system
            images = work.images
            history = work.history
            output = work.output
            stop = work.stop
            turn_context = work.turn_context
            context_var = work.context_var
            tag = work.tag
            context_token = context_var.set(turn_context)
            if stop.is_set():
                output.put((tag, "done", None))
                return
            kwargs = {"system": system, "images": images}
            if history:
                kwargs["history"] = history
            stream = client.stream(prompt, **kwargs)
            self._publish_stream(stream)
            if stop.is_set():
                if _cross_thread_cancel_safe(stream):
                    stream.cancel()
                output.put((tag, "done", None))
                return
            for token in stream:
                if stop.is_set():
                    break
                output.put((tag, "tok", token))
            output.put((tag, "done", None))
        except Exception as exc:
            if output is not None:
                try:
                    output.put(
                        (
                            work.tag if work is not None else "source",
                            "err",
                            type(exc).__name__,
                        )
                    )
                except BaseException:
                    pass
        except BaseException as exc:
            if output is not None:
                try:
                    output.put(
                        (
                            work.tag if work is not None else "source",
                            "err",
                            type(exc).__name__,
                        )
                    )
                except BaseException:
                    pass
        finally:
            with self._condition:
                self._active_stream = None
            try:
                closer = getattr(stream, "close", None)
            except BaseException as exc:
                closer = None
                cleanup_error_type = type(exc).__name__
            if callable(closer):
                try:
                    closer()
                except BaseException as exc:
                    cleanup_error_type = cleanup_error_type or type(exc).__name__
            if context_token is not None and context_var is not None:
                try:
                    context_var.reset(context_token)
                except BaseException as exc:
                    cleanup_error_type = cleanup_error_type or type(exc).__name__
            # Clear every call-payload alias before publishing worker return.
            work = client = prompt = system = images = history = output = stop = None
            turn_context = context_var = context_token = stream = kwargs = token = None
            closer = None
            self._publish_returned(cleanup_error_type)


class HedgeSourceLease:
    """Synchronous same-key lease for a direct local-only stream."""

    def __init__(self, key: HedgeSourceKey, registry: HedgeSourceRegistry) -> None:
        if type(key) is not HedgeSourceKey:
            raise TypeError("key must be an exact HedgeSourceKey")
        if type(registry) is not HedgeSourceRegistry:
            raise TypeError("registry must be an exact HedgeSourceRegistry")
        self.key = key
        self._registry = registry
        self._lock = threading.Lock()
        self._state = HedgeSourceLeaseState.ACTIVE
        self._references_cleared = False
        self._released = False
        self._error_type = ""

    def finish(self, *, references_cleared: bool, error_type: str = "") -> bool:
        """Release after successful close/reset/alias clearing; else retain."""

        if type(references_cleared) is not bool:
            raise TypeError("references_cleared must be an exact bool")
        if type(error_type) is not str:
            raise TypeError("error_type must be an exact string")
        with self._lock:
            if self._released:
                return True
            if self._state is HedgeSourceLeaseState.RETAINED:
                return False
            self._references_cleared = references_cleared
            if error_type or not references_cleared:
                self._error_type = error_type or "ReferencesNotCleared"
                self._state = HedgeSourceLeaseState.RETAINED
                return False
        try:
            released = self._registry.release(self.key, self)
        except BaseException as exc:
            released = False
            release_error = type(exc).__name__
        else:
            release_error = "RegistryReleaseError"
        with self._lock:
            if not released:
                self._error_type = release_error
                self._state = HedgeSourceLeaseState.RETAINED
                return False
            self._released = True
            self._state = HedgeSourceLeaseState.CLOSED
            return True

    def snapshot(self) -> HedgeSourceLeaseSnapshot:
        with self._lock:
            return HedgeSourceLeaseSnapshot(
                key=self.key,
                state=self._state,
                references_cleared=self._references_cleared,
                released=self._released,
                error_type=self._error_type,
            )


def _reap_current(registry: HedgeSourceRegistry, key: HedgeSourceKey) -> bool:
    current = registry.current(key)
    if current is None:
        return True
    if type(current) is HedgeSourceOwner:
        return current.try_reap()
    return False


def try_claim_hedge_source_owner(
    key: HedgeSourceKey,
    work: HedgeSourceWork,
    *,
    registry: HedgeSourceRegistry,
    thread_factory: Optional[Callable[..., object]] = None,
) -> HedgeSourceOwner | None:
    """Publish one exact worker owner without waiting or starting it."""

    if not _reap_current(registry, key):
        return None
    reservation = registry.try_reserve(key)
    if reservation is None:
        return None
    try:
        candidate = HedgeSourceOwner(
            key,
            work,
            registry=registry,
            thread_factory=thread_factory,
        )
        if not registry.publish(key, reservation, candidate):
            raise RuntimeError("hedge source reservation was lost")
        try:
            candidate._install_thread()
        except BaseException as exc:
            candidate._retain_install_failure(exc)
            raise
        return candidate
    finally:
        registry.cancel_reservation(key, reservation)


def try_claim_hedge_source_lease(
    key: HedgeSourceKey,
    *,
    registry: HedgeSourceRegistry,
) -> HedgeSourceLease | None:
    """Claim a no-thread direct lease without waiting for the same source."""

    if not _reap_current(registry, key):
        return None
    reservation = registry.try_reserve(key)
    if reservation is None:
        return None
    try:
        candidate = HedgeSourceLease(key, registry)
        if not registry.publish(key, reservation, candidate):
            raise RuntimeError("hedge source reservation was lost")
        return candidate
    finally:
        registry.cancel_reservation(key, reservation)
