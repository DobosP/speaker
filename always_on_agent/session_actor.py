"""Bounded, device-neutral ownership for one conversational session.

Capture and recognition deliberately stay outside this module.  A device
endpoint may publish a typed final, but only this actor allocates the session
turn and cancellation identities used by tasks, mutating tools, TTS, and
playback.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Event, Lock
from typing import Mapping
import uuid


_MAX_ID_LENGTH = 160


def _opaque_id(
    name: str,
    value: str,
    *,
    max_length: int = _MAX_ID_LENGTH,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or len(value) > max_length:
        raise ValueError(f"{name} must contain 1-{max_length} characters")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ValueError(f"{name} cannot contain control characters")
    return value


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class TurnIdentity:
    """One recognizer revision admitted into a conversational session."""

    session_id: str
    turn_id: str
    revision: int

    def __post_init__(self) -> None:
        _opaque_id("session_id", self.session_id)
        _opaque_id("turn_id", self.turn_id)
        _non_negative_int("revision", self.revision)


@dataclass(frozen=True)
class TaskIdentity:
    """Required causal identity shared by task and speech work."""

    session_id: str
    turn_id: str
    revision: int
    cancel_generation: int

    def __post_init__(self) -> None:
        _opaque_id("session_id", self.session_id)
        _opaque_id("turn_id", self.turn_id)
        _non_negative_int("revision", self.revision)
        _non_negative_int("cancel_generation", self.cancel_generation)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TaskIdentity":
        """Parse the four required fields; missing identity fails closed."""

        try:
            return cls(
                session_id=payload["session_id"],  # type: ignore[arg-type]
                turn_id=payload["turn_id"],  # type: ignore[arg-type]
                revision=payload["revision"],  # type: ignore[arg-type]
                cancel_generation=payload["cancel_generation"],  # type: ignore[arg-type]
            )
        except KeyError as exc:
            raise ValueError(f"missing session identity field: {exc.args[0]}") from exc

    def to_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "revision": self.revision,
            "cancel_generation": self.cancel_generation,
        }


@dataclass(frozen=True)
class ToolAdmission:
    """One explicitly cancellable mutating tool call."""

    task_id: str
    tool_call_id: str
    idempotency_key: str
    identity: TaskIdentity
    cancel_event: Event


@dataclass(frozen=True)
class PlaybackAdmission:
    """One bounded handoff from accepted TTS text to an output sink."""

    playback_admission_id: str
    identity: TaskIdentity


@dataclass(frozen=True)
class SessionSnapshot:
    """Small diagnostic snapshot with no transcript or user data."""

    session_id: str
    cancel_generation: int
    active_tasks: int
    active_tools: int
    active_playbacks: int
    remembered_tasks: int
    remembered_idempotency_keys: int


class AdmissionRejected(RuntimeError):
    """A bounded or stale work item was not admitted."""


@dataclass
class _ToolState:
    admission: ToolAdmission
    started: bool = False


@dataclass(frozen=True)
class _PlaybackState:
    task_id: str
    identity: TaskIdentity


class SessionActor:
    """Single lock-protected owner of post-recognition session work.

    The actor performs only bounded in-memory transitions.  It never waits for
    audio, models, providers, or playback callbacks, so capture loops can remain
    thin producers.
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        max_active_tasks: int = 6,
        max_active_tools: int = 6,
        max_active_playbacks: int = 64,
        max_remembered_tasks: int = 2048,
        max_idempotency_keys: int = 2048,
    ) -> None:
        self.session_id = _opaque_id(
            "session_id", session_id or f"session-{uuid.uuid4().hex}"
        )
        self._max_active_tasks = max(1, int(max_active_tasks))
        self._max_active_tools = max(1, int(max_active_tools))
        self._max_active_playbacks = max(1, int(max_active_playbacks))
        self._max_remembered_tasks = max(1, int(max_remembered_tasks))
        self._max_idempotency_keys = max(1, int(max_idempotency_keys))
        self._lock = Lock()
        self._next_turn = 0
        self._next_playback = 0
        self._cancel_generation = 0
        self._shutdown = False
        self._task_identities: dict[str, TaskIdentity] = {}
        self._task_order: deque[str] = deque()
        self._retired_task_bindings: set[str] = set()
        self._active_tasks: dict[str, TaskIdentity] = {}
        self._active_tools: dict[str, _ToolState] = {}
        self._active_playbacks: dict[str, _PlaybackState] = {}
        self._idempotency_calls: dict[str, str] = {}
        self._idempotency_order: deque[str] = deque()

    @property
    def cancel_generation(self) -> int:
        with self._lock:
            return self._cancel_generation

    @property
    def active_task_count(self) -> int:
        with self._lock:
            return len(self._active_tasks)

    @property
    def active_tool_count(self) -> int:
        with self._lock:
            return len(self._active_tools)

    @property
    def active_playback_count(self) -> int:
        with self._lock:
            return len(self._active_playbacks)

    def open_turn(
        self,
        *,
        revision: int,
        turn_id: str | None = None,
    ) -> TurnIdentity:
        """Allocate one session-local turn after a final wins ingress."""

        revision = _non_negative_int("revision", revision)
        with self._lock:
            if self._shutdown:
                raise AdmissionRejected("session is shutting down")
            if turn_id is None:
                self._next_turn += 1
                turn_id = f"turn-{self._next_turn}"
            else:
                _opaque_id("turn_id", turn_id)
            return TurnIdentity(self.session_id, turn_id, revision)

    def bind_task(self, task_id: str, turn: TurnIdentity) -> TaskIdentity:
        """Bind a task once and capture the current speech-cancel generation."""

        task_id = _opaque_id("task_id", task_id)
        self._validate_turn(turn)
        with self._lock:
            if self._shutdown:
                raise AdmissionRejected("session is shutting down")
            existing = self._task_identities.get(task_id)
            if existing is not None:
                if task_id in self._retired_task_bindings:
                    raise AdmissionRejected(
                        f"task_id {task_id!r} binding is retired"
                    )
                if (
                    existing.session_id != turn.session_id
                    or existing.turn_id != turn.turn_id
                    or existing.revision != turn.revision
                ):
                    raise AdmissionRejected(
                        f"task_id {task_id!r} is already bound to another turn"
                    )
                return existing
            self._make_task_room_locked()
            identity = TaskIdentity(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                revision=turn.revision,
                cancel_generation=self._cancel_generation,
            )
            self._task_identities[task_id] = identity
            self._task_order.append(task_id)
            return identity

    def identity_for_task(self, task_id: str) -> TaskIdentity | None:
        """Return only a binding that remains eligible for new actor work."""

        with self._lock:
            if task_id in self._retired_task_bindings:
                return None
            return self._task_identities.get(task_id)

    def admit_task(self, task_id: str, identity: TaskIdentity) -> bool:
        """Admit a bound task without exceeding the global task ceiling."""

        task_id = _opaque_id("task_id", task_id)
        with self._lock:
            if not self._identity_current_locked(identity):
                return False
            if task_id in self._retired_task_bindings:
                return False
            if self._task_identities.get(task_id) != identity:
                return False
            existing = self._active_tasks.get(task_id)
            if existing is not None:
                return False
            if len(self._active_tasks) >= self._max_active_tasks:
                return False
            self._active_tasks[task_id] = identity
            return True

    def finish_task(self, task_id: str) -> None:
        with self._lock:
            self._active_tasks.pop(task_id, None)
            self._reclaim_retired_task_bindings_locked()

    def task_is_admitted(self, task_id: str, identity: TaskIdentity) -> bool:
        with self._lock:
            return bool(
                not self._shutdown
                and task_id not in self._retired_task_bindings
                and self._active_tasks.get(task_id) == identity
            )

    def release_task_binding(self, task_id: str) -> bool:
        """Mark external task ownership terminal and reclaim when actor-safe."""

        task_id = _opaque_id("task_id", task_id)
        with self._lock:
            if task_id not in self._task_identities:
                return False
            self._retired_task_bindings.add(task_id)
            self._reclaim_retired_task_bindings_locked()
            return True

    def cancel_speech(self) -> int:
        """Invalidate speech and mutating calls that have not started providers.

        This lock is also the provider-start claim lock. Whichever transition
        wins is authoritative: cancellation sets the tool Event before a late
        claim, while a provider that already claimed start remains independent
        and requires explicit tool cancellation.
        """

        with self._lock:
            self._cancel_generation += 1
            for state in self._active_tools.values():
                if not state.started:
                    state.admission.cancel_event.set()
            return self._cancel_generation

    def shutdown(self) -> int:
        """Fence new work and request cancellation of every active tool.

        Returns the number of tool providers/admissions still owned by the
        actor. Their coordinators release ownership when the provider exits.
        """

        with self._lock:
            self._shutdown = True
            for state in self._active_tools.values():
                state.admission.cancel_event.set()
            return len(self._active_tools)

    def admit_playback(
        self,
        *,
        task_id: str,
        identity: TaskIdentity,
    ) -> PlaybackAdmission | None:
        """Reserve a sink handoff only for an actor-bound task identity."""

        task_id = _opaque_id("task_id", task_id)
        with self._lock:
            if not self._identity_current_locked(identity):
                return None
            if task_id in self._retired_task_bindings:
                return None
            if self._task_identities.get(task_id) != identity:
                return None
            if len(self._active_playbacks) >= self._max_active_playbacks:
                return None
            self._next_playback += 1
            playback_id = f"speech-{self._next_playback}"
            self._active_playbacks[playback_id] = _PlaybackState(
                task_id=task_id,
                identity=identity,
            )
            return PlaybackAdmission(playback_id, identity)

    def finish_playback(self, playback_admission_id: str) -> None:
        with self._lock:
            self._active_playbacks.pop(playback_admission_id, None)
            self._reclaim_retired_task_bindings_locked()

    def admit_tool(
        self,
        *,
        task_id: str,
        identity: TaskIdentity,
        tool_call_id: str,
        idempotency_key: str,
    ) -> ToolAdmission:
        """Reserve a mutating call once; duplicates never reach the provider."""

        task_id = _opaque_id("task_id", task_id)
        tool_call_id = _opaque_id("tool_call_id", tool_call_id)
        idempotency_key = _opaque_id(
            "idempotency_key", idempotency_key, max_length=512
        )
        with self._lock:
            if self._shutdown:
                raise AdmissionRejected("session is shutting down")
            if not self._identity_current_locked(identity):
                raise AdmissionRejected("tool call belongs to stale speech generation")
            if task_id in self._retired_task_bindings:
                raise AdmissionRejected("tool call belongs to a retired task binding")
            if self._task_identities.get(task_id) != identity:
                raise AdmissionRejected("tool call does not belong to a bound task")
            prior = self._idempotency_calls.get(idempotency_key)
            if prior is not None:
                raise AdmissionRejected(
                    f"idempotency key already admitted by {prior}"
                )
            if tool_call_id in self._active_tools:
                raise AdmissionRejected(f"tool_call_id already active: {tool_call_id}")
            if len(self._active_tools) >= self._max_active_tools:
                raise AdmissionRejected("mutating tool admission is at capacity")
            self._make_idempotency_room_locked()
            admission = ToolAdmission(
                task_id=task_id,
                tool_call_id=tool_call_id,
                idempotency_key=idempotency_key,
                identity=identity,
                cancel_event=Event(),
            )
            self._active_tools[tool_call_id] = _ToolState(admission)
            self._idempotency_calls[idempotency_key] = tool_call_id
            self._idempotency_order.append(idempotency_key)
            return admission

    def claim_tool_start(self, tool_call_id: str) -> bool:
        """Linearize a mutating provider start against explicit cancellation."""

        with self._lock:
            state = self._active_tools.get(tool_call_id)
            if (
                self._shutdown
                or state is None
                or state.started
                or state.admission.cancel_event.is_set()
            ):
                return False
            state.started = True
            return True

    def cancel_tool(self, tool_call_id: str) -> bool:
        """Explicitly cancel one call, including one whose provider has started."""

        with self._lock:
            state = self._active_tools.get(tool_call_id)
            if state is None:
                return False
            state.admission.cancel_event.set()
            return True

    def fence_task_tools(self, task_id: str) -> int:
        """Terminally fence one task from tools and cancel unstarted calls.

        Retirement and cancellation share the actor lock with ``admit_tool``
        and ``claim_tool_start``. Work paused before admission therefore cannot
        enter after this returns; an admitted start race either wins first and
        retains the provider-started policy, or observes its cancellation.
        """

        task_id = _opaque_id("task_id", task_id)
        with self._lock:
            if task_id in self._task_identities:
                self._retired_task_bindings.add(task_id)
            cancelled = 0
            for state in self._active_tools.values():
                if (
                    state.admission.task_id == task_id
                    and not state.started
                    and not state.admission.cancel_event.is_set()
                ):
                    state.admission.cancel_event.set()
                    cancelled += 1
            self._reclaim_retired_task_bindings_locked()
            return cancelled

    def finish_tool(self, tool_call_id: str) -> None:
        with self._lock:
            self._active_tools.pop(tool_call_id, None)
            self._reclaim_retired_task_bindings_locked()

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            return SessionSnapshot(
                session_id=self.session_id,
                cancel_generation=self._cancel_generation,
                active_tasks=len(self._active_tasks),
                active_tools=len(self._active_tools),
                active_playbacks=len(self._active_playbacks),
                remembered_tasks=len(self._task_identities),
                remembered_idempotency_keys=len(self._idempotency_calls),
            )

    def _validate_turn(self, turn: TurnIdentity) -> None:
        if turn.session_id != self.session_id:
            raise AdmissionRejected("turn belongs to another session")

    def _identity_current_locked(self, identity: TaskIdentity) -> bool:
        return bool(
            not self._shutdown
            and identity.session_id == self.session_id
            and identity.cancel_generation == self._cancel_generation
        )

    def _make_task_room_locked(self) -> None:
        self._reclaim_retired_task_bindings_locked()
        if len(self._task_identities) >= self._max_remembered_tasks:
            raise AdmissionRejected("remembered task identity table is at capacity")

    def _reclaim_retired_task_bindings_locked(self) -> None:
        """Reclaim only bindings whose controller explicitly released them."""

        active_playback_task_ids = {
            state.task_id for state in self._active_playbacks.values()
        }
        for task_id in tuple(self._task_order):
            if task_id not in self._retired_task_bindings:
                continue
            if (
                task_id in self._active_tasks
                or any(
                    state.admission.task_id == task_id
                    for state in self._active_tools.values()
                )
                or task_id in active_playback_task_ids
            ):
                continue
            self._task_order.remove(task_id)
            self._retired_task_bindings.discard(task_id)
            self._task_identities.pop(task_id, None)

    def _make_idempotency_room_locked(self) -> None:
        while len(self._idempotency_calls) >= self._max_idempotency_keys:
            victim = next(
                (
                    key
                    for key in self._idempotency_order
                    if self._idempotency_calls.get(key)
                    not in self._active_tools
                ),
                None,
            )
            if victim is None:
                raise AdmissionRejected("idempotency table is at capacity")
            self._idempotency_order.remove(victim)
            self._idempotency_calls.pop(victim, None)
