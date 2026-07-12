"""Thread-safe playout ledger for receipt-capable audio engines.

The model and event bus can enqueue sentence fragments much faster than a
speaker can play them.  This ledger keeps those *attempts* separate from text
an engine has attested reached its output sink.  It deliberately knows nothing
about tasks or memory implementations: the runtime registers fragments, closes
their producer group, and consumes the resulting :class:`PlaybackCommit`.

Partial words are never inferred from sample ratios.  Only an engine-provided
``safe_text_prefix`` is eligible for conversational history.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .engine import PlaybackOutcome, PlaybackReceipt


@dataclass(frozen=True)
class PlaybackCommit:
    """One role-ordered conversation item released by playback sequencing."""

    role: str
    text: str
    is_followup: bool
    schedule_followup: bool
    epoch: int
    input_generation: int
    followup_generation: int


@dataclass(frozen=True)
class PlaybackResolution:
    """The accepted portion of one terminal receipt."""

    fragment_id: str
    requested_text: str
    safe_text_prefix: str
    played: bool
    commits: tuple[PlaybackCommit, ...] = ()


@dataclass(frozen=True)
class PlaybackContext:
    """Stable task/input identity for one tracked playback group."""

    task_id: str
    epoch: int
    input_generation: int
    remember: bool


@dataclass(frozen=True)
class PlaybackFinalization:
    """One finalized group's causal identity and aggregate sink outcome."""

    context: PlaybackContext
    outcome: PlaybackOutcome


@dataclass
class _Fragment:
    fragment_id: str
    requested_text: str
    order: int
    group_id: int
    started: bool = False
    terminal: bool = False
    receipt: Optional[PlaybackReceipt] = None


@dataclass
class _Group:
    group_id: int
    task_id: str
    epoch: int
    input_generation: int
    followup_generation: int
    remember: bool
    is_followup: bool
    streaming: bool
    source_closed: bool
    interrupted: bool = False
    fragments: list[str] = field(default_factory=list)


class PlaybackHistory:
    """Own fragment/group lifecycle until playout truth is terminal.

    All methods are safe from engine worker callbacks.  Unknown and duplicate
    callbacks are ignored; an engine bug therefore cannot duplicate memory or
    re-arm a follow-up.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_fragment = 0
        self._next_group = 0
        self._fragments: dict[str, _Fragment] = {}
        self._groups: dict[int, _Group] = {}
        self._stream_groups: dict[tuple[str, int], int] = {}
        self._stream_metadata: dict[tuple[str, int], bool] = {}
        # Conversation commits are released in group-registration order even
        # when terminal callbacks arrive out of order.  Without this, a delayed
        # receipt from an interrupted old reply could be appended *after* a
        # newer answer and reverse recent-conversation chronology.
        self._remember_order: deque[int] = deque()
        self._ready_commits: dict[int, Optional[PlaybackCommit]] = {}
        self._finalizations: deque[PlaybackFinalization] = deque()

    @property
    def pending(self) -> bool:
        with self._lock:
            return bool(self._groups)

    @property
    def conversation_pending(self) -> bool:
        """Whether an unresolved group can still contribute assistant history."""

        with self._lock:
            return any(group.remember for group in self._groups.values())

    @property
    def pending_fragments(self) -> int:
        with self._lock:
            return sum(not f.terminal for f in self._fragments.values())

    def register(
        self,
        *,
        task_id: str,
        epoch: int,
        input_generation: int,
        followup_generation: int,
        text: str,
        remember: bool,
        is_followup: bool,
        streaming: bool,
    ) -> str:
        """Register before calling ``engine.speak_tracked`` and return its ID."""

        with self._lock:
            key = (task_id, int(epoch))
            group_id = self._stream_groups.get(key) if streaming else None
            if group_id is None:
                self._next_group += 1
                group_id = self._next_group
                group = _Group(
                    group_id=group_id,
                    task_id=task_id,
                    epoch=int(epoch),
                    input_generation=int(input_generation),
                    followup_generation=int(followup_generation),
                    remember=bool(remember),
                    is_followup=(
                        self._stream_metadata.get(key, bool(is_followup))
                        if streaming
                        else bool(is_followup)
                    ),
                    streaming=bool(streaming),
                    source_closed=not streaming,
                )
                self._groups[group_id] = group
                if remember:
                    self._remember_order.append(group_id)
                if streaming:
                    self._stream_groups[key] = group_id
            else:
                group = self._groups[group_id]
                # A later lifecycle event may have supplied follow-up metadata
                # after the first sentence was emitted.
                group.is_followup = self._stream_metadata.get(
                    key, group.is_followup or bool(is_followup)
                )

            self._next_fragment += 1
            fragment_id = f"playback-{self._next_fragment}"
            fragment = _Fragment(
                fragment_id=fragment_id,
                requested_text=text,
                order=self._next_fragment,
                group_id=group_id,
            )
            self._fragments[fragment_id] = fragment
            group.fragments.append(fragment_id)
            return fragment_id

    def note_stream_metadata(
        self, task_id: str, epoch: int, *, is_followup: bool
    ) -> None:
        """Attach completion metadata without closing ahead of queued TTS events."""

        with self._lock:
            key = (task_id, int(epoch))
            self._stream_metadata[key] = bool(is_followup)
            group_id = self._stream_groups.get(key)
            if group_id is not None and group_id in self._groups:
                self._groups[group_id].is_followup = bool(is_followup)

    def stage_user(
        self,
        text: str,
        *,
        epoch: int,
        input_generation: int,
        followup_generation: int,
    ) -> tuple[PlaybackCommit, ...]:
        """Place a user turn between older and later assistant groups.

        The marker is ready immediately but the shared order queue holds it
        behind any earlier unresolved assistant receipt. Assistant groups
        registered after this call are appended after the user marker.
        """

        cleaned = (text or "").strip()
        if not cleaned:
            return ()
        with self._lock:
            self._next_group += 1
            sequence_id = self._next_group
            self._remember_order.append(sequence_id)
            self._ready_commits[sequence_id] = PlaybackCommit(
                role="user",
                text=cleaned,
                is_followup=False,
                schedule_followup=False,
                epoch=int(epoch),
                input_generation=int(input_generation),
                followup_generation=int(followup_generation),
            )
            return self._drain_ready_locked()

    def mark_started(self, fragment_id: str) -> Optional[str]:
        """Record exact sink onset; return attempted text on the first signal."""

        with self._lock:
            fragment = self._fragments.get(fragment_id)
            if fragment is None or fragment.terminal or fragment.started:
                return None
            fragment.started = True
            return fragment.requested_text

    def fragment_context(self, fragment_id: str) -> Optional[PlaybackContext]:
        with self._lock:
            fragment = self._fragments.get(fragment_id)
            if fragment is None:
                return None
            group = self._groups.get(fragment.group_id)
            if group is None:
                return None
            return PlaybackContext(
                task_id=group.task_id,
                epoch=group.epoch,
                input_generation=group.input_generation,
                remember=group.remember,
            )

    def stream_context(self, task_id: str, epoch: int) -> Optional[PlaybackContext]:
        with self._lock:
            group_id = self._stream_groups.get((task_id, int(epoch)))
            group = self._groups.get(group_id) if group_id is not None else None
            if group is None:
                return None
            return PlaybackContext(
                task_id=group.task_id,
                epoch=group.epoch,
                input_generation=group.input_generation,
                remember=group.remember,
            )

    def drain_finalizations(self) -> tuple[PlaybackFinalization, ...]:
        """Consume group-terminal attestations in finalization order."""

        with self._lock:
            finalizations = tuple(self._finalizations)
            self._finalizations.clear()
            return finalizations

    def resolve(self, receipt: PlaybackReceipt) -> Optional[PlaybackResolution]:
        """Accept one terminal receipt and finalize its group when possible."""

        with self._lock:
            fragment = self._fragments.get(receipt.fragment_id)
            if fragment is None or fragment.terminal:
                return None
            fragment.terminal = True
            fragment.receipt = receipt
            safe = (receipt.safe_text_prefix or "").strip()
            played_samples = receipt.played_samples
            played = bool(
                fragment.started
                or safe
                or (played_samples is not None and played_samples > 0)
            )
            commits = self._finalize_ready_locked(fragment.group_id)
            return PlaybackResolution(
                fragment_id=fragment.fragment_id,
                requested_text=fragment.requested_text,
                safe_text_prefix=safe,
                played=played,
                commits=commits,
            )

    def close_stream(
        self,
        task_id: str,
        epoch: int,
        *,
        interrupted: bool = False,
    ) -> tuple[PlaybackCommit, ...]:
        """Close after ``TTS_STREAM_END`` (or cancellation) without inventing audio."""

        with self._lock:
            key = (task_id, int(epoch))
            group_id = self._stream_groups.get(key)
            self._stream_metadata.pop(key, None)
            if group_id is None or group_id not in self._groups:
                return ()
            group = self._groups[group_id]
            group.source_closed = True
            group.interrupted = group.interrupted or bool(interrupted)
            return self._finalize_ready_locked(group_id)

    def close_task(
        self, task_id: str, *, interrupted: bool = True
    ) -> tuple[PlaybackCommit, ...]:
        """Close every open stream group owned by a task."""

        with self._lock:
            commits: list[PlaybackCommit] = []
            group_ids = [
                group_id
                for group_id, group in self._groups.items()
                if group.task_id == task_id and group.streaming
            ]
            for group_id in group_ids:
                group = self._groups.get(group_id)
                if group is None:
                    continue
                group.source_closed = True
                group.interrupted = group.interrupted or bool(interrupted)
                commits.extend(self._finalize_ready_locked(group_id))
            return tuple(commits)

    def interrupt_all(self) -> tuple[PlaybackCommit, ...]:
        """Close all producers; unresolved fragments still await terminal receipts."""

        with self._lock:
            commits: list[PlaybackCommit] = []
            for group_id in tuple(self._groups):
                group = self._groups.get(group_id)
                if group is None:
                    continue
                group.source_closed = True
                group.interrupted = True
                commits.extend(self._finalize_ready_locked(group_id))
            self._stream_metadata.clear()
            return tuple(commits)

    def _finalize_ready_locked(
        self, group_id: int
    ) -> tuple[PlaybackCommit, ...]:
        group = self._groups.get(group_id)
        if group is None or not group.source_closed:
            return ()
        fragments = [self._fragments[fid] for fid in group.fragments]
        if any(not fragment.terminal for fragment in fragments):
            return ()

        safe_parts = [
            fragment.receipt.safe_text_prefix.strip()
            for fragment in fragments
            if fragment.receipt is not None
            and fragment.receipt.safe_text_prefix.strip()
        ]
        all_completed = bool(fragments) and all(
            fragment.receipt is not None
            and fragment.receipt.outcome == PlaybackOutcome.COMPLETED
            for fragment in fragments
        )
        outcomes = tuple(
            fragment.receipt.outcome
            for fragment in fragments
            if fragment.receipt is not None
        )
        outcome_priority = {
            PlaybackOutcome.COMPLETED: 0,
            PlaybackOutcome.DROPPED: 1,
            PlaybackOutcome.INTERRUPTED: 2,
            PlaybackOutcome.FAILED: 3,
        }
        aggregate_outcome = max(outcomes, key=outcome_priority.__getitem__)
        text = " ".join(safe_parts).strip()
        commit: Optional[PlaybackCommit] = None
        if group.remember and text:
            commit = PlaybackCommit(
                role="assistant",
                text=text,
                is_followup=group.is_followup,
                schedule_followup=(not group.interrupted and all_completed),
                epoch=group.epoch,
                input_generation=group.input_generation,
                followup_generation=group.followup_generation,
            )

        for fragment in fragments:
            self._fragments.pop(fragment.fragment_id, None)
        self._groups.pop(group_id, None)
        key = (group.task_id, group.epoch)
        if self._stream_groups.get(key) == group_id:
            self._stream_groups.pop(key, None)
        self._stream_metadata.pop(key, None)
        self._finalizations.append(
            PlaybackFinalization(
                context=PlaybackContext(
                    task_id=group.task_id,
                    epoch=group.epoch,
                    input_generation=group.input_generation,
                    remember=group.remember,
                ),
                outcome=aggregate_outcome,
            )
        )
        if not group.remember:
            return ()
        self._ready_commits[group_id] = commit
        return self._drain_ready_locked()

    def _drain_ready_locked(self) -> tuple[PlaybackCommit, ...]:
        commits: list[PlaybackCommit] = []
        while (
            self._remember_order
            and self._remember_order[0] in self._ready_commits
        ):
            ready_id = self._remember_order.popleft()
            ready = self._ready_commits.pop(ready_id)
            if ready is not None:
                commits.append(ready)
        return tuple(commits)
