from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace
from threading import Lock, Timer
from typing import Callable, Mapping, Optional

log = logging.getLogger("speaker.supervisor")

# Per-mode wall-clock backstops (seconds) after which an active task is reaped as
# hung. Generous: a normal turn finishes far sooner; these only catch a
# capability that has blocked uninterruptibly (a hung generate / a network read
# with no timeout). RESEARCH/agent turns get the longest budget (web search +
# multi-step). 0 disables the deadline for a mode.
DEFAULT_TASK_TIMEOUTS: dict[str, float] = {
    "assistant": 25.0,
    "search": 30.0,
    "research": 120.0,
    "command": 30.0,
    "dictation": 30.0,
    "meeting": 30.0,
}
_FALLBACK_TASK_TIMEOUT = 60.0
_TOOL_SHUTDOWN_DRAIN_SEC = 0.25

# Spoken when a turn is reaped for exceeding its deadline, so a timed-out turn
# isn't just dead air.
_TIMEOUT_APOLOGY = "Sorry, that took too long -- let's try again."

# Spoken when a turn FAILS outright (capability/LLM error), so a failed turn
# isn't just dead air either (sr-2). Mirror in core/conversation.py's excluded
# replies so it isn't ingested as conversational content.
_FAILURE_APOLOGY = "Sorry, I ran into a problem with that -- let's try again."

_ACK_THEN_THINK_POLICY = "ack_then_think"
_STREAM_POLICIES = {"stream_main", "stream_research"}
_CLARIFY_POLICY = "clarify"
_CLARIFY_TEXT = "Could you say a bit more?"
_SILENT_INGEST_POLICY = "silent_ingest"
_ACK_THEN_THINK_TEXT = "I'll check that now."
_RESERVED_CONTINUATION_KEYS = (
    "reserved_continuation",
    "continuation_arrival_kind",
    "continuation_merged_text",
    "continuation_of",
    "continue_after",
    "continuation_origin",
    "continuation_addons",
    "continuation_record_origin",
    "continuation_unrecorded_addons",
    "continuation_victim_owner_verified",
    "continuation_victim_origin",
    "continuation_victim_metrics_turn_token",
    "skip_user_memory",
)

from .capabilities import CapabilityRegistry, create_default_capabilities
from .continuation import (
    CONTINUE,
    ContinuationClassifier,
    ContinuationConfig,
    HeuristicContinuationClassifier,
)
from .event_bus import EventBus
from .events import AgentEvent, EventKind, Mode
from .followups import FollowupConfig, FollowupState
from .memory import Memory, SessionMemory
from .models import IntentDecision, IntentKind, SpeechObservation
from .origin import Origin, combine
from .session_actor import (
    AdmissionRejected,
    PlaybackAdmission,
    SessionActor,
    TaskIdentity,
    TurnIdentity,
)
from .speech_analyzer import LiveSpeechAnalyzer
from .tasks import AgentTask, TaskRuntime


@dataclass
class SupervisorState:
    mode: Mode = Mode.PASSIVE
    last_partial: str = ""
    # transcript_log stays a plain list: it is SLICED (log[ln:]) by the
    # live_session / noise_stress drivers, and deque has no slicing. The other
    # high-churn histories are bounded deques (maxlen >> one turn's worth) so a
    # long-running process can't grow them without bound (rc-6/aq-7); only the
    # most recent N matter for snapshot/diagnostics/continuation.
    transcript_log: list[str] = field(default_factory=list)
    observations: "deque[SpeechObservation]" = field(default_factory=lambda: deque(maxlen=256))
    decisions: "deque[IntentDecision]" = field(default_factory=lambda: deque(maxlen=256))
    active_tasks: dict[str, AgentTask] = field(default_factory=dict)
    # Completed tasks whose first real answer TTS is queued but has not yet been
    # admitted to engine playback. They remain continuation/cancellation victims
    # until runtime TTS admission marks actual first audio.
    pending_audio_tasks: dict[str, AgentTask] = field(default_factory=dict)
    pending_aux_tts: dict[
        str, tuple[int, int | None, int | None, TaskIdentity, str]
    ] = field(default_factory=dict)
    queued_tasks: list[AgentTask] = field(default_factory=list)
    pending_confirmations: dict[str, AgentTask] = field(default_factory=dict)
    # Speaker-ID trust of the CURRENT turn's utterance (action chokepoint). Set per
    # final from the event payload; stamped onto each task created for the turn so
    # the capability layer can refuse a side-effecting action that isn't
    # owner-verified live audio. Default FAIL-CLOSED.
    turn_owner_verified: bool = False
    turn_origin: str = "unknown"
    turn_metadata: dict[str, object] = field(default_factory=dict)
    turn_identity: TurnIdentity | None = None
    spoken_outputs: "deque[str]" = field(default_factory=lambda: deque(maxlen=512))
    event_log: "deque[AgentEvent]" = field(default_factory=lambda: deque(maxlen=1024))
    failures: "deque[str]" = field(default_factory=lambda: deque(maxlen=128))


@dataclass(frozen=True)
class ArrivalContinuation:
    """Lineage reserved when an add-on arrives during an active answer.

    The runtime carries this value through cancellable final preprocessing.  It
    is deliberately data-only: memory and metrics effects happen only after the
    replacement final wins terminal ownership and starts its task.
    """

    victim_task_id: str
    origin: str
    addons: tuple[str, ...]
    recorded_addon_count: int
    merge_before_audio: bool
    record_origin: bool
    victim_owner_verified: bool
    victim_origin: str
    victim_metrics_turn_token: int | None
    awaiting_addon: bool = False


class AgentSupervisor:
    """High-level mode and task coordinator for always-on voice events."""

    def __init__(
        self,
        bus: EventBus | None = None,
        analyzer: LiveSpeechAnalyzer | None = None,
        capabilities: CapabilityRegistry | None = None,
        memory: Memory | None = None,
        stream_tts: bool = False,
        followup_config: FollowupConfig | None = None,
        continuation_config: ContinuationConfig | None = None,
        continuation: ContinuationClassifier | None = None,
        task_timeouts: Mapping[str, float] | None = None,
        confirmation_ttl_sec: float = 180.0,
        max_queued_tasks: int = 32,
        load_fraction: Optional[Callable[[], Optional[float]]] = None,
        admission_load_ceiling: float = 0.85,
        on_turn_merged: Optional[Callable[[Optional[int]], None]] = None,
        on_continuation_admitted: Optional[Callable[[int], None]] = None,
        on_input_resolved: Optional[Callable[[int], None]] = None,
        runtime_owns_stop: bool = False,
        defer_output_until_tts_admission: bool = False,
        defer_output_until_playback_receipt: bool = False,
        record_user_memory: Optional[Callable[[str], None]] = None,
        on_input_claimed: Optional[Callable[[int], None]] = None,
    ):
        self.bus = bus or EventBus()
        self.state = SupervisorState()
        self.memory = memory or SessionMemory()
        self.capabilities = capabilities or create_default_capabilities(self.memory)
        self.analyzer = analyzer or LiveSpeechAnalyzer()
        self.session_actor = SessionActor()
        self.tasks = TaskRuntime(
            self.publish,
            self.capabilities,
            stream_tts=stream_tts,
            session_actor=self.session_actor,
        )
        self.followups = followup_config or FollowupConfig()
        # ADD-ON / continuation: detect a follow-up that extends the in-flight
        # turn and merge it instead of spawning a competing cold task. Default
        # config is disabled, so a supervisor built without it behaves exactly as
        # before (existing tests stay byte-identical); the shipped config.json
        # opts in. The classifier is the cheap deterministic heuristic unless a
        # caller injects one (e.g. a scripted fake in tests, or a future LLM
        # upgrade). Built only when enabled so the disabled path allocates nothing.
        self._continuation_cfg = continuation_config or ContinuationConfig()
        self._continuation = continuation
        if self._continuation is None and self._continuation_cfg.enabled:
            self._continuation = HeuristicContinuationClassifier(self._continuation_cfg)
        # Per-mode wall-clock task deadlines (the never-stuck backstop). Defaults
        # are merged with any config override; a hung task past its deadline is
        # reaped by reap_overdue_tasks (driven off the watchdog tick).
        # Load-elastic admission (control-plane-2): a cheap () -> Optional[float]
        # system-load reader (core.sysinfo.SystemMonitor.load_fraction). Under
        # sustained load the concurrent-task ceiling tightens to 1 so a second
        # turn can't thrash an already-saturated CPU/GPU. None (default) -> inert,
        # so a supervisor built without it behaves exactly as before.
        self._load_fraction = load_fraction
        self._admission_load_ceiling = float(admission_load_ceiling)
        self._on_turn_merged = on_turn_merged
        self._on_continuation_admitted = on_continuation_admitted
        self._on_input_claimed = on_input_claimed
        self._on_input_resolved = on_input_resolved
        self._record_user_memory = record_user_memory
        self._runtime_owns_stop = bool(runtime_owns_stop)
        self._defer_output_until_tts_admission = bool(
            defer_output_until_tts_admission
        )
        # Receipt-capable runtimes commit assistant history only after the
        # output sink terminally attests what played.  Legacy runtimes leave
        # this off and retain admission-time behavior byte-for-byte.
        self._defer_output_until_playback_receipt = bool(
            defer_output_until_playback_receipt
        )
        self._task_timeouts = dict(DEFAULT_TASK_TIMEOUTS)
        if isinstance(task_timeouts, Mapping):
            for key, value in task_timeouts.items():
                try:
                    self._task_timeouts[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        # TTL for staged owner confirmations (never-stuck backstop, same family as
        # task_timeouts): an abandoned "Confirm command: ..." used to wait forever,
        # so a stray later "yes" could approve a long-forgotten action. 0 disables.
        try:
            self._confirmation_ttl_sec = max(0.0, float(confirmation_ttl_sec))
        except (TypeError, ValueError):
            self._confirmation_ttl_sec = 180.0
        # Bounded queued-task admission (backlog: unbounded queue under an input
        # storm). Every queue append funnels through _queue_task, which drops the
        # OLDEST droppable turn past this cap. Mirrors the bounded-deque histories
        # above and the playback queue's drop-oldest overflow (sherpa).
        self._max_queued_tasks = max(1, int(max_queued_tasks))
        self._queue_overflow_announced = False
        self._next_aux_tts = 0
        self._pending_output_bindings: dict[str, TaskIdentity] = {}
        self._retired_tts_tasks: set[tuple[str, int]] = set()
        self._retired_tts_order: deque[tuple[str, int]] = deque()
        # Runtime subscribers execute after the supervisor.  Preserve both the
        # exact actor binding resolved here (including for identity-less legacy
        # events) and whether TASK_COMPLETED is still outward-current.  A stale
        # completion remains authorized for interrupted cleanup only.
        self._accepted_terminal_events: deque[
            tuple[AgentEvent, TaskIdentity | None, bool]
        ] = deque(maxlen=2048)
        self._followup_state = FollowupState(
            markers=self.followups.markers, max_followups=self.followups.max_followups
        )
        self._followup_timer: Timer | None = None
        self._followup_lock = Lock()
        self._followup_shutdown = False
        # Ownership for timer callbacks and delayed playback commits.  A user
        # speech/reset advances generation; replacing/cancelling a timer advances
        # token.  Both identities ride FOLLOWUP_TICK so queued stale ticks drop.
        self._followup_generation = 0
        self._followup_token = 0
        # Set by shutdown(): a Timer callback that was already in flight when the
        # runtime stopped must not publish a late FOLLOWUP_TICK into a dead bus,
        # and nothing may arm a new timer afterwards (backlog: shutdown guard).
        self._stopped = False
        # Barge-in determinism (realtime-concurrency-1): cancellation can be
        # requested from the audio thread (runtime barge-in/stop) *and* the bus
        # thread (CONTROL_STOP). This lock makes the cancel atomic and guards
        # the monotonic speech epoch. It is held only around in-memory bookkeeping
        # -- never across an engine call, ``stop_speaking()``, or a thread join.
        self._cancel_lock = Lock()
        # Bumped on every cancel_all. A TTS_REQUEST stamped with an older epoch
        # (i.e. produced by an interrupted turn) must not reach ``engine.speak``.
        self.speech_epoch = self.session_actor.cancel_generation
        # Bumped only when external control (barge/stop/shutdown) invalidates
        # finals that may already be dequeued but have not registered a task.
        # Internal newest-turn/continuation cancellation still advances the
        # speech epoch, but must not discard later ordered input events.
        self.input_epoch = 0
        # Monotonic runtime-final identity. A later committed final invalidates
        # an older STT_FINAL that is queued/dequeued but has not started a task.
        self.latest_input_generation = 0
        # Advances at substantive final ARRIVAL, before its gates. This closes
        # the committed-publish -> bus-task-registration gap while the newer
        # final is still preprocessing.
        self.latest_arrival_generation = 0
        self.bus.subscribe(self.handle_event)

    def publish(self, event: AgentEvent) -> bool:
        # Keep shutdown quiescent: task/failure/timer workers may finish after
        # cancellation, but none may enqueue a new lifecycle or TTS event after
        # shutdown has committed. bus.publish is non-blocking and does not call
        # back into the supervisor, so it is safe inside this short lock.
        with self._cancel_lock:
            if self._stopped:
                return False
            return self.bus.publish(event)

    def has_live_tasks(self) -> bool:
        """Whether uncancelled active/queued work can still produce output."""
        with self._cancel_lock:
            speech_work = any(
                not task.cancel_event.is_set()
                for task in self.state.active_tasks.values()
            ) or any(
                not task.cancel_event.is_set()
                for task in self.state.pending_audio_tasks.values()
            ) or any(
                not task.cancel_event.is_set()
                for task in self.state.queued_tasks
            )
        return bool(speech_work or self.session_actor.active_tool_count)

    @property
    def followup_generation(self) -> int:
        with self._followup_lock:
            return self._followup_generation

    def looks_like_realtime_continuation(self, text: str) -> bool:
        """Audio-thread-safe continuation check for arrival-time preemption.

        Unknown/future classifiers are treated as non-realtime and therefore
        cannot block the engine callback; the shipped heuristic advertises the
        explicit safe marker.
        """
        if not getattr(self._continuation, "realtime_safe", False):
            return False
        return self.looks_like_continuation(text)

    def reserve_arrival_continuation(
        self,
        text: str,
    ) -> ArrivalContinuation | None:
        """Reserve a clear add-on and atomically silence an unheard victim.

        This runs on the engine callback while the runtime holds its terminal
        effect lock.  Only classifiers that explicitly advertise a bounded,
        local ``realtime_safe`` implementation are eligible.  The speech epoch
        bump on the pre-audio branch makes any sentence emitted by the victim in
        the narrow race after classification stale before TTS admission.  Once
        real answer audio has started, the victim remains live and the reserved
        contextual continuation will queue behind it after preprocessing.
        """
        if (
            self._continuation is None
            or not self._continuation_cfg.enabled
            or not getattr(self._continuation, "realtime_safe", False)
        ):
            return None
        with self._cancel_lock:
            if self._stopped:
                return None
            active = [
                task
                for task in self.state.active_tasks.values()
                if not task.cancel_event.is_set()
            ]
            active.extend(
                task
                for task_id, task in self.state.pending_audio_tasks.items()
                if (
                    task_id not in self.state.active_tasks
                    and not task.cancel_event.is_set()
                )
            )
            if len(active) != 1:
                return None
            victim = active[0]
            if (
                victim.mode != Mode.ASSISTANT
                or victim.metadata.get("post_barge_response_only")
            ):
                return None
            queued_continuation = (
                self._pending_continuation_behind(victim.task_id)
                if victim.started_speaking
                else None
            )
            lineage_task = queued_continuation or victim
            if lineage_task.metadata.get("post_barge_response_only"):
                return None
            try:
                continuation = (
                    self._continuation.classify(text, lineage_task.input_text)
                    == CONTINUE
                )
            except Exception:  # noqa: BLE001 - fail closed to normal preemption
                log.exception("arrival continuation classifier failed")
                return None
            if not continuation:
                return None

            origin, prior_addons = self._continuation_lineage(lineage_task)
            if queued_continuation is not None:
                if queued_continuation not in self.state.queued_tasks:
                    return None
                self.state.queued_tasks.remove(queued_continuation)
                queued_continuation.cancel()
                self._release_task_binding_if_unowned_locked(
                    queued_continuation.task_id,
                    queued_continuation.identity,
                )
                return ArrivalContinuation(
                    victim_task_id=victim.task_id,
                    origin=origin,
                    addons=(*prior_addons, text),
                    recorded_addon_count=len(prior_addons),
                    merge_before_audio=False,
                    record_origin=False,
                    victim_owner_verified=(
                        queued_continuation.metadata.get("owner_verified") is True
                    ),
                    victim_origin=str(
                        queued_continuation.metadata.get("origin", "unknown")
                    ),
                    victim_metrics_turn_token=(
                        int(token)
                        if (
                            token := queued_continuation.metadata.get(
                                "metrics_turn_token"
                            )
                        ) is not None
                        else None
                    ),
                    awaiting_addon=False,
                )
            # Revalidate the identity immediately before the linearization
            # point.  Every check and mutation is under the cancellation lock.
            if (
                self.state.active_tasks.get(victim.task_id) is not victim
                and self.state.pending_audio_tasks.get(victim.task_id) is not victim
            ) or victim.cancel_event.is_set():
                return None
            merge_before_audio = not victim.started_speaking
            if merge_before_audio:
                self._advance_output_epoch_locked()
                self.state.active_tasks.pop(victim.task_id, None)
                self.state.pending_audio_tasks.pop(victim.task_id, None)
                self._pending_output_bindings.pop(victim.task_id, None)
                victim.cancel()
                self._release_task_binding_if_unowned_locked(
                    victim.task_id,
                    victim.identity,
                )
            return ArrivalContinuation(
                victim_task_id=victim.task_id,
                origin=origin,
                addons=(*prior_addons, text),
                recorded_addon_count=len(prior_addons),
                merge_before_audio=merge_before_audio,
                record_origin=merge_before_audio and not prior_addons,
                victim_owner_verified=(victim.metadata.get("owner_verified") is True),
                victim_origin=str(victim.metadata.get("origin", "unknown")),
                victim_metrics_turn_token=(
                    int(token)
                    if (token := victim.metadata.get("metrics_turn_token"))
                    is not None
                    else None
                ),
                awaiting_addon=False,
            )

    def reserve_unheard_for_partial(self) -> ArrivalContinuation | None:
        """Silence every unheard task; reserve lineage only when unambiguous.

        A partial is enough to suppress queued/pre-token audio regardless of
        mode.  Continuation, however, is assistant-specific and needs exactly
        one victim, so multi-task/research cases are cancelled without guessing
        conversational lineage.
        """
        with self._cancel_lock:
            if self._stopped:
                return None
            candidates = [
                task
                for task in self.state.active_tasks.values()
                if not task.cancel_event.is_set() and not task.started_speaking
            ]
            candidates.extend(
                task
                for task_id, task in self.state.pending_audio_tasks.items()
                if (
                    task_id not in self.state.active_tasks
                    and not task.cancel_event.is_set()
                )
            )
            candidates.extend(
                task
                for task in self.state.queued_tasks
                if not task.cancel_event.is_set()
            )
            # Deduplicate tasks that may be visible during an atomic active ->
            # pending-audio move, preserving deterministic insertion order.
            unique = {task.task_id: task for task in candidates}
            candidates = list(unique.values())
            if not candidates:
                return None
            victim = candidates[0] if len(candidates) == 1 else None
            reservation: ArrivalContinuation | None = None
            if (
                victim is not None
                and victim.mode == Mode.ASSISTANT
                and not victim.metadata.get("post_barge_response_only")
            ):
                origin, prior_addons = self._continuation_lineage(victim)
                token = victim.metadata.get("metrics_turn_token")
                reservation = ArrivalContinuation(
                    victim_task_id=victim.task_id,
                    origin=origin,
                    addons=tuple(prior_addons),
                    recorded_addon_count=len(prior_addons),
                    merge_before_audio=True,
                    record_origin=not prior_addons,
                    victim_owner_verified=(
                        victim.metadata.get("owner_verified") is True
                    ),
                    victim_origin=str(victim.metadata.get("origin", "unknown")),
                    victim_metrics_turn_token=(
                        int(token) if token is not None else None
                    ),
                    awaiting_addon=True,
                )
            self._advance_output_epoch_locked()
            for task in candidates:
                self.state.active_tasks.pop(task.task_id, None)
                self.state.pending_audio_tasks.pop(task.task_id, None)
                self._pending_output_bindings.pop(task.task_id, None)
                task.cancel()
                self._retire_task_tts_locked(task.task_id, task.identity)
            self.state.queued_tasks = [
                task
                for task in self.state.queued_tasks
                if task.task_id not in unique
            ]
            for task in candidates:
                self._release_task_binding_if_unowned_locked(
                    task.task_id,
                    task.identity,
                )
            return reservation

    @property
    def continuation_enabled(self) -> bool:
        return bool(self._continuation is not None and self._continuation_cfg.enabled)

    def extend_arrival_continuation(
        self,
        reservation: ArrivalContinuation,
        text: str,
    ) -> ArrivalContinuation | None:
        """Extend a still-preprocessing reservation with one newer add-on."""
        if (
            self._continuation is None
            or not self._continuation_cfg.enabled
            or not getattr(self._continuation, "realtime_safe", False)
        ):
            return None
        previous = self._continuation_cfg.merge_template.format(
            prev=reservation.origin,
            addon=self._render_addons(list(reservation.addons)),
        )
        try:
            continuation = self._continuation.classify(text, previous) == CONTINUE
        except Exception:  # noqa: BLE001 - fail closed to a fresh turn
            log.exception("arrival continuation classifier failed")
            return None
        if not continuation:
            return None
        return ArrivalContinuation(
            victim_task_id=reservation.victim_task_id,
            origin=reservation.origin,
            addons=(*reservation.addons, text),
            recorded_addon_count=reservation.recorded_addon_count,
            merge_before_audio=reservation.merge_before_audio,
            record_origin=reservation.record_origin,
            victim_owner_verified=reservation.victim_owner_verified,
            victim_origin=reservation.victim_origin,
            victim_metrics_turn_token=reservation.victim_metrics_turn_token,
            awaiting_addon=False,
        )

    @staticmethod
    def coalesce_arrival_continuation(
        reservation: ArrivalContinuation,
        text: str,
    ) -> ArrivalContinuation:
        """Replace the newest raw fragment with the dispatcher's merged text."""
        addons = list(reservation.addons)
        if addons:
            addons[-1] = text
        else:
            addons.append(text)
        return ArrivalContinuation(
            victim_task_id=reservation.victim_task_id,
            origin=reservation.origin,
            addons=tuple(addons),
            recorded_addon_count=reservation.recorded_addon_count,
            merge_before_audio=reservation.merge_before_audio,
            record_origin=reservation.record_origin,
            victim_owner_verified=reservation.victim_owner_verified,
            victim_origin=reservation.victim_origin,
            victim_metrics_turn_token=reservation.victim_metrics_turn_token,
            awaiting_addon=reservation.awaiting_addon,
        )

    def materialize_arrival_continuation(
        self,
        reservation: ArrivalContinuation,
        addon: str,
    ) -> tuple[str, dict[str, object]]:
        """Build the replacement prompt and JSON-safe supervisor metadata."""
        addons = list(reservation.addons)
        if addons:
            # Only the newest add-on reached this winning cleaner pass.  Keep
            # earlier raw utterances verbatim and use the cleaned winning text.
            addons[-1] = addon
        else:  # defensive: reservations created above always contain one add-on
            addons.append(addon)
        template = (
            self._continuation_cfg.merge_template
            if reservation.merge_before_audio
            else self._continuation_cfg.continue_template
        )
        merged = template.format(
            prev=reservation.origin,
            addon=self._render_addons(addons),
        )
        metadata: dict[str, object] = {
            "reserved_continuation": True,
            "continuation_arrival_kind": (
                "merge" if reservation.merge_before_audio else "continue"
            ),
            "continuation_merged_text": merged,
            "continuation_of": reservation.victim_task_id,
            "continuation_origin": reservation.origin,
            "continuation_addons": addons,
            "continuation_record_origin": reservation.record_origin,
            "continuation_unrecorded_addons": addons[
                reservation.recorded_addon_count:
            ],
            "continuation_victim_owner_verified": (
                reservation.victim_owner_verified
            ),
            "continuation_victim_origin": reservation.victim_origin,
            "continuation_victim_metrics_turn_token": (
                reservation.victim_metrics_turn_token
            ),
            "skip_user_memory": True,
        }
        if not reservation.merge_before_audio:
            metadata["continue_after"] = reservation.victim_task_id
        return merged, metadata

    def should_preempt_for_final_arrival(self, text: str) -> bool:
        """Cheap arrival policy: newest input fences old output immediately."""
        with self._cancel_lock:
            if self.state.pending_confirmations:
                return False
            has_live = any(
                not task.cancel_event.is_set()
                for task in self.state.active_tasks.values()
            ) or any(
                not task.cancel_event.is_set()
                for task in self.state.pending_audio_tasks.values()
            ) or any(
                not task.cancel_event.is_set()
                for task in self.state.queued_tasks
            )
        if not has_live:
            return False
        return not self.looks_like_realtime_continuation(text)

    def commit_input_generation(self, generation: int) -> bool:
        """Publish the newest task-producing runtime final identity.

        Returns false for an out-of-order stale commit. The task-start path
        rechecks this identity, closing the publish-to-registration race.
        """
        generation = int(generation)
        with self._cancel_lock:
            if self._stopped:
                return False
            if (
                generation < self.latest_arrival_generation
                or generation < self.latest_input_generation
            ):
                return False
            self.latest_input_generation = generation
            return True

    def note_input_arrival(self, generation: int) -> bool:
        """Fence older published/dequeued finals before this final's gates."""
        generation = int(generation)
        with self._cancel_lock:
            if self._stopped or generation < self.latest_arrival_generation:
                return False
            self.latest_arrival_generation = generation
            return True

    def _input_metadata_current_locked(self, metadata: Mapping[str, object]) -> bool:
        if self._stopped:
            return False
        input_epoch = metadata.get("input_epoch")
        if input_epoch is not None and int(input_epoch) != self.input_epoch:
            return False
        generation = metadata.get("input_generation")
        if (
            generation is not None
            and (
                int(generation) != self.latest_input_generation
                or int(generation) != self.latest_arrival_generation
            )
        ):
            return False
        return True

    def handle_event(self, event: AgentEvent) -> None:
        # STT_PARTIAL is the highest-volume kind (one per ASR frame) and no
        # consumer reads partials back out of the log -- keep it out so the
        # bounded event_log isn't churned through by interim transcripts
        # (rc-6/aq-7). The STT_PARTIAL dispatch below still runs.
        if event.kind != EventKind.STT_PARTIAL:
            self.state.event_log.append(event)

        if event.kind == EventKind.MAILBOX_FAULT:
            self.state.failures.append(
                "event mailbox fault; runtime restart required"
            )
            # The mailbox discarded queued work when serialized delivery could
            # no longer be guaranteed. Retire every actor/task binding directly;
            # late worker publications are gated by _stopped.
            self.shutdown()
            return
        if event.kind == EventKind.STT_PARTIAL:
            self._handle_speech(str(event.payload.get("text", "")), is_final=False)
            return
        if event.kind == EventKind.STT_FINAL:
            metadata = event.payload.get("metadata")
            final_metadata = (
                dict(metadata) if isinstance(metadata, Mapping) else {}
            )
            # Revision is controller-owned acoustic identity, not caller
            # metadata. Untyped/legacy finals explicitly enter revision zero.
            raw_revision = event.payload.get("revision", 0)
            try:
                revision = int(raw_revision)
            except (TypeError, ValueError):
                log.warning("dropping final with invalid revision %r", raw_revision)
                return
            if isinstance(raw_revision, bool) or revision < 0:
                log.warning("dropping final with invalid revision %r", raw_revision)
                return
            final_metadata["revision"] = revision
            generation = (
                final_metadata.get("input_generation")
            )
            try:
                if self._on_input_claimed is not None and generation is not None:
                    self._on_input_claimed(int(generation))
                self._handle_speech(
                    str(event.payload.get("text", "")),
                    is_final=True,
                    owner_verified=(event.payload.get("owner_verified") is True),
                    origin=str(event.payload.get("origin", "unknown")),
                    metadata=final_metadata,
                )
            finally:
                if self._on_input_resolved is not None and generation is not None:
                    self._on_input_resolved(int(generation))
            return
        if event.kind == EventKind.FOLLOWUP_TICK:
            self._emit_followup(
                expected_speech_epoch=event.payload.get("speech_epoch"),
                expected_input_generation=event.payload.get("input_generation"),
                expected_followup_generation=event.payload.get(
                    "followup_generation"
                ),
                timer_token=event.payload.get("timer_token"),
            )
            return
        if event.kind == EventKind.MEMORY_COMMIT:
            if event.payload.get("source") == "playback_receipt":
                schedule_followup = bool(
                    event.payload.get("schedule_followup", False)
                )
                self.record_admitted_output(
                    str(event.payload.get("text", "")),
                    is_followup=bool(event.payload.get("is_followup", False)),
                    # Scheduling has a separate epoch-linearized step below:
                    # heard history remains true after a cut, but an old commit
                    # must never re-arm cadence after cancel_all reset it.
                    schedule_followup=False,
                )
                if schedule_followup:
                    self._schedule_followup_if_playback_current(
                        event.payload.get("epoch"),
                        event.payload.get("input_generation"),
                        event.payload.get("followup_generation"),
                    )
            elif event.payload.get("source") == "playback_ordered_user":
                text = str(event.payload.get("text", "")).strip()
                if text:
                    self.memory.add(text, tags=("user",))
            return
        if event.kind == EventKind.CONTROL_STOP:
            if self._runtime_owns_stop:
                return
            if event.payload.get("already_cancelled", False):
                return
            self.cancel_all_if_current(event.payload)
            return
        if event.kind == EventKind.CONTROL_MODE:
            self._set_mode(
                str(event.payload.get("mode", "")),
                expected_control=event.payload,
            )
            return
        if event.kind == EventKind.CONTROL_CONFIRM:
            self._confirm_next(
                owner_verified=(event.payload.get("owner_verified") is True),
                origin=str(event.payload.get("origin", "unknown")),
                direct_user_instruction=(
                    event.payload.get("direct_user_instruction") is True
                ),
                input_generation=event.payload.get("input_generation"),
                input_epoch=event.payload.get("input_epoch"),
                expected_control=event.payload,
            )
            return
        if event.kind == EventKind.CONTROL_DENY:
            self._deny_all(expected_control=event.payload)
            return
        if event.kind == EventKind.TASK_COMPLETED:
            self._handle_task_completed(event)
            return
        if event.kind == EventKind.TASK_CANCELLED:
            task_id = str(event.payload.get("task_id", ""))
            task_matches_event = False
            with self._cancel_lock:
                task = self.state.active_tasks.get(task_id)
                if task is None:
                    task = self.state.pending_audio_tasks.get(task_id)
                matches = self._terminal_event_matches_locked(
                    task_id,
                    event.payload,
                    task,
                )
                if not matches:
                    log.warning(
                        "dropping TASK_CANCELLED with stale task identity: %s",
                        task_id,
                    )
                else:
                    event_identity = self._resolved_terminal_identity_locked(
                        event.payload,
                        task,
                    )
                    task_matches_event = bool(
                        task is not None
                        and task.identity == event_identity
                    )
                    if task_matches_event:
                        self.state.active_tasks.pop(task_id, None)
                        self.state.pending_audio_tasks.pop(task_id, None)
                    self._retire_task_tts_locked(task_id, event_identity)
                    self._release_task_binding_if_unowned_locked(
                        task_id,
                        event_identity,
                    )
                    self._accepted_terminal_events.append(
                        (event, event_identity, True)
                    )
            if not matches:
                return
            if task_matches_event:
                self._start_queued_tasks()
            return
        if event.kind == EventKind.TASK_FAILED:
            task_id = str(event.payload.get("task_id", ""))
            task_matches_event = False
            with self._cancel_lock:
                task = self.state.active_tasks.get(task_id)
                if task is None:
                    task = self.state.pending_audio_tasks.get(task_id)
                matches = self._terminal_event_matches_locked(
                    task_id,
                    event.payload,
                    task,
                )
                if not matches:
                    log.warning(
                        "dropping TASK_FAILED with stale task identity: %s",
                        task_id,
                    )
                    outward_stale = True
                else:
                    event_identity = self._resolved_terminal_identity_locked(
                        event.payload,
                        task,
                    )
                    task_matches_event = bool(
                        task is not None
                        and task.identity == event_identity
                    )
                    if task_matches_event:
                        self.state.active_tasks.pop(task_id, None)
                        self.state.pending_audio_tasks.pop(task_id, None)
                    self._retire_task_tts_locked(task_id, event_identity)
                    known = self.session_actor.identity_for_task(task_id)
                    event_epoch = event.payload.get("epoch")
                    outward_stale = bool(
                        (
                            task_matches_event
                            and task is not None
                            and (
                                task.cancel_event.is_set()
                                or known != task.identity
                            )
                        )
                        or (
                            not task_matches_event
                            and any(
                                field in event.payload
                                for field in TaskIdentity.PAYLOAD_FIELDS
                            )
                        )
                        or (
                            event_epoch is not None
                            and int(event_epoch) < self.speech_epoch
                        )
                    )
                    self._accepted_terminal_events.append(
                        (event, event_identity, True)
                    )
            if not matches:
                return
            if not outward_stale:
                self.state.failures.append(str(event.payload.get("error", "")))
            # Don't leave dead air on a failed turn -- speak an apology, but only
            # for a turn that would have spoken (not dictation / meeting notes /
            # a proactive follow-up). Epoch-stamped so a barge-in still
            # suppresses it (mirrors the reap path's _TIMEOUT_APOLOGY; sr-2).
            # Older events may lack the new payload keys -> default speak=True
            # and fall back to the current epoch.
            if (
                not outward_stale
                and event.payload.get("speak", True)
                and not event.payload.get("followup")
            ):
                self.state.spoken_outputs.append(_FAILURE_APOLOGY)
                aux_tts_id = self.register_aux_tts(
                    task_id,
                    speech_epoch=event.payload.get("epoch"),
                    input_generation=event.payload.get("input_generation"),
                    input_epoch=event.payload.get("input_epoch"),
                )
                self.publish(
                    AgentEvent(
                        EventKind.TTS_REQUEST,
                        {
                            "task_id": task_id,
                            "text": _FAILURE_APOLOGY,
                            "epoch": event.payload.get("epoch", self.speech_epoch),
                            "auxiliary_tts": True,
                            "aux_tts_id": aux_tts_id,
                            **self._event_identity_payload(event.payload),
                        },
                    )
                )
            with self._cancel_lock:
                self._release_task_binding_if_unowned_locked(
                    task_id,
                    event_identity,
                )
            if task_matches_event:
                self._start_queued_tasks()
            return
        if event.kind == EventKind.TTS_STREAM_END:
            self._handle_tts_stream_end(event)
            return

    def drain(self) -> int:
        return self.bus.drain()

    def control_event_current(self, payload: Mapping[str, object]) -> bool:
        """Validate analyzer-derived control identity at its actual effect."""
        with self._cancel_lock:
            return self._control_event_current_locked(payload)

    @property
    def pending_output_count(self) -> int:
        with self._cancel_lock:
            return len(self._pending_output_bindings)

    def _control_event_current_locked(self, payload: Mapping[str, object]) -> bool:
        generation = payload.get("input_generation")
        input_epoch = payload.get("input_epoch")
        if generation is not None and int(generation) != self.latest_arrival_generation:
            return False
        if input_epoch is not None and int(input_epoch) != self.input_epoch:
            return False
        return not self._stopped

    def control_generation_current(self, payload: Mapping[str, object]) -> bool:
        """Runtime-side check after supervisor STOP may have advanced epoch."""
        generation = payload.get("input_generation")
        with self._cancel_lock:
            return bool(
                not self._stopped
                and (
                    generation is None
                    or int(generation) == self.latest_arrival_generation
                )
            )

    def shutdown(self) -> None:
        """Cancel background timers (called on runtime stop). Idempotent; after
        this, follow-up scheduling and the timer callback are permanently inert."""
        with self._cancel_lock:
            if self._stopped:
                return
            self._stopped = True
        with self._followup_lock:
            self._followup_shutdown = True
            self._cancel_followup_locked()
        self.cancel_all()
        self.session_actor.shutdown()
        drain_deadline = time.monotonic() + _TOOL_SHUTDOWN_DRAIN_SEC
        snapshot = self.session_actor.snapshot()
        while (
            (
                snapshot.active_tasks
                or snapshot.active_providers
                or snapshot.active_tools
                or snapshot.active_playbacks
            )
            and time.monotonic() < drain_deadline
        ):
            time.sleep(0.005)
            snapshot = self.session_actor.snapshot()
        if (
            snapshot.active_tasks
            or snapshot.active_providers
            or snapshot.active_tools
            or snapshot.active_playbacks
        ):
            log.warning(
                "session shutdown bounded drain ended with %d task(s), "
                "%d provider(s), %d tool call(s), and %d playback "
                "admission(s) still active",
                snapshot.active_tasks,
                snapshot.active_providers,
                snapshot.active_tools,
                snapshot.active_playbacks,
            )
        # Lifecycle publications are intentionally gated once _stopped is set,
        # so no late TASK_CANCELLED event remains to reap these references.
        with self._cancel_lock:
            active_tasks = tuple(self.state.active_tasks.values())
            self.state.active_tasks.clear()
            for task in active_tasks:
                self._release_task_binding_if_unowned_locked(
                    task.task_id,
                    task.identity,
                )

    def _cancel_speech_locked(self) -> int:
        """Advance global actor and output fences under ``_cancel_lock``."""

        actor_generation = self.session_actor.cancel_speech()
        self.speech_epoch = max(self.speech_epoch + 1, actor_generation)
        return self.speech_epoch

    def _advance_output_epoch_locked(self) -> int:
        """Supersede selected output without invalidating sibling handles."""

        self.speech_epoch += 1
        return self.speech_epoch

    def cancel_tool_call(
        self,
        tool_call_id: str,
        identity: TaskIdentity,
    ) -> bool:
        """Explicitly cancel one mutating tool without changing speech state."""

        return self.tasks.cancel_tool_call(tool_call_id, identity)

    def cancel_all(self, *, invalidate_inputs: bool = True) -> None:
        """Preempt every active/queued/pending task. Safe to call from the audio
        thread (runtime barge-in/stop) or the bus thread (CONTROL_STOP).

        The speech epoch is bumped and every active task is cancelled *under the
        lock* so that, the instant this returns, no in-flight task can be treated
        as active and no prior-epoch sentence will be spoken. ``task.cancel()``
        only sets an :class:`Event`, so the lock is never held across blocking
        work (no engine call, no ``stop_speaking()``, no thread join). A
        provider-started mutation owns a separate tool Event and is cancelled
        only through :meth:`cancel_tool_call`; its later task result remains
        speech-suppressed. A mutation still waiting to claim provider start is
        cancelled by this speech-generation transition."""
        with self._cancel_lock:
            self._cancel_all_locked(invalidate_inputs=invalidate_inputs)
        self._after_cancel_all()

    def cancel_all_if_current(
        self,
        payload: Mapping[str, object],
        *,
        invalidate_inputs: bool = True,
    ) -> bool:
        """Atomically validate a control identity and apply cancellation."""
        with self._cancel_lock:
            if not self._control_event_current_locked(payload):
                return False
            self._cancel_all_locked(invalidate_inputs=invalidate_inputs)
        self._after_cancel_all()
        return True

    def _cancel_all_locked(self, *, invalidate_inputs: bool) -> None:
        self._cancel_speech_locked()
        release_candidates: dict[str, TaskIdentity] = {
            task.task_id: task.identity
            for task in self.state.pending_audio_tasks.values()
            if task.identity is not None
        }
        release_candidates.update(
            {
                task.task_id: task.identity
                for task in self.state.queued_tasks
                if task.identity is not None
            }
        )
        release_candidates.update(
            {
                task.task_id: task.identity
                for task in self.state.pending_confirmations.values()
                if task.identity is not None
            }
        )
        release_candidates.update(
            {
                pending[4]: pending[3]
                for pending in self.state.pending_aux_tts.values()
            }
        )
        release_candidates.update(self._pending_output_bindings)
        if invalidate_inputs:
            self.input_epoch += 1
        log.info(
            "cancel_all: speech_epoch=%d input_epoch=%d %d active, "
            "%d pending-audio, %d queued, %d pending-confirm",
            self.speech_epoch,
            self.input_epoch,
            len(self.state.active_tasks),
            len(self.state.pending_audio_tasks),
            len(self.state.queued_tasks),
            len(self.state.pending_confirmations),
        )
        for task in list(self.state.active_tasks.values()):
            task.cancel()
            self._retire_task_tts_locked(task.task_id, task.identity)
        for task in self.state.pending_audio_tasks.values():
            task.cancel()
            self._retire_task_tts_locked(task.task_id, task.identity)
        self.state.pending_audio_tasks.clear()
        self.state.pending_aux_tts.clear()
        self._pending_output_bindings.clear()
        for task in self.state.queued_tasks:
            task.cancel()
            self._retire_task_tts_locked(task.task_id, task.identity)
        self.state.queued_tasks.clear()
        for task in self.state.pending_confirmations.values():
            task.cancel()
            self._retire_task_tts_locked(task.task_id, task.identity)
        self.state.pending_confirmations.clear()
        for task_id, identity in release_candidates.items():
            self._release_task_binding_if_unowned_locked(task_id, identity)

    def _after_cancel_all(self) -> None:
        # Timer cancellation is independent of the in-memory cancel lock.
        self._reset_followups()
        self.state.spoken_outputs.append("[cancelled]")

    def tts_request_allowed(
        self,
        task_id: str,
        epoch: int | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> bool:
        """Whether a ``TTS_REQUEST`` still belongs to the current speech turn.

        Both emission paths stamp the speech ``epoch`` and a barge-in silences
        both by advancing the epoch via ``cancel_all``:

        * **Streaming** sentences carry the epoch captured when the task was
          started (``_start_task``). They are dropped once a later barge-in
          advances the epoch -- crucially *without* depending on the task still
          being in ``active_tasks``, because ``TASK_COMPLETED`` (priority 60) is
          dequeued before the trailing ``TTS_REQUEST``s (priority 100) and has
          already removed it.
        * **Completion** replies carry the epoch captured when the task
          finished; same rule.

        A request with no stamp (``epoch is None``: legacy/direct emits) falls
        back to the active-and-uncancelled check."""
        with self._cancel_lock:
            carries_identity = bool(
                payload is not None
                and any(
                    field in payload
                    for field in TaskIdentity.PAYLOAD_FIELDS
                )
            )
            carried: TaskIdentity | None = None
            if carries_identity:
                try:
                    carried = TaskIdentity.from_payload(payload or {})
                except (TypeError, ValueError):
                    return False
                expected = self._pending_output_bindings.get(task_id)
                task = self.state.active_tasks.get(task_id)
                if task is None:
                    task = self.state.pending_audio_tasks.get(task_id)
                if task is not None:
                    expected = task.identity
                if expected is None:
                    expected = self.session_actor.binding_identity(task_id)
                if expected != carried:
                    return False
                if self.session_actor.identity_for_task(task_id) != carried:
                    return False
                if (task_id, carried.binding_id) in self._retired_tts_tasks:
                    return False
            elif task_id and any(
                retired_task_id == task_id
                for retired_task_id, _binding_id in self._retired_tts_tasks
            ):
                return False
            if epoch is not None:
                return epoch >= self.speech_epoch
            if not task_id:
                return True
            task = self.state.active_tasks.get(task_id)
            if task is None:
                task = self.state.pending_audio_tasks.get(task_id)
            return task is not None and not task.cancel_event.is_set()

    def _retire_task_tts_locked(
        self,
        task_id: str,
        identity: TaskIdentity | None,
    ) -> None:
        key = (task_id, identity.binding_id if identity is not None else 0)
        if not task_id or key in self._retired_tts_tasks:
            return
        self._retired_tts_tasks.add(key)
        self._retired_tts_order.append(key)
        while len(self._retired_tts_order) > 2048:
            expired = self._retired_tts_order.popleft()
            self._retired_tts_tasks.discard(expired)

    def _task_binding_owned_locked(
        self,
        task_id: str,
        identity: TaskIdentity,
    ) -> bool:
        """Whether controller state can still emit/admit work for a binding."""

        active = self.state.active_tasks.get(task_id)
        pending_audio = self.state.pending_audio_tasks.get(task_id)
        confirmation = self.state.pending_confirmations.get(task_id)
        return bool(
            (active is not None and active.identity == identity)
            or (
                pending_audio is not None
                and pending_audio.identity == identity
            )
            or (
                confirmation is not None
                and confirmation.identity == identity
            )
            or self._pending_output_bindings.get(task_id) == identity
            or any(
                task.task_id == task_id and task.identity == identity
                for task in self.state.queued_tasks
            )
            or any(
                pending[4] == task_id and pending[3] == identity
                for pending in self.state.pending_aux_tts.values()
            )
        )

    def _release_task_binding_if_unowned_locked(
        self,
        task_id: str,
        identity: TaskIdentity | None,
    ) -> bool:
        if (
            not task_id
            or identity is None
            or self._task_binding_owned_locked(task_id, identity)
        ):
            return False
        return self.tasks.release_task_binding(task_id, identity)

    def _register_aux_tts_locked(
        self,
        task_id: str,
        *,
        speech_epoch: object = None,
        input_generation: object = None,
        input_epoch: object = None,
    ) -> str:
        if self._stopped:
            return ""
        self._next_aux_tts += 1
        aux_tts_id = f"{task_id}:{self._next_aux_tts}"
        identity = self.session_actor.identity_for_task(task_id)
        identity_task_id = task_id
        if identity is None:
            turn = self.state.turn_identity or self.session_actor.open_turn(
                revision=0
            )
            # Auxiliary sources (for example a due reminder) need identity too,
            # but need not occupy a task-execution slot.
            identity = self.session_actor.bind_task(aux_tts_id, turn)
            identity_task_id = aux_tts_id
        self.state.pending_aux_tts[aux_tts_id] = (
            int(speech_epoch) if speech_epoch is not None else self.speech_epoch,
            (
                int(input_generation)
                if input_generation is not None
                else self.latest_arrival_generation
            ),
            int(input_epoch) if input_epoch is not None else self.input_epoch,
            identity,
            identity_task_id,
        )
        return aux_tts_id

    def register_aux_tts(
        self,
        task_id: str,
        *,
        speech_epoch: object = None,
        input_generation: object = None,
        input_epoch: object = None,
    ) -> str:
        with self._cancel_lock:
            return self._register_aux_tts_locked(
                task_id or "aux",
                speech_epoch=speech_epoch,
                input_generation=input_generation,
                input_epoch=input_epoch,
            )

    def cancel_pending_aux_tts(self) -> None:
        """Retire queued ack/apology/clarify audio without touching tasks."""
        with self._cancel_lock:
            bindings = {
                (pending[4], pending[3])
                for pending in self.state.pending_aux_tts.values()
            }
            self.state.pending_aux_tts.clear()
            for binding_id, identity in bindings:
                self._release_task_binding_if_unowned_locked(
                    binding_id,
                    identity,
                )

    def auxiliary_tts_allowed(
        self,
        aux_tts_id: str,
        epoch: int | None = None,
    ) -> bool:
        with self._cancel_lock:
            identity = self.state.pending_aux_tts.get(aux_tts_id)
            if identity is None:
                return False
            (
                registered_speech_epoch,
                input_generation,
                input_epoch,
                _task_identity,
                _identity_task_id,
            ) = identity
            return bool(
                registered_speech_epoch >= self.speech_epoch
                and (epoch is None or int(epoch) >= self.speech_epoch)
                and (
                    input_generation is None
                    or input_generation == self.latest_arrival_generation
                )
                and (input_epoch is None or input_epoch == self.input_epoch)
            )

    def note_aux_tts_admitted(self, aux_tts_id: str) -> None:
        with self._cancel_lock:
            pending = self.state.pending_aux_tts.pop(aux_tts_id, None)
            if pending is not None:
                self._release_task_binding_if_unowned_locked(
                    pending[4],
                    pending[3],
                )

    def auxiliary_tts_identity_payload(
        self, aux_tts_id: str
    ) -> dict[str, object]:
        with self._cancel_lock:
            pending = self.state.pending_aux_tts.get(aux_tts_id)
            return pending[3].to_payload() if pending is not None else {}

    def admit_playback(
        self,
        payload: Mapping[str, object],
        *,
        task_id: str,
        auxiliary_tts: bool,
        aux_tts_id: str,
    ) -> PlaybackAdmission | None:
        """Require actor-owned identity at the final speech-output boundary."""

        with self._cancel_lock:
            identity: TaskIdentity | None = None
            identity_task_id = ""
            carried: TaskIdentity | None = None
            carries_identity = any(
                field in payload for field in TaskIdentity.PAYLOAD_FIELDS
            )
            if auxiliary_tts:
                pending = self.state.pending_aux_tts.get(aux_tts_id)
                if pending is not None:
                    identity = pending[3]
                    identity_task_id = pending[4]
                if carries_identity:
                    try:
                        carried = TaskIdentity.from_payload(payload)
                    except (TypeError, ValueError):
                        carried = None
                    if identity is None or carried != identity:
                        identity = None
            else:
                known = self.session_actor.identity_for_task(task_id)
                if carries_identity:
                    try:
                        carried = TaskIdentity.from_payload(payload)
                    except (TypeError, ValueError):
                        carried = None
                    if known is not None and carried == known:
                        identity = carried
                        identity_task_id = task_id
                else:
                    # Compatibility for direct in-process callers: production
                    # task emitters always carry the explicit identity payload.
                    identity = known
                    identity_task_id = task_id
                    if identity is None:
                        self._next_aux_tts += 1
                        turn = (
                            self.state.turn_identity
                            or self.session_actor.open_turn(revision=0)
                        )
                        identity = self.session_actor.bind_task(
                            f"legacy-tts:{task_id or 'direct'}:{self._next_aux_tts}",
                            turn,
                        )
                        identity_task_id = (
                            f"legacy-tts:{task_id or 'direct'}:{self._next_aux_tts}"
                        )
            if identity is None:
                if not auxiliary_tts:
                    # Identity-bearing stale/malformed callbacks must not
                    # consume a newer same-ID scope. Legacy in-process callers
                    # retain the old task-ID cleanup behavior.
                    if not carries_identity:
                        pending_identity = self._pending_output_bindings.pop(
                            task_id,
                            None,
                        )
                        task = self.state.pending_audio_tasks.get(task_id)
                        if (
                            task is not None
                            and not task.metadata.get("stream_audio_pending")
                        ):
                            self.state.pending_audio_tasks.pop(task_id, None)
                            self._retire_task_tts_locked(
                                task_id,
                                task.identity,
                            )
                            pending_identity = task.identity
                        self._release_task_binding_if_unowned_locked(
                            task_id,
                            pending_identity,
                        )
                return None
            admission = self.session_actor.admit_playback(
                task_id=identity_task_id,
                identity=identity,
            )
            if not auxiliary_tts:
                if self._pending_output_bindings.get(task_id) == identity:
                    self._pending_output_bindings.pop(task_id, None)
                if admission is None:
                    task = self.state.pending_audio_tasks.get(task_id)
                    if (
                        task is not None
                        and task.identity == identity
                        and not task.metadata.get("stream_audio_pending")
                    ):
                        self.state.pending_audio_tasks.pop(task_id, None)
                        self._retire_task_tts_locked(task_id, task.identity)
                self._release_task_binding_if_unowned_locked(
                    task_id,
                    identity,
                )
                if identity_task_id != task_id:
                    self._release_task_binding_if_unowned_locked(
                        identity_task_id,
                        identity,
                    )
            return admission

    def finish_playback(self, playback_admission_id: str) -> None:
        self.session_actor.finish_playback(playback_admission_id)

    def note_tts_rejected(
        self,
        task_id: str,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        """Release controller ownership for one terminally dropped TTS event."""

        if not task_id:
            return
        with self._cancel_lock:
            carries_identity = bool(
                payload is not None
                and any(
                    field in payload
                    for field in TaskIdentity.PAYLOAD_FIELDS
                )
            )
            carried = (
                self._task_identity_from_payload(payload or {})
                if carries_identity
                else None
            )
            if carries_identity and carried is None:
                return
            task = self.state.pending_audio_tasks.get(task_id)
            expected = self._pending_output_bindings.get(task_id)
            if task is not None:
                expected = task.identity
            if expected is None:
                expected = self.session_actor.binding_identity(task_id)
            if carried is not None and carried != expected:
                return
            identity = carried or expected
            if self._pending_output_bindings.get(task_id) == identity:
                self._pending_output_bindings.pop(task_id, None)
            if (
                task is not None
                and task.identity == identity
                and not task.metadata.get("stream_audio_pending")
            ):
                self.state.pending_audio_tasks.pop(task_id, None)
                self._retire_task_tts_locked(task_id, task.identity)
            self._release_task_binding_if_unowned_locked(task_id, identity)

    def note_tts_admitted(
        self,
        task_id: str,
        epoch: int | None = None,
        text: str = "",
        payload: Mapping[str, object] | None = None,
    ) -> tuple[str, bool] | None:
        """Mark first real answer audio only at the engine playback boundary."""
        if not task_id:
            return None
        with self._cancel_lock:
            if epoch is not None and int(epoch) < self.speech_epoch:
                return None
            task = self.state.active_tasks.get(task_id)
            if task is None:
                task = self.state.pending_audio_tasks.get(task_id)
            if task is None or task.cancel_event.is_set():
                return None
            if payload is not None and any(
                field in payload for field in TaskIdentity.PAYLOAD_FIELDS
            ):
                carried = self._task_identity_from_payload(payload)
                if carried is None or carried != task.identity:
                    return None
            task.started_speaking = True
            if text:
                fragments = task.metadata.setdefault(
                    "admitted_tts_fragments",
                    [],
                )
                if isinstance(fragments, list):
                    fragments.append(text)
            if not task.metadata.get("stream_audio_pending"):
                self.state.pending_audio_tasks.pop(task_id, None)
                self._release_task_binding_if_unowned_locked(
                    task_id,
                    task.identity,
                )
            pending_output = task.metadata.pop("pending_assistant_output", None)
            pending_followup = bool(
                task.metadata.pop("pending_assistant_followup", False)
            )
            if isinstance(pending_output, str) and pending_output:
                return pending_output, pending_followup
            return None

    def record_admitted_output(
        self,
        text: str,
        *,
        is_followup: bool,
        schedule_followup: bool = True,
    ) -> None:
        """Commit a reply after the runtime's configured playback boundary."""
        if text:
            self.state.spoken_outputs.append(text)
        if text and not is_followup:
            self.memory.add(text, tags=("assistant_output",))
        if text and schedule_followup:
            self._schedule_followup()

    # --- proactive follow-ups ------------------------------------------------
    def _schedule_followup(self) -> None:
        """Arm a timer that, after a silent delay, nudges the conversation."""
        if not self.followups.enabled or not self._followup_state.can_continue():
            return
        with self._cancel_lock:
            if self._stopped:
                return
            speech_epoch = self.speech_epoch
            input_generation = self.latest_arrival_generation
            with self._followup_lock:
                self._arm_followup_locked(
                    speech_epoch,
                    input_generation,
                    self._followup_generation,
                )

    def _schedule_followup_if_playback_current(
        self,
        epoch: object,
        input_generation: object,
        followup_generation: object,
    ) -> None:
        """Arm cadence only while the receipt still owns the silent interval.

        Heard text remains memory after a cut/new utterance, but its delayed
        commit may not start a new timer. Speech epoch covers stop/barge; input
        arrival covers runtime partials; follow-up generation covers every
        supervisor-level speech/reset. Locks linearize scheduling against each.
        """

        try:
            expected_epoch = int(epoch)
            expected_input = int(input_generation)
            expected_followup = int(followup_generation)
        except (TypeError, ValueError):
            return
        with self._cancel_lock:
            if (
                self._stopped
                or expected_epoch != self.speech_epoch
                or expected_input != self.latest_arrival_generation
            ):
                return
            with self._followup_lock:
                if expected_followup != self._followup_generation:
                    return
                self._arm_followup_locked(
                    expected_epoch,
                    expected_input,
                    expected_followup,
                )

    def _arm_followup_locked(
        self,
        speech_epoch: int,
        input_generation: int,
        followup_generation: int,
    ) -> None:
        """Create/replace one timer. Caller holds ``_followup_lock``."""

        if (
            self._followup_shutdown
            or self._stopped
            or not self.followups.enabled
            or not self._followup_state.can_continue()
        ):
            return
        self._cancel_followup_locked()
        token = self._followup_token
        timer = Timer(
            self.followups.delay_sec,
            self._tick_followup,
            args=(
                token,
                int(speech_epoch),
                int(input_generation),
                int(followup_generation),
            ),
        )
        timer.daemon = True
        self._followup_timer = timer
        timer.start()

    def _cancel_followup(self) -> None:
        with self._followup_lock:
            self._cancel_followup_locked()

    def _cancel_followup_locked(self) -> None:
        self._followup_token += 1
        timer = self._followup_timer
        if timer is not None:
            timer.cancel()
            self._followup_timer = None

    def _reset_followups(self) -> None:
        with self._followup_lock:
            self._followup_generation += 1
            self._cancel_followup_locked()
            self._followup_state.reset()

    def _tick_followup(
        self,
        timer_token: object = None,
        speech_epoch: object = None,
        input_generation: object = None,
        followup_generation: object = None,
    ) -> None:
        """Timer-thread callback: hand control back to the bus thread."""
        with self._cancel_lock:
            if self._stopped:
                return
            event_epoch = (
                self.speech_epoch if speech_epoch is None else int(speech_epoch)
            )
            event_input = (
                self.latest_arrival_generation
                if input_generation is None
                else int(input_generation)
            )
        with self._followup_lock:
            event_token = (
                self._followup_token
                if timer_token is None
                else int(timer_token)
            )
            event_followup = (
                self._followup_generation
                if followup_generation is None
                else int(followup_generation)
            )
            if (
                event_token != self._followup_token
                or event_followup != self._followup_generation
            ):
                return
            self._followup_timer = None
            if self._followup_shutdown or self._stopped:
                return
        self.publish(
            AgentEvent(
                EventKind.FOLLOWUP_TICK,
                {
                    "timer_token": event_token,
                    "speech_epoch": event_epoch,
                    "input_generation": event_input,
                    "followup_generation": event_followup,
                },
                priority=95,
            )
        )

    def _emit_followup(
        self,
        *,
        expected_speech_epoch: object = None,
        expected_input_generation: object = None,
        expected_followup_generation: object = None,
        timer_token: object = None,
    ) -> None:
        """Bus-thread handler: start a follow-up task if still warranted."""
        if not self.followups.enabled or not self._followup_state.can_continue():
            return
        try:
            expected_epoch = (
                None
                if expected_speech_epoch is None
                else int(expected_speech_epoch)
            )
            expected_input = (
                None
                if expected_input_generation is None
                else int(expected_input_generation)
            )
            expected_followup = (
                None
                if expected_followup_generation is None
                else int(expected_followup_generation)
            )
            expected_token = None if timer_token is None else int(timer_token)
        except (TypeError, ValueError):
            return
        with self._cancel_lock:
            if (
                self._stopped
                or (
                    expected_epoch is not None
                    and expected_epoch != self.speech_epoch
                )
                or (
                    expected_input is not None
                    and expected_input != self.latest_arrival_generation
                )
                or self.state.active_tasks
                or self.state.pending_audio_tasks
                or self.state.pending_aux_tts
                or self.state.queued_tasks
            ):
                return
            speech_epoch = self.speech_epoch
            # Bind the synthetic turn BEFORE task creation.  Stamping whatever
            # generation happened to be current afterward let a user arrival in
            # this gap make the follow-up look like part of the newer turn.
            input_epoch = self.input_epoch
            input_generation = self.latest_input_generation
            with self._followup_lock:
                if (
                    expected_followup is not None
                    and expected_followup != self._followup_generation
                ) or (
                    expected_token is not None
                    and expected_token != self._followup_token
                ):
                    return
        marker = self._followup_state.next_marker()
        decision = IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text=marker,
            reason="followup",
            mode=Mode.ASSISTANT,
            speak=True,
        )
        task = self._create_task(
            decision,
            turn_identity=self.session_actor.open_turn(revision=0),
        )
        task.metadata["followup"] = True
        # A proactive follow-up is assistant-initiated (a synthetic "[silent]"
        # marker the owner never spoke), so it must NEVER carry the prior turn's
        # owner-verified trust -- fail-close it so it can't drive an action.
        task.metadata["owner_verified"] = False
        task.metadata["origin"] = "system"
        task.metadata["input_epoch"] = input_epoch
        task.metadata["input_generation"] = input_generation
        self._start_task(task, expected_epoch=speech_epoch)

    def _handle_speech(
        self, text: str, *, is_final: bool,
        owner_verified: bool = False, origin: str = "unknown",
        metadata: object = None,
    ) -> None:
        # Record the current turn's speaker-ID trust (fail-closed) so tasks created
        # from this utterance carry it to the action chokepoint. Set on every final
        # (partials don't create tasks) -- a new turn always re-establishes trust,
        # never inherits a prior turn's.
        if is_final:
            turn_metadata = (
                {
                    key: metadata[key]
                    for key in (
                        "latency_policy",
                        "metrics_turn_token",
                        "input_epoch",
                        "input_generation",
                        "revision",
                        "reserved_continuation",
                        "continuation_arrival_kind",
                        "continuation_merged_text",
                        "continuation_of",
                        "continue_after",
                        "continuation_origin",
                        "continuation_addons",
                        "continuation_record_origin",
                        "continuation_unrecorded_addons",
                        "continuation_victim_owner_verified",
                        "continuation_victim_origin",
                        "continuation_victim_metrics_turn_token",
                        "skip_user_memory",
                        "post_barge_response_only",
                        "direct_user_instruction",
                    )
                    if key in metadata
                }
                if isinstance(metadata, Mapping)
                else {}
            )
            expected_input_epoch = turn_metadata.get("input_epoch")
            with self._cancel_lock:
                if self._stopped:
                    return
                if expected_input_epoch is None:
                    expected_input_epoch = self.input_epoch
                    turn_metadata["input_epoch"] = expected_input_epoch
                if int(expected_input_epoch) != self.input_epoch:
                    log.info("dropping stale pre-task final %r (input epoch)", text)
                    return
                input_generation = turn_metadata.get("input_generation")
                if input_generation is None:
                    turn_metadata["input_generation"] = self.latest_input_generation
                elif int(input_generation) < self.latest_arrival_generation:
                    log.info(
                        "dropping superseded pre-task final %r (arrival generation)",
                        text,
                    )
                    return
            if (
                turn_metadata.get("post_barge_response_only")
                and self.state.mode != Mode.ASSISTANT
            ):
                log.info(
                    "dropping response-only final %r after mode changed to %s",
                    text,
                    self.state.mode.value,
                )
                return
            if turn_metadata.get("post_barge_response_only"):
                # Defense in depth: metadata controls decision scope; it can
                # never carry caller-supplied speaker/action trust.
                owner_verified = False
                origin = "unknown"
        observation = self.analyzer.observe(text, is_final=is_final)
        if is_final:
            # An analyzer may block while a newer acoustic partial establishes
            # its arrival fence. Recheck after that unbounded call and before
            # mutating observations, transcript history, trust, decisions, or
            # tasks. A matching abort can then reissue the claimed event on the
            # fence generation without duplicating any of those effects.
            # Direct supervisor callers historically commit an input generation
            # without a separate note_input_arrival(), so reject only an
            # observation that became older rather than requiring exact equality.
            with self._cancel_lock:
                post_analysis_generation = turn_metadata.get(
                    "input_generation"
                )
                post_analysis_epoch = turn_metadata.get("input_epoch")
                if (
                    self._stopped
                    or (
                        post_analysis_epoch is not None
                        and int(post_analysis_epoch) != self.input_epoch
                    )
                    or (
                        post_analysis_generation is not None
                        and int(post_analysis_generation)
                        < self.latest_arrival_generation
                    )
                ):
                    log.info(
                        "dropping superseded post-analysis final %r "
                        "(input generation)",
                        text,
                    )
                    return
            self.state.turn_owner_verified = owner_verified is True
            self.state.turn_origin = str(origin)
            self.state.turn_metadata = turn_metadata
            self.state.turn_identity = self.session_actor.open_turn(
                revision=int(turn_metadata.get("revision", 0))
            )
        self.state.observations.append(observation)
        if not observation.normalized:
            return
        # The user spoke: stop any pending proactive follow-up and restart the
        # silence cadence from zero.
        self._reset_followups()
        if is_final:
            self.state.transcript_log.append(text)
        else:
            self.state.last_partial = text

        decision = self.analyzer.decide(
            observation,
            self.state.mode,
            has_pending_confirmation=bool(self.state.pending_confirmations),
        )
        if is_final and self.state.turn_metadata.get(
            "post_barge_response_only"
        ):
            # The runtime admitted this transcript solely because it followed
            # a confirmed acoustic interruption.  That provenance permits a
            # conversational answer and nothing else: never reinterpret its
            # words as stop/confirm/mode/search/research/command authority.
            decision = IntentDecision(
                IntentKind.ASSISTANT,
                1.0,
                observation.text,
                "post_barge_response_only",
                mode=Mode.ASSISTANT,
            )
        if is_final and decision.kind in {
            IntentKind.STOP,
            IntentKind.CONFIRM,
            IntentKind.DENY,
            IntentKind.MODE_SWITCH,
        }:
            with self._cancel_lock:
                generation = self.state.turn_metadata.get("input_generation")
                if (
                    generation is not None
                    and int(generation) != self.latest_arrival_generation
                ):
                    log.info(
                        "dropping superseded control final %r (generation)",
                        text,
                    )
                    return
        self.state.decisions.append(decision)
        log.debug(
            "decision: kind=%s confidence=%.2f reason=%s mode=%s text=%r",
            decision.kind.value, decision.confidence, decision.reason,
            self.state.mode.value, decision.text,
        )
        self.publish(
            AgentEvent(
                EventKind.INTENT_DECISION,
                {
                    "kind": decision.kind.value,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "text": decision.text,
                },
                priority=55,
            )
        )
        try:
            self._execute_decision(decision)
        finally:
            # Reservation metadata belongs to exactly this final.  Tasks already
            # copied it in _create_task; leaving it in turn_metadata would let a
            # later proactive follow-up inherit synthetic lineage/trust policy.
            if is_final and self.state.turn_metadata.get("reserved_continuation"):
                for key in _RESERVED_CONTINUATION_KEYS:
                    self.state.turn_metadata.pop(key, None)

    def _create_task(
        self,
        decision: IntentDecision,
        *,
        turn_identity: TurnIdentity | None = None,
    ) -> AgentTask:
        """Create a task and stamp it with the current turn's speaker-ID trust, so
        the capability layer's action chokepoint (always_on_agent.origin) sees
        owner_verified/origin. Fail-closed: defaults to the not-verified turn trust."""
        task = self.tasks.create_task(
            decision,
            turn_identity=turn_identity or self.state.turn_identity,
        )
        task.metadata.update(self.state.turn_metadata)
        # Stamp authority AFTER generic turn metadata so no custom analyzer or
        # event payload can override the controller-owned trust verdict.
        task.metadata["owner_verified"] = self.state.turn_owner_verified is True
        task.metadata["origin"] = str(self.state.turn_origin)
        task.metadata["direct_user_instruction"] = (
            self.state.turn_metadata.get("direct_user_instruction") is True
        )
        task.metadata["confirmed"] = False
        return task

    @staticmethod
    def _task_identity_payload(task: AgentTask) -> dict[str, object]:
        if task.identity is None:
            raise RuntimeError("task has no session identity")
        return task.identity.to_payload()

    @staticmethod
    def _event_identity_payload(
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        # Only a truly legacy event may remain identity-less. Preserve partial
        # or malformed fields verbatim so playback sees that identity was
        # attempted and fails closed instead of taking the legacy upgrade path.
        return {
            field: payload[field]
            for field in TaskIdentity.PAYLOAD_FIELDS
            if field in payload
        }

    def _terminal_event_matches_locked(
        self,
        task_id: str,
        payload: Mapping[str, object],
        task: AgentTask | None,
    ) -> bool:
        """Authenticate identity-bearing lifecycle events before state mutation."""

        if not any(field in payload for field in TaskIdentity.PAYLOAD_FIELDS):
            return True
        try:
            carried = TaskIdentity.from_payload(payload)
        except (TypeError, ValueError):
            return False
        if task is not None and task.identity == carried:
            return True
        return bool(
            self.session_actor.binding_matches(task_id, carried)
            or (task_id, carried.binding_id) in self._retired_tts_tasks
        )

    def _resolved_terminal_identity_locked(
        self,
        payload: Mapping[str, object],
        task: AgentTask | None,
    ) -> TaskIdentity | None:
        """Resolve carried identity before falling back to current legacy state."""

        if any(field in payload for field in TaskIdentity.PAYLOAD_FIELDS):
            return self._task_identity_from_payload(payload)
        return task.identity if task is not None else None

    def consume_terminal_event_acceptance(
        self,
        event: AgentEvent,
    ) -> tuple[bool, TaskIdentity | None, bool]:
        """Return runtime authorization, resolved identity, and success scope.

        The final flag is false only for an authenticated TASK_COMPLETED that
        lost its turn before dispatch.  The runtime must then close its exact
        playback/voice ownership as interrupted without applying completion
        metrics, metadata, continuity, or memory effects.
        """

        with self._cancel_lock:
            for index, (accepted, identity, success_current) in enumerate(
                self._accepted_terminal_events
            ):
                if accepted is event:
                    del self._accepted_terminal_events[index]
                    return True, identity, success_current
            # Direct legacy events carry no task identity. Production task
            # emitters carry all five fields and require supervisor acceptance.
            legacy = not any(
                field in event.payload
                for field in TaskIdentity.PAYLOAD_FIELDS
            )
            return legacy, None, legacy

    @staticmethod
    def _task_identity_from_payload(
        payload: Mapping[str, object],
    ) -> TaskIdentity | None:
        try:
            return TaskIdentity.from_payload(payload)
        except (TypeError, ValueError):
            return None

    def _execute_decision(self, decision: IntentDecision) -> None:
        if (
            self.state.turn_metadata.get("reserved_continuation")
            and decision.kind != IntentKind.ASSISTANT
        ):
            generation = self.state.turn_metadata.get("input_generation")
            for key in _RESERVED_CONTINUATION_KEYS:
                self.state.turn_metadata.pop(key, None)
            if (
                self._on_continuation_admitted is not None
                and generation is not None
            ):
                self._on_continuation_admitted(int(generation))
        if decision.kind == IntentKind.IGNORE:
            return
        generation = self.state.turn_metadata.get("input_generation")
        input_epoch = self.state.turn_metadata.get("input_epoch")
        control_identity = {
            "input_generation": (
                int(generation) if generation is not None else None
            ),
            "input_epoch": int(input_epoch) if input_epoch is not None else None,
        }
        if decision.kind == IntentKind.STOP:
            self.publish(AgentEvent.stop(decision.reason, **control_identity))
            return
        if decision.kind == IntentKind.CONFIRM:
            # Carry the speaker-ID trust of THIS "yes" so an owner-verified staged
            # action can only be approved by an owner-verified confirm.
            self.publish(
                AgentEvent.confirm(
                    owner_verified=self.state.turn_owner_verified,
                    origin=self.state.turn_origin,
                    direct_user_instruction=(
                        self.state.turn_metadata.get("direct_user_instruction") is True
                    ),
                    **control_identity,
                )
            )
            return
        if decision.kind == IntentKind.DENY:
            self.publish(AgentEvent.deny(**control_identity))
            return
        if decision.kind == IntentKind.MODE_SWITCH and decision.target_mode is not None:
            self.publish(AgentEvent.mode(decision.target_mode, **control_identity))
            return
        if not decision.starts_task:
            return
        # ADD-ON / continuation: a follow-up that extends the in-flight turn is
        # merged into it (or queued behind it) rather than spawning a competing
        # cold task. Runs only here -- strictly *after* the deterministic
        # STOP/CONFIRM/DENY/MODE_SWITCH forks above -- so a real control phrase
        # can never be misread as a continuation. Returns True when it handled
        # the follow-up; otherwise we fall through to the normal start path.
        if self._maybe_continue(decision):
            return
        task = self._create_task(decision)
        if self._handle_latency_policy(task):
            return
        if task.metadata.get("requires_confirmation"):
            if self._confirmation_ttl_sec > 0:
                task.metadata["confirmation_expires_at"] = (
                    time.monotonic() + self._confirmation_ttl_sec
                )
            with self._cancel_lock:
                if not self._input_metadata_current_locked(task.metadata):
                    task.cancel()
                    self._release_task_binding_if_unowned_locked(
                        task.task_id,
                        task.identity,
                    )
                    return
                self.state.pending_confirmations[task.task_id] = task
                epoch = self.speech_epoch
                confirmation_text = f"Confirm command: {task.input_text}"
                self.state.spoken_outputs.append(confirmation_text)
                aux_tts_id = self._register_aux_tts_locked(
                    task.task_id,
                    speech_epoch=epoch,
                    input_generation=task.metadata.get("input_generation"),
                    input_epoch=task.metadata.get("input_epoch"),
                )
            self.publish(
                AgentEvent(
                    EventKind.TTS_REQUEST,
                    {
                        "task_id": task.task_id,
                        "text": confirmation_text,
                        "epoch": epoch,
                        "confirmation_prompt": True,
                        "auxiliary_tts": True,
                        "aux_tts_id": aux_tts_id,
                        **self._task_identity_payload(task),
                    },
                )
            )
            return
        if self._should_queue(task):
            if self._queue_task(task):
                self.state.spoken_outputs.append(
                    f"Queued {task.mode.value}: {task.input_text}"
                )
            return
        self._start_task(task)

    def _handle_latency_policy(self, task: AgentTask) -> bool:
        policy = str(task.metadata.get("latency_policy", "") or "")
        if policy == _SILENT_INGEST_POLICY:
            task.metadata["speak"] = False
            return False
        if policy in _STREAM_POLICIES:
            task.metadata["stream_tts"] = True
            return False
        if policy != _CLARIFY_POLICY:
            return False
        if not task.metadata.get("speak", True) or task.metadata.get("followup"):
            return False
        with self._cancel_lock:
            if not self._input_metadata_current_locked(task.metadata):
                task.cancel()
                self._release_task_binding_if_unowned_locked(
                    task.task_id,
                    task.identity,
                )
                return True
            epoch = self.speech_epoch
            self.state.spoken_outputs.append(_CLARIFY_TEXT)
            aux_tts_id = self._register_aux_tts_locked(
                task.task_id,
                speech_epoch=epoch,
                input_generation=task.metadata.get("input_generation"),
                input_epoch=task.metadata.get("input_epoch"),
            )
        self.publish(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {
                    "task_id": task.task_id,
                    "text": _CLARIFY_TEXT,
                    "epoch": epoch,
                    "latency_clarify": True,
                    "auxiliary_tts": True,
                    "aux_tts_id": aux_tts_id,
                    **self._task_identity_payload(task),
                },
            )
        )
        return True

    def _start_task(
        self,
        task: AgentTask,
        expected_epoch: int | None = None,
        pending_confirmation_id: str | None = None,
    ) -> bool:
        # Capture the current speech epoch and register the task atomically with
        # respect to cancel_all (realtime-concurrency-1): the streaming sentences
        # this task emits are stamped with this epoch, so a barge-in that
        # advances the epoch drops them even though TASK_COMPLETED (priority 60)
        # is dequeued before the trailing TTS_REQUESTs (priority 100) and has
        # already removed the task from active_tasks.
        with self._cancel_lock:
            if not self._input_metadata_current_locked(task.metadata):
                if pending_confirmation_id is None:
                    task.cancel()
                    self._release_task_binding_if_unowned_locked(
                        task.task_id,
                        task.identity,
                    )
                return False
            if (
                pending_confirmation_id is not None
                and self.state.pending_confirmations.get(
                    pending_confirmation_id
                ) is not task
            ):
                self._release_task_binding_if_unowned_locked(
                    task.task_id,
                    task.identity,
                )
                return False
            # Drop a task whose cancel_event was set after it was queued (e.g. a
            # cancel_all / _cancel_one landed between the queue check and here):
            # never register or spawn a worker for an already-cancelled turn
            # (rc-4). The re-check under the lock is the authoritative drop.
            if task.cancel_event.is_set():
                self._release_task_binding_if_unowned_locked(
                    task.task_id,
                    task.identity,
                )
                return False
            # rc-4 residual window: a task snapshotted by _start_queued_tasks is
            # invisible to cancel_all (it's neither in active_tasks nor
            # queued_tasks during the drain), so cancel_all can't set its
            # cancel_event. Re-check the epoch under the lock too: if a barge-in /
            # supersede advanced it since the drain started, drop the task rather
            # than resurrect a turn the user moved past.
            if expected_epoch is not None and self.speech_epoch != expected_epoch:
                self._release_task_binding_if_unowned_locked(
                    task.task_id,
                    task.identity,
                )
                return False
            if not self.tasks.admit_task(task):
                task.cancel()
                self._release_task_binding_if_unowned_locked(
                    task.task_id,
                    task.identity,
                )
                return False
            if pending_confirmation_id is not None:
                self.state.pending_confirmations.pop(
                    pending_confirmation_id,
                    None,
                )
            task.speech_epoch = self.speech_epoch
            timeout = self._timeout_for(task.mode)
            if timeout > 0:
                task.deadline_at = time.monotonic() + timeout
            self.state.active_tasks[task.task_id] = task
            epoch = task.speech_epoch
        try:
            self._emit_latency_ack(task, epoch)
            started = self.tasks.start(task)
        except BaseException:
            self._rollback_task_start(task)
            raise
        if started is not False:
            return True
        self._rollback_task_start(task)
        return False

    def _rollback_task_start(self, task: AgentTask) -> None:
        """Undo controller and actor ownership after synchronous start failure."""

        with self._cancel_lock:
            if self.state.active_tasks.get(task.task_id) is task:
                self.state.active_tasks.pop(task.task_id, None)
            if self.state.pending_audio_tasks.get(task.task_id) is task:
                self.state.pending_audio_tasks.pop(task.task_id, None)
            self._pending_output_bindings.pop(task.task_id, None)
            aux_ids = tuple(
                aux_id
                for aux_id, pending in self.state.pending_aux_tts.items()
                if pending[4] == task.task_id
            )
            for aux_id in aux_ids:
                self.state.pending_aux_tts.pop(aux_id, None)
            self._retire_task_tts_locked(task.task_id, task.identity)
            task.cancel()
        self.tasks.retire_task_admission(task)
        with self._cancel_lock:
            self._release_task_binding_if_unowned_locked(
                task.task_id,
                task.identity,
            )

    def _emit_latency_ack(self, task: AgentTask, epoch: int) -> None:
        if task.metadata.get("latency_policy") != _ACK_THEN_THINK_POLICY:
            return
        if not task.metadata.get("speak", True) or task.metadata.get("followup"):
            return
        if task.metadata.get("continuation_of"):
            # A merged turn's lineage was already acknowledged once; re-acking
            # the same filler after every add-on sounds robotic.
            return
        # ack_spoken, NOT started_speaking: the ack is filler, not answer audio.
        # started_speaking's contract ("first real spoken sentence", tasks.py)
        # gates the continuation MERGE in _maybe_continue -- flipping it here
        # made an add-on during a slow acked turn queue behind the unheard
        # answer instead of merging into it.
        task.ack_spoken = True
        self.state.spoken_outputs.append(_ACK_THEN_THINK_TEXT)
        aux_tts_id = self.register_aux_tts(
            task.task_id,
            speech_epoch=epoch,
            input_generation=task.metadata.get("input_generation"),
            input_epoch=task.metadata.get("input_epoch"),
        )
        self.publish(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {
                    "task_id": task.task_id,
                    "text": _ACK_THEN_THINK_TEXT,
                    "epoch": epoch,
                    "latency_ack": True,
                    "auxiliary_tts": True,
                    "aux_tts_id": aux_tts_id,
                    **self._task_identity_payload(task),
                },
            )
        )

    def _timeout_for(self, mode: Mode) -> float:
        return self._task_timeouts.get(mode.value, _FALLBACK_TASK_TIMEOUT)

    def reap_overdue_tasks(self) -> int:
        """Cancel + remove any active task past its wall-clock deadline.

        The controller's "never get stuck waiting for output" backstop: a
        capability that blocks uninterruptibly inside a step (a hung generate, a
        network read with no timeout) is cancelled and dropped from active_tasks
        here, so the supervisor stops treating it as live.  TaskRuntime's
        registered coordinator exits promptly; its bounded daemon provider may
        remain blocked until its own I/O returns or times out.

        Safe to call from the watchdog tick thread: the active_tasks mutation is
        done under ``_cancel_lock`` exactly like ``cancel_all`` and ``cancel``
        only sets an Event. Queue draining is handed back to the bus thread via a
        republished TASK_CANCELLED (so ``queued_tasks`` is only ever touched
        there) rather than draining here off-thread.

        Unlike ``cancel_all`` / ``_cancel_one`` this deliberately does NOT advance
        the global speech epoch: a reap can hit one of several concurrent tasks,
        and bumping the shared epoch would strand a sibling's queued TTS. A hung
        task (the usual reap target) has no queued audio anyway; the rare partial
        from a slow-but-producing task may play its tail before the apology."""
        now = time.monotonic()
        reaped: list[AgentTask] = []
        with self._cancel_lock:
            for task in list(self.state.active_tasks.values()):
                if task.deadline_at and now >= task.deadline_at:
                    # Claim task terminality and actor tool terminality as one
                    # transition. The callback runs task-lock -> actor-lock;
                    # SessionActor never calls into AgentTask, so there is no
                    # inverse nesting. A completed/failed task that won first is
                    # left in supervisor state for its queued lifecycle event.
                    if task.cancel_and_claim_terminal(
                        lambda task_id=task.task_id: (
                            self.tasks.fence_task_tool_calls(
                                task_id,
                                task.identity,
                            )
                        )
                    ):
                        self.state.active_tasks.pop(task.task_id, None)
                        self._retire_task_tts_locked(
                            task.task_id,
                            task.identity,
                        )
                        reaped.append(task)
            epoch = self.speech_epoch
        for task in reaped:
            log.warning(
                "reaped overdue task %s (mode=%s, capability=%s) -- exceeded its "
                "%.0fs deadline; the controller is moving on",
                task.task_id, task.mode.value, task.capability, self._timeout_for(task.mode),
            )
            self.state.failures.append(f"task {task.task_id} timed out")
            # Tell the user the turn was dropped, instead of leaving dead air --
            # but only for a turn that would have spoken (not dictation / meeting
            # notes / a proactive follow-up). Stamped with the current epoch so a
            # barge-in still suppresses it.
            if task.metadata.get("speak", True) and not task.metadata.get("followup"):
                aux_tts_id = ""
                try:
                    aux_tts_id = self.register_aux_tts(
                        task.task_id,
                        speech_epoch=epoch,
                        input_generation=task.metadata.get("input_generation"),
                        input_epoch=task.metadata.get("input_epoch"),
                    )
                    if not aux_tts_id:
                        raise AdmissionRejected(
                            "timeout apology identity was not admitted"
                        )
                    self.publish(
                        AgentEvent(
                            EventKind.TTS_REQUEST,
                            {
                                "task_id": task.task_id,
                                "text": _TIMEOUT_APOLOGY,
                                "epoch": epoch,
                                "auxiliary_tts": True,
                                "aux_tts_id": aux_tts_id,
                                **self.auxiliary_tts_identity_payload(
                                    aux_tts_id
                                ),
                            },
                        )
                    )
                except AdmissionRejected:
                    log.warning(
                        "timeout apology omitted: actor identity capacity unavailable"
                    )
                    if aux_tts_id:
                        self.note_aux_tts_admitted(aux_tts_id)
                except Exception:  # noqa: BLE001 - lifecycle must still publish
                    log.exception("timeout apology publication failed")
                    if aux_tts_id:
                        self.note_aux_tts_admitted(aux_tts_id)
                else:
                    self.state.spoken_outputs.append(_TIMEOUT_APOLOGY)
            # Drain the queue on the bus thread (TASK_CANCELLED -> _start_queued_tasks
            # there). priority 20 matches the normal cancellation lifecycle event.
            self.publish(
                AgentEvent(
                    EventKind.TASK_CANCELLED,
                    {
                        "task_id": task.task_id,
                        "mode": task.mode.value,
                        "reaped": True,
                        **self._task_identity_payload(task),
                    },
                    priority=20,
                )
            )
        return len(reaped)

    def sweep_expired_confirmations(self, now: float | None = None) -> int:
        """Drop staged confirmations past their TTL and say so (never-stuck backstop,
        the pending-confirmation sibling of ``reap_overdue_tasks``).

        Without this an abandoned "Confirm command: ..." waited forever, so a stray
        "yes" minutes later could approve a long-forgotten action — the TTL closes
        that window. Safe from the watchdog tick thread: the dict mutation happens
        under ``_cancel_lock`` exactly like ``cancel_all``; the spoken cancellation
        is published outside the lock, stamped with the current epoch so a barge-in
        still suppresses it."""
        if self._confirmation_ttl_sec <= 0:
            return 0
        now = time.monotonic() if now is None else now
        expired: list[AgentTask] = []
        with self._cancel_lock:
            for task_id, task in list(self.state.pending_confirmations.items()):
                deadline = task.metadata.get("confirmation_expires_at")
                if deadline is not None and now >= float(deadline):
                    self.state.pending_confirmations.pop(task_id, None)
                    task.cancel()
                    self._retire_task_tts_locked(
                        task.task_id,
                        task.identity,
                    )
                    expired.append(task)
            epoch = self.speech_epoch
        for task in expired:
            log.info(
                "expired pending confirmation %s after %.0fs: %r",
                task.task_id, self._confirmation_ttl_sec, task.input_text,
            )
            text = f"Confirmation expired: {task.input_text}"
            self.state.spoken_outputs.append(text)
            if task.metadata.get("speak", True):
                aux_tts_id = self.register_aux_tts(
                    task.task_id,
                    speech_epoch=epoch,
                    input_generation=task.metadata.get("input_generation"),
                    input_epoch=task.metadata.get("input_epoch"),
                )
                self.publish(
                    AgentEvent(
                        EventKind.TTS_REQUEST,
                        {
                            "task_id": task.task_id,
                            "text": text,
                            "epoch": epoch,
                            "auxiliary_tts": True,
                            "aux_tts_id": aux_tts_id,
                            **self.auxiliary_tts_identity_payload(aux_tts_id),
                        },
                    )
                )
        with self._cancel_lock:
            for task in expired:
                self._release_task_binding_if_unowned_locked(
                    task.task_id,
                    task.identity,
                )
        return len(expired)

    def looks_like_continuation(self, text: str) -> bool:
        """Whether ``text`` would extend the single in-flight assistant turn.

        Read-only mirror of the _maybe_continue gate, for the runtime's input
        gate (core/runtime.py): a short add-on the addressing classifier would
        otherwise INGEST as ambient speech is still *addressed* when a turn is in
        flight, so the gate consults this to let a genuine continuation through
        to the brain instead of silently dropping it.

        Called from the engine/capture thread (``_on_final``), so it snapshots
        active_tasks defensively -- a concurrent mutation from the bus thread
        just yields a conservative ``False`` (the add-on then takes the normal
        gate path) rather than raising."""
        if self._continuation is None or not self._continuation_cfg.enabled:
            return False
        try:
            tasks = list(self.state.active_tasks.values())
        except RuntimeError:  # dict mutated under us on the bus thread
            return False
        if len(tasks) != 1:
            return False
        victim = tasks[0]
        if (
            victim.mode != Mode.ASSISTANT
            or victim.cancel_event.is_set()
            or victim.metadata.get("post_barge_response_only")
        ):
            return False
        return self._continuation.classify(text, victim.input_text) == CONTINUE

    @staticmethod
    def _continuation_lineage(task: AgentTask) -> tuple[str, list[str]]:
        """The original user question + the raw add-ons folded into ``task``.

        A continuation carries its true origin and the list of raw add-on
        utterances in metadata, so a *chain* of add-ons always rebuilds the
        prompt from (origin + all addons) rather than nesting one synthetic
        template inside the next. A plain (non-continuation) task is its own
        origin with no add-ons yet."""
        origin = task.metadata.get("continuation_origin")
        if isinstance(origin, str):
            addons = [str(a) for a in (task.metadata.get("continuation_addons") or [])]
            return origin, addons
        return task.input_text, []

    @staticmethod
    def _compose_continuation_trust(
        task: AgentTask,
        *,
        prior_owner_verified: object,
        prior_origin: object,
    ) -> None:
        """Intersect authorization provenance across synthetic prompt lineage."""
        origin = combine(task.metadata.get("origin"), prior_origin)
        task.metadata["origin"] = origin.value
        task.metadata["owner_verified"] = bool(
            task.metadata.get("owner_verified") is True
            and prior_owner_verified is True
            and origin is Origin.LIVE_AUDIO
        )

    @staticmethod
    def _render_addons(addons: list[str]) -> str:
        return " ".join(a for a in addons if a)

    def _pending_continuation_behind(self, parent_id: str) -> AgentTask | None:
        for task in self.state.queued_tasks:
            if task.metadata.get("continue_after") == parent_id:
                return task
        return None

    def _maybe_continue(self, decision: IntentDecision) -> bool:
        """Fold a follow-up into the in-flight ASSISTANT turn if it's an add-on.

        Returns True when the follow-up was handled as a continuation -- either
        *merged* into one fresh turn (before any audio) or *queued behind* the
        speaking turn (after audio). Returns False to let the caller start it as
        an independent task (today's behaviour). Gated hard so it only fires for
        an unambiguous add-on to a single, live, same-mode assistant turn.
        """
        if self._continuation is None or not self._continuation_cfg.enabled:
            return False
        if decision.kind != IntentKind.ASSISTANT:
            return False
        if self.state.turn_metadata.get("post_barge_response_only"):
            # This final itself may answer once, but can never enter a synthetic
            # continuation lineage whose later task loses the restriction.
            return False
        if self.state.turn_metadata.get("reserved_continuation"):
            merged = self.state.turn_metadata.get("continuation_merged_text")
            if not isinstance(merged, str) or not merged:
                return True
            task = self._create_task(replace(decision, text=merged))
            self._compose_continuation_trust(
                task,
                prior_owner_verified=task.metadata.get(
                    "continuation_victim_owner_verified",
                    False,
                ),
                prior_origin=task.metadata.get(
                    "continuation_victim_origin",
                    "unknown",
                ),
            )
            arrival_kind = task.metadata.get("continuation_arrival_kind")
            queued = False
            if arrival_kind == "continue" and self._should_queue(task):
                if not self._queue_task(task):
                    return True
                queued = True
            elif not self._start_task(task):
                return True
            unrecorded = [
                str(addon)
                for addon in (
                    task.metadata.get("continuation_unrecorded_addons") or []
                )
            ]
            origin = task.metadata.get("continuation_origin")
            if (
                task.metadata.get("continuation_record_origin")
                and isinstance(origin, str)
            ):
                self._record_addon(origin)
            for addon in unrecorded:
                self._record_addon(addon)
            log.info(
                "continuation ARRIVAL-%s: victim %s -> %s",
                str(arrival_kind).upper(),
                task.metadata.get("continuation_of"),
                task.task_id,
            )
            if arrival_kind == "merge" and self._on_turn_merged is not None:
                try:
                    token = task.metadata.get(
                        "continuation_victim_metrics_turn_token"
                    )
                    self._on_turn_merged(
                        int(token) if token is not None else None
                    )
                except Exception:  # noqa: BLE001 - diagnostics must not break merge
                    log.exception("continuation merge callback failed")
            if queued:
                self._start_queued_tasks()
            generation = task.metadata.get("input_generation")
            if (
                self._on_continuation_admitted is not None
                and generation is not None
            ):
                self._on_continuation_admitted(int(generation))
            return True
        expected_input_epoch = self.state.turn_metadata.get("input_epoch")
        # Exactly one live turn to extend. With >1 active task we don't guess a
        # victim, and it keeps the single-victim epoch bump in _cancel_one safe
        # (no sibling stream to strand); the follow-up falls back to a clean task.
        # Snapshot once: the reap (watchdog thread) can pop a task between a
        # len() and a next(iter()), and an emptied dict's next(iter()) raises
        # StopIteration -- which, uncaught, would kill the bus thread.
        active = list(self.state.active_tasks.values())
        if len(active) != 1:
            return False
        victim = active[0]
        if (
            victim.mode != Mode.ASSISTANT
            or victim.cancel_event.is_set()
            or victim.metadata.get("post_barge_response_only")
        ):
            return False
        addon = decision.text
        if self._continuation.classify(addon, victim.input_text) != CONTINUE:
            return False
        with self._cancel_lock:
            if (
                expected_input_epoch is not None
                and int(expected_input_epoch) != self.input_epoch
            ) or (
                self.state.active_tasks.get(victim.task_id) is not victim
                or victim.cancel_event.is_set()
            ):
                return True
        cfg = self._continuation_cfg
        origin, prior_addons = self._continuation_lineage(victim)
        addons = prior_addons + [addon]
        if not victim.started_speaking:
            # BEFORE first audio: nothing has been spoken, so cancel the
            # not-yet-heard turn and answer ONE merged prompt rebuilt from the
            # original ask + every add-on so far -- a single coherent reply, one
            # stream, never a nested template. _cancel_one bumps the epoch so any
            # sentence the victim races out before it exits goes stale and drops.
            merged = cfg.merge_template.format(prev=origin, addon=self._render_addons(addons))
            if not self._cancel_one(
                victim,
                expected_input_epoch=expected_input_epoch,
            ):
                return True
            task = self._create_task(replace(decision, text=merged))
            with self._cancel_lock:
                task.metadata["input_generation"] = self.latest_input_generation
            task.metadata["continuation_of"] = victim.task_id
            task.metadata["continuation_origin"] = origin
            task.metadata["continuation_addons"] = addons
            task.metadata["skip_user_memory"] = True
            self._compose_continuation_trust(
                task,
                prior_owner_verified=victim.metadata.get(
                    "owner_verified",
                    False,
                ),
                prior_origin=victim.metadata.get("origin", "unknown"),
            )
            if not self._start_task(task):
                return True
            if not prior_addons:
                # First add-on in this lineage: the victim may have been cancelled
                # before its worker ingested the original ask, so record it here
                # to guarantee the prior turn is captured. If the victim DID
                # already ingest it, this is a duplicate user entry -- cosmetic
                # only (recall is off by default; the smart-memory layer dedupes
                # when enabled). Guaranteeing capture beats risking a lost turn.
                self._record_addon(origin)
            self._record_addon(addon)
            log.info(
                "continuation MERGE: victim %s superseded by merged turn %s",
                victim.task_id, task.task_id,
            )
            if self._on_turn_merged is not None:
                try:
                    token = victim.metadata.get("metrics_turn_token")
                    self._on_turn_merged(
                        int(token) if token is not None else None
                    )
                except Exception:  # noqa: BLE001 - diagnostics must not break merge
                    log.exception("continuation merge callback failed")
            return True
        # AFTER first audio: the victim is already talking -- don't cancel or
        # re-speak it. If a continuation is ALREADY queued behind this victim,
        # fold the new add-on into it (it hasn't started, so mutating its text is
        # safe and it stays a single follow-up answer -- no second queued task
        # that would start in parallel). Otherwise queue a fresh continuation
        # strictly behind the victim; it starts only when the victim completes.
        with self._cancel_lock:
            if (
                expected_input_epoch is not None
                and int(expected_input_epoch) != self.input_epoch
            ) or (
                self.state.active_tasks.get(victim.task_id) is not victim
                or victim.cancel_event.is_set()
            ):
                return True
            pending = self._pending_continuation_behind(victim.task_id)
        if pending is not None:
            with self._cancel_lock:
                if (
                    expected_input_epoch is not None
                    and int(expected_input_epoch) != self.input_epoch
                ) or pending not in self.state.queued_tasks:
                    return True
                p_origin, p_addons = self._continuation_lineage(pending)
                p_addons = p_addons + [addon]
                pending.metadata["continuation_addons"] = p_addons
                pending.input_text = cfg.continue_template.format(
                    prev=p_origin, addon=self._render_addons(p_addons)
                )
                # FOLD bypasses _create_task, so intersect the new add-on's
                # provenance with the queued synthetic lineage explicitly.
                self._compose_continuation_trust(
                    pending,
                    prior_owner_verified=self.state.turn_owner_verified,
                    prior_origin=self.state.turn_origin,
                )
                if "metrics_turn_token" in self.state.turn_metadata:
                    pending.metadata["metrics_turn_token"] = self.state.turn_metadata[
                        "metrics_turn_token"
                    ]
                else:
                    pending.metadata.pop("metrics_turn_token", None)
                pending.metadata["input_epoch"] = expected_input_epoch
                pending.metadata["input_generation"] = self.latest_input_generation
            self._record_addon(addon)
            log.info("continuation FOLD: add-on merged into queued %s", pending.task_id)
            return True
        cont_text = cfg.continue_template.format(prev=origin, addon=self._render_addons(addons))
        task = self._create_task(replace(decision, text=cont_text))
        task.metadata["continuation_of"] = victim.task_id
        task.metadata["continue_after"] = victim.task_id
        task.metadata["continuation_origin"] = origin
        task.metadata["continuation_addons"] = addons
        task.metadata["skip_user_memory"] = True
        self._compose_continuation_trust(
            task,
            prior_owner_verified=victim.metadata.get("owner_verified", False),
            prior_origin=victim.metadata.get("origin", "unknown"),
        )
        with self._cancel_lock:
            task.metadata["input_generation"] = self.latest_input_generation
        if not self._queue_task(task):
            return True
        self._record_addon(addon)
        log.info(
            "continuation QUEUE: %s queued behind speaking turn %s",
            task.task_id, victim.task_id,
        )
        # Drain now in case the victim finished between the active-task check and
        # here: then 'continue_after' no longer matches and it starts at once;
        # while the victim is still active the parent gate keeps it queued.
        if task is not None:
            self._start_queued_tasks()
        return True

    def _cancel_one(
        self,
        task: AgentTask,
        *,
        expected_input_epoch: object = None,
    ) -> bool:
        """Cancel a single active task and supersede its speech.

        The single-task cousin of :meth:`cancel_all`: bump the epoch (so the
        victim's queued / in-flight sentences go stale and drop via
        ``tts_request_allowed``) and remove it from ``active_tasks``, under the
        cancel lock. Bumping the *global* epoch is safe here only because the
        caller verified the victim is the SOLE active task -- no sibling stream
        can be stranded. Like ``cancel_all`` the lock is held only around
        in-memory bookkeeping, never across engine I/O or a thread join."""
        with self._cancel_lock:
            if (
                expected_input_epoch is not None
                and int(expected_input_epoch) != self.input_epoch
            ) or (
                self.state.active_tasks.get(task.task_id) is not task
                or task.cancel_event.is_set()
            ):
                return False
            self._advance_output_epoch_locked()
            self.state.active_tasks.pop(task.task_id, None)
            self._pending_output_bindings.pop(task.task_id, None)
            task.cancel()
            self._release_task_binding_if_unowned_locked(
                task.task_id,
                task.identity,
            )
            return True

    def _record_addon(self, addon: str) -> None:
        """Ingest the raw add-on as a user turn (the merged prompt is synthetic).

        The continuation task carries ``skip_user_memory`` so the capability
        won't ingest its folded prompt; we record the real user utterance here
        instead. The prior turn was already ingested by the victim, so memory
        keeps both real utterances without the synthetic duplicate. Best-effort."""
        try:
            if self._record_user_memory is not None:
                self._record_user_memory(addon)
            else:
                self.memory.add(addon, tags=("user",))
        except Exception:  # noqa: BLE001 - memory is best-effort, never fatal
            log.debug("continuation: memory.add(addon) failed", exc_info=True)

    def _should_queue(self, task: AgentTask) -> bool:
        # A continuation queued behind a still-speaking turn waits until that
        # parent leaves active_tasks (TASK_COMPLETED drains the queue), so the
        # two never overlap. Once the parent is gone this falls through to the
        # normal checks and the continuation starts.
        parent = task.metadata.get("continue_after")
        if isinstance(parent, str) and parent in self.state.active_tasks:
            return True
        # Global backstop across all modes (realtime-concurrency-4): if the task
        # runtime is at its concurrent-thread ceiling, queue the turn; the
        # existing _start_queued_tasks drains it as threads free up.
        if self.tasks.at_capacity():
            return True
        # Load-elastic admission (control-plane-2): under sustained system load,
        # drop the effective ceiling to 1 -- a second concurrent turn would only
        # thrash an already-saturated CPU/GPU (worst on the weak on-device tiers).
        # Queue (never drop); the queue drains EVENT-driven as the active task
        # completes/cancels (_start_queued_tasks runs on TASK_COMPLETED), not on a
        # load edge -- and a hung active task is reaped on its wall-clock deadline,
        # so a queued turn can never starve permanently. The first turn always
        # admits (no active task to be the "1"); inert without a load reader or
        # below the ceiling, so default behaviour is unchanged.
        if self._load_fraction is not None and self.state.active_tasks:
            load = self._load_fraction()
            if load is not None and load >= self._admission_load_ceiling:
                return True
        if task.mode != Mode.RESEARCH:
            return False
        # Snapshot: the reap (watchdog thread) can pop mid-iteration, which would
        # raise "dictionary changed size during iteration" on the bus thread.
        active_research = sum(
            1 for active in list(self.state.active_tasks.values()) if active.mode == Mode.RESEARCH
        )
        return active_research >= self.analyzer.policy.research_parallel_tasks

    def _queue_task(self, task: AgentTask) -> bool:
        """Bounded FIFO admission for ``queued_tasks`` (backlog: unbounded queue).

        Past the cap, the OLDEST droppable turn is cancelled and removed —
        drop-oldest keeps the newest input (voice UX: the user's latest request
        wins), matching the playback queue's overflow policy. A continuation task
        extends the live turn, so a stale cold task is preferred as the victim.
        Mutation under ``_cancel_lock`` like every other queued_tasks writer; the
        spoken notice is once-per-storm (reset when a drain empties the queue)."""
        dropped: AgentTask | None = None
        with self._cancel_lock:
            if not self._input_metadata_current_locked(task.metadata):
                task.cancel()
                self._release_task_binding_if_unowned_locked(
                    task.task_id,
                    task.identity,
                )
                return False
            self.state.queued_tasks.append(task)
            if len(self.state.queued_tasks) > self._max_queued_tasks:
                idx = next(
                    (i for i, t in enumerate(self.state.queued_tasks)
                     if not t.metadata.get("continuation_of")),
                    0,
                )
                dropped = self.state.queued_tasks.pop(idx)
                dropped.cancel()
                self._release_task_binding_if_unowned_locked(
                    dropped.task_id,
                    dropped.identity,
                )
        if dropped is None:
            return True
        log.warning(
            "queued-task overflow: dropped oldest %s (mode=%s, %r); cap=%d",
            dropped.task_id, dropped.mode.value, dropped.input_text, self._max_queued_tasks,
        )
        self.state.failures.append(f"task {dropped.task_id} dropped (queue full)")
        if not self._queue_overflow_announced:
            self._queue_overflow_announced = True
            self.state.spoken_outputs.append(
                "I'm at capacity — dropping the oldest queued request."
            )
        return True

    def _start_queued_tasks(self) -> None:
        # Snapshot + swap the queue under the cancel lock so a concurrent
        # cancel_all (audio thread) cannot clear queued_tasks mid-iteration and
        # have this method reassign a stale `remaining` over the cleared list --
        # which would resurrect a cancelled turn (rc-4). Capture the epoch too:
        # if a barge-in / supersede advances it during the drain, the snapshot is
        # stale and the rest of the pass is dropped (newest-input-wins).
        with self._cancel_lock:
            start_epoch = self.speech_epoch
            pending = list(self.state.queued_tasks)
            self.state.queued_tasks = []
        remaining: list[AgentTask] = []
        for index, task in enumerate(pending):
            # A barge-in/stop landed mid-drain (cancel_all/_cancel_one bumped the
            # epoch and already cleared the live queue + cancelled active work):
            # drop the rest of this stale pass instead of starting/re-queuing it.
            if self.speech_epoch != start_epoch:
                with self._cancel_lock:
                    # Earlier items may already have been classified into
                    # ``remaining`` while this snapshot was invisible to
                    # cancel_all. They are just as stale as the unvisited tail.
                    for stale_task in (*remaining, *pending[index:]):
                        stale_task.cancel()
                        self._release_task_binding_if_unowned_locked(
                            stale_task.task_id,
                            stale_task.identity,
                        )
                return
            if task.cancel_event.is_set():
                with self._cancel_lock:
                    self._release_task_binding_if_unowned_locked(
                        task.task_id,
                        task.identity,
                    )
                continue  # cancelled while queued -- drop it
            if self._should_queue(task):
                remaining.append(task)
                continue
            # Start INLINE (outside the lock; _start_task re-acquires it). Doing
            # it here -- not in a deferred second pass -- means each start updates
            # active_tasks / the thread registry BEFORE the next _should_queue
            # check, so a single drain pass cannot over-admit past the per-mode
            # RESEARCH cap or the global thread ceiling. _start_task drops the
            # task if it was cancelled or the epoch advanced in the meantime.
            try:
                self._start_task(task, expected_epoch=start_epoch)
            except BaseException:
                # _start_task owns rollback of the task it attempted.  This
                # drain still owns every previously deferred item and the
                # unvisited tail hidden by the snapshot/swap above.  Preserve
                # those siblings in FIFO order when the turn is current; if a
                # cancellation advanced the epoch, fence and release them.
                stranded = [*remaining, *pending[index + 1 :]]
                with self._cancel_lock:
                    if self.speech_epoch == start_epoch:
                        self.state.queued_tasks = (
                            stranded + self.state.queued_tasks
                        )
                    else:
                        for stale_task in stranded:
                            stale_task.cancel()
                            self._release_task_binding_if_unowned_locked(
                                stale_task.task_id,
                                stale_task.identity,
                            )
                raise
        with self._cancel_lock:
            # A barge landing after the loop but before this re-queue would
            # otherwise strand `remaining` back onto a queue the user moved past;
            # re-check the epoch atomically with the re-queue and drop if stale.
            if self.speech_epoch != start_epoch:
                for stale_task in remaining:
                    stale_task.cancel()
                    self._release_task_binding_if_unowned_locked(
                        stale_task.task_id,
                        stale_task.identity,
                    )
                return
            # Re-queue the not-yet-runnable tasks AHEAD of anything appended
            # while we were unlocked, preserving FIFO order.
            self.state.queued_tasks = remaining + self.state.queued_tasks
            if not self.state.queued_tasks:
                # Storm over: the next overflow may speak its one-line notice again.
                self._queue_overflow_announced = False

    def _confirm_next(
        self,
        owner_verified: bool = False,
        *,
        origin: str = "unknown",
        direct_user_instruction: bool = False,
        input_generation: object = None,
        input_epoch: object = None,
        expected_control: Mapping[str, object] | None = None,
    ) -> None:
        with self._cancel_lock:
            if (
                expected_control is not None
                and not self._control_event_current_locked(expected_control)
            ):
                return
            if not self.state.pending_confirmations:
                self.state.spoken_outputs.append("Nothing to confirm.")
                return
            task_id, task = next(iter(self.state.pending_confirmations.items()))
        # Bind the confirm to the specific staged action (oldest pending) and read it
        # back, instead of a blind "yes confirms whatever". If that action was staged
        # from an OWNER-VERIFIED command, the confirm must ALSO be owner-verified --
        # so an ambient/leaked "yes" can't approve the owner's pending destructive
        # action (the spoofable-confirm race). A non-owner-gated staged task (e.g.
        # legacy / pre-enrollment) confirms as before; its execution is still
        # refused by the capability chokepoint if it isn't owner-verified.
        # Strict `is True` (mirrors origin.is_action_allowed): a truthy non-bool
        # never counts as owner-verified, so the confirm boundary fails closed
        # identically to the action chokepoint regardless of caller coercion.
        with self._cancel_lock:
            if (
                expected_control is not None
                and not self._control_event_current_locked(expected_control)
            ):
                return
            if (
                self.state.pending_confirmations.get(task_id) is not task
                or task.cancel_event.is_set()
                or self._stopped
            ):
                return
            if (
                task.metadata.get("owner_verified") is True
                and owner_verified is not True
            ):
                self.state.spoken_outputs.append(
                    f"I can't confirm '{task.input_text}' without "
                    "verified-owner authorization."
                )
                return
            if (
                str(task.metadata.get("origin", "unknown")) == "live_audio"
                and not (
                    origin == "live_audio"
                    and direct_user_instruction is True
                )
            ):
                self.state.spoken_outputs.append(
                    f"I can't confirm '{task.input_text}' without a direct "
                    "live voice confirmation."
                )
                return
            # Confirmation is a new, separately validated input. Rebind only
            # the cancellation identity; preserve the staged action's origin /
            # owner authorization and all other metadata.
            task.metadata["input_epoch"] = (
                int(input_epoch)
                if input_epoch is not None
                else self.state.turn_metadata.get("input_epoch", self.input_epoch)
            )
            task.metadata["input_generation"] = (
                int(input_generation)
                if input_generation is not None
                else self.state.turn_metadata.get(
                    "input_generation",
                    self.latest_input_generation,
                )
            )
            task.metadata["confirmed"] = True
        if self._start_task(
            task,
            pending_confirmation_id=task_id,
        ):
            self.state.spoken_outputs.append(f"Confirmed: {task.input_text}")

    def _deny_all(
        self,
        *,
        expected_control: Mapping[str, object] | None = None,
    ) -> None:
        with self._cancel_lock:
            if (
                expected_control is not None
                and not self._control_event_current_locked(expected_control)
            ):
                return
            if not self.state.pending_confirmations:
                self.state.spoken_outputs.append("Nothing to cancel.")
                return
            denied = tuple(self.state.pending_confirmations.values())
            for task in denied:
                task.cancel()
                self._retire_task_tts_locked(task.task_id, task.identity)
            self.state.pending_confirmations.clear()
            for task in denied:
                self._release_task_binding_if_unowned_locked(
                    task.task_id,
                    task.identity,
                )
        self.state.spoken_outputs.append("Command cancelled.")

    def _handle_task_completed(self, event: AgentEvent) -> None:
        task_id = str(event.payload.get("task_id", ""))
        text = str(event.payload.get("text", "")).strip()
        speak = bool(event.payload.get("speak", True))
        data = event.payload.get("data", {})
        streamed = bool(isinstance(data, dict) and data.get("streamed"))
        # Drop a superseded completion: if this task finished and enqueued its
        # TASK_COMPLETED, but a continuation merge (or a barge-in) then advanced
        # the speech epoch, the answer belongs to a turn the user has moved on
        # from. Neither speak it nor remember it -- the merged/continuation turn
        # owns the reply now. A completion stamped with the *current* epoch (the
        # common case: nothing preempted it) is unaffected.
        completed_epoch = event.payload.get("epoch")
        with self._cancel_lock:
            candidate = self.state.active_tasks.get(task_id)
            if not self._terminal_event_matches_locked(
                task_id,
                event.payload,
                candidate,
            ):
                log.warning(
                    "dropping TASK_COMPLETED with stale task identity: %s",
                    task_id,
                )
                return
            event_identity = self._resolved_terminal_identity_locked(
                event.payload,
                candidate,
            )
            task = (
                candidate
                if (
                    candidate is not None
                    and candidate.identity == event_identity
                )
                else None
            )
            if task is not None:
                self.state.active_tasks.pop(task_id, None)
            known = self.session_actor.identity_for_task(task_id)
            stale = (
                (
                    completed_epoch is not None
                    and int(completed_epoch) < self.speech_epoch
                )
                or (
                    task is not None
                    and (
                        task.cancel_event.is_set()
                        or known != task.identity
                    )
                )
                or (
                    task is None
                    and any(
                        field in event.payload
                        for field in TaskIdentity.PAYLOAD_FIELDS
                    )
                )
            )
            output_event_pending = bool(
                not stale
                and text
                and speak
                and not streamed
                and self._defer_output_until_tts_admission
                and task is not None
                and known == task.identity
            )
            if output_event_pending:
                assert task is not None and task.identity is not None
                self._pending_output_bindings[task_id] = task.identity
            stream_audio_pending = bool(
                not stale
                and task is not None
                and streamed
                and speak
            )
            if stream_audio_pending:
                # TASK_COMPLETED outranks already-queued sentence TTS events.
                # Keep the completed stream visible until its explicit end
                # marker drains, so newer input can still cancel/continue it.
                self.state.pending_audio_tasks[task_id] = task
                task.metadata["stream_audio_pending"] = True
                task.metadata["pending_stream_is_followup"] = bool(
                    event.payload.get("followup")
                )
                task.metadata["pending_stream_should_followup"] = bool(
                    speak and text
                )
            if (
                not stale
                and task is not None
                and text
                and speak
                and not streamed
                and not task.started_speaking
                and self._defer_output_until_tts_admission
            ):
                # Move atomically: arrival-time cancellation sees either the
                # active task or this queued-audio victim, never a gap.
                self.state.pending_audio_tasks[task_id] = task
                task.metadata["pending_assistant_output"] = text
                task.metadata["pending_assistant_followup"] = bool(
                    event.payload.get("followup")
                )
                defer_memory = True
            else:
                defer_memory = False
            if stale:
                self._retire_task_tts_locked(task_id, event_identity)
            self._release_task_binding_if_unowned_locked(
                task_id,
                event_identity,
            )
            self._accepted_terminal_events.append(
                (event, event_identity, not stale)
            )
        self._start_queued_tasks()
        if stale:
            log.info("dropping superseded completion %s (stale epoch)", task_id)
            return
        is_followup = bool(event.payload.get("followup"))
        # Proactive follow-ups are conversational filler, not facts: keep them
        # out of long-term memory.
        if text and not is_followup and not defer_memory and not streamed:
            self.memory.add(text, tags=("assistant_output",))
        if text and speak:
            if not defer_memory and not streamed:
                self.state.spoken_outputs.append(text)
            # When the capability streamed sentence-by-sentence, the audio has
            # already been spoken during the task; don't re-speak the whole text.
            if not streamed:
                # Carry the task's own epoch (captured at start) so the runtime
                # drops this reply if a barge-in/stop/merge advanced the epoch
                # before it is spoken (realtime-concurrency-1). A completion
                # without a stamp (legacy/manual publish) keeps the prior
                # behaviour of using the current epoch.
                if completed_epoch is not None:
                    epoch = int(completed_epoch)
                else:
                    with self._cancel_lock:
                        epoch = self.speech_epoch
                self.publish(
                    AgentEvent(
                        EventKind.TTS_REQUEST,
                        {
                            "task_id": task_id,
                            "text": text,
                            "epoch": epoch,
                            "input_generation": event.payload.get(
                                "input_generation"
                            ),
                            "metrics_turn_token": event.payload.get(
                                "metrics_turn_token"
                            ),
                            **self._event_identity_payload(event.payload),
                        },
                    )
                )
        # Arm (or continue) the silence cadence once the assistant has spoken.
        if speak and text and not defer_memory and not stream_audio_pending:
            self._schedule_followup()

    def _handle_tts_stream_end(self, event: AgentEvent) -> None:
        """Release a completed stream after every queued sentence was handled."""
        task_id = str(event.payload.get("task_id", ""))
        with self._cancel_lock:
            candidate = self.state.pending_audio_tasks.get(task_id)
            if not self._terminal_event_matches_locked(
                task_id,
                event.payload,
                candidate,
            ):
                log.warning(
                    "dropping TTS_STREAM_END with stale task identity: %s",
                    task_id,
                )
                return
            event_identity = self._resolved_terminal_identity_locked(
                event.payload,
                candidate,
            )
            task = (
                candidate
                if (
                    candidate is not None
                    and candidate.identity == event_identity
                )
                else None
            )
            if task is not None:
                self.state.pending_audio_tasks.pop(task_id, None)
            # Every legitimate sentence was published before this lower-priority
            # marker. Tombstone the producer now so a buggy/late emitter cannot
            # reopen an unclosable playback-history group after stream end.
            self._retire_task_tts_locked(task_id, event_identity)
            schedule_followup = bool(
                task is not None
                and not task.cancel_event.is_set()
                and task.metadata.pop("pending_stream_should_followup", False)
            )
            is_followup = bool(
                task is not None
                and task.metadata.pop("pending_stream_is_followup", False)
            )
            fragments = (
                task.metadata.pop("admitted_tts_fragments", [])
                if task is not None
                else []
            )
            if task is not None:
                task.metadata.pop("stream_audio_pending", None)
            self._release_task_binding_if_unowned_locked(
                task_id,
                event_identity,
            )
            self._accepted_terminal_events.append(
                (event, event_identity, True)
            )
        admitted_text = " ".join(
            str(fragment).strip()
            for fragment in fragments
            if str(fragment).strip()
        )
        if self._defer_output_until_playback_receipt:
            # TTS_STREAM_END means only that the model stopped PRODUCING
            # sentences.  A receipt-capable runtime owns the later sink-drain
            # boundary and publishes one MEMORY_COMMIT from the safe prefixes.
            return
        if admitted_text:
            self.state.spoken_outputs.append(admitted_text)
            if not is_followup:
                self.memory.add(admitted_text, tags=("assistant_output",))
        if schedule_followup:
            self._schedule_followup()

    def _set_mode(
        self,
        mode_value: str,
        *,
        expected_control: Mapping[str, object] | None = None,
    ) -> None:
        with self._cancel_lock:
            if (
                expected_control is not None
                and not self._control_event_current_locked(expected_control)
            ):
                return
            try:
                self.state.mode = Mode(mode_value)
            except ValueError:
                return
        self._reset_followups()
        self.state.spoken_outputs.append(f"Mode: {self.state.mode.value}")
