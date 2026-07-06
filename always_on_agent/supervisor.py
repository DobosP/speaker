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

# Spoken when a turn is reaped for exceeding its deadline, so a timed-out turn
# isn't just dead air.
_TIMEOUT_APOLOGY = "Sorry, that took too long -- let's try again."

# Spoken when a turn FAILS outright (capability/LLM error), so a failed turn
# isn't just dead air either (sr-2). Mirror in core/conversation.py's excluded
# replies so it isn't ingested as conversational content.
_FAILURE_APOLOGY = "Sorry, I ran into a problem with that -- let's try again."

_ACK_THEN_THINK_POLICY = "ack_then_think"
_ACK_THEN_THINK_TEXT = "I'll check that now."

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
    queued_tasks: list[AgentTask] = field(default_factory=list)
    pending_confirmations: dict[str, AgentTask] = field(default_factory=dict)
    # Speaker-ID trust of the CURRENT turn's utterance (action chokepoint). Set per
    # final from the event payload; stamped onto each task created for the turn so
    # the capability layer can refuse a side-effecting action that isn't
    # owner-verified live audio. Default FAIL-CLOSED.
    turn_owner_verified: bool = False
    turn_origin: str = "unknown"
    turn_metadata: dict[str, object] = field(default_factory=dict)
    spoken_outputs: "deque[str]" = field(default_factory=lambda: deque(maxlen=512))
    event_log: "deque[AgentEvent]" = field(default_factory=lambda: deque(maxlen=1024))
    failures: "deque[str]" = field(default_factory=lambda: deque(maxlen=128))


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
        load_fraction: Optional[Callable[[], Optional[float]]] = None,
        admission_load_ceiling: float = 0.85,
        on_turn_merged: Optional[Callable[[], None]] = None,
    ):
        self.bus = bus or EventBus()
        self.state = SupervisorState()
        self.memory = memory or SessionMemory()
        self.capabilities = capabilities or create_default_capabilities(self.memory)
        self.analyzer = analyzer or LiveSpeechAnalyzer()
        self.tasks = TaskRuntime(self.bus.publish, self.capabilities, stream_tts=stream_tts)
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
        self._task_timeouts = dict(DEFAULT_TASK_TIMEOUTS)
        if isinstance(task_timeouts, Mapping):
            for key, value in task_timeouts.items():
                try:
                    self._task_timeouts[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        self._followup_state = FollowupState(
            markers=self.followups.markers, max_followups=self.followups.max_followups
        )
        self._followup_timer: Timer | None = None
        # Barge-in determinism (realtime-concurrency-1): cancellation can be
        # requested from the audio thread (runtime barge-in/stop) *and* the bus
        # thread (CONTROL_STOP). This lock makes the cancel atomic and guards
        # the monotonic speech epoch. It is held only around in-memory bookkeeping
        # -- never across an engine call, ``stop_speaking()``, or a thread join.
        self._cancel_lock = Lock()
        # Bumped on every cancel_all. A TTS_REQUEST stamped with an older epoch
        # (i.e. produced by an interrupted turn) must not reach ``engine.speak``.
        self.speech_epoch = 0
        self.bus.subscribe(self.handle_event)

    def publish(self, event: AgentEvent) -> None:
        self.bus.publish(event)

    def handle_event(self, event: AgentEvent) -> None:
        # STT_PARTIAL is the highest-volume kind (one per ASR frame) and no
        # consumer reads partials back out of the log -- keep it out so the
        # bounded event_log isn't churned through by interim transcripts
        # (rc-6/aq-7). The STT_PARTIAL dispatch below still runs.
        if event.kind != EventKind.STT_PARTIAL:
            self.state.event_log.append(event)

        if event.kind == EventKind.STT_PARTIAL:
            self._handle_speech(str(event.payload.get("text", "")), is_final=False)
            return
        if event.kind == EventKind.STT_FINAL:
            self._handle_speech(
                str(event.payload.get("text", "")),
                is_final=True,
                owner_verified=bool(event.payload.get("owner_verified", False)),
                origin=str(event.payload.get("origin", "unknown")),
                metadata=event.payload.get("metadata"),
            )
            return
        if event.kind == EventKind.FOLLOWUP_TICK:
            self._emit_followup()
            return
        if event.kind == EventKind.CONTROL_STOP:
            self.cancel_all()
            return
        if event.kind == EventKind.CONTROL_MODE:
            self._set_mode(str(event.payload.get("mode", "")))
            return
        if event.kind == EventKind.CONTROL_CONFIRM:
            self._confirm_next(owner_verified=bool(event.payload.get("owner_verified", False)))
            return
        if event.kind == EventKind.CONTROL_DENY:
            self._deny_all()
            return
        if event.kind == EventKind.TASK_COMPLETED:
            self._handle_task_completed(event)
            return
        if event.kind == EventKind.TASK_CANCELLED:
            self.state.active_tasks.pop(str(event.payload.get("task_id", "")), None)
            self._start_queued_tasks()
            return
        if event.kind == EventKind.TASK_FAILED:
            task_id = str(event.payload.get("task_id", ""))
            self.state.active_tasks.pop(task_id, None)
            self.state.failures.append(str(event.payload.get("error", "")))
            # Don't leave dead air on a failed turn -- speak an apology, but only
            # for a turn that would have spoken (not dictation / meeting notes /
            # a proactive follow-up). Epoch-stamped so a barge-in still
            # suppresses it (mirrors the reap path's _TIMEOUT_APOLOGY; sr-2).
            # Older events may lack the new payload keys -> default speak=True
            # and fall back to the current epoch.
            if event.payload.get("speak", True) and not event.payload.get("followup"):
                self.state.spoken_outputs.append(_FAILURE_APOLOGY)
                self.publish(
                    AgentEvent(
                        EventKind.TTS_REQUEST,
                        {
                            "task_id": task_id,
                            "text": _FAILURE_APOLOGY,
                            "epoch": event.payload.get("epoch", self.speech_epoch),
                        },
                    )
                )
            self._start_queued_tasks()

    def drain(self) -> int:
        return self.bus.drain()

    def shutdown(self) -> None:
        """Cancel background timers (called on runtime stop)."""
        self._cancel_followup()

    def cancel_all(self) -> None:
        """Preempt every active/queued/pending task. Safe to call from the audio
        thread (runtime barge-in/stop) or the bus thread (CONTROL_STOP).

        The speech epoch is bumped and every active task is cancelled *under the
        lock* so that, the instant this returns, no in-flight task can be treated
        as active and no prior-epoch sentence will be spoken. ``task.cancel()``
        only sets an :class:`Event`, so the lock is never held across blocking
        work (no engine call, no ``stop_speaking()``, no thread join)."""
        with self._cancel_lock:
            self.speech_epoch += 1
            log.info(
                "cancel_all: epoch=%d %d active, %d queued, %d pending-confirm",
                self.speech_epoch,
                len(self.state.active_tasks), len(self.state.queued_tasks),
                len(self.state.pending_confirmations),
            )
            for task in list(self.state.active_tasks.values()):
                task.cancel()
            # Cancel queued (not-yet-started) tasks too, BEFORE clearing the
            # list (rc-4): a concurrent _start_queued_tasks (bus thread) may
            # already hold a reference to one of these, so setting cancel_event
            # ensures _start_task drops it instead of resurrecting it as active
            # after a barge-in.
            for task in self.state.queued_tasks:
                task.cancel()
            self.state.queued_tasks.clear()
            self.state.pending_confirmations.clear()
        # Out of the lock: timer cancel touches a Timer thread and the spoken log
        # is independent of the cancel bookkeeping above.
        self._reset_followups()
        self.state.spoken_outputs.append("[cancelled]")

    def tts_request_allowed(self, task_id: str, epoch: int | None = None) -> bool:
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
            if epoch is not None:
                return epoch >= self.speech_epoch
            if not task_id:
                return True
            task = self.state.active_tasks.get(task_id)
            return task is not None and not task.cancel_event.is_set()

    # --- proactive follow-ups ------------------------------------------------
    def _schedule_followup(self) -> None:
        """Arm a timer that, after a silent delay, nudges the conversation."""
        if not self.followups.enabled or not self._followup_state.can_continue():
            return
        self._cancel_followup()
        timer = Timer(self.followups.delay_sec, self._tick_followup)
        timer.daemon = True
        self._followup_timer = timer
        timer.start()

    def _cancel_followup(self) -> None:
        timer = self._followup_timer
        if timer is not None:
            timer.cancel()
            self._followup_timer = None

    def _reset_followups(self) -> None:
        self._cancel_followup()
        self._followup_state.reset()

    def _tick_followup(self) -> None:
        """Timer-thread callback: hand control back to the bus thread."""
        self.publish(AgentEvent(EventKind.FOLLOWUP_TICK, priority=95))

    def _emit_followup(self) -> None:
        """Bus-thread handler: start a follow-up task if still warranted."""
        if not self.followups.enabled or not self._followup_state.can_continue():
            return
        if self.state.active_tasks or self.state.queued_tasks:
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
        task = self._create_task(decision)
        task.metadata["followup"] = True
        # A proactive follow-up is assistant-initiated (a synthetic "[silent]"
        # marker the owner never spoke), so it must NEVER carry the prior turn's
        # owner-verified trust -- fail-close it so it can't drive an action.
        task.metadata["owner_verified"] = False
        task.metadata["origin"] = "system"
        self._start_task(task)

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
            self.state.turn_owner_verified = bool(owner_verified)
            self.state.turn_origin = str(origin)
            self.state.turn_metadata = (
                {"latency_policy": metadata["latency_policy"]}
                if isinstance(metadata, Mapping) and "latency_policy" in metadata
                else {}
            )
        observation = self.analyzer.observe(text, is_final=is_final)
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
        self._execute_decision(decision)

    def _create_task(self, decision: IntentDecision) -> AgentTask:
        """Create a task and stamp it with the current turn's speaker-ID trust, so
        the capability layer's action chokepoint (always_on_agent.origin) sees
        owner_verified/origin. Fail-closed: defaults to the not-verified turn trust."""
        task = self.tasks.create_task(decision)
        task.metadata["owner_verified"] = bool(self.state.turn_owner_verified)
        task.metadata["origin"] = str(self.state.turn_origin)
        task.metadata.update(self.state.turn_metadata)
        return task

    def _execute_decision(self, decision: IntentDecision) -> None:
        if decision.kind == IntentKind.IGNORE:
            return
        if decision.kind == IntentKind.STOP:
            self.publish(AgentEvent.stop(decision.reason))
            return
        if decision.kind == IntentKind.CONFIRM:
            # Carry the speaker-ID trust of THIS "yes" so an owner-verified staged
            # action can only be approved by an owner-verified confirm.
            self.publish(AgentEvent.confirm(owner_verified=self.state.turn_owner_verified))
            return
        if decision.kind == IntentKind.DENY:
            self.publish(AgentEvent.deny())
            return
        if decision.kind == IntentKind.MODE_SWITCH and decision.target_mode is not None:
            self.publish(AgentEvent.mode(decision.target_mode))
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
        if task.metadata.get("requires_confirmation"):
            self.state.pending_confirmations[task.task_id] = task
            self.state.spoken_outputs.append(f"Confirm command: {task.input_text}")
            return
        if self._should_queue(task):
            self.state.queued_tasks.append(task)
            self.state.spoken_outputs.append(f"Queued {task.mode.value}: {task.input_text}")
            return
        self._start_task(task)

    def _start_task(self, task: AgentTask, expected_epoch: int | None = None) -> None:
        # Capture the current speech epoch and register the task atomically with
        # respect to cancel_all (realtime-concurrency-1): the streaming sentences
        # this task emits are stamped with this epoch, so a barge-in that
        # advances the epoch drops them even though TASK_COMPLETED (priority 60)
        # is dequeued before the trailing TTS_REQUESTs (priority 100) and has
        # already removed the task from active_tasks.
        with self._cancel_lock:
            # Drop a task whose cancel_event was set after it was queued (e.g. a
            # cancel_all / _cancel_one landed between the queue check and here):
            # never register or spawn a worker for an already-cancelled turn
            # (rc-4). The re-check under the lock is the authoritative drop.
            if task.cancel_event.is_set():
                return
            # rc-4 residual window: a task snapshotted by _start_queued_tasks is
            # invisible to cancel_all (it's neither in active_tasks nor
            # queued_tasks during the drain), so cancel_all can't set its
            # cancel_event. Re-check the epoch under the lock too: if a barge-in /
            # supersede advanced it since the drain started, drop the task rather
            # than resurrect a turn the user moved past.
            if expected_epoch is not None and self.speech_epoch != expected_epoch:
                return
            task.speech_epoch = self.speech_epoch
            timeout = self._timeout_for(task.mode)
            if timeout > 0:
                task.deadline_at = time.monotonic() + timeout
            self.state.active_tasks[task.task_id] = task
            epoch = task.speech_epoch
        self._emit_latency_ack(task, epoch)
        self.tasks.start(task)

    def _emit_latency_ack(self, task: AgentTask, epoch: int) -> None:
        if task.metadata.get("latency_policy") != _ACK_THEN_THINK_POLICY:
            return
        if not task.metadata.get("speak", True) or task.metadata.get("followup"):
            return
        task.started_speaking = True
        self.state.spoken_outputs.append(_ACK_THEN_THINK_TEXT)
        self.publish(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {
                    "task_id": task.task_id,
                    "text": _ACK_THEN_THINK_TEXT,
                    "epoch": epoch,
                    "latency_ack": True,
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
        here, so the supervisor stops treating it as live -- even though the
        daemon worker may still be blocked (it exits when its own I/O finally
        returns, or it leaks harmlessly as a daemon).

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
                    self.state.active_tasks.pop(task.task_id, None)
                    task.cancel()
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
                self.state.spoken_outputs.append(_TIMEOUT_APOLOGY)
                self.publish(
                    AgentEvent(
                        EventKind.TTS_REQUEST,
                        {"task_id": task.task_id, "text": _TIMEOUT_APOLOGY, "epoch": epoch},
                    )
                )
            # Drain the queue on the bus thread (TASK_CANCELLED -> _start_queued_tasks
            # there). priority 20 matches the normal cancellation lifecycle event.
            self.publish(
                AgentEvent(
                    EventKind.TASK_CANCELLED,
                    {"task_id": task.task_id, "mode": task.mode.value, "reaped": True},
                    priority=20,
                )
            )
        return len(reaped)

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
        if victim.mode != Mode.ASSISTANT or victim.cancel_event.is_set():
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
        if victim.mode != Mode.ASSISTANT or victim.cancel_event.is_set():
            return False
        addon = decision.text
        if self._continuation.classify(addon, victim.input_text) != CONTINUE:
            return False
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
            self._cancel_one(victim)
            task = self._create_task(replace(decision, text=merged))
            task.metadata["continuation_of"] = victim.task_id
            task.metadata["continuation_origin"] = origin
            task.metadata["continuation_addons"] = addons
            task.metadata["skip_user_memory"] = True
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
                    self._on_turn_merged()
                except Exception:  # noqa: BLE001 - diagnostics must not break merge
                    log.exception("continuation merge callback failed")
            self._start_task(task)
            return True
        # AFTER first audio: the victim is already talking -- don't cancel or
        # re-speak it. If a continuation is ALREADY queued behind this victim,
        # fold the new add-on into it (it hasn't started, so mutating its text is
        # safe and it stays a single follow-up answer -- no second queued task
        # that would start in parallel). Otherwise queue a fresh continuation
        # strictly behind the victim; it starts only when the victim completes.
        pending = self._pending_continuation_behind(victim.task_id)
        if pending is not None:
            p_origin, p_addons = self._continuation_lineage(pending)
            p_addons = p_addons + [addon]
            pending.metadata["continuation_addons"] = p_addons
            pending.input_text = cfg.continue_template.format(
                prev=p_origin, addon=self._render_addons(p_addons)
            )
            # FOLD is the one task-mutation site that bypasses _create_task, so it
            # must re-stamp trust too -- DEMOTE-only: an UNVERIFIED add-on folded
            # into an owner-verified pending revokes that pending's owner trust
            # (an unverified utterance can't keep authorizing an action it is now
            # changing). Trust survives only if BOTH the pending and the add-on are
            # owner-verified.
            if not self.state.turn_owner_verified:
                pending.metadata["owner_verified"] = False
                pending.metadata["origin"] = self.state.turn_origin
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
        self._record_addon(addon)
        self.state.queued_tasks.append(task)
        log.info(
            "continuation QUEUE: %s queued behind speaking turn %s",
            task.task_id, victim.task_id,
        )
        # Drain now in case the victim finished between the active-task check and
        # here: then 'continue_after' no longer matches and it starts at once;
        # while the victim is still active the parent gate keeps it queued.
        self._start_queued_tasks()
        return True

    def _cancel_one(self, task: AgentTask) -> None:
        """Cancel a single active task and supersede its speech.

        The single-task cousin of :meth:`cancel_all`: bump the epoch (so the
        victim's queued / in-flight sentences go stale and drop via
        ``tts_request_allowed``) and remove it from ``active_tasks``, under the
        cancel lock. Bumping the *global* epoch is safe here only because the
        caller verified the victim is the SOLE active task -- no sibling stream
        can be stranded. Like ``cancel_all`` the lock is held only around
        in-memory bookkeeping, never across engine I/O or a thread join."""
        with self._cancel_lock:
            self.speech_epoch += 1
            self.state.active_tasks.pop(task.task_id, None)
            task.cancel()

    def _record_addon(self, addon: str) -> None:
        """Ingest the raw add-on as a user turn (the merged prompt is synthetic).

        The continuation task carries ``skip_user_memory`` so the capability
        won't ingest its folded prompt; we record the real user utterance here
        instead. The prior turn was already ingested by the victim, so memory
        keeps both real utterances without the synthetic duplicate. Best-effort."""
        try:
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
        for task in pending:
            # A barge-in/stop landed mid-drain (cancel_all/_cancel_one bumped the
            # epoch and already cleared the live queue + cancelled active work):
            # drop the rest of this stale pass instead of starting/re-queuing it.
            if self.speech_epoch != start_epoch:
                return
            if task.cancel_event.is_set():
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
            self._start_task(task, expected_epoch=start_epoch)
        with self._cancel_lock:
            # A barge landing after the loop but before this re-queue would
            # otherwise strand `remaining` back onto a queue the user moved past;
            # re-check the epoch atomically with the re-queue and drop if stale.
            if self.speech_epoch != start_epoch:
                return
            # Re-queue the not-yet-runnable tasks AHEAD of anything appended
            # while we were unlocked, preserving FIFO order.
            self.state.queued_tasks = remaining + self.state.queued_tasks

    def _confirm_next(self, owner_verified: bool = False) -> None:
        if not self.state.pending_confirmations:
            self.state.spoken_outputs.append("Nothing to confirm.")
            return
        # Bind the confirm to the specific staged action (oldest pending) and read it
        # back, instead of a blind "yes confirms whatever". If that action was staged
        # from an OWNER-VERIFIED command, the confirm must ALSO be owner-verified --
        # so an ambient/leaked "yes" can't approve the owner's pending destructive
        # action (the spoofable-confirm race). A non-owner-gated staged task (e.g.
        # legacy / pre-enrollment) confirms as before; its execution is still
        # refused by the capability chokepoint if it isn't owner-verified.
        task_id, task = next(iter(self.state.pending_confirmations.items()))
        # Strict `is True` (mirrors origin.is_action_allowed): a truthy non-bool
        # never counts as owner-verified, so the confirm boundary fails closed
        # identically to the action chokepoint regardless of caller coercion.
        if bool(task.metadata.get("owner_verified", False)) and owner_verified is not True:
            self.state.spoken_outputs.append(
                f"I can't confirm '{task.input_text}' without verified-owner authorization."
            )
            return
        self.state.pending_confirmations.pop(task_id, None)
        self.state.spoken_outputs.append(f"Confirmed: {task.input_text}")
        self._start_task(task)

    def _deny_all(self) -> None:
        if not self.state.pending_confirmations:
            self.state.spoken_outputs.append("Nothing to cancel.")
            return
        self.state.pending_confirmations.clear()
        self.state.spoken_outputs.append("Command cancelled.")

    def _handle_task_completed(self, event: AgentEvent) -> None:
        task_id = str(event.payload.get("task_id", ""))
        self.state.active_tasks.pop(task_id, None)
        self._start_queued_tasks()
        # Drop a superseded completion: if this task finished and enqueued its
        # TASK_COMPLETED, but a continuation merge (or a barge-in) then advanced
        # the speech epoch, the answer belongs to a turn the user has moved on
        # from. Neither speak it nor remember it -- the merged/continuation turn
        # owns the reply now. A completion stamped with the *current* epoch (the
        # common case: nothing preempted it) is unaffected.
        completed_epoch = event.payload.get("epoch")
        if completed_epoch is not None:
            with self._cancel_lock:
                if int(completed_epoch) < self.speech_epoch:
                    log.info("dropping superseded completion %s (stale epoch)", task_id)
                    return
        text = str(event.payload.get("text", "")).strip()
        speak = bool(event.payload.get("speak", True))
        data = event.payload.get("data", {})
        streamed = bool(isinstance(data, dict) and data.get("streamed"))
        is_followup = bool(event.payload.get("followup"))
        # Proactive follow-ups are conversational filler, not facts: keep them
        # out of long-term memory.
        if text and not is_followup:
            self.memory.add(text, tags=("assistant_output",))
        if text and speak:
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
                        {"task_id": task_id, "text": text, "epoch": epoch},
                    )
                )
        # Arm (or continue) the silence cadence once the assistant has spoken.
        if speak and text:
            self._schedule_followup()

    def _set_mode(self, mode_value: str) -> None:
        try:
            self.state.mode = Mode(mode_value)
        except ValueError:
            return
        self._reset_followups()
        self.state.spoken_outputs.append(f"Mode: {self.state.mode.value}")
