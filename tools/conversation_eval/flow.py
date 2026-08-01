"""Deterministic end-to-end conversation-flow acceptance journeys.

The fixtures in this module replace only hardware, network, and model
boundaries.  Every journey still uses the production runtime assembler,
serialized event bus, supervisor, session actor, task runtime, capability
registry, and tracked playback receipts.  The lifecycle journeys exercise
:class:`core.voice_session.VoiceSession`; barge recovery deliberately reuses
the unchanged v4 control-plane journey.

Known fixture text is plumbing evidence only.  Nothing here measures ASR
quality, physical barge-in, real synthesis, or perceived latency.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Event, Lock, Thread
import time
from typing import Callable, Mapping

from always_on_agent.acoustic import (
    AcousticLineage,
    AcousticSource,
    AcousticSpan,
    EndpointReason,
)
from always_on_agent.capabilities import CapabilityResult
from always_on_agent.events import Mode
from always_on_agent.session_actor import SessionSnapshot, TaskIdentity
from core.app import build_runtime
from core.config import deep_merge
from core.endpointing import (
    AdaptiveEndpointPolicy,
    CompletionScoreState,
    EndpointConfig,
    TurnDecisionBasis,
    TurnObservation,
    decide_turn,
)
from core.engine import FinalTranscript, OwnerVerification, PartialTranscript
from core.metrics import SPEECH_END
from core.routing import build_router
from core.voice_session import VoiceSession, VoiceSessionPlan

from .flow_schema import FLOW_SCENARIO_IDS, FlowCheck, FlowScenarioResult
from .runner import TracedScriptedEngine, deterministic_models, run_scenario
from .schema import TraceEvent
from .scenarios import selected
from .trace import TraceRecorder


_WAIT_SEC = 5.0
_STOP_JOIN_SEC = 1.0
_DESCRIPTIONS = {
    "turn_incremental": (
        "A typed acoustic partial is held without work, then one higher-revision "
        "final streams multiple receipt-owned fragments and fences a stale final."
    ),
    "barge_recovery": (
        "The deterministic v4 barge/redirect journey proves control-plane cut and "
        "recovery ownership; physical acoustic interruption is explicitly unproven."
    ),
    "slow_tool_complete_followup": (
        "A slow read-only tool is acknowledged, finishes before its task terminal, "
        "and leaves receipt-backed context for a clean follow-up."
    ),
    "stop_restart_fence": (
        "A stopped session fences a still-returning provider while a fresh session "
        "runs independently under a distinct actor identity."
    ),
}


def _check(
    name: str,
    expected: object,
    actual: object,
    *,
    ok: bool | None = None,
) -> FlowCheck:
    return FlowCheck(
        name=name,
        ok=bool(expected == actual if ok is None else ok),
        expected=expected,
        actual=actual,
    )


def _wait_until(predicate: Callable[[], bool], timeout: float = _WAIT_SEC) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout))
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def _headless_config(config: Mapping[str, object], *, stream_tts: bool) -> dict:
    """Retain production routing while terminally disabling external state."""

    return deep_merge(
        dict(config),
        {
            "local_only": True,
            "warm_on_start": False,
            "voice_session": {"audio_egress": "device_only"},
            "memory": {
                "backend": "inmemory",
                "embeddings": False,
                "recall_enabled": False,
                "profile_enabled": False,
                "procedural_enabled": False,
                "cross_session_continuity": False,
                "persist_assistant": False,
            },
            "llm": {
                "router_model": None,
                "live_routing": False,
                "cloud": {"enabled": False, "strategy": "local_only"},
            },
            "tts": {"streaming": bool(stream_tts)},
            "web_search": {"enabled": False, "base_url": ""},
            "obsidian": {"enabled": False},
            "reminders": {"enabled": False},
            "trusted_apps": {"enabled": False, "apps": {}},
            "watch": {"enabled": False, "grants": []},
            "gui_actions": {"enabled": False},
            "screen_capture": {"enabled": False, "memorize": False},
            "agent_brain": {"local_only": True, "offline": True, "os_mode": False},
        },
    )


class _FlowEngine(TracedScriptedEngine):
    """Scripted boundary with an explicit user-speech metric helper."""

    def mark_user_speech_end(self, *, at: float) -> None:
        self._cb.on_metric(SPEECH_END, at=at)

    def typed_partial(
        self,
        text: str,
        *,
        acoustic: AcousticLineage,
        revision: int,
    ) -> None:
        self.partial_result(
            PartialTranscript(text=text, acoustic=acoustic, revision=revision)
        )

    def typed_final(
        self,
        text: str,
        *,
        acoustic: AcousticLineage,
        revision: int,
    ) -> None:
        self.final_result(
            FinalTranscript(
                text=text,
                owner_verification=OwnerVerification.UNKNOWN,
                origin="live_audio",
                acoustic=acoustic,
                revision=revision,
            )
        )


@dataclass
class _Harness:
    config: dict
    models: object
    trace: TraceRecorder
    engine: _FlowEngine
    runtime: object
    session: VoiceSession
    _remove_observer: Callable[[], None]
    started: bool = False
    observer_removed: bool = False

    @classmethod
    def build(
        cls,
        config: Mapping[str, object],
        trace: TraceRecorder,
        *,
        stream_tts: bool,
        hold_speech: bool = False,
        observe_capabilities: bool = True,
    ) -> "_Harness":
        runtime_config = _headless_config(config, stream_tts=stream_tts)
        models = deterministic_models()
        engine = _FlowEngine(trace, hold_speech=hold_speech)
        runtime = build_runtime(
            runtime_config,
            engine=engine,
            llm=models.main,
            fast_llm=models.fast,
            router=build_router(runtime_config),
            start_mode=Mode.ASSISTANT,
            agent_on=False,
            gui_actions_on=False,
            force_stream_tts=stream_tts,
            load_fraction=None,
        )
        remove = (
            runtime.supervisor.capabilities.observe_invocations(
                trace.on_capability_invocation
            )
            if observe_capabilities
            else (lambda: None)
        )
        runtime.bus.subscribe(trace.on_agent_event)
        plan = VoiceSessionPlan.resolve(
            config=runtime_config,
            requested_session="console",
        )
        return cls(
            runtime_config,
            models,
            trace,
            engine,
            runtime,
            VoiceSession(plan, runtime),
            remove,
        )

    def start(self) -> None:
        self.runtime.start(run_bus=True)
        self.started = True

    def stop(self) -> None:
        self.session.stop()

    def remove_observer(self) -> None:
        if not self.observer_removed:
            self._remove_observer()
            self.observer_removed = True


class _BlockingCapability:
    """Pure in-memory provider used to expose real actor child ownership."""

    def __init__(self, result_text: str) -> None:
        self.result_text = result_text
        self.started = Event()
        self.release = Event()
        self.finished = Event()
        self._lock = Lock()
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, query: str, context: dict[str, object]) -> CapabilityResult:
        with self._lock:
            self.calls.append((query, dict(context)))
        self.started.set()
        try:
            if not self.release.wait(timeout=10.0):
                return CapabilityResult(False, "", error="flow fixture timed out")
            return CapabilityResult(
                True,
                self.result_text,
                data={
                    "source": "flow-fixture",
                    "external_effect_performed": False,
                },
            )
        finally:
            self.finished.set()

    def call_count(self) -> int:
        with self._lock:
            return len(self.calls)

    def call_rows(self) -> tuple[tuple[str, dict[str, object]], ...]:
        with self._lock:
            return tuple((query, dict(context)) for query, context in self.calls)


def _lineage(label: str, *, turn_index: int) -> AcousticLineage:
    emitted_at = time.perf_counter()
    return AcousticLineage.single(
        AcousticSpan(
            stream_id=f"flow-{label}",
            utterance_id=f"utterance-{label}",
            turn_index=turn_index,
            capture_epoch=1,
            capture_generation=1,
            source=AcousticSource.SCRIPTED,
            speech_start_at=max(0.0, emitted_at - 0.25),
            speech_end_at=max(0.0, emitted_at - 0.05),
            endpoint_committed_at=emitted_at,
            emitted_at=emitted_at,
            sample_rate_hz=16000,
            owned_sample_count=4000,
            endpoint_reason=EndpointReason.VAD_SILENCE,
        )
    )


def _event(
    events: tuple[TraceEvent, ...],
    kind: str,
    predicate: Callable[[TraceEvent], bool] | None = None,
) -> TraceEvent | None:
    accept = predicate or (lambda _event: True)
    return next((item for item in events if item.kind == kind and accept(item)), None)


def _events(
    events: tuple[TraceEvent, ...],
    kind: str,
    predicate: Callable[[TraceEvent], bool] | None = None,
) -> tuple[TraceEvent, ...]:
    accept = predicate or (lambda _event: True)
    return tuple(item for item in events if item.kind == kind and accept(item))


def _trace_timing_check(events: tuple[TraceEvent, ...]) -> FlowCheck:
    invalid: list[tuple[object, ...]] = []
    previous_elapsed = -1.0
    for expected_sequence, event in enumerate(events, start=1):
        valid_elapsed = bool(
            isinstance(event.elapsed_ms, (int, float))
            and not isinstance(event.elapsed_ms, bool)
            and math.isfinite(float(event.elapsed_ms))
            and float(event.elapsed_ms) >= previous_elapsed
        )
        if event.sequence != expected_sequence or not valid_elapsed:
            invalid.append(
                (
                    "event",
                    expected_sequence,
                    event.sequence,
                    event.elapsed_ms,
                )
            )
        if valid_elapsed:
            previous_elapsed = float(event.elapsed_ms)
    return _check("causal_trace_timestamps_finite", (), tuple(invalid))


def _active_counts(snapshot: SessionSnapshot) -> tuple[int, int, int, int]:
    return (
        snapshot.active_tasks,
        snapshot.active_providers,
        snapshot.active_tools,
        snapshot.active_playbacks,
    )


def _receipt_texts(events: tuple[TraceEvent, ...]) -> tuple[str, ...]:
    return tuple(
        str(event.payload.get("text", "")).strip()
        for event in events
        if event.kind == "memory.commit"
        and event.payload.get("source") == "playback_receipt"
        and str(event.payload.get("text", "")).strip()
    )


def _fragment_ownership_check(events: tuple[TraceEvent, ...]) -> FlowCheck:
    kinds = (
        "playback.requested",
        "playback.started",
        "playback.terminal",
        "playback.attributed",
    )
    rows = tuple(
        (
            kind,
            tuple(
                sorted(
                    str(event.payload.get("fragment_id", ""))
                    for event in events
                    if event.kind == kind
                )
            ),
        )
        for kind in kinds
    )
    requested = rows[0][1]
    one_to_one = bool(
        requested
        and all(requested == identifiers for _, identifiers in rows[1:])
        and all(requested)
        and len(requested) == len(set(requested))
    )
    return _check(
        "playback_fragment_ownership_one_to_one",
        "same unique nonempty IDs at request/start/terminal/attribution",
        rows,
        ok=one_to_one,
    )


def _memory_snapshot(runtime: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((item.text, tuple(item.tags)) for item in runtime.memory.all())


def _identity_rows(events: tuple[TraceEvent, ...]) -> tuple[tuple[object, ...], ...]:
    kinds = {
        "task.started",
        "task.progress",
        "task.completed",
        "task.cancelled",
        "task.failed",
        "tts.request",
        "tts.stream_end",
    }
    rows: list[tuple[object, ...]] = []
    for event in events:
        if event.kind not in kinds or not event.task_id:
            continue
        if not all(field in event.payload for field in TaskIdentity.PAYLOAD_FIELDS):
            rows.append(("missing", event.kind, event.task_id))
            continue
        rows.append(
            tuple(event.payload[field] for field in TaskIdentity.PAYLOAD_FIELDS)
        )
    return tuple(rows)


def _metric_values(runtime: object) -> dict[str, tuple[float, ...]]:
    names = (
        "first_audio_latency",
        "endpoint_latency",
        "final_to_first_token",
        "first_token_to_audio",
    )
    records = tuple(record.as_dict() for record in runtime.metrics.records())
    return {
        name: tuple(
            float(record[name])
            for record in records
            if isinstance(record.get(name), (int, float))
            and not isinstance(record.get(name), bool)
            and math.isfinite(float(record[name]))
            and float(record[name]) >= 0.0
        )
        for name in names
    }


def _run_turn_incremental(
    config: Mapping[str, object],
    trace: TraceRecorder,
) -> tuple[FlowCheck, ...]:
    harness = _Harness.build(config, trace, stream_tts=True)
    checks: list[FlowCheck] = []
    final_text = "Tell me a lighthouse story."
    lineage = _lineage("incremental", turn_index=1)
    policy = AdaptiveEndpointPolicy(
        EndpointConfig(
            enabled=True,
            min_silence_sec=0.5,
            max_silence_sec=1.6,
        )
    )
    try:
        harness.start()
        speech_end = lineage.spans[-1].speech_end_at
        assert speech_end is not None
        harness.engine.mark_user_speech_end(at=speech_end)

        hold = decide_turn(
            TurnObservation(
                acoustic_endpoint=True,
                vad_active=False,
                early_endpoint_allowed=True,
                trailing_silence_sec=1.0,
                completion_state=CompletionScoreState.SCORED,
                completion_score=0.1,
            ),
            policy=policy,
        )
        trace.mark(
            "flow.endpoint.hold",
            {"commit": hold.commit, "basis": hold.basis.value, "revision": 1},
        )
        if not hold.commit:
            harness.engine.typed_partial(
                "Tell me a lighthouse story and",
                acoustic=lineage,
                revision=1,
            )
        partial_seen = _wait_until(
            lambda: bool(_events(trace.events(), "stt.partial")), timeout=1.0
        )
        held_events = trace.events()
        held_work = tuple(
            event.kind
            for event in held_events
            if event.kind.startswith("task.")
            or event.kind in {"tts.request", "playback.requested"}
        )
        checks.extend(
            (
                _check(
                    "endpoint_hold_decision",
                    (False, TurnDecisionBasis.SEMANTIC_HOLD.value),
                    (hold.commit, hold.basis.value),
                ),
                _check("partial_reached_runtime", True, partial_seen),
                _check("endpoint_hold_created_no_work", (), held_work),
            )
        )

        commit = decide_turn(
            TurnObservation(
                acoustic_endpoint=True,
                vad_active=False,
                early_endpoint_allowed=True,
                trailing_silence_sec=1.7,
                completion_state=CompletionScoreState.SCORED,
                completion_score=0.9,
            ),
            policy=policy,
        )
        trace.mark(
            "flow.endpoint.commit",
            {"commit": commit.commit, "basis": commit.basis.value, "revision": 2},
        )
        harness.models.stream_gate.arm()
        if commit.commit:
            harness.engine.typed_final(
                final_text,
                acoustic=lineage,
                revision=2,
            )

        continuation_withheld = harness.models.stream_gate.paused.wait(
            timeout=_WAIT_SEC
        )
        first_playback = harness.engine.playback_started.wait(timeout=_WAIT_SEC)
        paused_events = trace.events()
        playback_started = _event(paused_events, "playback.started")
        answer_finished = _event(
            paused_events,
            "capability.finished",
            lambda event: event.payload.get("name") == "assistant.answer",
        )
        premature_terminal = tuple(
            event.kind
            for event in paused_events
            if event.kind in {"task.completed", "task.cancelled", "task.failed"}
        )
        checks.extend(
            (
                _check("endpoint_commit_decision", True, commit.commit),
                _check(
                    "first_fragment_before_model_continuation",
                    True,
                    bool(
                        continuation_withheld
                        and first_playback
                        and playback_started is not None
                        and answer_finished is None
                    ),
                ),
                _check("no_terminal_while_stream_paused", (), premature_terminal),
            )
        )

        harness.models.stream_gate.open()
        quiescent = harness.runtime.wait_idle(timeout=_WAIT_SEC)
        calls_closed = harness.models.wait_calls_closed(timeout=1.0)
        before_stale = trace.events()
        before_counts = (
            len(_events(before_stale, "task.started")),
            len(_events(before_stale, "playback.requested")),
            len(_receipt_texts(before_stale)),
        )
        harness.engine.typed_final(
            "This stale revision must not run.",
            acoustic=lineage,
            revision=1,
        )
        stale_quiet = harness.runtime.wait_idle(timeout=1.0)
        final_events = trace.events()
        after_counts = (
            len(_events(final_events, "task.started")),
            len(_events(final_events, "playback.requested")),
            len(_receipt_texts(final_events)),
        )
        fragments = _events(
            final_events,
            "playback.attributed",
            lambda event: not bool(event.payload.get("auxiliary_tts", False)),
        )
        receipts = _receipt_texts(final_events)
        final_texts = tuple(
            str(event.payload.get("text", ""))
            for event in final_events
            if event.kind == "stt.final"
        )
        answer_queries = tuple(
            str(event.payload.get("query", ""))
            for event in final_events
            if event.kind == "capability.started"
            and event.payload.get("name") == "assistant.answer"
            and event.payload.get("planner_tool") is False
        )
        identity_rows = _identity_rows(final_events)
        identity_set = set(identity_rows)
        metrics = _metric_values(harness.runtime)
        snapshot = harness.runtime.supervisor.session_actor.snapshot()
        checks.extend(
            (
                _check(
                    "incremental_fragment_count",
                    ">=2",
                    len(fragments),
                    ok=len(fragments) >= 2,
                ),
                _check(
                    "one_task_started",
                    1,
                    len(_events(final_events, "task.started")),
                ),
                _check(
                    "one_task_completed",
                    1,
                    len(_events(final_events, "task.completed")),
                ),
                _check(
                    "one_stream_end",
                    1,
                    len(_events(final_events, "tts.stream_end")),
                ),
                _check("exact_final_transcript_text", (final_text,), final_texts),
                _check(
                    "exact_assistant_answer_query",
                    (final_text,),
                    answer_queries,
                ),
                _check(
                    "receipt_backed_memory",
                    True,
                    bool(
                        len(receipts) == 1
                        and "lighthouse" in receipts[0].lower()
                        and "keeper" in receipts[0].lower()
                    ),
                ),
                _check(
                    "task_identity_complete_and_consistent",
                    True,
                    bool(
                        identity_rows
                        and not any(
                            row and row[0] == "missing" for row in identity_rows
                        )
                        and len(identity_set) == 1
                        and next(iter(identity_set))[2] == 2
                    ),
                ),
                _check("stale_lower_revision_fenced", before_counts, after_counts),
                _check("runtime_quiescent", True, bool(quiescent and stale_quiet)),
                _check("model_calls_closed", True, calls_closed),
                _check(
                    "actor_children_drained",
                    (0, 0, 0, 0),
                    _active_counts(snapshot),
                ),
                _check(
                    "causal_metrics_finite",
                    True,
                    bool(metrics and all(metrics[name] for name in metrics)),
                ),
                _fragment_ownership_check(final_events),
                _trace_timing_check(final_events),
            )
        )
    finally:
        harness.models.stream_gate.open()
        try:
            harness.stop()
        finally:
            harness.remove_observer()
    return tuple(checks)


_BARGE_V4_CHECKS = frozenset(
    {
        "barge_cut_linearized",
        "post_cut_playback_attributed",
        "cancelled_task_tts_after_cut",
        "redirect_owns_every_post_cut_playback",
        "redirect_playback_input_generation",
        "barge_victim_is_cancelled_fast_answer",
        "barge_withheld_real_continuation",
        "redirect_task_terminals",
        "redirect_playback_outcomes",
        "redirect_is_response_only",
        "redirect_fragment_terminal_ownership",
        "redirect_response_only_input_ownership",
        "redirect_completion_input_ownership",
        "redirect_memory_receipt_ownership",
        "redirect_used_controller_answer",
        "redirect_bypassed_model",
        "runtime_quiescent",
        "invocations_closed",
        "model_calls_closed",
        "runner_error",
    }
)


def _run_barge_recovery(
    config: Mapping[str, object],
    _trace: TraceRecorder,
    *,
    run_index: int,
) -> tuple[tuple[FlowCheck, ...], tuple[TraceEvent, ...]]:
    scenario = selected(("barge_redirect",))[0]
    result = run_scenario(
        scenario,
        config=_headless_config(config, stream_tts=True),
        models=deterministic_models(),
        run_index=run_index,
    )
    observed_check_names = {check.name for check in result.checks}
    missing_check_names = tuple(sorted(_BARGE_V4_CHECKS - observed_check_names))
    checks = [
        FlowCheck(
            name=f"v4.{check.name}",
            ok=check.ok,
            expected=check.expected,
            actual=check.actual,
        )
        for check in result.checks
        if check.name in _BARGE_V4_CHECKS
    ]
    checks.extend(
        (
            _check(
                "v4_required_check_names_present",
                (),
                missing_check_names,
            ),
            _check("v4_runner_error", "", result.error),
            _check(
                "physical_acoustic_barge_in_proven",
                False,
                False,
            ),
            _check(
                "barge_evidence_scope",
                "scripted_control_plane_only",
                "scripted_control_plane_only",
            ),
            _trace_timing_check(result.trace),
        )
    )
    return tuple(checks), result.trace


def _invocation_pair_check(events: tuple[TraceEvent, ...]) -> FlowCheck:
    phases: dict[int, list[str]] = {}
    for event in events:
        if event.kind not in {"capability.started", "capability.finished"}:
            continue
        invocation_id = event.payload.get("invocation_id")
        if isinstance(invocation_id, int) and not isinstance(invocation_id, bool):
            phases.setdefault(invocation_id, []).append(event.kind.split(".", 1)[1])
    actual = tuple((key, tuple(value)) for key, value in sorted(phases.items()))
    ok = bool(actual and all(value == ("started", "finished") for _, value in actual))
    return _check(
        "capability_invocations_paired",
        "started_then_finished",
        actual,
        ok=ok,
    )


def _run_slow_tool_complete_followup(
    config: Mapping[str, object],
    trace: TraceRecorder,
) -> tuple[FlowCheck, ...]:
    harness = _Harness.build(config, trace, stream_tts=False)
    blocking = _BlockingCapability(
        "Pipecat is an open-source framework for realtime voice agents."
    )
    checks: list[FlowCheck] = []
    harness.runtime.supervisor.capabilities.register("web.search", blocking)
    try:
        harness.start()
        first_lineage = _lineage("slow-tool", turn_index=1)
        harness.engine.mark_user_speech_end(
            at=first_lineage.spans[-1].speech_end_at or time.perf_counter()
        )
        harness.engine.typed_final(
            "Look up Pipecat using your tools.",
            acoustic=first_lineage,
            revision=1,
        )
        tool_started = blocking.started.wait(timeout=_WAIT_SEC)
        ack_terminal = _wait_until(
            lambda: bool(
                _events(
                    trace.events(),
                    "playback.attributed",
                    lambda event: bool(event.payload.get("auxiliary_tts", False)),
                )
            ),
            timeout=_WAIT_SEC,
        )
        blocked_events = trace.events()
        ack = _event(
            blocked_events,
            "tts.request",
            lambda event: bool(event.payload.get("latency_ack", False)),
        )
        web_started = _event(
            blocked_events,
            "capability.started",
            lambda event: event.payload.get("name") == "web.search",
        )
        ack_playback = _event(blocked_events, "playback.started")
        premature = tuple(
            event.kind
            for event in blocked_events
            if event.kind in {"task.completed", "task.cancelled", "task.failed"}
        )
        checks.extend(
            (
                _check("blocking_tool_started", True, tool_started),
                _check(
                    "latency_ack_before_tool_wait",
                    True,
                    bool(
                        ack_terminal
                        and ack is not None
                        and ack_playback is not None
                        and web_started is not None
                        and ack_playback.sequence < web_started.sequence
                    ),
                ),
                _check("no_premature_task_terminal", (), premature),
            )
        )

        blocking.release.set()
        provider_finished = blocking.finished.wait(timeout=_WAIT_SEC)
        first_idle = harness.runtime.wait_idle(timeout=_WAIT_SEC)
        first_events = trace.events()
        web_finished = _event(
            first_events,
            "capability.finished",
            lambda event: event.payload.get("name") == "web.search",
        )
        first_terminal = _event(first_events, "task.completed")
        first_receipts = _receipt_texts(first_events)
        checks.extend(
            (
                _check("provider_finished", True, provider_finished),
                _check(
                    "tool_finished_before_task_terminal",
                    True,
                    bool(
                        web_finished is not None
                        and first_terminal is not None
                        and web_finished.sequence < first_terminal.sequence
                    ),
                ),
                _check("first_turn_quiescent", True, first_idle),
                _check(
                    "first_turn_receipt_backed",
                    True,
                    bool(
                        len(first_receipts) == 1
                        and "pipecat" in first_receipts[0].lower()
                    ),
                ),
            )
        )

        second_lineage = _lineage("slow-tool-followup", turn_index=2)
        harness.engine.mark_user_speech_end(
            at=second_lineage.spans[-1].speech_end_at or time.perf_counter()
        )
        harness.engine.typed_final(
            "Which category does the previously discussed framework belong to?",
            acoustic=second_lineage,
            revision=1,
        )
        second_idle = harness.runtime.wait_idle(timeout=_WAIT_SEC)
        invocations_closed = trace.wait_invocations_closed(timeout=1.0)
        model_calls_closed = harness.models.wait_calls_closed(timeout=1.0)
        final_events = trace.events()
        receipts = _receipt_texts(final_events)
        snapshot = harness.runtime.supervisor.session_actor.snapshot()
        checks.extend(
            (
                _check("blocking_provider_called_once", 1, blocking.call_count()),
                _check(
                    "two_tasks_completed",
                    2,
                    len(_events(final_events, "task.completed")),
                ),
                _check(
                    "followup_receipt_uses_prior_context",
                    True,
                    bool(len(receipts) == 2 and "pipecat" in receipts[-1].lower()),
                ),
                _invocation_pair_check(final_events),
                _check("invocation_observer_quiescent", True, invocations_closed),
                _check("model_calls_closed", True, model_calls_closed),
                _check("followup_quiescent", True, second_idle),
                _check(
                    "actor_children_drained",
                    (0, 0, 0, 0),
                    _active_counts(snapshot),
                ),
                _trace_timing_check(final_events),
            )
        )
    finally:
        blocking.release.set()
        try:
            harness.stop()
        finally:
            harness.remove_observer()
    return tuple(checks)


def _run_stop_restart_fence(
    config: Mapping[str, object],
    trace: TraceRecorder,
) -> tuple[FlowCheck, ...]:
    first = _Harness.build(config, trace, stream_tts=False)
    second: _Harness | None = None
    blocking = _BlockingCapability(
        "The synthetic command completed without any external effect."
    )
    checks: list[FlowCheck] = []
    # Re-registering without a spec deliberately preserves command.stage's
    # production side-effecting metadata and therefore its actor-owned tool
    # admission. The provider itself remains a pure in-memory fixture.
    first.runtime.supervisor.capabilities.register("command.stage", blocking)
    stop_done = Event()
    stop_error: list[str] = []
    stop_elapsed_ms = 0.0

    def stop_first() -> None:
        nonlocal stop_elapsed_ms
        started = time.monotonic()
        try:
            first.stop()
        except Exception as exc:  # surfaced as a red check without losing cleanup
            stop_error.append(f"{type(exc).__name__}: {exc}")
        finally:
            stop_elapsed_ms = (time.monotonic() - started) * 1000.0
            stop_done.set()

    stop_thread: Thread | None = None
    try:
        first.start()
        trace.mark(
            "flow.session.started",
            {
                "label": "A",
                "session_id": first.runtime.supervisor.session_actor.session_id,
            },
        )
        command_text = "Run a harmless test command."
        lineage_a = _lineage("restart-a-command", turn_index=1)
        first.engine.typed_final(
            command_text,
            acoustic=lineage_a,
            revision=1,
        )
        confirmation_seen = _wait_until(
            lambda: bool(
                _events(
                    trace.events(),
                    "tts.request",
                    lambda event: bool(
                        event.payload.get("confirmation_prompt", False)
                    ),
                )
                and _events(trace.events(), "playback.attributed")
            ),
            timeout=_WAIT_SEC,
        )
        staged_events = trace.events()
        confirmation_request = _event(
            staged_events,
            "tts.request",
            lambda event: bool(event.payload.get("confirmation_prompt", False)),
        )
        confirmation_attribution = (
            _event(
                staged_events,
                "playback.attributed",
                lambda event: bool(
                    confirmation_request is not None
                    and event.payload.get("tts_sequence")
                    == confirmation_request.sequence
                ),
            )
            if confirmation_request is not None
            else None
        )
        confirmation_terminal = (
            _event(
                staged_events,
                "playback.terminal",
                lambda event: bool(
                    confirmation_attribution is not None
                    and event.payload.get("fragment_id")
                    == confirmation_attribution.payload.get("fragment_id")
                ),
            )
            if confirmation_attribution is not None
            else None
        )
        pending_before_confirmation = len(
            first.runtime.supervisor.state.pending_confirmations
        )
        no_start_before_confirmation = bool(
            not blocking.started.is_set()
            and not _events(staged_events, "task.started")
            and not _events(staged_events, "capability.started")
        )

        confirm_lineage = _lineage("restart-a-confirm", turn_index=2)
        first.engine.typed_final(
            "Yes",
            acoustic=confirm_lineage,
            revision=1,
        )
        tool_started = blocking.started.wait(timeout=_WAIT_SEC)
        blocked_events = trace.events()
        blocked_snapshot = first.runtime.supervisor.session_actor.snapshot()
        command_task_started = _event(
            blocked_events,
            "task.started",
            lambda event: event.payload.get("capability") == "command.stage",
        )
        command_invocation_started = _event(
            blocked_events,
            "capability.started",
            lambda event: event.payload.get("name") == "command.stage",
        )
        call_rows = blocking.call_rows()
        provider_context = call_rows[0][1] if len(call_rows) == 1 else {}
        task_identity = tuple(
            command_task_started.payload.get(field)
            if command_task_started is not None
            else None
            for field in TaskIdentity.PAYLOAD_FIELDS
        )
        prompt_identity = tuple(
            confirmation_request.payload.get(field)
            if confirmation_request is not None
            else None
            for field in TaskIdentity.PAYLOAD_FIELDS
        )
        provider_identity = tuple(
            provider_context.get(field) for field in TaskIdentity.PAYLOAD_FIELDS
        )
        tool_call_id = str(
            command_invocation_started.payload.get("tool_call_id", "")
            if command_invocation_started is not None
            else ""
        )
        idempotency_key = str(
            command_invocation_started.payload.get("idempotency_key", "")
            if command_invocation_started is not None
            else ""
        )
        checks.extend(
            (
                _check(
                    "session_a_confirmation_prompt_receipted",
                    True,
                    bool(
                        confirmation_seen
                        and confirmation_request is not None
                        and confirmation_attribution is not None
                        and confirmation_terminal is not None
                        and confirmation_terminal.payload.get("outcome")
                        == "completed"
                    ),
                ),
                _check(
                    "session_a_pending_confirmation_exact",
                    1,
                    pending_before_confirmation,
                ),
                _check(
                    "session_a_provider_not_started_before_confirmation",
                    True,
                    no_start_before_confirmation,
                ),
                _check("session_a_tool_started", True, tool_started),
                _check(
                    "session_a_active_tool_while_blocked",
                    1,
                    blocked_snapshot.active_tools,
                ),
                _check(
                    "session_a_staged_task_identity_exact",
                    True,
                    bool(
                        command_task_started is not None
                        and confirmation_request is not None
                        and command_invocation_started is not None
                        and command_task_started.task_id
                        == confirmation_request.task_id
                        == command_invocation_started.task_id
                        and task_identity == prompt_identity == provider_identity
                        and all(
                            field in command_task_started.payload
                            for field in TaskIdentity.PAYLOAD_FIELDS
                        )
                    ),
                ),
                _check(
                    "session_a_tool_ids_nonempty",
                    True,
                    bool(
                        tool_call_id
                        and idempotency_key
                        and provider_context.get("tool_call_id") == tool_call_id
                        and provider_context.get("idempotency_key")
                        == idempotency_key
                    ),
                ),
            )
        )
        stop_thread = Thread(target=stop_first, name="flow-stop-session-a", daemon=True)
        stop_thread.start()
        stop_thread.join(timeout=_STOP_JOIN_SEC)
        bounded_before_release = bool(
            stop_done.is_set() and not blocking.finished.is_set()
        )
        checks.extend(
            (
                _check(
                    "session_a_stop_bounded_before_provider_release",
                    True,
                    bounded_before_release,
                ),
                _check("session_a_stop_error", (), tuple(stop_error)),
                _check(
                    "session_a_stop_duration_finite",
                    True,
                    bool(math.isfinite(stop_elapsed_ms) and stop_elapsed_ms >= 0.0),
                ),
            )
        )
        if not bounded_before_release:
            raise RuntimeError("session A did not stop within the bounded drain")

        first_session_id = first.runtime.supervisor.session_actor.session_id
        first_output_after_stop = tuple(first.engine.spoken)
        first_memory_after_stop = _memory_snapshot(first.runtime)

        # The second registry intentionally has no invocation observer: registry
        # invocation counters restart at one, while TraceRecorder's open set is
        # global. Bus, task, and playback events still carry the exact new actor
        # identity and are sufficient for the isolation proof.
        second = _Harness.build(
            config,
            trace,
            stream_tts=True,
            observe_capabilities=False,
        )
        second.start()
        second_session_id = second.runtime.supervisor.session_actor.session_id
        trace.mark(
            "flow.session.started",
            {"label": "B", "session_id": second_session_id},
        )
        lineage_b = _lineage("restart-b", turn_index=1)
        second.engine.typed_final(
            "What is the capital of Japan?",
            acoustic=lineage_b,
            revision=1,
        )
        second_idle = second.runtime.wait_idle(timeout=_WAIT_SEC)
        second_events_before_release = trace.events()
        second_task = _event(
            second_events_before_release,
            "task.completed",
            lambda event: event.payload.get("session_id") == second_session_id,
        )
        second_receipts = _receipt_texts(second_events_before_release)

        old_trace_count = len(trace.events())
        first.engine.typed_final(
            "This callback belongs to the stopped session.",
            acoustic=_lineage("restart-a-late", turn_index=2),
            revision=1,
        )
        old_callback_fenced = len(trace.events()) == old_trace_count

        second_trace_count = len(trace.events())
        second_output = tuple(second.engine.spoken)
        second_memory = _memory_snapshot(second.runtime)
        blocking.release.set()
        provider_finished = blocking.finished.wait(timeout=_WAIT_SEC)
        invocations_closed = first.trace.wait_invocations_closed(timeout=1.0)
        first_actor_drained = _wait_until(
            lambda: _active_counts(
                first.runtime.supervisor.session_actor.snapshot()
            )
            == (0, 0, 0, 0),
            timeout=2.0,
        )
        # A's capability-finished observer is expected to add one trace row. B's
        # own task/output/memory surfaces must remain byte-for-byte unchanged.
        no_cross_session_effect = bool(
            tuple(second.engine.spoken) == second_output
            and _memory_snapshot(second.runtime) == second_memory
            and len(
                tuple(
                    event
                    for event in trace.events()[second_trace_count:]
                    if event.payload.get("session_id") == second_session_id
                )
            )
            == 0
        )
        first_post_release_stable = bool(
            tuple(first.engine.spoken) == first_output_after_stop
            and _memory_snapshot(first.runtime) == first_memory_after_stop
        )
        first_snapshot = first.runtime.supervisor.session_actor.snapshot()
        second_snapshot = second.runtime.supervisor.session_actor.snapshot()
        released_events = trace.events()
        command_invocation_finished = _event(
            released_events,
            "capability.finished",
            lambda event: event.payload.get("name") == "command.stage",
        )
        started_invocation_identity = (
            command_invocation_started.task_id,
            command_invocation_started.payload.get("invocation_id"),
            command_invocation_started.payload.get("tool_call_id"),
            command_invocation_started.payload.get("idempotency_key"),
        ) if command_invocation_started is not None else ()
        finished_invocation_identity = (
            command_invocation_finished.task_id,
            command_invocation_finished.payload.get("invocation_id"),
            command_invocation_finished.payload.get("tool_call_id"),
            command_invocation_finished.payload.get("idempotency_key"),
        ) if command_invocation_finished is not None else ()
        finished_result = (
            command_invocation_finished.payload.get("result", {})
            if command_invocation_finished is not None
            else {}
        )
        finished_data = (
            finished_result.get("data", {})
            if isinstance(finished_result, Mapping)
            else {}
        )
        checks.extend(
            (
                _check(
                    "fresh_actor_session_identity",
                    True,
                    first_session_id != second_session_id,
                ),
                _check(
                    "session_b_completed_with_receipt",
                    True,
                    bool(
                        second_idle
                        and second_task is not None
                        and second_task.payload.get("session_id") == second_session_id
                        and second_receipts
                        and "tokyo" in second_receipts[-1].lower()
                    ),
                ),
                _check("old_engine_callback_fenced", True, old_callback_fenced),
                _check(
                    "session_a_provider_finished_after_b",
                    True,
                    provider_finished,
                ),
                _check(
                    "session_a_command_provider_called_once",
                    1,
                    blocking.call_count(),
                ),
                _check(
                    "session_a_tool_lifecycle_identity_paired",
                    started_invocation_identity,
                    finished_invocation_identity,
                    ok=bool(
                        started_invocation_identity
                        and started_invocation_identity
                        == finished_invocation_identity
                        and tool_call_id
                        and idempotency_key
                    ),
                ),
                _check(
                    "synthetic_command_external_effect_performed",
                    False,
                    finished_data.get("external_effect_performed")
                    if isinstance(finished_data, Mapping)
                    else None,
                ),
                _check("session_a_invocation_closed", True, invocations_closed),
                _check(
                    "provider_release_did_not_mutate_session_b",
                    True,
                    no_cross_session_effect,
                ),
                _check(
                    "provider_release_produced_no_old_output",
                    True,
                    first_post_release_stable,
                ),
                _check("session_a_actor_drained", True, first_actor_drained),
                _check(
                    "session_a_active_tools_zero_after_release",
                    0,
                    first_snapshot.active_tools,
                ),
                _check(
                    "session_a_children_zero",
                    (0, 0, 0, 0),
                    _active_counts(first_snapshot),
                ),
                _check(
                    "session_b_children_zero",
                    (0, 0, 0, 0),
                    _active_counts(second_snapshot),
                ),
                _check("session_a_stopped", True, first.session.stopped),
                _trace_timing_check(trace.events()),
            )
        )
    finally:
        blocking.release.set()
        if stop_thread is not None and stop_thread.is_alive():
            stop_thread.join(timeout=_STOP_JOIN_SEC)
        try:
            if not first.session.stopped:
                first.stop()
        finally:
            first.remove_observer()
        if second is not None:
            try:
                second.stop()
            finally:
                second.remove_observer()
    return tuple(checks)


def run_flow_scenario(
    scenario_id: str,
    *,
    config: Mapping[str, object],
    run_index: int,
) -> FlowScenarioResult:
    """Run one deterministic production-runtime journey."""

    if scenario_id not in FLOW_SCENARIO_IDS:
        choices = ", ".join(FLOW_SCENARIO_IDS)
        raise ValueError(
            f"unknown flow scenario {scenario_id!r}; choose one of: {choices}"
        )
    if isinstance(run_index, bool) or not isinstance(run_index, int) or run_index < 1:
        raise ValueError("run_index must be a positive integer")

    started = time.monotonic()
    trace = TraceRecorder()
    trace.mark(
        "flow.fixture",
        {
            "scenario_id": scenario_id,
            "fixture_transcript_classification": "plumbing_only",
            "stt_quality_evidence": False,
        },
        turn_id=run_index,
    )
    checks: tuple[FlowCheck, ...] = ()
    result_trace: tuple[TraceEvent, ...] | None = None
    error = ""
    try:
        if scenario_id == "turn_incremental":
            checks = _run_turn_incremental(config, trace)
        elif scenario_id == "barge_recovery":
            checks, result_trace = _run_barge_recovery(
                config,
                trace,
                run_index=run_index,
            )
        elif scenario_id == "slow_tool_complete_followup":
            checks = _run_slow_tool_complete_followup(config, trace)
        else:
            checks = _run_stop_restart_fence(config, trace)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        checks = (*checks, _check("runner_error", "", error))

    checks_ok = bool(checks and not error and all(check.ok for check in checks))
    terminal_payload = {
        "scenario_id": scenario_id,
        "checks_ok": checks_ok,
        "error": error,
    }
    if result_trace is None:
        trace.mark("flow.complete", terminal_payload, turn_id=run_index)
        observed_trace = trace.events()
    else:
        terminal_elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
        observed_trace = (
            *result_trace,
            TraceEvent(
                sequence=len(result_trace) + 1,
                elapsed_ms=terminal_elapsed_ms,
                kind="flow.complete",
                turn_id=run_index,
                payload=terminal_payload,
            ),
        )
    duration_ms = (time.monotonic() - started) * 1000.0
    return FlowScenarioResult(
        scenario_id=scenario_id,
        description=_DESCRIPTIONS[scenario_id],
        run_index=run_index,
        passed=checks_ok,
        duration_ms=duration_ms,
        checks=checks,
        trace=observed_trace,
        error=error,
    )


def run_flow_suite(
    *,
    config: Mapping[str, object],
    runs: int = 3,
) -> tuple[FlowScenarioResult, ...]:
    """Run every flow scenario for ``runs`` repetitions in stable order."""

    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    results: list[FlowScenarioResult] = []
    for run_index in range(1, runs + 1):
        for scenario_id in FLOW_SCENARIO_IDS:
            started = time.monotonic()
            try:
                result = run_flow_scenario(
                    scenario_id,
                    config=config,
                    run_index=run_index,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                trace = TraceRecorder()
                trace.mark(
                    "flow.fixture",
                    {
                        "scenario_id": scenario_id,
                        "fixture_transcript_classification": "plumbing_only",
                        "stt_quality_evidence": False,
                    },
                    turn_id=run_index,
                )
                checks = (_check("runner_error", "", error),)
                trace.mark(
                    "flow.complete",
                    {
                        "scenario_id": scenario_id,
                        "checks_ok": False,
                        "error": error,
                    },
                    turn_id=run_index,
                )
                result = FlowScenarioResult(
                    scenario_id=scenario_id,
                    description=_DESCRIPTIONS[scenario_id],
                    run_index=run_index,
                    passed=False,
                    duration_ms=(time.monotonic() - started) * 1000.0,
                    checks=checks,
                    trace=trace.events(),
                    error=error,
                )
            results.append(result)
    return tuple(results)
