"""Response-only admission for the first terminal final after a real barge."""
from __future__ import annotations

import threading
import time
from typing import Iterator

from always_on_agent.events import AgentEvent, EventKind, Mode
from always_on_agent.continuation import ContinuationConfig
from always_on_agent.memory import SessionMemory
from always_on_agent.models import IntentKind
from always_on_agent.post_barge import PostBargeResponseGate
from always_on_agent.react import PlannerConfig
from core.addressing import ACT, INGEST, ScriptedAddressingClassifier
from core.app import _post_barge_response_window
from core.capability_router import RouteDecision, SIMPLE
from core.engine import FinalTranscript, OwnerVerification
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.metrics import HANDLED_LOCAL
from core.capabilities import RecallConfig
from core.resume import ResumeConfig
from core.runtime import VoiceRuntime
from core.turn_merge import TurnMergeConfig


_RUN_144211_FINAL = (
    "Segwam though Carry me about through roman arcticture instead"
)


def _live_final(
    engine: ScriptedEngine,
    text: str,
    *,
    verified: bool = True,
) -> None:
    callback = engine._cb.on_final_result
    assert callback is not None
    callback(
        FinalTranscript(
            text,
            owner_verification=(
                OwnerVerification.VERIFIED
                if verified
                else OwnerVerification.UNKNOWN
            ),
            origin="live_audio",
        )
    )


def _barge_while_speaking(engine: ScriptedEngine) -> None:
    engine.speak("the interrupted old reply")
    assert engine.is_speaking
    engine.barge_in()
    assert not engine.is_speaking
    # Let the answer produced by the post-barge final complete normally.
    engine._hold = False


def test_gate_is_one_shot_epoch_bound_and_rearm_safe():
    gate = PostBargeResponseGate(window_sec=8.0)
    gate.arm(3, now=10.0)
    first = gate.inspect(3, "live_audio", now=17.9)
    assert first is not None and first.response_eligible

    # A newer barge cannot be consumed by the older preprocessing snapshot.
    gate.arm(4, now=18.0)
    assert not gate.consume(first)
    second = gate.inspect(4, "live_audio", now=18.1)
    assert second is not None and gate.consume(second)
    assert gate.inspect(4, "live_audio", now=18.2) is None


def test_gate_expires_and_foreign_origin_consumes_without_bypass():
    gate = PostBargeResponseGate(window_sec=8.0)
    gate.arm(1, now=1.0)
    assert gate.inspect(1, "live_audio", now=9.0) is None

    gate.arm(2, now=20.0)
    foreign = gate.inspect(2, "unknown", now=20.1)
    assert foreign is not None and not foreign.response_eligible
    assert gate.consume(foreign)
    assert gate.inspect(2, "live_audio", now=20.2) is None


def test_invalid_windows_fail_closed_in_app_and_gate():
    invalid = (-1.0, float("nan"), float("inf"), float("-inf"), "bad", None)
    for value in invalid:
        parsed = _post_barge_response_window(
            {"post_barge_response": {"enabled": True, "window_sec": value}}
        )
        assert parsed == 0.0
        gate = PostBargeResponseGate(value)  # type: ignore[arg-type]
        gate.arm(1, now=1.0)
        assert gate.inspect(1, "live_audio", now=1.0) is None
    assert _post_barge_response_window(
        {"post_barge_response": {"enabled": False, "window_sec": 99.0}}
    ) == 0.0


def test_run_144211_held_override_bypasses_ingest_once_for_response():
    engine = ScriptedEngine(hold_speech=True)
    addressing = ScriptedAddressingClassifier(default=INGEST)
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="Roman architecture answer"),
        start_mode=Mode.ASSISTANT,
        addressing=addressing,
        unsure_acts=False,
        turn_merge_config=TurnMergeConfig(
            enabled=True,
            hold_sec=0.2,
            max_hold_sec=1.0,
            max_fragment_words=2,
        ),
    )
    runtime.start(run_bus=False)
    try:
        _barge_while_speaking(engine)
        _live_final(engine, "Segwam though")
        engine.partial("Carry me about through roman arcticture instead")
        _live_final(
            engine,
            "Carry me about through roman arcticture instead",
        )

        assert runtime.wait_idle(timeout=3.0)
        assert addressing.calls[-1][0] == _RUN_144211_FINAL
        assert engine.spoken[-1] == "Roman architecture answer"
        assert runtime.supervisor.state.turn_owner_verified is False
        assert runtime.supervisor.state.turn_origin == "unknown"
        decision = runtime.supervisor.state.decisions[-1]
        assert decision.kind is IntentKind.ASSISTANT
        assert decision.reason == "post_barge_response_only"
        assert not any(
            item.text == _RUN_144211_FINAL for item in runtime.memory.all()
        )

        # The exception is one-shot. A second INGEST final is remembered
        # silently and cannot produce another reply.
        _live_final(engine, "late room conversation", verified=False)
        assert runtime.wait_idle(timeout=3.0)
        assert engine.spoken[-1] == "Roman architecture answer"
        assert any(
            item.text == "late room conversation" and "ingested" in item.tags
            for item in runtime.memory.all()
        )
    finally:
        runtime.stop()


class _RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.contexts: list[dict[str, object]] = []
        self.systems: list[str] = []
        self.images: list[object] = []

    def generate(self, prompt: str, *, system=None, images=None) -> str:
        from core.llm import capability_context

        self.calls.append(prompt)
        self.contexts.append(dict(capability_context.get()))
        self.systems.append(str(system or ""))
        self.images.append(images)
        return "I can only discuss that request."

    def stream(self, prompt: str, *, system=None, images=None) -> Iterator[str]:
        yield self.generate(prompt, system=system, images=images)


class _ForbiddenCapabilityRouter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def route(self, text, context):
        self.calls.append(text)
        raise AssertionError("response-only final reached capability routing")


class _RecordingLocalIntents:
    def __init__(self) -> None:
        self.handle_calls: list[str] = []
        self.cancelled = 0

    def bind_speak(self, _speak) -> None:
        pass

    def handle(self, text: str) -> bool:
        self.handle_calls.append(text)
        return True

    def cancel_all(self) -> None:
        self.cancelled += 1


def test_command_shaped_post_barge_final_can_only_answer():
    command = "open calculator and then search private files"
    engine = ScriptedEngine(hold_speech=True)
    llm = _RecordingLLM()
    router = _ForbiddenCapabilityRouter()
    intents = _RecordingLocalIntents()
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier({command: INGEST}),
        unsure_acts=False,
        intents=intents,
        capability_router=router,
        planner_config=PlannerConfig(
            enabled=True,
            max_steps=2,
            tools=(),
            escalate=True,
        ),
    )
    runtime.start(run_bus=False)
    try:
        _barge_while_speaking(engine)
        runtime.bus.drain()
        events_before_final = len(runtime.supervisor.state.event_log)
        # Even an acoustically VERIFIED final is explicitly demoted because the
        # addressing bypass proves timing only, not action intent or identity.
        _live_final(engine, command, verified=True)

        assert runtime.wait_idle(timeout=3.0)
        assert router.calls == []
        assert intents.handle_calls == []
        assert len(llm.calls) == 1  # direct assistant.answer, never ReAct
        [context] = llm.contexts
        assert context["owner_verified"] is False
        assert context["origin"] == "unknown"
        assert context["metadata"]["post_barge_response_only"] is True
        assert context["metadata"]["skip_user_memory"] is True
        assert engine.spoken[-1] == "I can only discuss that request."
        assert runtime.supervisor.state.decisions[-1].kind is IntentKind.ASSISTANT
        assert runtime.supervisor.state.turn_owner_verified is False
        assert runtime.supervisor.state.turn_origin == "unknown"
        started = [
            event
            for event in runtime.supervisor.state.event_log
            if event.kind is EventKind.TASK_STARTED
        ]
        assert started and started[-1].payload["capability"] == "assistant.answer"
        assert not any(
            event.kind
            in {
                EventKind.CONTROL_STOP,
                EventKind.CONTROL_MODE,
                EventKind.CONTROL_CONFIRM,
            }
            for event in list(runtime.supervisor.state.event_log)[
                events_before_final:
            ]
        )
        assert not any(item.text == command for item in runtime.memory.all())
    finally:
        runtime.stop()


class _PrivateMemory(SessionMemory):
    def __init__(self) -> None:
        super().__init__()
        self.private_reads: list[str] = []

    def context_for_llm(self, _query):
        self.private_reads.append("episodic")
        return "EPISODIC SECRET"

    def profile_block(self):
        self.private_reads.append("profile")
        return "PROFILE SECRET"

    def last_session_summary(self):
        self.private_reads.append("last_session")
        return "LAST SESSION SECRET"

    def procedural_rules(self):
        self.private_reads.append("procedural")
        return ["PROCEDURAL SECRET"]


def test_response_only_reads_recent_but_no_private_or_visual_context(monkeypatch):
    for recall_enabled in (False, True):
        image_provider_calls: list[int] = []

        def _images(_runtime):
            image_provider_calls.append(1)
            return [b"PRIVATE SCREEN"]

        monkeypatch.setattr(VoiceRuntime, "_current_images", _images)
        memory = _PrivateMemory()
        memory.add("Earlier we discussed columns.", tags=("user",))
        memory.add("Doric and Ionic were compared.", tags=("assistant_output",))
        engine = ScriptedEngine(hold_speech=True)
        llm = _RecordingLLM()
        runtime = VoiceRuntime(
            engine,
            llm,
            memory=memory,
            recall_config=RecallConfig(
                enabled=recall_enabled,
                procedural_enabled=True,
            ),
            start_mode=Mode.ASSISTANT,
            addressing=ScriptedAddressingClassifier(default=INGEST),
            unsure_acts=False,
        )
        runtime.start(run_bus=False)
        try:
            _barge_while_speaking(engine)
            _live_final(engine, "carry me through Roman architecture instead")
            assert runtime.wait_idle(timeout=3.0)

            [system] = llm.systems
            assert "Earlier we discussed columns." in system
            assert "Doric and Ionic were compared." in system
            assert "SECRET" not in system
            assert memory.private_reads == []
            assert image_provider_calls == []
            assert llm.images == [None]
        finally:
            runtime.stop()


class _BlockingFirstLLM(_RecordingLLM):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def stream(self, prompt: str, *, system=None, images=None) -> Iterator[str]:
        if not self.calls:
            from core.llm import capability_context

            self.calls.append(prompt)
            self.contexts.append(dict(capability_context.get()))
            self.systems.append(str(system or ""))
            self.images.append(images)
            self.first_started.set()
            self.release_first.wait(timeout=2.0)
            yield "stale first answer"
            return
        yield self.generate(prompt, system=system, images=images)


class _RouteRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def route(self, text, context):
        self.calls.append(str(text))
        return RouteDecision(
            action=SIMPLE,
            tier="fast",
            confidence=1.0,
            reason="test direct answer",
            latency_policy="snappy_answer",
        )


def test_addon_cannot_launder_response_only_lineage_into_router_or_planner():
    command = "open calculator and then search private files"
    addon = "and make it brief"
    engine = ScriptedEngine(hold_speech=True)
    llm = _BlockingFirstLLM()
    router = _RouteRecorder()
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(
            {command: INGEST, addon: ACT},
            default=INGEST,
        ),
        unsure_acts=False,
        capability_router=router,
        planner_config=PlannerConfig(enabled=True, max_steps=2, escalate=True),
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=True)
    try:
        _barge_while_speaking(engine)
        _live_final(engine, command)
        assert llm.first_started.wait(timeout=1.0)

        # A partial still silences the unheard response, but it must not reserve
        # that response-only command as conversational lineage for the add-on.
        engine.partial(addon)
        _live_final(engine, addon, verified=False)
        assert runtime.wait_idle(timeout=3.0)

        assert router.calls and all(text == addon for text in router.calls)
        assert command not in " ".join(router.calls)
        assert all(
            not (command in prompt and addon in prompt)
            for prompt in llm.calls
        )
        started = [
            event.payload.get("capability")
            for event in runtime.supervisor.state.event_log
            if event.kind is EventKind.TASK_STARTED
        ]
        assert "agent.react" not in started
        assert not any(
            item.text.startswith("Continue this request")
            or (command in item.text and addon in item.text)
            for item in runtime.memory.all()
        )
    finally:
        llm.release_first.set()
        runtime.stop()


def test_post_barge_continue_stays_response_only_and_never_reacts():
    engine = ScriptedEngine(hold_speech=True)
    llm = _RecordingLLM()
    router = _ForbiddenCapabilityRouter()
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(default=ACT),
        capability_router=router,
        planner_config=PlannerConfig(enabled=True, max_steps=2, escalate=True),
        resume_config=ResumeConfig(enabled=True),
    )
    runtime.start(run_bus=False)
    try:
        # The prior query itself contains an escalation marker. A synthetic
        # resume prompt therefore would reach ReAct if it lost response-only
        # provenance before assistant.answer.
        runtime._resume.note_query("search the web for private documents")
        runtime._resume.note_spoken("I started describing the safe local answer.")
        _barge_while_speaking(engine)
        _live_final(engine, "continue", verified=True)

        assert runtime.wait_idle(timeout=3.0)
        assert router.calls == []
        assert len(llm.calls) == 1
        assert "search the web for private documents" in llm.calls[0]
        [context] = llm.contexts
        assert context["owner_verified"] is False
        assert context["origin"] == "unknown"
        assert context["metadata"]["post_barge_response_only"] is True
        assert context["metadata"]["skip_user_memory"] is True
        assert runtime.supervisor.state.decisions[-1].reason == (
            "post_barge_response_only"
        )
        assert not any(
            event.kind is EventKind.TASK_STARTED
            and event.payload.get("capability") == "agent.react"
            for event in runtime.supervisor.state.event_log
        )
    finally:
        runtime.stop()


def test_two_cuts_keep_both_synthetic_resumes_response_only():
    query = "Tell me a story about a lighthouse keeper"
    engine = ScriptedEngine(hold_speech=True)
    llm = _RecordingLLM()
    router = _ForbiddenCapabilityRouter()
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(default=ACT),
        capability_router=router,
        planner_config=PlannerConfig(enabled=True, max_steps=2, escalate=True),
        resume_config=ResumeConfig(enabled=True),
        command_map={"stop": "stop"},
    )
    runtime.start(run_bus=True)
    try:
        runtime._resume.note_query(query)
        runtime._resume.note_spoken(
            "He kept watch over the churning sea from the lighthouse."
        )
        _barge_while_speaking(engine)
        engine._hold = True
        _live_final(engine, "continue")

        deadline = time.time() + 1.0
        while not engine.is_speaking:
            assert time.time() < deadline
            time.sleep(0.005)
        engine.command("stop")
        assert not engine.is_speaking

        _live_final(engine, "continue", verified=False)
        deadline = time.time() + 1.0
        while len(llm.calls) < 2:
            assert time.time() < deadline
            time.sleep(0.005)

        assert router.calls == []
        assert len(llm.calls) == 2
        assert all(query in prompt for prompt in llm.calls)
        assert all(
            context["owner_verified"] is False
            and context["origin"] == "unknown"
            and context["metadata"]["post_barge_response_only"] is True
            and context["metadata"]["skip_user_memory"] is True
            for context in llm.contexts
        )
        assert not any(
            event.kind is EventKind.TASK_STARTED
            and event.payload.get("capability") == "agent.react"
            for event in runtime.supervisor.state.event_log
        )
    finally:
        runtime.stop()


def test_foreign_continue_under_grant_drops_and_clears_retry_lineage():
    engine = ScriptedEngine(hold_speech=True)
    llm = _RecordingLLM()
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(default=INGEST),
        unsure_acts=False,
        resume_config=ResumeConfig(enabled=True),
    )
    runtime.start(run_bus=False)
    try:
        runtime._resume.note_query("Tell me a private story")
        runtime._resume.note_spoken("The story had already begun.")
        _barge_while_speaking(engine)
        callback = engine._cb.on_final_result
        assert callback is not None
        callback(
            FinalTranscript(
                "continue",
                owner_verification=OwnerVerification.UNKNOWN,
                origin="remote_audio",
            )
        )
        assert runtime.wait_idle(timeout=3.0)

        # The foreign observation consumed the one-shot grant and discarded
        # resumable lineage, so a later live retry is just ordinary INGEST.
        _live_final(engine, "continue", verified=False)
        assert runtime.wait_idle(timeout=3.0)
        assert llm.calls == []
        assert engine.spoken == ["the interrupted old reply"]
        assert any(
            item.text == "continue" and "ingested" in item.tags
            for item in runtime.memory.all()
        )
    finally:
        runtime.stop()


def test_mode_invalidated_continue_drops_and_clears_retry_lineage():
    engine = ScriptedEngine(hold_speech=True)
    llm = _RecordingLLM()
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(default=INGEST),
        unsure_acts=False,
        resume_config=ResumeConfig(enabled=True),
    )
    runtime.start(run_bus=False)
    try:
        runtime._resume.note_query("Tell me a private story")
        runtime._resume.note_spoken("The story had already begun.")
        _barge_while_speaking(engine)
        runtime.bus.publish(AgentEvent.mode(Mode.DICTATION, source="ui"))
        runtime.bus.drain()
        _live_final(engine, "continue", verified=False)
        assert runtime.wait_idle(timeout=3.0)

        runtime.bus.publish(AgentEvent.mode(Mode.ASSISTANT, source="ui"))
        runtime.bus.drain()
        _live_final(engine, "continue", verified=False)
        assert runtime.wait_idle(timeout=3.0)
        assert llm.calls == []
        assert engine.spoken == ["the interrupted old reply"]
        assert any(
            item.text == "continue" and "ingested" in item.tags
            for item in runtime.memory.all()
        )
    finally:
        runtime.stop()


def test_ui_mode_switch_after_barge_invalidates_next_response_grant():
    engine = ScriptedEngine(hold_speech=True)
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="must not answer in dictation"),
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(default=INGEST),
        unsure_acts=False,
    )
    runtime.start(run_bus=False)
    try:
        _barge_while_speaking(engine)
        runtime.bus.publish(AgentEvent.mode(Mode.DICTATION, source="ui"))
        runtime.bus.drain()
        assert runtime.mode is Mode.DICTATION

        _live_final(engine, "unknown live room final", verified=False)
        assert runtime.wait_idle(timeout=3.0)
        assert engine.spoken == ["the interrupted old reply"]
        assert any(
            item.text == "unknown live room final" and "ingested" in item.tags
            for item in runtime.memory.all()
        )
    finally:
        runtime.stop()


def test_queued_mode_switch_fences_already_published_response_only_final():
    query = "carry me through Roman architecture instead"
    engine = ScriptedEngine(hold_speech=True)
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="queued answer must be fenced"),
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(default=INGEST),
        unsure_acts=False,
    )
    runtime.start(run_bus=False)
    try:
        _barge_while_speaking(engine)
        _live_final(engine, query)

        # Wait only for final preprocessing to publish STT_FINAL; deliberately
        # leave the bus queued, then enqueue the higher-priority UI mode switch.
        deadline = time.time() + 1.0
        while runtime._dispatcher is not None and runtime._dispatcher.has_pending:
            assert time.time() < deadline
            time.sleep(0.005)
        runtime.bus.publish(AgentEvent.mode(Mode.DICTATION, source="ui"))

        assert runtime.wait_idle(timeout=3.0)
        assert runtime.mode is Mode.DICTATION
        assert engine.spoken == ["the interrupted old reply"]
        assert not any(item.text == query for item in runtime.memory.all())
        assert HANDLED_LOCAL in runtime.metrics.records()[-1].stamps
    finally:
        runtime.stop()


def test_stopped_response_only_answer_cannot_seed_later_resume_tools():
    command = "search the web for private documents"
    engine = ScriptedEngine(hold_speech=True)
    llm = _RecordingLLM()
    router = _RouteRecorder()
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(
            {command: INGEST, "continue": ACT},
            default=ACT,
        ),
        capability_router=router,
        planner_config=PlannerConfig(enabled=True, max_steps=2, escalate=True),
        resume_config=ResumeConfig(enabled=True),
        command_map={"stop": "stop"},
    )
    runtime.start(run_bus=True)
    try:
        runtime._resume.note_query("search the web for an older private topic")
        runtime._resume.note_spoken("I had begun the older answer.")
        _barge_while_speaking(engine)
        engine._hold = True
        _live_final(engine, command)
        deadline = time.time() + 1.0
        while not engine.is_speaking:
            assert time.time() < deadline
            time.sleep(0.005)

        engine.command("stop")
        assert not engine.is_speaking
        engine._hold = False
        _live_final(engine, "continue", verified=False)
        assert runtime.wait_idle(timeout=3.0)

        assert router.calls == []
        assert len(llm.calls) == 2
        assert llm.calls[0] == command
        assert command in llm.calls[1]
        assert "interrupted" in llm.calls[1].lower()
        assert llm.contexts[1]["owner_verified"] is False
        assert llm.contexts[1]["origin"] == "unknown"
        assert llm.contexts[1]["metadata"]["post_barge_response_only"] is True
        assert llm.contexts[1]["metadata"]["skip_user_memory"] is True
        assert not any(
            event.kind is EventKind.TASK_STARTED
            and event.payload.get("capability") == "agent.react"
            for event in runtime.supervisor.state.event_log
        )
        assert not any(item.text == command for item in runtime.memory.all())
    finally:
        runtime.stop()


def test_explicit_rejected_final_invalidates_grant_and_never_bypasses():
    engine = ScriptedEngine(hold_speech=True)
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="must not speak"),
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(default=INGEST),
        unsure_acts=False,
    )
    runtime.start(run_bus=False)
    try:
        _barge_while_speaking(engine)
        callback = engine._cb.on_final_result
        assert callback is not None
        callback(
            FinalTranscript(
                "rejected nearby voice",
                owner_verification=OwnerVerification.REJECTED,
                origin="live_audio",
            )
        )
        assert runtime.wait_idle(timeout=3.0)
        _live_final(engine, "later ambient speech", verified=False)

        assert runtime.wait_idle(timeout=3.0)
        assert engine.spoken == ["the interrupted old reply"]
        assert any(
            item.text == "rejected nearby voice" and "ingested" in item.tags
            for item in runtime.memory.all()
        )
        assert any(
            item.text == "later ambient speech" and "ingested" in item.tags
            for item in runtime.memory.all()
        )
    finally:
        runtime.stop()


def test_act_classified_first_final_consumes_grant_without_elevation():
    first = "tell me about arches"
    second = "late ambient words"
    engine = ScriptedEngine(hold_speech=True)
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ordinary answer"),
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(
            {first: ACT, second: INGEST},
            default=INGEST,
        ),
        unsure_acts=False,
    )
    runtime.start(run_bus=False)
    try:
        _barge_while_speaking(engine)
        _live_final(engine, first, verified=False)
        assert runtime.wait_idle(timeout=3.0)
        assert engine.spoken[-1] == "ordinary answer"
        assert runtime.supervisor.state.decisions[-1].reason == "assistant_mode"
        assert runtime.supervisor.state.turn_owner_verified is False
        assert runtime.supervisor.state.turn_origin == "live_audio"

        # ACT used up the one post-barge observation normally. The next INGEST
        # cannot claim an exception merely because the first final did not need it.
        _live_final(engine, second, verified=False)
        assert runtime.wait_idle(timeout=3.0)
        assert engine.spoken[-1] == "ordinary answer"
        assert any(
            item.text == second and "ingested" in item.tags
            for item in runtime.memory.all()
        )
    finally:
        runtime.stop()


class _BlockingIngest:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def classify(self, _text, recent=()):
        self.started.set()
        self.release.wait(timeout=2.0)
        return INGEST


def test_invalidated_token_drops_stale_preprocessing_without_memory():
    stale = "garbled override that must disappear"
    engine = ScriptedEngine(hold_speech=True)
    addressing = _BlockingIngest()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="must not speak"),
        start_mode=Mode.ASSISTANT,
        addressing=addressing,
        unsure_acts=False,
    )
    runtime.start(run_bus=False)
    try:
        _barge_while_speaking(engine)
        _live_final(engine, stale, verified=False)
        assert addressing.started.wait(timeout=1.0)

        # Carrier noise invalidates the grant while the older addressing lease
        # is still running. Its later INGEST result may neither answer nor write
        # the superseded garble into memory.
        _live_final(engine, ".", verified=False)
        addressing.release.set()
        assert runtime.wait_idle(timeout=3.0)
        assert engine.spoken == ["the interrupted old reply"]
        assert not any(item.text == stale for item in runtime.memory.all())
    finally:
        addressing.release.set()
        runtime.stop()


def test_empty_or_self_echo_final_invalidates_post_barge_grant():
    echo = "The interrupted old reply"
    engine = ScriptedEngine(hold_speech=True)
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="must not speak"),
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(default=INGEST),
        unsure_acts=False,
        resume_config=ResumeConfig(
            echo_guard_enabled=True,
            echo_window_sec=8.0,
        ),
    )
    runtime.start(run_bus=False)
    try:
        runtime._resume.note_spoken(echo)
        _barge_while_speaking(engine)
        _live_final(engine, echo, verified=True)
        # The echo consumes/invalidates the grant despite a false-positive
        # VERIFIED verdict; the following room final remains ordinary INGEST.
        _live_final(engine, "late ambient words", verified=False)
        assert runtime.wait_idle(timeout=3.0)
        assert engine.spoken == ["the interrupted old reply"]

        # A fresh barge followed by punctuation-only/no-content likewise cannot
        # leave a grant hanging for a later ambient final.
        engine._hold = True
        _barge_while_speaking(engine)
        _live_final(engine, ".", verified=False)
        _live_final(engine, "more ambient words", verified=False)
        assert runtime.wait_idle(timeout=3.0)
        assert engine.spoken == ["the interrupted old reply", "the interrupted old reply"]
    finally:
        runtime.stop()
