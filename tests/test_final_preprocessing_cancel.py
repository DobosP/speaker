"""Newest-input cancellation for the pre-task final-processing chain.

These tests cover the gap before AgentTask exists: addressing, cleanup, and
LLM-assisted routing can block on a provider. The FinalDispatcher must retire
that old chain, keep unknown abandoned calls bounded, and let only the latest
lease commit side effects.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from always_on_agent.continuation import ContinuationConfig
from always_on_agent.events import AgentEvent, EventKind, Mode
from always_on_agent.memory import SessionMemory
from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.speech_analyzer import LiveSpeechAnalyzer
from always_on_agent.supervisor import AgentSupervisor
from core.addressing import ACT, LLMAddressingClassifier
from core.capability_router import (
    HeuristicCapabilityRouter,
    LLMCapabilityRouter,
    RESEARCH,
)
from core.cleanup import LLMTranscriptCleaner, ScriptedTranscriptCleaner
from core.engines.scripted import ScriptedEngine
from core.llm import (
    EchoLLM,
    LLMCallCancelled,
    OllamaLLM,
    capability_context,
    collect_llm_text,
)
from core.metrics import ASR_FINAL, MERGED, SUPERSEDED
from core.resume import ResumeConfig
from core.runtime import VoiceRuntime
from core.turn_merge import FinalDispatcher, FinalDispatchLease, TurnMergeConfig
from tests.test_ollama_async_cancel import (
    FakeAsyncChunks,
    FakeAsyncClientFactory,
)


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _cancellable_dispatcher(dispatch, *, max_active: int = 6) -> FinalDispatcher:
    dispatcher = FinalDispatcher(
        dispatch,
        TurnMergeConfig(enabled=False),
        cancellable=True,
        max_active_dispatches=max_active,
    )
    dispatcher.start()
    return dispatcher


def test_newer_final_cancels_cooperative_preprocessing_and_only_latest_commits():
    old_started = threading.Event()
    committed: list[str] = []
    old_cancelled: list[bool] = []

    def dispatch(text, lease):
        if text == "old question":
            old_started.set()
            assert lease.cancel_event.wait(1.0)
            old_cancelled.append(True)
        if lease.claim_commit():
            committed.append(text)

    dispatcher = _cancellable_dispatcher(dispatch)
    try:
        dispatcher.submit("old question")
        assert old_started.wait(1.0)

        dispatcher.submit("latest question")

        assert _wait_until(lambda: committed == ["latest question"])
        assert old_cancelled == [True]
    finally:
        dispatcher.stop()


def test_unknown_cancel_ignoring_providers_are_bounded_and_capacity_recovers():
    releases = {
        "blocked zero": threading.Event(),
        "blocked one": threading.Event(),
    }
    started: list[str] = []
    committed: list[str] = []
    lock = threading.Lock()

    def dispatch(text, lease):
        with lock:
            started.append(text)
        release = releases.get(text)
        if release is not None:
            release.wait(2.0)  # deliberately ignores lease.cancel_event
        if lease.claim_commit():
            committed.append(text)

    dispatcher = _cancellable_dispatcher(dispatch, max_active=2)
    try:
        dispatcher.submit("blocked zero")
        assert _wait_until(lambda: started == ["blocked zero"])
        dispatcher.submit("blocked one")
        assert _wait_until(lambda: started == ["blocked zero", "blocked one"])

        dispatcher.submit("healthy latest")
        time.sleep(0.05)
        assert "healthy latest" not in started
        assert len(started) == 2, "provider bulkhead exceeded its configured cap"

        releases["blocked zero"].set()
        assert _wait_until(lambda: "healthy latest" in started)
        assert _wait_until(lambda: committed == ["healthy latest"])
    finally:
        for release in releases.values():
            release.set()
        dispatcher.stop()


def test_stop_cancels_uncommitted_preprocessing_without_a_late_commit():
    started = threading.Event()
    exited = threading.Event()
    committed: list[str] = []

    def dispatch(text, lease):
        started.set()
        assert lease.cancel_event.wait(1.0)
        if lease.claim_commit():
            committed.append(text)
        exited.set()

    dispatcher = _cancellable_dispatcher(dispatch)
    dispatcher.submit("old question")
    assert started.wait(1.0)

    dispatcher.stop()

    assert exited.wait(1.0)
    assert committed == []


def test_partial_after_explicit_cancel_cannot_resurrect_the_old_final():
    old_started = threading.Event()
    release_old = threading.Event()
    latest_started = threading.Event()
    committed: list[str] = []

    def dispatch(text, lease):
        if text == "old complete question":
            old_started.set()
            assert release_old.wait(1.0)
        else:
            latest_started.set()
        if lease.claim_commit():
            committed.append(text)

    dispatcher = FinalDispatcher(
        dispatch,
        TurnMergeConfig(enabled=True, hold_sec=0.1, max_hold_sec=1.0),
        cancellable=True,
    )
    dispatcher.start()
    try:
        dispatcher.submit("old complete question")
        assert old_started.wait(1.0)
        # Hold the condition across these public calls so the coordinator cannot
        # clear the old identity and make the resurrection race disappear.
        with dispatcher._cv:
            dispatcher.cancel_pending()
            dispatcher.note_partial()
            assert dispatcher._pending is None
            dispatcher.submit("new interruption words")

        assert latest_started.wait(1.0)
        assert _wait_until(lambda: committed == ["new interruption words"])
        assert dispatcher.merged_count == 0
    finally:
        release_old.set()
        dispatcher.stop()


def test_committed_dispatch_linearizes_before_a_newer_submission():
    claimed = threading.Event()
    release_terminal = threading.Event()
    committed: list[str] = []

    def dispatch(text, lease):
        if text == "first":
            assert lease.claim_commit()
            claimed.set()
            release_terminal.wait(1.0)
            committed.append(text)
            return
        if lease.claim_commit():
            committed.append(text)

    dispatcher = _cancellable_dispatcher(dispatch)
    try:
        dispatcher.submit("first")
        assert claimed.wait(1.0)
        dispatcher.cancel_pending()
        dispatcher.submit("second")
        time.sleep(0.05)
        assert committed == []
        release_terminal.set()
        assert _wait_until(lambda: committed == ["first", "second"])
    finally:
        release_terminal.set()
        dispatcher.stop()


def test_active_fragment_cannot_merge_after_the_total_hold_window():
    old_started = threading.Event()
    release_old = threading.Event()
    committed: list[str] = []

    def dispatch(text, lease):
        if text == "A story about":
            old_started.set()
            release_old.wait(1.0)
        if lease.claim_commit():
            committed.append(text)

    dispatcher = FinalDispatcher(
        dispatch,
        TurnMergeConfig(enabled=True, hold_sec=0.01, max_hold_sec=0.03),
        cancellable=True,
    )
    dispatcher.start()
    try:
        dispatcher.submit("A story about")
        assert old_started.wait(1.0)
        time.sleep(0.05)

        dispatcher.submit("what time is it")

        assert _wait_until(lambda: committed == ["what time is it"])
        assert dispatcher.merged_count == 0
    finally:
        release_old.set()
        dispatcher.stop()


def test_retired_provider_finalizer_cannot_finish_the_next_generation():
    old_started = threading.Event()
    release_old = threading.Event()
    latest_started = threading.Event()
    release_latest = threading.Event()
    committed: list[str] = []

    def dispatch(text, lease):
        if text == "old":
            old_started.set()
            assert release_old.wait(1.0)
        else:
            latest_started.set()
            assert release_latest.wait(1.0)
        if lease.claim_commit():
            committed.append(text)

    dispatcher = _cancellable_dispatcher(dispatch)
    try:
        dispatcher.submit("old")
        assert old_started.wait(1.0)
        dispatcher.submit("latest")
        assert latest_started.wait(1.0)

        # Generation one retires after generation two has rebound the
        # coordinator's loop locals. Its finalizer must signal only its own
        # completion event, never make generation two appear idle.
        release_old.set()
        assert _wait_until(
            lambda: not any(
                thread.name == "speaker-final-provider-1"
                for thread in threading.enumerate()
            )
        )
        assert dispatcher.has_pending

        release_latest.set()
        assert _wait_until(lambda: committed == ["latest"])
    finally:
        release_old.set()
        release_latest.set()
        dispatcher.stop()


def test_active_held_fragment_merges_with_a_late_continuation():
    old_started = threading.Event()
    committed: list[str] = []

    def dispatch(text, lease):
        if text == "A long story about":
            old_started.set()
            assert lease.cancel_event.wait(1.0)
        if lease.claim_commit():
            committed.append(text)

    dispatcher = FinalDispatcher(
        dispatch,
        TurnMergeConfig(enabled=True, hold_sec=0.02, max_hold_sec=1.0),
        cancellable=True,
    )
    dispatcher.start()
    try:
        dispatcher.submit("A long story about")
        assert old_started.wait(1.0), "fragment never entered preprocessing"

        dispatcher.submit("a lighthouse")

        assert _wait_until(
            lambda: committed == ["A long story about a lighthouse"]
        )
        assert dispatcher.merged_count == 1
    finally:
        dispatcher.stop()


def test_partial_requeues_complete_active_final_for_one_forced_merge():
    old_started = threading.Event()
    committed: list[str] = []

    def dispatch(text, lease):
        if text == "Tell me a story":
            old_started.set()
            assert lease.cancel_event.wait(1.0)
        if lease.claim_commit():
            committed.append(text)

    dispatcher = FinalDispatcher(
        dispatch,
        TurnMergeConfig(enabled=True, hold_sec=0.1, max_hold_sec=1.0),
        cancellable=True,
    )
    dispatcher.start()
    try:
        dispatcher.submit("Tell me a story")
        assert old_started.wait(1.0)
        dispatcher.note_partial()
        dispatcher.submit("about a lighthouse")

        assert _wait_until(
            lambda: committed == ["Tell me a story about a lighthouse"]
        )
        assert dispatcher.merged_count == 1
    finally:
        dispatcher.stop()


def test_stopped_dispatcher_rejects_late_submissions():
    committed: list[str] = []
    dispatcher = _cancellable_dispatcher(
        lambda text, lease: committed.append(text)
        if lease.claim_commit()
        else None
    )

    dispatcher.stop()
    dispatcher.submit("late final")

    assert not dispatcher.has_pending
    assert committed == []


def test_timed_out_stop_keeps_live_coordinator_handle_until_it_exits():
    claimed = threading.Event()
    release = threading.Event()

    def dispatch(_text, lease):
        assert lease.claim_commit()
        claimed.set()
        release.wait(1.0)

    dispatcher = _cancellable_dispatcher(dispatch)
    dispatcher.submit("committed")
    assert claimed.wait(1.0)
    coordinator = dispatcher._thread
    assert coordinator is not None

    dispatcher.stop(timeout=0.01)
    assert coordinator.is_alive()
    assert dispatcher._thread is coordinator

    dispatcher.start()
    assert dispatcher._thread is coordinator

    release.set()
    assert _wait_until(lambda: not coordinator.is_alive())
    dispatcher.start()
    try:
        assert dispatcher._thread is not coordinator
        assert dispatcher._thread is not None
        assert dispatcher._thread.is_alive()
    finally:
        dispatcher.stop()


def test_runtime_newest_final_cancels_blocked_ollama_addressing_and_recovers():
    blocked = FakeAsyncChunks(block_after_chunks=True)
    healthy = FakeAsyncChunks("ACT")
    factory = FakeAsyncClientFactory(blocked, healthy)
    gate_llm = OllamaLLM(
        model="minicpm5-1b:q8",
        async_client_factory=factory,
    )
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        addressing=LLMAddressingClassifier(gate_llm),
        unsure_acts=False,
    )
    runtime.start(run_bus=True)
    try:
        engine.final("old question")
        assert blocked.waiting.wait(1.0)

        engine.final("latest question")

        assert blocked.task_cancelled.wait(1.0)
        assert factory.clients[0].closed.wait(1.0)
        assert runtime.wait_idle(timeout=3.0)
        spoken = " ".join(engine.spoken)
        assert "latest question" in spoken
        assert "old question" not in spoken
        assert len(factory.clients) == 2
        assert factory.clients[1].closed.is_set()
    finally:
        runtime.stop()


@pytest.mark.parametrize("control", ["command", "barge_in"])
def test_runtime_control_retires_blocked_preprocessing_without_late_speech(control):
    blocked = FakeAsyncChunks(block_after_chunks=True)
    factory = FakeAsyncClientFactory(blocked)
    gate_llm = OllamaLLM(
        model="minicpm5-1b:q8",
        async_client_factory=factory,
    )
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        addressing=LLMAddressingClassifier(gate_llm),
        unsure_acts=False,
    )
    runtime.start(run_bus=True)
    try:
        engine.final("old question")
        assert blocked.waiting.wait(1.0)

        if control == "command":
            engine.command("stop")
        else:
            engine.barge_in()

        assert blocked.task_cancelled.wait(1.0)
        assert factory.clients[0].closed.wait(1.0)
        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == []
        assert runtime.supervisor.state.active_tasks == {}
    finally:
        runtime.stop()


def test_punctuation_noise_does_not_retire_a_valid_preprocessing_turn():
    blocked = FakeAsyncChunks("ACT", block_after_chunks=True)
    factory = FakeAsyncClientFactory(blocked)
    gate_llm = OllamaLLM(
        model="minicpm5-1b:q8",
        async_client_factory=factory,
    )
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        addressing=LLMAddressingClassifier(gate_llm),
        unsure_acts=False,
    )
    runtime.start(run_bus=True)
    try:
        engine.final("valid old question")
        assert blocked.waiting.wait(1.0)

        engine.final(".")
        assert not blocked.task_cancelled.is_set()
        blocked.release_naturally()

        assert runtime.wait_idle(timeout=2.0)
        spoken = " ".join(engine.spoken)
        assert "valid old question" in spoken
        assert "You said: ." not in spoken
    finally:
        runtime.stop()


def test_self_echo_does_not_retire_a_valid_preprocessing_turn():
    blocked = FakeAsyncChunks("ACT", block_after_chunks=True)
    factory = FakeAsyncClientFactory(blocked)
    gate_llm = OllamaLLM(
        model="minicpm5-1b:q8",
        async_client_factory=factory,
    )
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        addressing=LLMAddressingClassifier(gate_llm),
        unsure_acts=False,
        resume_config=ResumeConfig(
            echo_guard_enabled=True,
            echo_min_words=3,
        ),
    )
    runtime.start(run_bus=True)
    try:
        echoed_reply = "Here is the answer I just spoke"
        runtime._resume.note_spoken(echoed_reply)
        runtime._resume.note_playback_end()

        engine.final("valid old question")
        assert blocked.waiting.wait(1.0)

        engine.final(echoed_reply)
        assert not blocked.task_cancelled.is_set()
        blocked.release_naturally()

        assert runtime.wait_idle(timeout=2.0)
        spoken = " ".join(engine.spoken)
        assert "valid old question" in spoken
        assert echoed_reply not in runtime.supervisor.state.transcript_log
    finally:
        runtime.stop()


def test_barge_after_terminal_claim_cannot_resurrect_a_published_final():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        addressing=LLMAddressingClassifier(_GenerateMustNotRun("ACT")),
        unsure_acts=False,
    )
    mark_entered = threading.Event()
    release_mark = threading.Event()
    original_mark = runtime.metrics.mark
    blocked_once = False

    def blocking_mark(stage, *args, **kwargs):
        nonlocal blocked_once
        if stage == ASR_FINAL and not blocked_once:
            blocked_once = True
            mark_entered.set()
            assert release_mark.wait(1.0)
        return original_mark(stage, *args, **kwargs)

    runtime.metrics.mark = blocking_mark
    runtime.start(run_bus=False)
    try:
        engine.final("old question")
        assert mark_entered.wait(1.0), "final never reached terminal commit"

        barge = threading.Thread(target=engine.barge_in, daemon=True)
        barge.start()
        time.sleep(0.02)
        assert barge.is_alive(), "barge should linearize behind terminal effects"

        release_mark.set()
        barge.join(timeout=1.0)
        assert not barge.is_alive()
        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == []
        assert runtime.supervisor.state.active_tasks == {}
    finally:
        release_mark.set()
        runtime.stop()


def test_bus_backlog_starts_only_the_newest_committed_final():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        addressing=LLMAddressingClassifier(_GenerateMustNotRun("ACT")),
        unsure_acts=False,
    )
    runtime.start(run_bus=False)
    try:
        engine.final("first question")
        assert _wait_until(
            lambda: not runtime._dispatcher.has_pending
            and runtime.bus._queue.unfinished_tasks == 1
        )

        engine.final("second question")
        assert _wait_until(
            lambda: not runtime._dispatcher.has_pending
            and runtime.bus._queue.unfinished_tasks == 2
        )

        assert runtime.wait_idle(timeout=2.0)
        spoken = " ".join(engine.spoken)
        assert "second question" in spoken
        assert "first question" not in spoken
    finally:
        runtime.stop()


def test_superseded_control_final_cannot_cancel_the_latest_backlogged_turn():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        addressing=LLMAddressingClassifier(_GenerateMustNotRun("ACT")),
        unsure_acts=False,
    )
    runtime.start(run_bus=False)
    try:
        engine.final("stop")
        assert _wait_until(
            lambda: not runtime._dispatcher.has_pending
            and runtime.bus._queue.unfinished_tasks == 1
        )
        engine.final("latest question")
        assert _wait_until(
            lambda: not runtime._dispatcher.has_pending
            and runtime.bus._queue.unfinished_tasks == 2
        )

        assert runtime.wait_idle(timeout=2.0)
        spoken = " ".join(engine.spoken)
        assert "latest question" in spoken
        assert runtime.supervisor.input_epoch == 0
    finally:
        runtime.stop()


class _FirstCallBlockingRouter:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self._lock = threading.Lock()

    def score(self, query, context):
        return 0.0

    def choose(self, query, context):
        with self._lock:
            self.calls += 1
            call = self.calls
        if call == 1:
            self.started.set()
            self.release.wait(1.0)
        return "fast"


def test_router_only_runtime_is_off_thread_cancellable_and_recovers():
    router = _FirstCallBlockingRouter()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM(), router=router)
    runtime.start(run_bus=True)
    try:
        engine.final("old question")
        assert router.started.wait(1.0)

        engine.final("latest question")

        assert runtime.wait_idle(timeout=2.0)
        spoken = " ".join(engine.spoken)
        assert "latest question" in spoken
        assert "old question" not in spoken
    finally:
        router.release.set()
        runtime.stop()


def test_tts_admission_and_speak_are_atomic_against_barge_in():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM())
    allowed_entered = threading.Event()
    release_allowed = threading.Event()
    calls: list[str] = []
    original_allowed = runtime.supervisor.tts_request_allowed
    original_speak = engine.speak
    original_stop = engine.stop_speaking

    def gated_allowed(task_id, epoch=None):
        allowed = original_allowed(task_id, epoch)
        allowed_entered.set()
        assert release_allowed.wait(1.0)
        return allowed

    def recording_speak(text, on_done=None):
        calls.append("speak")
        return original_speak(text, on_done)

    def recording_stop():
        calls.append("stop")
        return original_stop()

    runtime.supervisor.tts_request_allowed = gated_allowed
    engine.speak = recording_speak
    engine.stop_speaking = recording_stop
    runtime.start(run_bus=True)
    try:
        runtime.bus.publish(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {"task_id": "late", "text": "late sentence", "epoch": 0},
            )
        )
        assert allowed_entered.wait(1.0)

        barge = threading.Thread(target=engine.barge_in, daemon=True)
        barge.start()
        time.sleep(0.02)
        assert barge.is_alive()

        release_allowed.set()
        barge.join(timeout=1.0)
        assert not barge.is_alive()
        assert runtime.wait_idle(timeout=2.0)
        assert calls[:2] == ["speak", "stop"]
        assert "speak" not in calls[calls.index("stop") + 1 :]
    finally:
        release_allowed.set()
        runtime.stop()


def test_unstamped_keyword_event_cannot_start_a_task_after_barge_in():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM())
    decide_entered = threading.Event()
    release_decide = threading.Event()
    original_decide = runtime.supervisor.analyzer.decide

    def gated_decide(*args, **kwargs):
        decide_entered.set()
        assert release_decide.wait(1.0)
        return original_decide(*args, **kwargs)

    runtime.supervisor.analyzer.decide = gated_decide
    runtime.start(run_bus=True)
    try:
        engine.command("unknown keyword")
        assert decide_entered.wait(1.0)

        engine.barge_in()
        release_decide.set()

        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == []
        assert runtime.supervisor.state.active_tasks == {}
    finally:
        release_decide.set()
        runtime.stop()


class _SlowACT:
    def classify(self, text, recent=()):
        time.sleep(0.05)
        return ACT


def test_final_to_first_token_includes_preprocessing_latency():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        addressing=_SlowACT(),
        unsure_acts=False,
    )
    runtime.start(run_bus=False)
    try:
        engine.final("timed question")
        assert runtime.wait_idle(timeout=2.0)
        [record] = runtime.metrics.records()
        assert record.final_to_first_token is not None
        assert record.final_to_first_token >= 0.04
    finally:
        runtime.stop()


class _BlockedAnswerLLM:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, prompt, *, system=None, images=None):
        return "".join(self.stream(prompt, system=system, images=images))

    def stream(self, prompt, *, system=None, images=None):
        self.started.set()
        self.release.wait(1.0)
        yield "old answer"


class _OneLocalIntent:
    def __init__(self, speak) -> None:
        self.speak = speak

    def handle(self, text):
        if text != "local answer":
            return False
        self.speak("local reply")
        return True

    def cancel_all(self):
        return None


def test_local_intent_supersedes_an_older_active_answer():
    llm = _BlockedAnswerLLM()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        intents=_OneLocalIntent(engine.speak),
    )
    runtime.start(run_bus=True)
    try:
        engine.final("old question")
        assert llm.started.wait(1.0)
        assert runtime.supervisor.state.active_tasks

        engine.final("local answer")
        assert all(
            task.cancel_event.is_set()
            for task in runtime.supervisor.state.active_tasks.values()
        )
        llm.release.set()

        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == ["local reply"]
    finally:
        llm.release.set()
        runtime.stop()


class _SecondCallBlockingAddressing:
    def __init__(self) -> None:
        self.calls = 0
        self.second_started = threading.Event()
        self.release_second = threading.Event()

    def classify(self, text, recent=()):
        self.calls += 1
        if self.calls == 2:
            self.second_started.set()
            self.release_second.wait(1.0)
        return ACT


class _FirstBlockedEcho:
    def __init__(self) -> None:
        self.calls = 0
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def generate(self, prompt, *, system=None, images=None):
        return "".join(self.stream(prompt, system=system, images=images))

    def stream(self, prompt, *, system=None, images=None):
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
            self.release_first.wait(1.0)
        yield prompt


def test_new_final_fences_active_answer_before_its_gate_completes():
    llm = _FirstBlockedEcho()
    gate = _SecondCallBlockingAddressing()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        addressing=gate,
        unsure_acts=False,
    )
    runtime.start(run_bus=True)
    try:
        engine.final("old question")
        assert llm.first_started.wait(1.0)
        assert runtime.supervisor.state.active_tasks

        engine.final("latest question")
        assert gate.second_started.wait(1.0)
        llm.release_first.set()
        time.sleep(0.05)
        assert engine.spoken == []

        gate.release_second.set()
        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == ["latest question"]
    finally:
        llm.release_first.set()
        gate.release_second.set()
        runtime.stop()


class _FirstPretokenBlockedEcho:
    """Block only the old answer and expose when its generator is retired."""

    def __init__(self) -> None:
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.first_closed = threading.Event()
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def generate(self, prompt, *, system=None, images=None):
        return "".join(self.stream(prompt, system=system, images=images))

    def stream(self, prompt, *, system=None, images=None):
        with self._lock:
            self.prompts.append(prompt)
            call = len(self.prompts)
        if call != 1:
            yield prompt
            return
        self.first_started.set()
        try:
            assert self.release_first.wait(2.0)
            # If arrival-time continuation handling fails to fence this task,
            # this distinctive sentence is what would reach playback while the
            # continuation remains stuck in its addressing gate.
            yield "STALE OLD ANSWER."
        finally:
            self.first_closed.set()


def test_continuation_fences_pretoken_answer_before_its_gate_completes():
    """A pre-audio add-on owns playback even while its own gate is blocked."""
    llm = _FirstPretokenBlockedEcho()
    gate = _SecondCallBlockingAddressing()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        addressing=gate,
        unsure_acts=False,
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=True)
    try:
        engine.final("explain the moon phases")
        assert llm.first_started.wait(1.0)
        assert runtime.supervisor.state.active_tasks
        victim = next(iter(runtime.supervisor.state.active_tasks.values()))

        engine.final("make it shorter")
        assert gate.second_started.wait(1.0)
        assert victim.cancel_event.is_set()
        assert victim.task_id not in runtime.supervisor.state.active_tasks

        # The add-on is still preprocessing, but its arrival must already have
        # retired the unheard answer. Fully resume/close the old generator and
        # drain its bus fallout before asserting that no stale audio escaped.
        llm.release_first.set()
        assert llm.first_closed.wait(1.0)
        assert _wait_until(lambda: runtime.supervisor.tasks.active_count == 0)
        assert _wait_until(runtime.bus.idle)
        time.sleep(0.02)
        assert engine.spoken == []

        gate.release_second.set()
        assert runtime.wait_idle(timeout=2.0)

        assert len(engine.spoken) == 1
        assert "explain the moon phases" in engine.spoken[0]
        assert "make it shorter" in engine.spoken[0]
        assert len(llm.prompts) == 2
        assert "explain the moon phases" in llm.prompts[1]
        assert "make it shorter" in llm.prompts[1]
    finally:
        llm.release_first.set()
        gate.release_second.set()
        runtime.stop()


def test_preaudio_reservation_marks_victim_superseded_before_gate_finishes():
    """The watchdog must not diagnose a knowingly cancelled victim as stuck."""
    llm = _FirstPretokenBlockedEcho()
    gate = _SecondCallBlockingAddressing()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        addressing=gate,
        unsure_acts=False,
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=True)
    try:
        engine.final("explain the moon phases")
        assert llm.first_started.wait(1.0)

        engine.final("make it shorter")
        assert gate.second_started.wait(1.0)

        [victim_record] = runtime.metrics.records()
        assert SUPERSEDED in victim_record.stamps
        assert MERGED not in victim_record.stamps

        gate.release_second.set()
        llm.release_first.set()
        assert runtime.wait_idle(timeout=2.0)
        assert MERGED in runtime.metrics.records()[0].stamps
    finally:
        gate.release_second.set()
        llm.release_first.set()
        runtime.stop()


@pytest.mark.parametrize(
    ("victim_owner_verified", "victim_origin"),
    [(False, "live_audio"), (True, "screen")],
)
def test_arrival_continuation_cannot_upgrade_untrusted_victim_lineage(
    victim_owner_verified,
    victim_origin,
):
    """A verified add-on must not launder an untrusted original request."""
    supervisor = AgentSupervisor(
        continuation_config=ContinuationConfig(enabled=True),
    )
    victim = supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text="summarize the text on screen",
            reason="test",
            mode=Mode.ASSISTANT,
        )
    )
    victim.metadata["owner_verified"] = victim_owner_verified
    victim.metadata["origin"] = victim_origin
    supervisor.state.active_tasks[victim.task_id] = victim

    reservation = supervisor.reserve_arrival_continuation("make it shorter")
    assert reservation is not None
    merged, metadata = supervisor.materialize_arrival_continuation(
        reservation,
        "make it shorter",
    )

    # The add-on is verified live speech, but the synthetic replacement still
    # contains the untrusted victim prompt and therefore must remain demoted.
    supervisor.state.turn_owner_verified = True
    supervisor.state.turn_origin = "live_audio"
    supervisor.state.turn_metadata = metadata
    started = []
    supervisor._start_task = lambda task, *args, **kwargs: (  # type: ignore[method-assign]
        started.append(task) or True
    )
    supervisor._execute_decision(
        IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text="make it shorter",
            reason="test",
            mode=Mode.ASSISTANT,
        )
    )

    assert len(started) == 1
    replacement = started[0]
    assert replacement.input_text == merged
    if not victim_owner_verified:
        assert replacement.metadata["owner_verified"] is False
    if victim_origin != "live_audio":
        assert replacement.metadata["origin"] != "live_audio"
    assert not (
        replacement.metadata["owner_verified"] is True
        and replacement.metadata["origin"] == "live_audio"
    )


def test_unrelated_pending_confirmation_does_not_disable_clear_reservation():
    """A staged action must not reopen a pre-audio continuation race."""
    supervisor = AgentSupervisor(
        continuation_config=ContinuationConfig(enabled=True),
    )
    pending = supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.COMMAND,
            confidence=1.0,
            text="open the browser",
            reason="test",
            mode=Mode.COMMAND,
            requires_confirmation=True,
        )
    )
    supervisor.state.pending_confirmations[pending.task_id] = pending
    victim = supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text="explain the moon phases",
            reason="test",
            mode=Mode.ASSISTANT,
        )
    )
    supervisor.state.active_tasks[victim.task_id] = victim

    reservation = supervisor.reserve_arrival_continuation("make it shorter")

    assert reservation is not None
    assert reservation.victim_task_id == victim.task_id
    assert victim.cancel_event.is_set()
    assert victim.task_id not in supervisor.state.active_tasks
    assert supervisor.state.pending_confirmations == {pending.task_id: pending}
    assert not pending.cancel_event.is_set()


def test_cleaner_self_echo_drop_clears_its_owned_arrival_reservation():
    """A terminal drop cannot leave cancelled lineage for a later add-on."""
    raw_addon = "make it shorter"
    echoed_reply = "Keep it brief"
    runtime = VoiceRuntime(
        ScriptedEngine(),
        EchoLLM(),
        cleaner=ScriptedTranscriptCleaner({raw_addon: echoed_reply}),
        continuation_config=ContinuationConfig(enabled=True),
        resume_config=ResumeConfig(echo_guard_enabled=True),
    )
    victim = runtime.supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text="explain the moon phases",
            reason="test",
            mode=Mode.ASSISTANT,
        )
    )
    runtime.supervisor.state.active_tasks[victim.task_id] = victim
    reservation = runtime.supervisor.reserve_arrival_continuation(raw_addon)
    assert reservation is not None
    generation = 7
    runtime._note_arrival_continuation(generation, reservation)
    runtime._resume.note_spoken(echoed_reply)
    runtime._resume.note_playback_end()

    try:
        runtime._process_final(raw_addon, input_generation=generation)

        assert runtime._get_arrival_continuation(generation) is None
        assert runtime._latest_arrival_continuation() is None
        assert runtime._resume.echo_dropped == 1
    finally:
        runtime.stop()


def test_reserving_queued_audio_does_not_remember_the_unheard_answer():
    supervisor = AgentSupervisor(
        continuation_config=ContinuationConfig(enabled=True),
        defer_output_until_tts_admission=True,
    )
    victim = supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text="explain the moon phases",
            reason="test",
            mode=Mode.ASSISTANT,
        )
    )
    victim.speech_epoch = supervisor.speech_epoch
    supervisor.state.active_tasks[victim.task_id] = victim
    supervisor.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": victim.task_id,
                "text": "UNHEARD OLD ANSWER",
                "speak": True,
                "epoch": victim.speech_epoch,
            },
            priority=60,
        )
    )
    assert supervisor.bus.drain_once()
    assert victim.task_id in supervisor.state.pending_audio_tasks

    reservation = supervisor.reserve_arrival_continuation("make it shorter")

    assert reservation is not None
    assert victim.cancel_event.is_set()
    assert victim.task_id not in supervisor.state.pending_audio_tasks
    assert all(
        item.text != "UNHEARD OLD ANSWER" for item in supervisor.memory.all()
    )


def test_chained_continuations_keep_lineage_while_an_earlier_gate_is_blocked():
    """A newer add-on inherits a cancelled preprocessing lease's full lineage."""
    llm = _FirstPretokenBlockedEcho()
    gate = _SecondCallBlockingAddressing()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        addressing=gate,
        unsure_acts=False,
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=True)
    try:
        engine.final("explain the moon phases")
        assert llm.first_started.wait(1.0)

        engine.final("make it shorter")
        assert gate.second_started.wait(1.0)
        engine.final("and also use bullets")

        assert _wait_until(lambda: len(llm.prompts) >= 2)
        assert _wait_until(lambda: len(engine.spoken) == 1)
        spoken = engine.spoken[0]
        assert "explain the moon phases" in spoken
        assert "make it shorter" in spoken
        assert "and also use bullets" in spoken

        gate.release_second.set()
        llm.release_first.set()
        assert runtime.wait_idle(timeout=2.0)
        assert len(engine.spoken) == 1

        memory_items = runtime.memory.all()
        memory_texts = [item.text for item in memory_items]
        assert "explain the moon phases" in memory_texts
        assert "make it shorter" in memory_texts
        assert "and also use bullets" in memory_texts
        assert not any(
            "The user first asked:" in item.text and "user" in item.tags
            for item in memory_items
        )

        records = runtime.metrics.records()
        assert MERGED in records[0].stamps
        assert SUPERSEDED in records[0].stamps
    finally:
        gate.release_second.set()
        llm.release_first.set()
        runtime.stop()


def test_new_turn_abandons_a_blocked_continuation_reservation():
    """A fresh final must not resurrect or remember abandoned add-on lineage."""
    llm = _FirstPretokenBlockedEcho()
    gate = _SecondCallBlockingAddressing()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        addressing=gate,
        unsure_acts=False,
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=True)
    try:
        engine.final("explain the moon phases")
        assert llm.first_started.wait(1.0)

        engine.final("make it shorter")
        assert gate.second_started.wait(1.0)
        assert runtime._latest_arrival_continuation() is not None

        engine.final("what time is it")

        assert _wait_until(lambda: engine.spoken == ["what time is it"])
        gate.release_second.set()
        llm.release_first.set()
        assert runtime.wait_idle(timeout=2.0)
        assert runtime._latest_arrival_continuation() is None
        assert engine.spoken == ["what time is it"]
        memory_texts = [item.text for item in runtime.memory.all()]
        assert "make it shorter" not in memory_texts
        assert not any("The user first asked:" in text for text in memory_texts)
    finally:
        gate.release_second.set()
        llm.release_first.set()
        runtime.stop()


def test_barge_during_blocked_reservation_cannot_publish_or_record_addon():
    llm = _FirstPretokenBlockedEcho()
    gate = _SecondCallBlockingAddressing()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        addressing=gate,
        unsure_acts=False,
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=True)
    try:
        engine.final("explain the moon phases")
        assert llm.first_started.wait(1.0)

        engine.final("make it shorter")
        assert gate.second_started.wait(1.0)
        assert runtime._latest_arrival_continuation() is not None

        engine.barge_in()
        assert runtime._latest_arrival_continuation() is None
        gate.release_second.set()
        llm.release_first.set()

        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == []
        memory_texts = [item.text for item in runtime.memory.all()]
        assert "make it shorter" not in memory_texts
        assert not any("The user first asked:" in text for text in memory_texts)
    finally:
        gate.release_second.set()
        llm.release_first.set()
        runtime.stop()


def test_shutdown_discards_reserved_lineage_without_memory_side_effects():
    runtime = VoiceRuntime(
        ScriptedEngine(),
        EchoLLM(),
        cleaner=ScriptedTranscriptCleaner(),
        continuation_config=ContinuationConfig(enabled=True),
    )
    victim = runtime.supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text="explain the moon phases",
            reason="test",
            mode=Mode.ASSISTANT,
        )
    )
    runtime.supervisor.state.active_tasks[victim.task_id] = victim
    reservation = runtime.supervisor.reserve_arrival_continuation(
        "make it shorter"
    )
    assert reservation is not None
    runtime._note_arrival_continuation(1, reservation)

    runtime.stop()

    assert runtime._latest_arrival_continuation() is None
    assert runtime.supervisor.state.active_tasks == {}
    assert runtime.supervisor.state.pending_audio_tasks == {}
    assert runtime.supervisor.state.queued_tasks == []
    assert runtime.supervisor.state.pending_confirmations == {}
    assert runtime.memory.all() == []


@pytest.mark.parametrize(
    ("fragment", "tail", "combined"),
    [
        ("in", "Romanian", "in Romanian"),
        ("make it", "shorter", "make it shorter"),
    ],
)
def test_coalesced_continuation_fragment_keeps_origin_without_duplicate_addon(
    fragment,
    tail,
    combined,
):
    llm = _FirstPretokenBlockedEcho()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        continuation_config=ContinuationConfig(enabled=True),
        turn_merge_config=TurnMergeConfig(
            enabled=True,
            hold_sec=0.1,
            max_hold_sec=1.0,
        ),
    )
    runtime.start(run_bus=True)
    try:
        engine.final("explain the moon phases")
        assert llm.first_started.wait(1.0)

        engine.final(fragment)
        assert runtime._latest_arrival_continuation() is not None
        engine.partial(tail)
        engine.final(tail)

        assert _wait_until(lambda: len(llm.prompts) >= 2)
        assert _wait_until(lambda: len(engine.spoken) == 1)
        merged_prompt = llm.prompts[-1]
        assert "explain the moon phases" in merged_prompt
        assert merged_prompt.lower().count(combined.lower()) == 1
        assert engine.spoken[0].lower().count(combined.lower()) == 1

        llm.release_first.set()
        assert runtime.wait_idle(timeout=2.0)
        assert len(engine.spoken) == 1
    finally:
        llm.release_first.set()
        runtime.stop()


def test_new_final_fences_committed_lease_before_terminal_effects(monkeypatch):
    claimed = threading.Event()
    release_claim = threading.Event()
    original_claim = FinalDispatchLease.claim_commit

    def gated_claim(lease):
        result = original_claim(lease)
        if result and lease.input_generation == 1:
            claimed.set()
            assert release_claim.wait(2.0)
        return result

    monkeypatch.setattr(FinalDispatchLease, "claim_commit", gated_claim)
    gate = _SecondCallBlockingAddressing()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        addressing=gate,
        unsure_acts=False,
    )
    runtime.start(run_bus=True)
    try:
        engine.final("old question")
        assert claimed.wait(1.0)

        engine.final("latest question")
        release_claim.set()
        assert gate.second_started.wait(1.0)
        assert engine.spoken == []
        assert runtime.supervisor.state.active_tasks == {}

        gate.release_second.set()
        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == ["You said: latest question"]
    finally:
        release_claim.set()
        gate.release_second.set()
        runtime.stop()


def test_new_arrival_fences_a_published_final_before_bus_task_registration():
    gate = _SecondCallBlockingAddressing()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        addressing=gate,
        unsure_acts=False,
    )
    runtime.start(run_bus=False)
    try:
        engine.final("old question")
        assert _wait_until(
            lambda: not runtime._dispatcher.has_pending
            and runtime.bus._queue.unfinished_tasks == 1
        )

        engine.final("latest question")
        assert gate.second_started.wait(1.0)

        # The old STT_FINAL is already published, but generation two arrived.
        # Draining it while generation two is still gated must not start a task.
        runtime.bus.drain()
        assert engine.spoken == []
        assert runtime.supervisor.state.active_tasks == {}

        gate.release_second.set()
        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == ["You said: latest question"]
    finally:
        gate.release_second.set()
        runtime.stop()


def test_stale_published_control_cannot_cancel_a_newer_gated_final():
    gate = _SecondCallBlockingAddressing()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        addressing=gate,
        unsure_acts=False,
    )
    runtime.start(run_bus=False)
    try:
        engine.final("stop")
        assert _wait_until(
            lambda: not runtime._dispatcher.has_pending
            and runtime.bus._queue.unfinished_tasks == 1
        )
        # Publish CONTROL_STOP from generation one, but leave that control event
        # queued until after generation two has arrived.
        assert runtime.bus.drain_once()

        engine.final("latest question")
        assert gate.second_started.wait(1.0)
        runtime.bus.drain()
        assert runtime.supervisor.input_epoch == 0

        gate.release_second.set()
        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == ["You said: latest question"]
    finally:
        gate.release_second.set()
        runtime.stop()


def test_queued_tts_is_still_preaudio_until_runtime_admission():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        continuation_config=ContinuationConfig(enabled=True),
    )
    victim = runtime.supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text="explain the moon phases",
            reason="test",
            mode=Mode.ASSISTANT,
        )
    )
    victim.speech_epoch = runtime.supervisor.speech_epoch
    runtime.supervisor.state.active_tasks[victim.task_id] = victim
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_REQUEST,
            {
                "task_id": victim.task_id,
                "text": "OLD ANSWER.",
                "epoch": victim.speech_epoch,
            },
        )
    )
    assert engine.spoken == []
    assert not victim.started_speaking

    reservation = runtime.supervisor.reserve_arrival_continuation(
        "make it shorter"
    )

    try:
        assert reservation is not None
        assert reservation.merge_before_audio
        assert victim.cancel_event.is_set()
        runtime.bus.drain()
        assert engine.spoken == []
        assert not victim.started_speaking
    finally:
        runtime.stop()


def test_new_speech_partial_silences_pretoken_answer_until_final_arrives():
    llm = _FirstPretokenBlockedEcho()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=True)
    try:
        engine.final("explain the moon phases")
        assert llm.first_started.wait(1.0)
        victim = next(iter(runtime.supervisor.state.active_tasks.values()))

        engine.partial("make it shorter")
        assert victim.cancel_event.is_set()
        assert victim.task_id not in runtime.supervisor.state.active_tasks

        llm.release_first.set()
        assert _wait_until(lambda: runtime.supervisor.tasks.active_count == 0)
        time.sleep(0.02)
        assert engine.spoken == []

        engine.final("make it shorter")
        assert runtime.wait_idle(timeout=2.0)
        assert len(engine.spoken) == 1
        assert "explain the moon phases" in engine.spoken[0]
        assert engine.spoken[0].count("make it shorter") == 1
    finally:
        llm.release_first.set()
        runtime.stop()


def test_new_speech_partial_silences_pretoken_answer_when_merge_is_disabled():
    llm = _FirstPretokenBlockedEcho()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, llm)
    runtime.start(run_bus=True)
    try:
        engine.final("old question")
        assert llm.first_started.wait(1.0)
        victim = next(iter(runtime.supervisor.state.active_tasks.values()))

        engine.partial("latest question")
        assert victim.cancel_event.is_set()
        llm.release_first.set()
        assert _wait_until(lambda: runtime.supervisor.tasks.active_count == 0)
        time.sleep(0.02)
        assert engine.spoken == []

        engine.final("latest question")
        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == ["latest question"]
    finally:
        llm.release_first.set()
        runtime.stop()


def test_partial_continues_a_committed_final_still_waiting_on_the_bus():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=False)
    try:
        engine.final("explain the moon phases")
        assert runtime.bus._queue.unfinished_tasks == 1

        engine.partial("make it shorter")
        engine.final("make it shorter")

        assert runtime.wait_idle(timeout=2.0)
        assert len(engine.spoken) == 1
        assert "explain the moon phases" in engine.spoken[0]
        assert engine.spoken[0].count("make it shorter") == 1
        assert runtime._published_unheard == {}
    finally:
        runtime.stop()


def test_final_only_continues_a_committed_final_still_waiting_on_the_bus():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=False)
    try:
        engine.final("explain the moon phases")
        engine.final("make it shorter")

        assert runtime.wait_idle(timeout=2.0)
        assert len(engine.spoken) == 1
        assert "explain the moon phases" in engine.spoken[0]
        assert engine.spoken[0].count("make it shorter") == 1
        assert runtime._published_unheard == {}
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    "first_final",
    [
        "research quantum computing",
        "search quantum computing",
        "dictate a note about quantum computing",
        "open the browser",
    ],
)
def test_backlogged_nonassistant_final_cannot_become_continuation_origin(
    first_final,
):
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=False)
    try:
        engine.final(first_final)
        assert runtime._published_unheard == {}

        engine.final("make it shorter")

        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == ["You said: make it shorter"]
        assert not any(
            "The user first asked:" in output for output in engine.spoken
        )
    finally:
        runtime.stop()


class _ArchiveResearchAnalyzer(LiveSpeechAnalyzer):
    def decide(
        self,
        observation,
        current_mode,
        *,
        has_pending_confirmation=False,
    ):
        if observation.normalized == "archive this":
            return IntentDecision(
                IntentKind.RESEARCH,
                1.0,
                observation.text,
                "custom_research",
                mode=Mode.RESEARCH,
            )
        return super().decide(
            observation,
            current_mode,
            has_pending_confirmation=has_pending_confirmation,
        )


def test_custom_analyzer_fails_closed_for_published_unheard_lineage():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.supervisor.analyzer = _ArchiveResearchAnalyzer()
    runtime.start(run_bus=False)
    try:
        engine.final("archive this")
        assert runtime._published_unheard == {}

        engine.final("make it shorter")

        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == ["You said: make it shorter"]
    finally:
        runtime.stop()


def test_partial_silences_an_unheard_research_task_without_lineage_guessing():
    runtime = VoiceRuntime(ScriptedEngine(), EchoLLM())
    victim = runtime.supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.RESEARCH,
            confidence=1.0,
            text="old research",
            reason="test",
            mode=Mode.RESEARCH,
        )
    )
    runtime.supervisor.state.active_tasks[victim.task_id] = victim
    try:
        runtime._on_partial("new user speech")

        assert victim.cancel_event.is_set()
        assert victim.task_id not in runtime.supervisor.state.active_tasks
        assert runtime.supervisor.speech_epoch == 1
        assert runtime._latest_arrival_continuation() is None
    finally:
        runtime.stop()


def test_mapped_mode_command_retires_an_active_pretoken_answer():
    llm = _FirstPretokenBlockedEcho()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        command_map={"command mode": "mode:command"},
    )
    runtime.start(run_bus=True)
    try:
        engine.final("old question")
        assert llm.first_started.wait(1.0)
        victim = next(iter(runtime.supervisor.state.active_tasks.values()))

        engine.command("command mode")
        assert _wait_until(lambda: runtime.mode == Mode.COMMAND)
        assert victim.cancel_event.is_set()

        llm.release_first.set()
        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == []
    finally:
        llm.release_first.set()
        runtime.stop()


def test_unmapped_command_uses_normal_newest_input_preemption():
    llm = _FirstPretokenBlockedEcho()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, llm, command_map={"stop": "stop"})
    runtime.start(run_bus=True)
    try:
        engine.final("old question")
        assert llm.first_started.wait(1.0)
        victim = next(iter(runtime.supervisor.state.active_tasks.values()))

        engine.command("latest question")
        assert victim.cancel_event.is_set()
        llm.release_first.set()

        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == ["latest question"]
    finally:
        llm.release_first.set()
        runtime.stop()


def test_failed_task_retires_its_already_queued_stream_sentence():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM())
    task = runtime.supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text="old question",
            reason="test",
            mode=Mode.ASSISTANT,
        )
    )
    runtime.supervisor.state.active_tasks[task.task_id] = task
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_REQUEST,
            {
                "task_id": task.task_id,
                "text": "STALE QUEUED SENTENCE",
                "epoch": task.speech_epoch,
            },
        )
    )
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_FAILED,
            {
                "task_id": task.task_id,
                "error": "boom",
                "speak": False,
                "epoch": task.speech_epoch,
            },
            priority=25,
        )
    )
    try:
        runtime.bus.drain()

        assert engine.spoken == []
        assert task.task_id not in runtime.supervisor.state.active_tasks
    finally:
        runtime.stop()


def test_stream_completion_remembers_only_sentences_admitted_to_playback():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM())
    task = runtime.supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text="question",
            reason="test",
            mode=Mode.ASSISTANT,
        )
    )
    runtime.supervisor.state.active_tasks[task.task_id] = task
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_REQUEST,
            {
                "task_id": task.task_id,
                "text": "Sentence one.",
                "epoch": task.speech_epoch,
            },
        )
    )
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "Sentence one. Sentence two was never queued.",
                "speak": True,
                "followup": False,
                "data": {"streamed": True},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_STREAM_END,
            {"task_id": task.task_id, "epoch": task.speech_epoch},
            priority=110,
        )
    )
    try:
        runtime.bus.drain()

        assert engine.spoken == ["Sentence one."]
        assistant_memory = [
            item.text
            for item in runtime.memory.all()
            if "assistant_output" in item.tags
        ]
        assert assistant_memory == ["Sentence one."]
        assert runtime.supervisor.state.pending_audio_tasks == {}
    finally:
        runtime.stop()


def test_rejected_auxiliary_tts_consumes_its_registry_id():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM())
    runtime.supervisor.cancel_all()
    aux_tts_id = runtime.supervisor.register_aux_tts(
        "old-ack",
        speech_epoch=0,
        input_generation=0,
        input_epoch=0,
    )
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_REQUEST,
            {
                "task_id": "old-ack",
                "text": "STALE ACK",
                "epoch": 0,
                "auxiliary_tts": True,
                "aux_tts_id": aux_tts_id,
            },
        )
    )
    try:
        runtime.bus.drain()

        assert engine.spoken == []
        assert runtime.supervisor.state.pending_aux_tts == {}
    finally:
        runtime.stop()


def test_confirmation_stays_staged_if_a_new_arrival_wins_start_gap():
    supervisor = AgentSupervisor()
    assert supervisor.note_input_arrival(1)
    assert supervisor.commit_input_generation(1)
    task = supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.COMMAND,
            confidence=1.0,
            text="do the thing",
            reason="test",
            mode=Mode.COMMAND,
        )
    )
    task.metadata.update({"input_epoch": 0, "input_generation": 1})
    supervisor.state.pending_confirmations[task.task_id] = task
    original_start = supervisor._start_task

    def newer_arrival_then_start(candidate, *args, **kwargs):
        assert supervisor.note_input_arrival(2)
        return original_start(candidate, *args, **kwargs)

    supervisor._start_task = newer_arrival_then_start  # type: ignore[method-assign]

    supervisor._confirm_next(input_generation=1, input_epoch=0)

    assert supervisor.state.pending_confirmations == {task.task_id: task}
    assert not task.cancel_event.is_set()
    assert supervisor.state.active_tasks == {}


def test_stop_retires_a_held_final_before_its_submission_epoch_can_refresh():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        turn_merge_config=TurnMergeConfig(
            enabled=True,
            hold_sec=0.2,
            max_hold_sec=0.5,
        ),
    )
    runtime.start(run_bus=False)
    try:
        engine.final("tell me about")
        assert runtime._dispatcher is not None
        assert runtime._dispatcher.has_pending

        runtime.bus.publish(AgentEvent.stop("test"))
        runtime.bus.drain()
        time.sleep(0.25)
        runtime.bus.drain()

        assert not runtime._dispatcher.has_pending
        assert engine.spoken == []
    finally:
        runtime.stop()


class _IngestThenAct:
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, _text, recent=()):
        self.calls += 1
        return "INGEST" if self.calls == 1 else ACT


class _BlockingIngestMemory(SessionMemory):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def add(self, text: str, tags: tuple[str, ...] = ()) -> None:
        if text == "ambient old":
            self.started.set()
            assert self.release.wait(2.0)
        super().add(text, tags)


def test_irreversible_ingest_write_owns_terminal_before_newer_dispatch():
    memory = _BlockingIngestMemory()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        memory=memory,
        addressing=_IngestThenAct(),
        unsure_acts=False,
    )
    runtime.start(run_bus=True)
    try:
        engine.final("ambient old")
        assert memory.started.wait(1.0)

        engine.final("latest question")
        time.sleep(0.05)
        # The old lease claimed before entering irreversible memory I/O, so the
        # dispatcher orders the newer turn behind that committed effect.
        assert engine.spoken == []

        memory.release.set()
        assert runtime.wait_idle(timeout=2.0)
        assert any(
            item.text == "ambient old" and "ingested" in item.tags
            for item in memory.all()
        )
        assert engine.spoken == ["You said: latest question"]
    finally:
        memory.release.set()
        runtime.stop()


def test_shutdown_rejects_late_aux_registration_and_publication():
    supervisor = AgentSupervisor()
    supervisor.shutdown()

    assert supervisor.register_aux_tts("late", speech_epoch=0) == ""
    supervisor.publish(
        AgentEvent(
            EventKind.TTS_REQUEST,
            {"task_id": "late", "text": "late"},
        )
    )

    assert supervisor.state.pending_aux_tts == {}
    assert supervisor.bus.idle()


def test_backlogged_reserved_final_keeps_lineage_for_a_second_addon():
    llm = _FirstPretokenBlockedEcho()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=False)
    try:
        engine.final("explain the moon phases")
        assert runtime.bus.drain_once()
        assert llm.first_started.wait(1.0)

        engine.final("make it shorter")
        assert runtime._latest_arrival_continuation() is not None
        engine.final("and also use bullets")
        assert runtime._latest_arrival_continuation() is not None

        llm.release_first.set()
        assert runtime.wait_idle(timeout=2.0)
        assert len(engine.spoken) == 1
        spoken = engine.spoken[0]
        assert "explain the moon phases" in spoken
        assert spoken.count("make it shorter") == 1
        assert spoken.count("and also use bullets") == 1
        assert runtime._latest_arrival_continuation() is None
    finally:
        llm.release_first.set()
        runtime.stop()


def test_nonassistant_reserved_final_discards_synthetic_metadata():
    admitted: list[int] = []
    supervisor = AgentSupervisor(
        continuation_config=ContinuationConfig(enabled=True),
        on_continuation_admitted=admitted.append,
    )
    victim = supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text="explain the moon phases",
            reason="test",
            mode=Mode.ASSISTANT,
        )
    )
    supervisor.state.active_tasks[victim.task_id] = victim
    raw = "research that too"
    reservation = supervisor.reserve_arrival_continuation(raw)
    assert reservation is not None
    _merged, metadata = supervisor.materialize_arrival_continuation(
        reservation,
        raw,
    )
    generation = 3
    metadata.update(
        {
            "input_epoch": supervisor.input_epoch,
            "input_generation": generation,
        }
    )
    assert supervisor.commit_input_generation(generation)
    started = []
    supervisor._start_task = lambda task, *args, **kwargs: (  # type: ignore[method-assign]
        started.append(task) or True
    )

    supervisor.handle_event(AgentEvent.final(raw, metadata=metadata))

    assert len(started) == 1
    task = started[0]
    assert task.mode == Mode.RESEARCH
    assert task.input_text == "that too"
    assert not any(
        key == "reserved_continuation"
        or key == "skip_user_memory"
        or key.startswith("continuation_")
        for key in task.metadata
    )
    assert admitted == [generation]


class _SpeakingThenBlockedEcho:
    """Speak the parent sentence, then keep its task live until released."""

    def __init__(self) -> None:
        self.first_spoken = threading.Event()
        self.release_first = threading.Event()
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def generate(self, prompt, *, system=None, images=None):
        return "".join(self.stream(prompt, system=system, images=images))

    def stream(self, prompt, *, system=None, images=None):
        with self._lock:
            self.prompts.append(prompt)
            call = len(self.prompts)
        if call == 1:
            yield "PARENT ANSWER."
            # Sentence splitting requires whitespace after the terminator; this
            # second chunk makes the first sentence audible before we block.
            yield " "
            self.first_spoken.set()
            assert self.release_first.wait(2.0)
            return
        yield prompt


def test_slow_gate_keeps_context_after_the_speaking_parent_finishes():
    """After-audio continuation context survives parent completion in a gate."""
    llm = _SpeakingThenBlockedEcho()
    gate = _SecondCallBlockingAddressing()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        addressing=gate,
        unsure_acts=False,
        stream_tts=True,
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=True)
    try:
        engine.final("explain the moon phases")
        assert llm.first_spoken.wait(1.0)
        assert _wait_until(lambda: engine.spoken == ["PARENT ANSWER."])
        victim = next(iter(runtime.supervisor.state.active_tasks.values()))
        assert victim.started_speaking

        engine.final("make it shorter")
        assert gate.second_started.wait(1.0)
        assert not victim.cancel_event.is_set()

        # The parent is allowed to finish while the add-on is still in its gate.
        llm.release_first.set()
        assert _wait_until(
            lambda: victim.task_id not in runtime.supervisor.state.active_tasks
        )
        assert engine.spoken == ["PARENT ANSWER."]

        gate.release_second.set()
        assert runtime.wait_idle(timeout=2.0)
        assert len(llm.prompts) == 2
        continuation = " ".join(engine.spoken[1:])
        assert "A moment ago you were asked" in continuation
        assert "explain the moon phases" in continuation
        assert "make it shorter" in continuation
        assert MERGED not in runtime.metrics.records()[0].stamps
    finally:
        gate.release_second.set()
        llm.release_first.set()
        runtime.stop()


class _GenerateMustNotRun:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.stream_calls = 0

    def generate(self, prompt, *, system=None, images=None):
        raise AssertionError("pre-task gate used blocking generate()")

    def stream(self, prompt, *, system=None, images=None):
        self.stream_calls += 1
        yield self.reply


@pytest.mark.parametrize(
    "invoke, reply, expected",
    [
        (lambda llm: LLMAddressingClassifier(llm).classify("hello"), "ACT", ACT),
        (
            lambda llm: LLMTranscriptCleaner(llm).clean("hello hello"),
            "hello",
            "hello",
        ),
        (
            lambda llm: LLMCapabilityRouter(
                HeuristicCapabilityRouter(),
                llm,
                confidence_threshold=0.9,
            ).route("an ambiguous medium length utterance here", {}).action,
            "RESEARCH",
            RESEARCH,
        ),
    ],
)
def test_preprocessing_gates_use_stream_not_generate(invoke, reply, expected):
    llm = _GenerateMustNotRun(reply)
    assert invoke(llm) == expected
    assert llm.stream_calls == 1


@pytest.mark.parametrize(
    "invoke",
    [
        lambda llm: LLMAddressingClassifier(llm).classify("hello"),
        lambda llm: LLMTranscriptCleaner(llm).clean("hello hello"),
        lambda llm: LLMCapabilityRouter(
            HeuristicCapabilityRouter(),
            llm,
            confidence_threshold=0.9,
        ).route("an ambiguous medium length utterance here", {}),
    ],
)
def test_preprocessing_gates_propagate_cancellation_instead_of_fallback(invoke):
    cancel = threading.Event()
    cancel.set()
    context_token = capability_context.set({"cancel_event": cancel})
    try:
        with pytest.raises(LLMCallCancelled):
            invoke(_GenerateMustNotRun("ACT"))
    finally:
        capability_context.reset(context_token)


def test_cancelled_router_result_is_not_cached_for_a_healthy_identical_retry():
    llm = _GenerateMustNotRun("RESEARCH")
    router = LLMCapabilityRouter(
        HeuristicCapabilityRouter(),
        llm,
        confidence_threshold=0.9,
    )
    text = "an ambiguous medium length utterance here"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(LLMCallCancelled):
        router.route(text, {"cancel_event": cancel})

    result = router.route(text, {})

    assert result.action == RESEARCH
    assert llm.stream_calls == 1


def test_collect_helper_restores_inherited_context_after_natural_completion():
    llm = _GenerateMustNotRun("ok")
    marker = object()
    context_token = capability_context.set({"marker": marker})
    try:
        assert collect_llm_text(llm, "hello") == "ok"
        assert capability_context.get().get("marker") is marker
    finally:
        capability_context.reset(context_token)


class _FailureAfterCancellation:
    def __init__(self, cancel: threading.Event) -> None:
        self.cancel = cancel
        self.closed = False

    def generate(self, prompt, *, system=None, images=None):
        raise AssertionError("generate() must not run")

    def stream(self, prompt, *, system=None, images=None):
        outer = self

        class _Stream:
            def __iter__(self):
                return self

            def __next__(self):
                outer.cancel.set()
                raise RuntimeError("provider lost the cancellation race")

            def close(self):
                outer.closed = True

        return _Stream()


def test_provider_failure_after_retirement_stays_dedicated_cancellation():
    cancel = threading.Event()
    llm = _FailureAfterCancellation(cancel)
    marker = object()
    context_token = capability_context.set({"marker": marker})
    try:
        with pytest.raises(LLMCallCancelled):
            collect_llm_text(llm, "hello", cancel_event=cancel)
        assert llm.closed
        assert capability_context.get() == {"marker": marker}
    finally:
        capability_context.reset(context_token)


class _CancelledCloseLLM:
    def generate(self, prompt, *, system=None, images=None):
        raise AssertionError("generate() must not run")

    def stream(self, prompt, *, system=None, images=None):
        class _Stream:
            def __iter__(self):
                return iter(())

            def close(self):
                raise asyncio.CancelledError()

        return _Stream()


def test_cancelled_close_cannot_leak_the_helper_context():
    marker = object()
    context_token = capability_context.set({"marker": marker})
    try:
        assert collect_llm_text(_CancelledCloseLLM(), "hello") == ""
        assert capability_context.get() == {"marker": marker}
    finally:
        capability_context.reset(context_token)


@pytest.mark.parametrize("cancel_source", ["inherited", "explicit"])
def test_helper_combines_distinct_inherited_and_explicit_cancellation(cancel_source):
    inherited = threading.Event()
    explicit = threading.Event()
    (inherited if cancel_source == "inherited" else explicit).set()
    context_token = capability_context.set(
        {"cancel_event": inherited, "marker": "kept"}
    )
    try:
        with pytest.raises(LLMCallCancelled):
            collect_llm_text(
                _GenerateMustNotRun("unused"),
                "hello",
                cancel_event=explicit,
            )
        assert capability_context.get() == {
            "cancel_event": inherited,
            "marker": "kept",
        }
    finally:
        capability_context.reset(context_token)
