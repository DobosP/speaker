"""Failure-cascade resilience tests for the core runtime.

A failure mid-turn must never wedge the assistant: the turn must terminate
(either a graceful fallback is spoken, or the task fails cleanly and leaves
``active_tasks``) AND the runtime must still process a SUBSEQUENT normal turn.
That second turn is the real proof of "no deadlock / no wedge" -- it can only
succeed if the failed turn released the controller back to idle.

Each test injects ONE failure at a time, drives it with the scripted engine and
deterministic fakes (no audio, no models), and waits on the runtime's own
completion signal (``wait_idle`` -> active_tasks/queued_tasks empty) with a
bounded timeout -- never an unbounded sleep. They pin the real try/except wiring
in:
  - always_on_agent/tasks.py:207-214  (_run_task wraps the plan; a capability /
    LLM raise becomes a TASK_FAILED instead of a silently-dead daemon thread)
  - always_on_agent/supervisor.py:153-156 (TASK_FAILED pops the task from
    active_tasks + records the error, so the controller returns to idle)
  - core/capabilities.py:260-290 (memory/recall errors are swallowed so the
    answer is still produced -- a graceful fallback, not a dropped turn)
"""

from __future__ import annotations

import threading
from typing import Iterator, Optional, Sequence

from always_on_agent.events import Mode

from core.capabilities import RecallConfig
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runtime import VoiceRuntime


# A generous-but-bounded ceiling for every wait so a genuine wedge fails the
# test (instead of hanging the whole suite) rather than passing on a long sleep.
WAIT = 4.0


class _RaiseMidTokenLLM:
    """Streaming fake whose stream yields one token, then raises -- the
    'LLM died mid-generation' case (Ollama dropped the socket, model crashed).

    ``generate`` is provided for protocol-completeness; the capability layer
    only calls ``stream``."""

    def __init__(self) -> None:
        self.stream_calls = 0

    def generate(self, prompt: str, *, system: Optional[str] = None,
                 images: Optional[Sequence[object]] = None) -> str:  # pragma: no cover
        return "unused"

    def stream(self, prompt: str, *, system: Optional[str] = None,
               images: Optional[Sequence[object]] = None) -> Iterator[str]:
        self.stream_calls += 1
        yield "Well, "          # a real token reaches the consumer first...
        raise RuntimeError("model stream exploded mid-token")  # ...then it dies


class _RaiseBeforeTokenLLM:
    """Streaming fake that raises on the very first ``next()`` -- the LLM (or
    the whole answering capability) blew up before producing anything."""

    def generate(self, prompt: str, *, system: Optional[str] = None,
                 images: Optional[Sequence[object]] = None) -> str:  # pragma: no cover
        raise RuntimeError("generate exploded")

    def stream(self, prompt: str, *, system: Optional[str] = None,
               images: Optional[Sequence[object]] = None) -> Iterator[str]:
        raise RuntimeError("stream exploded before first token")
        yield ""  # pragma: no cover - unreachable, marks this a generator


class _ExplodingMemory:
    """A Memory whose recall/recent-window reads raise ConnectionError -- the
    Postgres-backed memory is unreachable. Every read the capability layer makes
    (``all`` for recent context, ``add`` to ingest, ``context_for_llm`` for
    recall) detonates, so the test proves the turn survives a dead memory.

    ``add`` of an assistant reply (supervisor side) is allowed to succeed so the
    failure is isolated to the recall path the assistant capability touches."""

    def __init__(self) -> None:
        self.add_attempts = 0

    def add(self, text: str, tags: tuple[str, ...] = ()) -> None:
        self.add_attempts += 1
        # The assistant capability ingests the user query under ("user",);
        # detonate there to exercise the recall try/except. Assistant-output
        # writes (supervisor) are let through so memory of the reply still works.
        if "assistant_output" not in tags:
            raise ConnectionError("memory backend is down")

    def search(self, query: str, limit: int = 5):
        raise ConnectionError("memory backend is down")

    def all(self):
        raise ConnectionError("memory backend is down")

    def context_for_llm(self, query: str) -> str:
        raise ConnectionError("memory backend is down")

    def prune(self) -> int:
        return 0

    def close(self) -> None:
        return None


def _normal_turn_recovers(runtime: VoiceRuntime, engine: ScriptedEngine,
                          *, phrase: str, expected: str) -> None:
    """Drive a plain turn AFTER a failure and assert it is answered -- the
    proof that the prior failure did not wedge the controller. Mutates the
    EchoLLM's reply so the follow-up turn has a deterministic, distinct answer.
    """
    before = list(engine.spoken)
    engine.final(phrase)
    assert runtime.wait_idle(timeout=WAIT), "controller did not return to idle"
    new = engine.spoken[len(before):]
    assert new == [expected], f"subsequent normal turn not answered cleanly: {new!r}"


def test_llm_stream_raising_mid_token_does_not_wedge_the_runtime():
    """(a) The LLM stream raises after one token. The turn must terminate -- the
    task fails out of active_tasks and the error is recorded -- so the controller
    returns to idle instead of waiting on a dead daemon thread forever."""
    engine = ScriptedEngine()
    boom = _RaiseMidTokenLLM()
    runtime = VoiceRuntime(engine, boom, start_mode=Mode.ASSISTANT)
    runtime.start(run_bus=False)
    try:
        engine.final("tell me a joke")
        # Bounded wait: the failed task must leave active_tasks so the controller
        # is idle again -- the anti-wedge invariant.
        assert runtime.wait_idle(timeout=WAIT), "runtime wedged on a mid-token LLM crash"
        assert runtime.supervisor.state.active_tasks == {}
        assert runtime.supervisor.state.queued_tasks == []
        # The failure was surfaced (TASK_FAILED -> supervisor.failures), not
        # swallowed into a silent dead thread.
        assert any("exploded" in f or "RuntimeError" in f
                   for f in runtime.supervisor.state.failures), \
            runtime.supervisor.state.failures
        assert boom.stream_calls == 1, "the failing capability never actually ran"
    finally:
        runtime.stop()


def test_runtime_recovers_a_normal_turn_after_an_llm_crash():
    """The headline anti-wedge proof: after a turn whose LLM raised, a fresh
    turn on a HEALTHY model is answered normally.

    Wiring exploits the real two-model split (core/capabilities.py): ASSISTANT
    turns answer on ``fast_llm`` (here, the bomb), while a RESEARCH turn runs
    ``research.local`` on the main ``llm`` (here, healthy). One immutable runtime
    thus yields one crashed turn followed by one clean turn -- and the second can
    only succeed if the crash released the controller back to idle."""
    engine = ScriptedEngine()
    healthy = EchoLLM(reply="All good now.")
    bomb = _RaiseBeforeTokenLLM()
    runtime = VoiceRuntime(engine, healthy, fast_llm=bomb, start_mode=Mode.ASSISTANT)
    runtime.start(run_bus=False)
    try:
        # ASSISTANT turn -> fast tier (the bomb) -> raises before first token.
        engine.final("what is two plus two")
        assert runtime.wait_idle(timeout=WAIT), "runtime wedged on a pre-token LLM crash"
        assert runtime.supervisor.state.active_tasks == {}
        failures_after_crash = len(runtime.supervisor.state.failures)
        assert failures_after_crash >= 1, "the LLM crash was not surfaced as a failure"
        assert engine.spoken == [], "a crashed turn must not fabricate speech"

        # Recovery: a RESEARCH turn runs research.local on the MAIN (healthy)
        # llm, so it must complete and speak -- proving the controller is live
        # and unwedged after the crash.
        engine.final("research the weather on mars")
        assert runtime.wait_idle(timeout=WAIT), "controller wedged; recovery turn never ran"
        assert runtime.supervisor.state.active_tasks == {}
        assert engine.spoken, "no recovery turn was answered -- the runtime wedged"
        # No NEW failure was recorded for the healthy recovery turn.
        assert len(runtime.supervisor.state.failures) == failures_after_crash, \
            "the healthy recovery turn unexpectedly failed"
    finally:
        runtime.stop()


def test_capability_raising_fails_the_turn_then_runtime_serves_the_next():
    """(b) A registered capability raises a non-LLM error. The turn must fail
    cleanly (TASK_FAILED, controller back to idle) and the NEXT normal turn must
    be answered -- the failure does not poison the registry or wedge the bus."""
    engine = ScriptedEngine()
    llm = EchoLLM(reply="Healthy answer.")
    runtime = VoiceRuntime(engine, llm, start_mode=Mode.ASSISTANT)
    runtime.start(run_bus=False)

    # Overwrite the answering capability with one that raises a plain error
    # (distinct from an LLM stream death -- exercises the direct-call try path
    # in tasks._invoke/_run_plan, not the streaming-iteration one).
    def _boom(query, context):
        raise ValueError("capability blew up")

    runtime.supervisor.capabilities.register("assistant.answer", _boom)
    try:
        engine.final("anything at all")
        assert runtime.wait_idle(timeout=WAIT), "runtime wedged on a raising capability"
        assert runtime.supervisor.state.active_tasks == {}
        assert any("blew up" in f or "ValueError" in f
                   for f in runtime.supervisor.state.failures), \
            runtime.supervisor.state.failures
        assert engine.spoken == [], "a failed capability must not speak"

        # Repair the capability and prove the controller still serves a turn.
        def _ok(query, context):
            from always_on_agent.capabilities import CapabilityResult
            return CapabilityResult(True, "Recovered.")

        runtime.supervisor.capabilities.register("assistant.answer", _ok)
        _normal_turn_recovers(runtime, engine, phrase="now answer me",
                              expected="Recovered.")
    finally:
        runtime.stop()


def test_memory_recall_raising_still_answers_the_turn():
    """(c) The memory backend is unreachable (ConnectionError on every recall
    read). The assistant capability swallows it and answers WITHOUT context --
    a graceful fallback, NOT a dropped turn. Then a second turn still works.

    Recall is enabled so ``context_for_llm`` is actually exercised; without that
    gate the failing recall path would never be hit and the test would be a
    tautology."""
    engine = ScriptedEngine()
    llm = EchoLLM(reply="Answer despite dead memory.")
    dead_memory = _ExplodingMemory()
    runtime = VoiceRuntime(
        engine, llm,
        start_mode=Mode.ASSISTANT,
        memory=dead_memory,
        recall_config=RecallConfig(enabled=True),
    )
    runtime.start(run_bus=False)
    try:
        engine.final("what did we say earlier")
        assert runtime.wait_idle(timeout=WAIT), "runtime wedged when memory recall raised"
        assert runtime.supervisor.state.active_tasks == {}
        # The turn COMPLETED and spoke the answer -- the memory failure was a
        # graceful fallback, not a turn-killer.
        assert engine.spoken == ["Answer despite dead memory."], engine.spoken
        # No task failure was recorded: a memory error is swallowed, not fatal.
        assert runtime.supervisor.state.failures == [], runtime.supervisor.state.failures
        # The recall path was genuinely exercised (the capability tried to read
        # memory, which detonated) -- not silently skipped.
        assert dead_memory.add_attempts >= 1, "the recall path was never hit"

        # A second turn still completes despite the still-dead memory.
        engine.final("and again")
        assert runtime.wait_idle(timeout=WAIT), "second turn wedged on dead memory"
        assert engine.spoken[-1] == "Answer despite dead memory."
    finally:
        runtime.stop()
