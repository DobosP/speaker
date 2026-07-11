"""The interrupt / barge-in race (realtime-concurrency-1).

When a barge-in (or a stop command) and a stale TTS / completion reply race, the
runtime must NOT speak a turn the user already interrupted. The mechanism is a
monotonic **speech epoch**: ``AgentSupervisor.cancel_all`` advances it, every
TTS_REQUEST carries the epoch captured when its task started, and the runtime
gates playback on ``supervisor.tts_request_allowed(task_id, epoch)`` before it
ever calls ``engine.speak`` (see ``core/runtime.py`` ``_on_event`` and
``always_on_agent/supervisor.py``).

These tests pin that invariant at three levels with no audio, no models, and no
subprocess -- the scripted engine + a deterministic gated fake LLM only. They
deliberately complement the existing streaming-path regression in
``tests/test_core_runtime.py`` by also exercising the gate directly and on the
non-streaming completion-reply path.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator, Optional, Sequence

from always_on_agent.events import AgentEvent, EventKind, Mode

from core.engines.scripted import ScriptedEngine
from core.runtime import VoiceRuntime


# --- shared helpers ---------------------------------------------------------


class _RecordingEngine(ScriptedEngine):
    """ScriptedEngine that records the ordered speak / stop_speaking calls so a
    test can assert no stale ``speak()`` lands after a barge-in's
    ``stop_speaking()``."""

    def __init__(self, hold_speech: bool = False):
        super().__init__(hold_speech=hold_speech)
        self.calls: list[tuple[str, str]] = []
        self._calls_lock = threading.Lock()

    def speak(self, text, on_done=None):
        with self._calls_lock:
            self.calls.append(("speak", text))
        super().speak(text, on_done)

    def speak_tracked(self, speech, *, on_terminal, on_started=None):
        with self._calls_lock:
            self.calls.append(("speak", speech.text))
        super().speak_tracked(
            speech,
            on_terminal=on_terminal,
            on_started=on_started,
        )

    def stop_speaking(self):
        with self._calls_lock:
            self.calls.append(("stop_speaking", ""))
        super().stop_speaking()

    def call_log(self) -> list[tuple[str, str]]:
        with self._calls_lock:
            return list(self.calls)


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# --- 1. the epoch gate itself (single source of truth) ----------------------


def test_tts_request_allowed_drops_pre_barge_epoch():
    """``tts_request_allowed`` is the documented single source of truth that gates
    TTS on the speech epoch. A barge-in advances the epoch via ``cancel_all``; a
    request stamped with the *pre-barge* epoch must be rejected, while one stamped
    with the *current* epoch is allowed. We assert against the REAL supervisor, not
    a re-implementation of the rule."""
    runtime = VoiceRuntime(ScriptedEngine(), start_mode=Mode.ASSISTANT)
    sup = runtime.supervisor

    epoch_before = sup.speech_epoch
    # A turn that started under the current epoch is allowed to speak.
    assert sup.tts_request_allowed("task-A", epoch_before) is True

    # A barge-in cancels everything and advances the epoch.
    sup.cancel_all()
    assert sup.speech_epoch == epoch_before + 1

    # The sentence produced by the now-interrupted turn (still stamped with the
    # pre-barge epoch) is dropped...
    assert sup.tts_request_allowed("task-A", epoch_before) is False
    # ...while a fresh turn stamped with the new epoch is allowed.
    assert sup.tts_request_allowed("task-B", sup.speech_epoch) is True


def test_tts_request_without_stamp_falls_back_to_active_task():
    """A legacy / direct emit with no epoch stamp falls back to the
    active-and-uncancelled check: unknown task id -> rejected, no task id ->
    allowed (the documented fallback)."""
    runtime = VoiceRuntime(ScriptedEngine(), start_mode=Mode.ASSISTANT)
    sup = runtime.supervisor
    # No epoch, and no such active task -> not allowed.
    assert sup.tts_request_allowed("ghost-task", None) is False
    # No epoch and no task id at all -> legacy allow.
    assert sup.tts_request_allowed("", None) is True


# --- 2. the runtime drops a stale-epoch TTS_REQUEST before engine.speak -----


def test_runtime_drops_stale_epoch_tts_request_before_speaking():
    """End-to-end through the real runtime bus subscriber (``_on_event``): a
    TTS_REQUEST stamped with a superseded epoch must never reach ``engine.speak``,
    while a current-epoch request does. This proves the gate fires on the actual
    playback path, not just in isolation."""
    engine = _RecordingEngine()
    runtime = VoiceRuntime(engine, start_mode=Mode.ASSISTANT)
    runtime.start(run_bus=False)
    sup = runtime.supervisor

    stale_epoch = sup.speech_epoch
    # The user interrupts: the epoch advances past anything stamped before it.
    sup.cancel_all()
    current_epoch = sup.speech_epoch

    # A sentence from the interrupted turn (stale epoch) and a sentence from the
    # current turn are both published; only the current one should be spoken.
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_REQUEST,
            {"task_id": "old", "text": "stale interrupted reply", "epoch": stale_epoch},
        )
    )
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_REQUEST,
            {"task_id": "new", "text": "fresh reply", "epoch": current_epoch},
        )
    )
    runtime.bus.drain()

    assert engine.spoken == ["fresh reply"]
    assert "stale interrupted reply" not in engine.spoken


# --- 3. non-streaming completion reply is suppressed by a barge-in ----------


class _GatedGenerateLLM:
    """Non-streaming-style fake LLM whose token stream is one whole answer that
    is *gated*: it blocks mid-generation until the test releases it. This models
    a slow ``generate`` so a test can fire a barge-in while the answer is still
    being produced and prove the (now stale) completion is never spoken."""

    def __init__(self, reply: str):
        self._reply = reply
        self.stream_started = threading.Event()
        self.gate = threading.Event()

    def generate(self, prompt: str, *, system=None, images=None) -> str:  # pragma: no cover
        return self._reply

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[object]] = None,
    ) -> Iterator[str]:
        # Announce that generation is underway, then block before producing the
        # answer so the test can land a barge-in mid-generation.
        self.stream_started.set()
        self.gate.wait(timeout=2.0)
        # The cancel check lives in the capability's _collect loop between
        # tokens, so emit the answer in pieces to give cancellation a seam.
        for word in self._reply.split():
            yield word + " "


def test_barge_in_during_generation_suppresses_completion_reply():
    """Regression for the interrupt race on the NON-streaming completion path
    (``stream_tts=False``). The user barges in while the model is still
    generating; the interrupted turn must produce NO speech and NO speak() may
    land after the barge-in's stop_speaking()."""
    llm = _GatedGenerateLLM("here is a long winded answer to your question")
    engine = _RecordingEngine(hold_speech=True)
    runtime = VoiceRuntime(engine, llm, start_mode=Mode.ASSISTANT, stream_tts=False)
    runtime.start(run_bus=True)
    try:
        engine.final("tell me a story")
        # The model is mid-generation but has emitted nothing playable yet.
        assert _wait_until(lambda: llm.stream_started.is_set())
        assert _wait_until(lambda: bool(runtime.supervisor.state.active_tasks))
        assert engine.spoken == []

        epoch_before = runtime.supervisor.speech_epoch
        # Barge-in: cancellation is set (epoch bumped) before stop_speaking()
        # returns; then we release the generation gate.
        engine.barge_in()
        assert runtime.supervisor.speech_epoch == epoch_before + 1
        llm.gate.set()

        # Let the now-cancelled task finish unwinding; no active task should
        # remain and nothing from the interrupted turn may be spoken.
        assert _wait_until(lambda: not runtime.supervisor.state.active_tasks)
        time.sleep(0.05)

        assert engine.spoken == [], f"stale completion spoken: {engine.spoken}"

        # Hard ordering invariant: no speak() after the barge-in stop_speaking().
        calls = engine.call_log()
        stop_idx = [i for i, (kind, _) in enumerate(calls) if kind == "stop_speaking"]
        assert stop_idx, "expected a stop_speaking on barge-in"
        speak_after_stop = [
            text for kind, text in calls[stop_idx[0] + 1 :] if kind == "speak"
        ]
        assert speak_after_stop == [], f"stale speak after stop: {speak_after_stop}"
    finally:
        runtime.stop()


# --- 4. rc-4: queued-task cancellation / resurrection ------------------------


def _make_task(sup):
    from always_on_agent.models import IntentDecision, IntentKind

    decision = IntentDecision(IntentKind.ASSISTANT, 0.9, "x", "test", mode=Mode.ASSISTANT)
    return sup.tasks.create_task(decision)


def test_cancel_all_cancels_queued_tasks():
    """rc-4: cancel_all must cancel (not just drop) queued tasks, so a concurrent
    _start_queued_tasks holding a reference sees them cancelled."""
    from always_on_agent.supervisor import AgentSupervisor

    sup = AgentSupervisor()
    queued = _make_task(sup)
    sup.state.queued_tasks.append(queued)

    sup.cancel_all()

    assert queued.cancel_event.is_set()
    assert list(sup.state.queued_tasks) == []


def test_start_task_drops_a_precancelled_task():
    """rc-4: _start_task must drop a task whose cancel_event is already set --
    never register it as active nor spawn a worker thread."""
    from always_on_agent.supervisor import AgentSupervisor

    sup = AgentSupervisor()
    task = _make_task(sup)
    task.cancel()

    sup._start_task(task)  # noqa: SLF001 - exercising the guard directly

    assert task.task_id not in sup.state.active_tasks
    assert sup.tasks.active_count == 0


def test_start_queued_drops_cancelled_task_without_calling_start_task(monkeypatch):
    """rc-4: a queued task cancelled before the queue drains is dropped by
    _start_queued_tasks's OWN guard -- it is never even handed to _start_task.
    (Spying on _start_task makes this pin the drain-loop drop specifically, not
    _start_task's separate cancel re-check.)"""
    from always_on_agent.supervisor import AgentSupervisor

    sup = AgentSupervisor()
    passed_to_start: list = []
    monkeypatch.setattr(sup, "_start_task",
                        lambda task, **kw: passed_to_start.append(task.task_id))
    task = _make_task(sup)
    sup.state.queued_tasks.append(task)
    task.cancel()

    sup._start_queued_tasks()  # noqa: SLF001

    assert task.task_id not in passed_to_start  # dropped before _start_task
    assert task not in sup.state.queued_tasks


def test_start_queued_respects_research_cap_in_one_pass(monkeypatch):
    """rc-4 over-admission regression: a SINGLE _start_queued_tasks drain pass must
    not admit more RESEARCH tasks than the per-mode cap. The earlier deferred-start
    version read stale capacity counts and started all of them at once."""
    from always_on_agent.models import IntentDecision, IntentKind
    from always_on_agent.speech_analyzer import LiveSpeechAnalyzer, ModePolicy
    from always_on_agent.supervisor import AgentSupervisor

    sup = AgentSupervisor(analyzer=LiveSpeechAnalyzer(ModePolicy(research_parallel_tasks=2)))
    started: list = []
    monkeypatch.setattr(sup.tasks, "start", lambda task: started.append(task.task_id))

    for _ in range(3):
        d = IntentDecision(IntentKind.RESEARCH, 0.9, "x", "test", mode=Mode.RESEARCH)
        sup.state.queued_tasks.append(sup.tasks.create_task(d))

    sup._start_queued_tasks()  # noqa: SLF001

    assert len(started) == 2, f"over-admitted past the RESEARCH cap: started {len(started)}"
    assert len(sup.state.active_tasks) == 2
    assert len(sup.state.queued_tasks) == 1  # the 3rd stays queued


def test_barge_in_during_drain_does_not_resurrect_queued_task(monkeypatch):
    """rc-4 residual window: a barge-in (cancel_all) landing mid-drain -- after the
    queue snapshot, while _should_queue runs with the lock released -- must NOT
    start the snapshotted task; the epoch re-check in _start_task drops it."""
    from always_on_agent.supervisor import AgentSupervisor

    sup = AgentSupervisor()
    started: list = []
    monkeypatch.setattr(sup.tasks, "start", lambda task: started.append(task.task_id))
    task = _make_task(sup)
    sup.state.queued_tasks.append(task)

    orig_should_queue = sup._should_queue

    def racing_should_queue(t):
        sup.cancel_all()  # a barge-in lands inside the drain, advancing the epoch
        return orig_should_queue(t)

    monkeypatch.setattr(sup, "_should_queue", racing_should_queue)

    sup._start_queued_tasks()  # noqa: SLF001

    assert started == [], "a queued task was resurrected after a mid-drain barge-in"
    assert task.task_id not in sup.state.active_tasks
