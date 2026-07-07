"""End-to-end tests for the lean core runtime.

These exercise the full path engine -> brain -> capability -> TTS using the
scriptable engine and a deterministic fake LLM, so they need no audio hardware,
no sherpa-onnx, and no Ollama.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator, Optional, Sequence

from always_on_agent.continuation import ContinuationConfig
from always_on_agent.events import AgentEvent, EventKind, Mode

from core.engines.scripted import ScriptedEngine
from core.intents import LocalIntentHandler
from core.llm import EchoLLM
from core.metrics import HANDLED_LOCAL, LLM_FIRST_TOKEN
from core.runtime import VoiceRuntime


def _runtime(start_mode=Mode.ASSISTANT, hold_speech=False, reply=None, command_map=None):
    engine = ScriptedEngine(hold_speech=hold_speech)
    runtime = VoiceRuntime(
        engine, EchoLLM(reply=reply), start_mode=start_mode, command_map=command_map
    )
    runtime.start(run_bus=False)
    return runtime, engine


_COMMANDS = {"stop": "stop", "command mode": "mode:command", "yes do it": "confirm"}


def test_assistant_reply_is_spoken():
    runtime, engine = _runtime(reply="The time is noon.")
    engine.final("what time is it")
    assert runtime.wait_idle()
    assert engine.spoken == ["The time is noon."]


def test_stop_drops_tts_request_racing_shutdown():
    """A queued TTS_REQUEST must never start speaking once stop() has begun:
    the threaded bus keeps dispatching between stop()'s first line and
    bus.stop() (dispatcher/watchdog/supervisor teardown sit in between), so the
    gate lives in _on_event. Calling _on_event directly models the late
    dispatch deterministically -- no thread timing."""
    runtime, engine = _runtime()
    event = AgentEvent(EventKind.TTS_REQUEST, {"text": "late reply"})
    runtime._on_event(event)
    assert engine.spoken == ["late reply"]  # sanity: allowed before stop
    runtime.stop()
    runtime._on_event(event)
    assert engine.spoken == ["late reply"]  # dropped during/after shutdown


def test_llm_task_does_not_mark_handled_local():
    runtime, engine = _runtime(reply="The time is noon.")
    engine.final("what time is it")
    assert runtime.wait_idle()
    [record] = runtime.metrics.records()
    assert LLM_FIRST_TOKEN in record.stamps
    assert HANDLED_LOCAL not in record.stamps


def test_brain_local_task_marks_handled_local_for_watchdog():
    runtime, engine = _runtime(start_mode=Mode.MEETING)
    engine.final("we agreed to ship the local task metric")
    assert runtime.wait_idle()
    [record] = runtime.metrics.records()
    assert HANDLED_LOCAL in record.stamps
    assert LLM_FIRST_TOKEN not in record.stamps


def test_voice_mode_switch():
    runtime, engine = _runtime(start_mode=Mode.ASSISTANT)
    engine.final("research mode")
    assert runtime.wait_idle()
    assert runtime.mode == Mode.RESEARCH


def test_research_prefix_runs_multistep_plan_and_speaks():
    runtime, engine = _runtime()
    engine.final("research local speech to text engines")
    assert runtime.wait_idle()
    # Research plan = scope -> search.local -> research.local(synthesis, spoken).
    assert len(engine.spoken) == 2
    assert engine.spoken[0] == "I'll check that now."
    assert engine.spoken[1]  # non-empty synthesized answer


class _BlockingGenerateLLM:
    def __init__(self, reply: str = "final answer"):
        self.reply = reply
        self.started = threading.Event()
        self.gate = threading.Event()

    def generate(self, prompt: str, *, system=None, images=None) -> str:
        self.started.set()
        self.gate.wait(timeout=2.0)
        return self.reply

    def stream(self, prompt: str, *, system=None, images=None) -> Iterator[str]:
        yield self.generate(prompt, system=system)


def test_ack_then_think_speaks_before_slow_reasoning_finishes():
    llm = _BlockingGenerateLLM("Here is the comparison.")
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, llm, start_mode=Mode.ASSISTANT)
    runtime.start(run_bus=True)
    try:
        engine.final("compare the latest local speech to text engines")
        assert _wait_until(lambda: engine.spoken == ["I'll check that now."])
        assert _wait_until(lambda: llm.started.is_set())
        assert engine.spoken == ["I'll check that now."]
        llm.gate.set()
        assert runtime.wait_idle()
        assert engine.spoken == ["I'll check that now.", "Here is the comparison."]
    finally:
        runtime.stop()


def test_instant_local_intent_gets_no_latency_ack():
    engine = ScriptedEngine()
    intents = LocalIntentHandler(engine.speak)
    runtime = VoiceRuntime(
        engine, EchoLLM(reply="should not run"), start_mode=Mode.ASSISTANT, intents=intents
    )
    runtime.start(run_bus=False)
    engine.final("what time is it")
    assert runtime.wait_idle()
    assert len(engine.spoken) == 1
    assert engine.spoken[0].startswith("It's ")


def test_stop_phrase_cancels_and_halts_playback():
    runtime, engine = _runtime(hold_speech=True, reply="a long winded answer")
    engine.final("tell me a story")
    assert runtime.wait_idle()
    assert engine.is_speaking  # held mid-utterance

    engine.final("stop")
    runtime.wait_idle()
    assert not engine.is_speaking


def test_barge_in_stops_playback():
    runtime, engine = _runtime(hold_speech=True, reply="assistant talking")
    engine.final("hello")
    assert runtime.wait_idle()
    assert engine.is_speaking

    engine.barge_in()
    runtime.wait_idle()
    assert not engine.is_speaking


def test_command_fast_path_stop_halts_playback_without_llm():
    runtime, engine = _runtime(hold_speech=True, reply="a long winded answer", command_map=_COMMANDS)
    engine.final("tell me a story")
    assert runtime.wait_idle()
    assert engine.is_speaking  # held mid-utterance

    engine.command("stop")  # spotted keyword, not a transcript
    runtime.wait_idle()
    assert not engine.is_speaking


def test_command_fast_path_switches_mode():
    runtime, engine = _runtime(start_mode=Mode.ASSISTANT, command_map=_COMMANDS)
    engine.command("command mode")
    assert runtime.wait_idle()
    assert runtime.mode == Mode.COMMAND


def test_unmapped_command_falls_back_to_transcript():
    # A keyword with no action mapping must not be dropped: it should behave
    # like a normal final transcript and get a spoken reply.
    runtime, engine = _runtime(reply="Sure.", command_map=_COMMANDS)
    engine.command("what time is it")
    assert runtime.wait_idle()
    assert engine.spoken == ["Sure."]


def test_passive_mode_ignores_unaddressed_speech():
    runtime, engine = _runtime(start_mode=Mode.PASSIVE)
    engine.final("what time is it")
    assert runtime.wait_idle()
    assert engine.spoken == []  # no wake word -> ignored


def test_passive_mode_activates_on_wake_word():
    runtime, engine = _runtime(start_mode=Mode.PASSIVE, reply="Hi there.")
    engine.final("assistant please help")
    assert runtime.wait_idle()
    assert engine.spoken == ["Hi there."]


# --- deterministic barge-in on the streaming-TTS path (realtime-concurrency-1) ---


class _GatedStreamLLM:
    """Streaming fake LLM that yields sentence-terminated chunks and blocks on a
    gate between the first and the rest. Lets a test barge in *after* sentence
    one is being spoken but *before* the later sentences could stream, then
    release the gate to prove the interrupted turn produces no more speech."""

    def __init__(self, sentences: Sequence[str]):
        self._sentences = list(sentences)
        # Set once the first sentence has been emitted to the TTS path.
        self.first_emitted = threading.Event()
        # The test sets this to let the stream proceed past sentence one.
        self.gate = threading.Event()

    def generate(self, prompt: str, *, system=None, images=None) -> str:  # pragma: no cover
        return " ".join(self._sentences)

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[object]] = None,
    ) -> Iterator[str]:
        if self._sentences:
            yield self._sentences[0] + " "
            self.first_emitted.set()
            # Wait until the test has fired the barge-in (or a timeout, so a
            # failing test never wedges the bus thread forever).
            self.gate.wait(timeout=2.0)
        for sentence in self._sentences[1:]:
            yield sentence + " "


class _RecordingEngine(ScriptedEngine):
    """ScriptedEngine that records the ordered sequence of speak/stop_speaking
    calls so a test can assert no ``speak()`` lands after a ``stop_speaking()``."""

    def __init__(self, hold_speech: bool = False):
        super().__init__(hold_speech=hold_speech)
        self.calls: list[tuple[str, str]] = []
        self._calls_lock = threading.Lock()

    def speak(self, text, on_done=None):
        with self._calls_lock:
            self.calls.append(("speak", text))
        super().speak(text, on_done)

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


def test_clarify_latency_policy_short_circuits_to_spoken_prompt():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine, EchoLLM(reply="should not run"), start_mode=Mode.ASSISTANT
    )
    runtime.start(run_bus=False)
    try:
        engine.final("why?")
        assert runtime.wait_idle()
        assert engine.spoken == ["Could you say a bit more?"]
        [record] = runtime.metrics.records()
        assert LLM_FIRST_TOKEN not in record.stamps
    finally:
        runtime.stop()


def test_stream_latency_policy_enables_per_turn_tts_streaming_when_global_off():
    llm = _GatedStreamLLM(["First sentence.", "Second sentence."])
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine, llm, start_mode=Mode.ASSISTANT, stream_tts=False
    )
    runtime.start(run_bus=True)
    try:
        runtime.bus.publish(
            AgentEvent.final(
                "tell me the plan", metadata={"latency_policy": "stream_main"}
            )
        )
        assert _wait_until(lambda: llm.first_emitted.is_set())
        assert _wait_until(lambda: engine.spoken == ["First sentence."])
        llm.gate.set()
        assert runtime.wait_idle()
        assert engine.spoken == ["First sentence.", "Second sentence."]
    finally:
        runtime.stop()


def test_silent_ingest_latency_policy_suppresses_spoken_output():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine, EchoLLM(reply="stored silently"), start_mode=Mode.ASSISTANT
    )
    runtime.start(run_bus=False)
    try:
        runtime.bus.publish(
            AgentEvent.final(
                "ambient note", metadata={"latency_policy": "silent_ingest"}
            )
        )
        assert runtime.wait_idle()
        assert engine.spoken == []
    finally:
        runtime.stop()


def test_barge_in_drops_streaming_sentences_after_stop_on_threaded_bus():
    """Regression for realtime-concurrency-1.

    On the streaming-TTS path a stale sentence from an interrupted turn must
    never be spoken after barge-in. Run the bus on its own thread (production
    shape) so cancellation crosses the audio-thread/bus-thread boundary the way
    it does live, and assert no ``engine.speak()`` happens after the barge-in's
    ``stop_speaking()``."""
    llm = _GatedStreamLLM(
        ["First sentence.", "Second sentence.", "Third sentence."]
    )
    engine = _RecordingEngine(hold_speech=True)
    runtime = VoiceRuntime(engine, llm, start_mode=Mode.ASSISTANT, stream_tts=True)
    runtime.start(run_bus=True)
    try:
        engine.final("tell me a story")
        # The first sentence is being spoken (and held mid-utterance).
        assert _wait_until(lambda: engine.spoken == ["First sentence."])
        assert engine.is_speaking

        # Barge-in: cancellation is set before stop_speaking() returns, and the
        # gated stream is then released -- so the later sentences must not reach
        # the engine.
        engine.barge_in()
        assert _wait_until(lambda: not engine.is_speaking)
        llm.gate.set()

        # Give the (now-cancelled) stream and the bus ample time to (try to)
        # push the later sentences through; none must be spoken.
        assert _wait_until(
            lambda: not runtime.supervisor.state.active_tasks, timeout=2.0
        )
        time.sleep(0.05)

        assert engine.spoken == ["First sentence."]

        # Hard invariant: no speak() lands after the barge-in's stop_speaking().
        calls = engine.call_log()
        stop_indices = [i for i, (kind, _) in enumerate(calls) if kind == "stop_speaking"]
        assert stop_indices, "expected a stop_speaking on barge-in"
        first_stop = stop_indices[0]
        speak_after_stop = [
            text for kind, text in calls[first_stop + 1 :] if kind == "speak"
        ]
        assert speak_after_stop == [], f"stale sentence spoken after stop: {speak_after_stop}"
    finally:
        runtime.stop()


# --- ADD-ON / continuation end-to-end (no two competing answers) ----------------


class _PreGatedStreamLLM:
    """Streaming fake that blocks BEFORE emitting anything, then echoes the
    prompt as its single sentence. Lets a test land a follow-up while the first
    turn is mid-generation but has spoken NOTHING yet (started_speaking False) --
    the before-audio window in which a continuation merges into one turn."""

    def __init__(self):
        self.stream_started = threading.Event()
        self.gate = threading.Event()
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, system=None, images=None) -> str:  # pragma: no cover
        return prompt

    def stream(self, prompt: str, *, system=None, images=None) -> Iterator[str]:
        self.prompts.append(prompt)
        self.stream_started.set()
        # Block before the first token so the turn stays "not yet speaking".
        self.gate.wait(timeout=2.0)
        yield prompt  # echo so the test sees which turn produced the answer


def test_addon_before_audio_merges_into_single_answer():
    """The headline ADD-ON fix: a follow-up that lands before the assistant has
    spoken merges into ONE answer instead of racing a second competing turn."""
    from core.metrics import MERGED, SUPERSEDED

    llm = _PreGatedStreamLLM()
    engine = ScriptedEngine(hold_speech=False)
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        stream_tts=True,
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=True)
    try:
        engine.final("whats the weather")
        # First turn is generating but blocked before any sentence is emitted.
        assert _wait_until(lambda: llm.stream_started.is_set())
        assert _wait_until(lambda: bool(runtime.supervisor.state.active_tasks))

        # Follow-up arrives while nothing has been spoken -> it must MERGE.
        engine.final("and also the forecast")
        # The merged turn reaches the LLM (2nd stream call) -> release the gate.
        assert _wait_until(lambda: len(llm.prompts) >= 2)
        llm.gate.set()

        assert runtime.wait_idle()
        # A SINGLE merged prompt reached the model (the original folded together
        # with the add-on) -- the essence of the fix, not two competing prompts.
        assert any(
            "whats the weather" in p and "and also the forecast" in p
            for p in llm.prompts
        ), llm.prompts
        # Everything spoken comes from that one merged turn (the echoed merge
        # prompt, recognizable by its "first asked" framing): both the original
        # and the add-on are voiced together. The first turn was cancelled before
        # it spoke, so there is no second racing answer.
        spoken_joined = " ".join(s.strip() for s in engine.spoken)
        assert "whats the weather" in spoken_joined
        assert "and also the forecast" in spoken_joined
        assert "first asked" in spoken_joined, engine.spoken
        records = runtime.metrics.records()
        assert MERGED in records[0].stamps
        assert SUPERSEDED in records[0].stamps
    finally:
        runtime.stop()


class _PromptRecordingBlockingLLM:
    """Non-streaming fake that blocks inside generate() and records prompts.
    ack_then_think only fires on the non-streaming path (streaming turns get
    STREAM_* policies instead), so the ack/merge regression needs generate()."""

    def __init__(self):
        self.gate = threading.Event()
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, system=None, images=None) -> str:
        self.prompts.append(prompt)
        self.gate.wait(timeout=2.0)
        return prompt  # echo so the test sees which turn produced the answer

    def stream(self, prompt: str, *, system=None, images=None) -> Iterator[str]:
        yield self.generate(prompt, system=system)


def test_addon_after_latency_ack_still_merges():
    """Regression: the ack_then_think acknowledgement is filler, not answer
    audio. An add-on that lands AFTER the ack but BEFORE the answer must still
    take the continuation MERGE branch (one combined answer), not queue behind
    a reply the user has not heard. The ack used to flip started_speaking and
    silently disable the merge."""
    from core.metrics import MERGED, SUPERSEDED

    llm = _PromptRecordingBlockingLLM()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        continuation_config=ContinuationConfig(enabled=True),
    )
    runtime.start(run_bus=True)
    try:
        # Slow-thinker phrasing -> classified ack_then_think on the
        # non-streaming path, so the ack plays while generate() is blocked.
        engine.final("compare the latest local speech to text engines")
        assert _wait_until(lambda: "I'll check that now." in engine.spoken)
        assert _wait_until(lambda: len(llm.prompts) >= 1)
        active = list(runtime.supervisor.state.active_tasks.values())
        assert active and active[0].ack_spoken
        assert not active[0].started_speaking  # the ack is not answer audio

        # Add-on after the ack, before any answer audio -> must MERGE.
        engine.final("and also their licenses")
        assert _wait_until(lambda: len(llm.prompts) >= 2)
        llm.gate.set()

        assert runtime.wait_idle()
        assert any(
            "compare the latest local speech to text engines" in p
            and "and also their licenses" in p
            for p in llm.prompts
        ), llm.prompts
        # The lineage is acknowledged exactly once -- the merged turn must not
        # re-speak the same filler after the add-on.
        assert engine.spoken.count("I'll check that now.") == 1, engine.spoken
        records = runtime.metrics.records()
        assert MERGED in records[0].stamps
        assert SUPERSEDED in records[0].stamps
    finally:
        runtime.stop()


def test_addon_passes_input_gate_while_turn_in_flight():
    """A short add-on the addressing gate would INGEST as ambient must still reach
    the brain (and merge) while a turn is in flight -- otherwise the default-ON
    input gate silently swallows continuations."""
    from core.addressing import ACT, INGEST, ScriptedAddressingClassifier

    llm = _PreGatedStreamLLM()
    engine = ScriptedEngine(hold_speech=False)
    addressing = ScriptedAddressingClassifier(
        {"whats the weather": ACT, "make it brief": INGEST}, default=ACT
    )
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        stream_tts=True,
        continuation_config=ContinuationConfig(enabled=True),
        addressing=addressing,
    )
    runtime.start(run_bus=True)
    try:
        engine.final("whats the weather")
        assert _wait_until(lambda: llm.stream_started.is_set())
        assert _wait_until(lambda: bool(runtime.supervisor.state.active_tasks))

        # The gate classifies this INGEST, but a turn is in flight and "make it"
        # is a continuation cue -> the override lets it through and it merges.
        engine.final("make it brief")
        assert _wait_until(lambda: len(llm.prompts) >= 2), llm.prompts
        llm.gate.set()

        assert runtime.wait_idle()
        spoken_joined = " ".join(s.strip() for s in engine.spoken)
        assert "whats the weather" in spoken_joined
        assert "make it brief" in spoken_joined
    finally:
        runtime.stop()
