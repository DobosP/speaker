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
from always_on_agent.events import Mode

from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
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
    assert len(engine.spoken) == 1
    assert engine.spoken[0]  # non-empty synthesized answer


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
