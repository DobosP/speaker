"""End-to-end tests for the lean core runtime.

These exercise the full path engine -> brain -> capability -> TTS using the
scriptable engine and a deterministic fake LLM, so they need no audio hardware,
no sherpa-onnx, and no Ollama.
"""

from __future__ import annotations

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
