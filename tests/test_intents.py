"""Speech-to-intent fast-path: grammar, slot parsing, handler, runtime wiring.

No audio/models. Deterministic via an injected clock and a fake scheduler.
"""
from __future__ import annotations

from datetime import datetime

from core.engines.scripted import ScriptedEngine
from core.intents import (
    IntentGrammar,
    LocalIntentHandler,
    human_duration,
    parse_duration,
)
from core.llm import EchoLLM
from core.runtime import VoiceRuntime


# --- grammar -----------------------------------------------------------------


def test_grammar_matches_time_date_timer():
    g = IntentGrammar.default()
    assert g.match("what time is it").name == "time"
    assert g.match("what's the time").name == "time"
    assert g.match("what's the date").name == "date"
    assert g.match("what day is it").name == "date"
    timer = g.match("set a timer for 5 minutes")
    assert timer is not None and timer.name == "timer"
    assert timer.slots == {"value": "5", "unit": "minutes"}


def test_grammar_timer_word_number_and_alarm_alias():
    g = IntentGrammar.default()
    m = g.match("set an alarm for ten seconds")
    assert m is not None and m.name == "timer"
    assert m.slots == {"value": "ten", "unit": "seconds"}


def test_grammar_misses_are_none():
    g = IntentGrammar.default()
    assert g.match("please summarize the news") is None
    assert g.match("set a timer") is None  # no duration -> not a full match


def test_grammar_timer_does_not_trigger_time():
    # "timer" contains "time" but the \b guard must keep them distinct.
    assert IntentGrammar.default().match("set a timer for 2 minutes").name == "timer"


# --- duration helpers --------------------------------------------------------


def test_parse_duration():
    assert parse_duration("5", "minutes") == 300
    assert parse_duration("ten", "seconds") == 10
    assert parse_duration("2", "hours") == 7200
    assert parse_duration("mins", "mins") is None  # non-numeric value
    assert parse_duration("5", "fortnights") is None


def test_human_duration():
    assert human_duration(300) == "5 minutes"
    assert human_duration(60) == "1 minute"
    assert human_duration(1) == "1 second"
    assert human_duration(3600) == "1 hour"


# --- handler -----------------------------------------------------------------


def _fixed_clock(hour=12, minute=0):
    return lambda: datetime(2026, 5, 27, hour, minute)


def test_handler_speaks_time():
    spoken: list[str] = []
    h = LocalIntentHandler(spoken.append, clock=_fixed_clock(13, 5))
    assert h.handle("what time is it") is True
    assert spoken == ["It's 1:05 PM."]


def test_handler_speaks_date():
    spoken: list[str] = []
    h = LocalIntentHandler(spoken.append, clock=_fixed_clock())
    assert h.handle("what's the date") is True
    assert spoken == ["Today is Wednesday, May 27."]


def test_handler_timer_schedules_confirms_and_fires():
    spoken: list[str] = []
    scheduled: list[tuple[float, object]] = []
    cancels: list[bool] = []

    def sched(delay, fn):
        scheduled.append((delay, fn))
        return lambda: cancels.append(True)

    h = LocalIntentHandler(spoken.append, scheduler=sched)
    assert h.handle("set a timer for 5 minutes") is True
    assert spoken == ["Okay, timer set for 5 minutes."]
    assert scheduled[0][0] == 300.0
    scheduled[0][1]()  # the scheduled callback fires
    assert spoken[-1] == "Time's up. Your 5 minutes timer is done."
    h.cancel_all()
    assert cancels == [True]


def test_handler_custom_phrase():
    spoken: list[str] = []
    h = LocalIntentHandler(spoken.append, phrases={"lights on": "Turning the lights on."})
    assert h.handle("Lights on") is True  # case-insensitive
    assert spoken == ["Turning the lights on."]


def test_handler_miss_and_disabled_return_false():
    spoken: list[str] = []
    h = LocalIntentHandler(spoken.append)
    assert h.handle("tell me a joke") is False
    assert spoken == []
    off = LocalIntentHandler(spoken.append, enabled=False)
    assert off.handle("what time is it") is False
    assert spoken == []


# --- runtime integration -----------------------------------------------------


def test_runtime_intent_fast_path_bypasses_llm():
    engine = ScriptedEngine()
    handler = LocalIntentHandler(engine.speak, clock=_fixed_clock(12, 0))
    runtime = VoiceRuntime(engine, EchoLLM(reply="LLM ANSWER"), intents=handler)
    runtime.start(run_bus=False)
    engine.final("what time is it")
    assert runtime.wait_idle()
    assert engine.spoken == ["It's 12:00 PM."]  # not the LLM reply


def test_runtime_intent_miss_falls_through_to_llm():
    engine = ScriptedEngine()
    handler = LocalIntentHandler(engine.speak)
    runtime = VoiceRuntime(engine, EchoLLM(reply="LLM ANSWER"), intents=handler)
    runtime.start(run_bus=False)
    engine.final("please summarize the news")
    assert runtime.wait_idle()
    assert engine.spoken == ["LLM ANSWER"]


def test_runtime_without_intents_is_unchanged():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM(reply="hi"))  # intents=None
    runtime.start(run_bus=False)
    engine.final("what time is it")
    assert runtime.wait_idle()
    assert engine.spoken == ["hi"]  # no fast-path -> straight to the LLM
