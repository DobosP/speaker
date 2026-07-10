"""Hold-and-merge final dispatch (core/turn_merge.py).

Pins the fix for the "answers my incomplete questions" live failure
(run-20260609-234435): the endpoint committed fragment finals -- 'A running',
'Dear me', 'The fisherman', 'A long story about' -- and the brain answered each
one. The coalescer decisions are pure; the dispatcher tests use a tiny hold so
they stay fast and deterministic-enough (generous assert windows, no exact
timing pins).
"""
from __future__ import annotations

import time

import pytest

from core.turn_merge import (
    DEFAULT_EXEMPT_PHRASES,
    DEFAULT_HOLD_ENDINGS,
    FinalCoalescer,
    FinalDispatcher,
    TurnMergeConfig,
)


# --- FinalCoalescer (pure decisions) ------------------------------------------


def test_holds_the_recorded_fragment_finals():
    c = FinalCoalescer()
    # The four answered fragments from run-20260609-234435.
    assert c.should_hold("A long story about")  # ends on a preposition
    assert c.should_hold("A running")           # 2-word fragment
    assert c.should_hold("Dear me")
    assert c.should_hold("The fisherman")
    # And the 003800 fragment.
    assert c.should_hold("So they")


def test_complete_finals_are_not_held():
    c = FinalCoalescer()
    assert not c.should_hold("Can you hear me")
    assert not c.should_hold("Tell me a story about a lighthouse")
    assert not c.should_hold("What is the capital of France")


def test_control_words_are_exempt_even_though_short():
    c = FinalCoalescer()
    for phrase in ("yes", "No", "stop", "cancel that", "Okay", "never mind",
                   "Start again", "thank you", "da", "opreste"):
        assert not c.should_hold(phrase), f"{phrase!r} must dispatch immediately"


def test_incomplete_endings_hold_and_merge_reads_naturally():
    c = FinalCoalescer()
    assert c.should_hold("I want to")
    assert c.should_hold("tell me about the history of")
    merged = c.merge("A long story about", "And the lighthouse")
    assert merged == "A long story about And the lighthouse"


def test_config_from_dict_defaults_and_overrides():
    dflt = TurnMergeConfig.from_dict(None)
    assert dflt.enabled is False  # dataclass off; config.json opts in
    assert dflt.hold_sec == pytest.approx(1.2)
    assert dflt.hold_endings == DEFAULT_HOLD_ENDINGS
    assert dflt.exempt_phrases == DEFAULT_EXEMPT_PHRASES

    cfg = TurnMergeConfig.from_dict(
        {"enabled": True, "hold_sec": 0.5, "max_fragment_words": 1,
         "exempt_phrases": ["go"]}
    )
    assert cfg.enabled and cfg.hold_sec == 0.5
    coal = FinalCoalescer(cfg)
    assert not coal.should_hold("go")          # custom exempt
    assert not coal.should_hold("two words")   # max_fragment_words=1
    assert coal.should_hold("hey")             # 1 word, not exempt


# --- FinalDispatcher (threaded hold/merge) -------------------------------------


def _dispatcher(received, **overrides):
    cfg = TurnMergeConfig(
        enabled=True,
        hold_sec=overrides.pop("hold_sec", 0.15),
        max_hold_sec=overrides.pop("max_hold_sec", 1.0),
        **overrides,
    )
    d = FinalDispatcher(received.append, cfg)
    d.start()
    return d


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_complete_final_dispatches_immediately():
    received: list[str] = []
    d = _dispatcher(received)
    try:
        d.submit("Can you hear me")
        assert _wait_for(lambda: received == ["Can you hear me"], timeout=1.0)
        assert d.held_count == 0
    finally:
        d.stop()


def test_incomplete_final_merges_with_the_continuation():
    """The live failure shape: 'A long story about' <pause> 'And the lighthouse'
    must dispatch ONCE, merged -- not as two answered turns."""
    received: list[str] = []
    d = _dispatcher(received)
    try:
        d.submit("A long story about")
        time.sleep(0.05)  # inside the hold window
        assert received == []  # still held
        d.submit("And the lighthouse")
        assert _wait_for(lambda: len(received) == 1)
        assert received == ["A long story about And the lighthouse"]
        assert d.merged_count == 1
    finally:
        d.stop()


def test_held_final_dispatches_alone_when_the_user_is_done():
    received: list[str] = []
    d = _dispatcher(received)
    try:
        d.submit("The fisherman")
        assert _wait_for(lambda: received == ["The fisherman"])
    finally:
        d.stop()


def test_partial_extends_the_hold_until_the_next_final():
    received: list[str] = []
    d = _dispatcher(received)
    try:
        d.submit("A long story about")
        # The user keeps talking: partials arrive past the base hold window.
        for _ in range(6):
            time.sleep(0.05)
            d.note_partial()
        assert received == []  # the hold followed the partials
        d.submit("a lighthouse keeper")
        assert _wait_for(lambda: len(received) == 1)
        assert received == ["A long story about a lighthouse keeper"]
    finally:
        d.stop()


def test_max_hold_caps_a_partial_storm():
    received: list[str] = []
    d = _dispatcher(received, max_hold_sec=0.3)
    try:
        d.submit("A long story about")
        deadline = time.time() + 1.5
        while time.time() < deadline and not received:
            d.note_partial()  # endless partials must NOT hold forever
            time.sleep(0.02)
        assert received == ["A long story about"], (
            "the hold must dispatch at max_hold_sec even under a partial storm"
        )
    finally:
        d.stop()


def test_stop_flushes_a_held_final_instead_of_dropping_it():
    received: list[str] = []
    d = _dispatcher(received, hold_sec=5.0)  # long hold; stop() must not wait it out
    d.submit("A long story about")
    t0 = time.time()
    d.stop()
    assert time.time() - t0 < 3.0
    assert received == ["A long story about"]


def test_has_pending_reflects_the_hold():
    received: list[str] = []
    d = _dispatcher(received, hold_sec=0.3)
    try:
        assert not d.has_pending
        d.submit("The fisherman")
        assert d.has_pending
        assert _wait_for(lambda: not d.has_pending)
    finally:
        d.stop()


def test_hold_callback_fires_when_fragment_is_held():
    received: list[str] = []
    held = 0

    def on_hold() -> None:
        nonlocal held
        held += 1

    cfg = TurnMergeConfig(enabled=True, hold_sec=0.15, max_hold_sec=1.0)
    d = FinalDispatcher(received.append, cfg, on_hold=on_hold)
    d.start()
    try:
        d.submit("A long story about")
        assert _wait_for(lambda: held == 1, timeout=1.0)
        assert received == []
    finally:
        d.stop()


# --- VoiceRuntime integration ---------------------------------------------------


def test_runtime_merges_fragments_into_one_answered_turn():
    """End-to-end through VoiceRuntime: two fragment finals become ONE brain
    turn (one assistant reply), and the reply answers the merged text."""
    from core.engines.scripted import ScriptedEngine
    from core.llm import EchoLLM
    from core.runtime import VoiceRuntime

    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        turn_merge_config=TurnMergeConfig(enabled=True, hold_sec=0.15),
    )
    runtime.start(run_bus=False)
    try:
        runtime._on_final("A long story about")
        runtime._on_final("and the lighthouse")
        assert runtime.wait_idle(timeout=5.0)
        spoken = " ".join(engine.spoken)
        assert "A long story about and the lighthouse" in spoken
        # ONE turn, not two: the fragment alone was never answered.
        assert sum("A long story about" in s for s in engine.spoken) == 1
    finally:
        runtime.stop()


def test_runtime_disabled_turn_merge_is_byte_identical():
    from core.engines.scripted import ScriptedEngine
    from core.llm import EchoLLM
    from core.runtime import VoiceRuntime

    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM())  # no turn_merge_config
    runtime.start(run_bus=False)
    try:
        assert runtime._dispatcher is None
        runtime._on_final("A long story about")
        assert runtime.wait_idle(timeout=5.0)
        assert any("A long story about" in s for s in engine.spoken)
    finally:
        runtime.stop()
