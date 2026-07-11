from __future__ import annotations

import threading

import pytest

from always_on_agent.events import AgentEvent, EventKind
from core.engine import (
    AudioEngine,
    EngineCallbacks,
    PlaybackOutcome,
    PlaybackReceipt,
    TrackedSpeech,
)
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.metrics import TTS_FIRST_AUDIO
from core.runtime import VoiceRuntime


class _LegacyEngine(AudioEngine):
    def start(self, callbacks: EngineCallbacks) -> None:
        pass

    def stop(self) -> None:
        pass

    def speak(self, text: str, on_done=None) -> None:
        if on_done is not None:
            on_done()

    def stop_speaking(self) -> None:
        pass


def test_tracked_speech_is_opt_in_for_legacy_engines():
    engine = _LegacyEngine()

    assert engine.playback_capabilities.tracked_terminal is False
    with pytest.raises(NotImplementedError):
        engine.speak_tracked(
            TrackedSpeech("fragment-1", "hello"),
            on_terminal=lambda _receipt: None,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"played_samples": -1},
        {"total_samples": -1},
        {"played_samples": 2, "total_samples": 1},
        {"output_sample_rate": 0},
    ],
)
def test_playback_receipt_rejects_invalid_sample_accounting(kwargs):
    with pytest.raises(ValueError):
        PlaybackReceipt(
            fragment_id="fragment-1",
            outcome=PlaybackOutcome.INTERRUPTED,
            **kwargs,
        )


def test_scripted_tracked_speech_completes_with_attested_text():
    engine = ScriptedEngine()
    lifecycle: list[object] = []
    engine.start(
        EngineCallbacks(
            on_speech_start=lambda: lifecycle.append("speech-start"),
            on_speech_end=lambda: lifecycle.append("speech-end"),
            on_metric=lambda name: lifecycle.append(name),
        )
    )

    engine.speak_tracked(
        TrackedSpeech("fragment-1", "Hello there."),
        on_started=lambda fragment_id: lifecycle.append(("started", fragment_id)),
        on_terminal=lambda receipt: lifecycle.append(receipt),
    )

    receipt = lifecycle[-1]
    assert receipt.fragment_id == "fragment-1"
    assert receipt.outcome is PlaybackOutcome.COMPLETED
    assert receipt.completed is True and receipt.interrupted is False
    assert receipt.safe_text_prefix == "Hello there."
    assert receipt.played_samples is None
    assert receipt.total_samples is None
    assert lifecycle[:-1] == [
        "speech-start",
        TTS_FIRST_AUDIO,
        ("started", "fragment-1"),
        "speech-end",
    ]
    assert engine.spoken == ["Hello there."]
    assert engine.is_speaking is False


def test_scripted_interruption_has_one_terminal_and_no_guessed_prefix():
    engine = ScriptedEngine(hold_speech=True)
    receipts = []

    engine.speak_tracked(
        TrackedSpeech("fragment-1", "Words that may not all play."),
        on_terminal=receipts.append,
    )
    assert engine.is_speaking is True

    engine.stop_speaking()
    engine.stop_speaking()
    engine.finish_speaking()

    assert len(receipts) == 1
    assert receipts[0].outcome is PlaybackOutcome.INTERRUPTED
    assert receipts[0].completed is False and receipts[0].interrupted is True
    assert receipts[0].safe_text_prefix == ""
    assert engine.is_speaking is False


def test_scripted_finish_wins_stop_race_exactly_once():
    engine = ScriptedEngine(hold_speech=True)
    receipts = []
    engine.speak_tracked(
        TrackedSpeech("fragment-1", "Complete me."),
        on_terminal=receipts.append,
    )

    gate = threading.Barrier(3)

    def finish() -> None:
        gate.wait()
        engine.finish_speaking()

    def stop() -> None:
        gate.wait()
        engine.stop_speaking()

    finish_thread = threading.Thread(target=finish)
    stop_thread = threading.Thread(target=stop)
    finish_thread.start()
    stop_thread.start()
    gate.wait()
    finish_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    assert not finish_thread.is_alive() and not stop_thread.is_alive()
    assert len(receipts) == 1
    assert receipts[0].outcome in {
        PlaybackOutcome.COMPLETED,
        PlaybackOutcome.INTERRUPTED,
    }
    expected = "Complete me." if receipts[0].completed else ""
    assert receipts[0].safe_text_prefix == expected


def test_scripted_replacement_terminals_the_older_held_fragment():
    engine = ScriptedEngine(hold_speech=True)
    receipts = []
    engine.speak_tracked(
        TrackedSpeech("old", "Old fragment."),
        on_terminal=receipts.append,
    )
    engine.speak_tracked(
        TrackedSpeech("new", "New fragment."),
        on_terminal=receipts.append,
    )
    engine.finish_speaking()

    assert [(receipt.fragment_id, receipt.outcome) for receipt in receipts] == [
        ("old", PlaybackOutcome.INTERRUPTED),
        ("new", PlaybackOutcome.COMPLETED),
    ]


def test_scripted_start_callback_precedes_concurrent_terminal():
    engine = ScriptedEngine(hold_speech=True)
    start_entered = threading.Event()
    release_start = threading.Event()
    order = []

    def on_started(fragment_id: str) -> None:
        order.append(("started", fragment_id))
        start_entered.set()
        assert release_start.wait(1.0)

    speak_thread = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("fragment-1", "Hold start."),
            on_started=on_started,
            on_terminal=lambda receipt: order.append(("terminal", receipt.outcome)),
        )
    )
    speak_thread.start()
    assert start_entered.wait(1.0)

    stop_thread = threading.Thread(target=engine.stop_speaking)
    stop_thread.start()
    assert stop_thread.is_alive()  # serialized behind the in-flight start callback
    release_start.set()
    speak_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    assert order == [
        ("started", "fragment-1"),
        ("terminal", PlaybackOutcome.INTERRUPTED),
    ]


def test_concurrent_replacement_cannot_complete_new_fragment_before_its_start():
    engine = ScriptedEngine()
    first_started = threading.Event()
    release_first = threading.Event()
    lifecycle = []

    def on_first_started(fragment_id: str) -> None:
        lifecycle.append(("started", fragment_id))
        first_started.set()
        assert release_first.wait(1.0)

    first = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("first", "First."),
            on_started=on_first_started,
            on_terminal=lambda receipt: lifecycle.append(
                ("terminal", receipt.fragment_id, receipt.outcome)
            ),
        )
    )
    first.start()
    assert first_started.wait(1.0)

    second = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("second", "Second."),
            on_started=lambda fragment_id: lifecycle.append(
                ("started", fragment_id)
            ),
            on_terminal=lambda receipt: lifecycle.append(
                ("terminal", receipt.fragment_id, receipt.outcome)
            ),
        )
    )
    second.start()
    release_first.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive() and not second.is_alive()
    assert lifecycle == [
        ("started", "first"),
        ("terminal", "first", PlaybackOutcome.INTERRUPTED),
        ("started", "second"),
        ("terminal", "second", PlaybackOutcome.COMPLETED),
    ]


def test_switching_to_legacy_speak_terminals_a_held_tracked_fragment():
    engine = ScriptedEngine(hold_speech=True)
    receipts = []
    engine.speak_tracked(
        TrackedSpeech("tracked", "Tracked fragment."),
        on_terminal=receipts.append,
    )

    engine.speak("legacy")

    assert len(receipts) == 1
    assert receipts[0].outcome is PlaybackOutcome.INTERRUPTED


def test_scripted_legacy_on_done_behavior_is_unchanged():
    completed = []
    immediate = ScriptedEngine()
    immediate.speak("done", on_done=lambda: completed.append("done"))
    assert completed == ["done"]

    interrupted = []
    held = ScriptedEngine(hold_speech=True)
    held.speak("cut", on_done=lambda: interrupted.append("unexpected"))
    held.stop_speaking()
    held.finish_speaking()
    assert interrupted == []


def test_speak_only_scripted_subclass_keeps_legacy_runtime_override():
    class _SpeakOnlySubclass(ScriptedEngine):
        def __init__(self):
            super().__init__()
            self.override_calls = []

        def speak(self, text, on_done=None):
            self.override_calls.append(text)
            super().speak(text, on_done)

    engine = _SpeakOnlySubclass()
    runtime = VoiceRuntime(engine, EchoLLM())
    runtime.start(run_bus=False)
    try:
        assert engine.playback_capabilities.tracked_terminal is False
        runtime._on_event(
            AgentEvent(EventKind.TTS_REQUEST, {"text": "legacy override"})
        )
        assert engine.override_calls == ["legacy override"]
    finally:
        runtime.stop()
