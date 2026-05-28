"""Tests for the ASR transcript cleanup pass (core.cleanup) and its
integration with VoiceRuntime.

Two layers:

* unit tests for the trailing-repeat heuristic, the cleanup prompt builder,
  and the LLM reply sanitizer;
* end-to-end tests that drive a final into a real :class:`VoiceRuntime`
  with a :class:`ScriptedTranscriptCleaner` and assert (a) the brain saw
  the cleaned text, not the raw, and (b) the transcript carries BOTH
  entries (raw first, cleaned second with the raw retained in ``raw``).
"""
from __future__ import annotations

import logging
from typing import Iterator, Optional, Sequence

from always_on_agent.events import Mode

from core.addressing import ACT, ScriptedAddressingClassifier
from core.cleanup import (
    LLMTranscriptCleaner,
    ScriptedTranscriptCleaner,
    _sanitize_reply,
    detect_signals,
    has_trailing_repeat,
)
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runtime import VoiceRuntime


# --- Heuristic unit tests ----------------------------------------------------

def test_trailing_repeat_detects_single_word_repeat():
    assert has_trailing_repeat("tell me about Paris Paris")
    assert has_trailing_repeat("go go")
    assert has_trailing_repeat("stop. stop.")  # punctuation between is fine


def test_trailing_repeat_detects_two_word_phrase_repeat():
    assert has_trailing_repeat("the model the model")
    assert has_trailing_repeat("we should go we should go")


def test_trailing_repeat_is_case_insensitive():
    assert has_trailing_repeat("Paris paris")
    assert has_trailing_repeat("STOP stop")


def test_trailing_repeat_rejects_clean_text():
    assert not has_trailing_repeat("hello world")
    assert not has_trailing_repeat("tell me about Paris")
    assert not has_trailing_repeat("the algorithm is fast")
    assert not has_trailing_repeat("")
    assert not has_trailing_repeat("solo")


def test_detect_signals_lists_repeat_and_editing_terms():
    sigs = detect_signals("go to Paris no wait London London")
    text = " ".join(sigs)
    assert "word-repeat-at-end" in text
    assert "no wait" in text


def test_detect_signals_empty_on_clean_text():
    assert detect_signals("what is the weather today") == []


def test_sanitize_reply_strips_quotes_and_labels():
    assert _sanitize_reply('"the model"', fallback="raw") == "the model"
    assert _sanitize_reply("Cleaned: the model", fallback="raw") == "the model"
    assert _sanitize_reply("output: 'hi there'", fallback="raw") == "hi there"


def test_sanitize_reply_returns_fallback_on_empty():
    assert _sanitize_reply("", fallback="raw") == "raw"
    assert _sanitize_reply("   ", fallback="raw") == "raw"


# --- LLMTranscriptCleaner unit tests -----------------------------------------

class _StubLLM:
    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, Optional[str]]] = []

    def generate(self, prompt: str, *, system: Optional[str] = None, images=None) -> str:
        self.calls.append((prompt, system))
        if not self._replies:
            raise AssertionError("StubLLM ran out of scripted replies")
        return self._replies.pop(0)

    def stream(self, prompt: str, *, system=None, images=None) -> Iterator[str]:
        yield self.generate(prompt, system=system, images=images)


class _RaisingLLM:
    def generate(self, prompt, *, system=None, images=None):
        raise RuntimeError("llm down")

    def stream(self, prompt, *, system=None, images=None):
        raise RuntimeError("llm down")


def test_cleaner_returns_llm_output_for_dirty_input():
    llm = _StubLLM(["tell me about Paris"])
    cleaner = LLMTranscriptCleaner(llm)
    assert cleaner.clean("tell me about Paris Paris") == "tell me about Paris"


def test_cleaner_passes_raw_through_when_llm_raises():
    cleaner = LLMTranscriptCleaner(_RaisingLLM())
    assert cleaner.clean("anything") == "anything"


def test_cleaner_flags_word_repeat_signal_in_prompt():
    llm = _StubLLM(["tell me about Paris"])
    cleaner = LLMTranscriptCleaner(llm)
    cleaner.clean("tell me about Paris Paris")
    prompt, _system = llm.calls[0]
    assert "word-repeat-at-end" in prompt
    assert "tell me about Paris Paris" in prompt


def test_cleaner_flags_editing_term_in_prompt():
    llm = _StubLLM(["go to London"])
    cleaner = LLMTranscriptCleaner(llm)
    cleaner.clean("go to Paris I mean London")
    prompt, _ = llm.calls[0]
    assert "editing term" in prompt.lower()
    assert "i mean" in prompt.lower()


def test_cleaner_omits_signal_block_when_text_is_clean():
    llm = _StubLLM(["what is the weather"])
    cleaner = LLMTranscriptCleaner(llm)
    cleaner.clean("what is the weather")
    prompt, _ = llm.calls[0]
    assert "Detected signals" not in prompt


def test_cleaner_includes_recent_context():
    llm = _StubLLM(["yes"])
    cleaner = LLMTranscriptCleaner(llm, max_context=2)
    cleaner.clean("yes yes", recent=["are you there", "did you hear me"])
    prompt, _ = llm.calls[0]
    assert "are you there" in prompt
    assert "did you hear me" in prompt


def test_cleaner_returns_input_unchanged_on_empty_text():
    cleaner = LLMTranscriptCleaner(_StubLLM([]))  # never called
    assert cleaner.clean("") == ""
    assert cleaner.clean("   ") == "   "


def test_cleaner_falls_back_to_raw_on_empty_llm_reply():
    cleaner = LLMTranscriptCleaner(_StubLLM([""]))
    assert cleaner.clean("hello there") == "hello there"


# --- VoiceRuntime integration tests ------------------------------------------

def _runtime_with_cleaner(
    cleanups: Optional[dict[str, str]] = None,
    *,
    addressing_decisions: Optional[dict[str, str]] = None,
    reply: str = "ok",
):
    """Build a runtime with a scripted cleaner (and optionally a scripted
    addressing gate). Returns the runtime + engine + the cleaner spy."""
    engine = ScriptedEngine()
    cleaner = ScriptedTranscriptCleaner(cleanups or {})
    addressing = (
        ScriptedAddressingClassifier(addressing_decisions, default=ACT)
        if addressing_decisions is not None
        else None
    )
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply=reply),
        start_mode=Mode.ASSISTANT,
        addressing=addressing,
        cleaner=cleaner,
    )
    runtime.start(run_bus=False)
    return runtime, engine, cleaner


def test_cleaned_text_reaches_the_brain_not_the_raw(caplog):
    """The brain (and its LLM-backed reply) should see the cleaned text, not
    the raw one with the trailing repeat."""
    runtime, engine, cleaner = _runtime_with_cleaner(
        {"tell me about Paris Paris": "tell me about Paris"},
    )
    with caplog.at_level(logging.INFO, logger="speaker.runtime"):
        engine.final("tell me about Paris Paris")
        assert runtime.wait_idle()
    # The brain ran (EchoLLM with reply="ok" produced the speech).
    assert engine.spoken == ["ok"]
    # The cleaner was called with the raw text.
    assert cleaner.calls and cleaner.calls[0][0] == "tell me about Paris Paris"
    # The "cleaned" log line is present, showing both versions.
    cleaned_lines = [r for r in caplog.records if r.message.startswith("cleaned: ")]
    assert cleaned_lines and "Paris Paris" in cleaned_lines[0].getMessage()
    assert "tell me about Paris" in cleaned_lines[0].getMessage()


def test_clean_text_passes_through_untouched():
    """When the cleaner returns the input unchanged (no rewrite), no second
    transcript entry is emitted and the brain just sees the original."""
    runtime, engine, cleaner = _runtime_with_cleaner({})  # cleaner is a no-op
    engine.final("what is the weather")
    assert runtime.wait_idle()
    assert engine.spoken == ["ok"]
    # Cleaner was called once with the raw text but returned it unchanged.
    assert cleaner.calls and cleaner.calls[0][0] == "what is the weather"


def test_cleaner_runs_only_after_addressing_acts():
    """Ingested utterances must not pay the cleanup cost. Verifies that the
    cleaner is bypassed entirely when the addressing gate says INGEST."""
    runtime, engine, cleaner = _runtime_with_cleaner(
        cleanups={},
        addressing_decisions={"background noise here": "INGEST"},
    )
    engine.final("background noise here")
    assert runtime.wait_idle()
    assert engine.spoken == []  # ingested; the brain never replied
    assert cleaner.calls == []  # and the cleaner was skipped


def test_transcript_carries_raw_and_cleaned_entries():
    """The summary's transcript must show both the raw final AND the
    cleaned rewrite, with the raw retained on the cleaned entry."""
    runtime, engine, _ = _runtime_with_cleaner(
        {"go go": "go"},
    )
    # Spy on the log records that carry a 'transcript' extra (the same path
    # the run-log summary harvests).
    captured: list[dict] = []

    class _TranscriptSpy(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            tr = getattr(record, "transcript", None)
            if isinstance(tr, dict):
                captured.append(tr)

    spy = _TranscriptSpy()
    logging.getLogger("speaker.runtime").addHandler(spy)
    try:
        engine.final("go go")
        assert runtime.wait_idle()
    finally:
        logging.getLogger("speaker.runtime").removeHandler(spy)

    user_entries = [t for t in captured if t.get("role") == "user"]
    # Two entries: the raw final, then the cleaned rewrite.
    assert len(user_entries) == 2
    assert user_entries[0] == {"role": "user", "text": "go go", "mode": "assistant"}
    assert user_entries[1]["text"] == "go"
    assert user_entries[1]["raw"] == "go go"


def test_no_cleaner_preserves_legacy_behavior():
    """When no cleaner is wired in, _on_final is byte-identical to before."""
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM(reply="legacy"))
    runtime.start(run_bus=False)
    engine.final("hello hello")  # word-repeat, but no cleaner -> passes through
    assert runtime.wait_idle()
    assert engine.spoken == ["legacy"]


def test_intent_fast_path_sees_cleaned_text():
    """If the cleaner rewrites 'stop stop' into 'stop', the intent fast-path
    can now match where it would have missed the raw repeat."""
    engine = ScriptedEngine(hold_speech=True)
    cleaner = ScriptedTranscriptCleaner({"stop stop": "stop"})
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="a long answer"),
        cleaner=cleaner,
        command_map={"stop": "stop"},
    )
    runtime.start(run_bus=False)
    engine.final("tell me a story")
    assert runtime.wait_idle()
    assert engine.is_speaking
    # The user repeats "stop" as a correction; cleaned -> "stop" -> halt.
    engine.final("stop stop")
    runtime.wait_idle()
    assert not engine.is_speaking


def test_cleaner_failure_falls_back_to_raw():
    """A cleaner that raises must not break the turn; the raw text proceeds
    to the brain so the assistant still answers."""
    class _BoomCleaner:
        def clean(self, text, recent=()):
            raise RuntimeError("cleaner crashed")

    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM(reply="ok"), cleaner=_BoomCleaner())
    runtime.start(run_bus=False)
    engine.final("hello there")
    assert runtime.wait_idle()
    assert engine.spoken == ["ok"]
