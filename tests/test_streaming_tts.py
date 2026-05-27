"""Tests for sentence-level streaming TTS (the low-latency speak-as-you-go path).

Uses the scripted engine to assert that each generated sentence is spoken as a
separate utterance, in order, and that the consolidated answer is not re-spoken.
"""

from __future__ import annotations

from typing import Iterator

from core.engines.scripted import ScriptedEngine
from core.runtime import VoiceRuntime


class SentenceLLM:
    """Streams a list of sentences as word+space chunks (so sentence-end
    boundaries appear mid-stream, like a real token stream)."""

    def __init__(self, sentences: list[str]):
        self._sentences = sentences

    def _text(self) -> str:
        return " ".join(self._sentences)

    def generate(self, prompt: str, *, system=None, images=None) -> str:
        return self._text()

    def stream(self, prompt: str, *, system=None, images=None) -> Iterator[str]:
        words = self._text().split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "


def test_streaming_speaks_each_sentence_in_order():
    sentences = ["First one.", "Second two.", "Third three."]
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, SentenceLLM(sentences), stream_tts=True)
    runtime.start(run_bus=False)

    engine.final("tell me three things")
    assert runtime.wait_idle()
    assert engine.spoken == sentences


def test_non_streaming_speaks_one_consolidated_utterance():
    sentences = ["First one.", "Second two."]
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, SentenceLLM(sentences))  # stream_tts defaults off
    runtime.start(run_bus=False)

    engine.final("tell me things")
    assert runtime.wait_idle()
    assert engine.spoken == ["First one. Second two."]
