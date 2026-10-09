"""Sentence-granularity revocation uses the existing provider cleanup seam."""
from threading import Event

from core.capabilities import _stream_and_speak


class Tokens:
    def __init__(self, *values):
        self.values = iter(values)
        self.closed = False
    def __iter__(self):
        return self
    def __next__(self):
        return next(self.values)
    def close(self):
        self.closed = True


def test_one_provider_token_cannot_emit_successor_sentence_after_revocation():
    cancel = Event()
    tokens = Tokens("First sentence. Second sentence. Unfinished tail")
    emitted = []
    def emit(sentence):
        emitted.append(sentence)
        cancel.set()
    text, cancelled = _stream_and_speak(tokens, cancel, emit)
    assert emitted == ["First sentence."]
    assert text == "First sentence. Second sentence. Unfinished tail"
    assert cancelled is True
    assert tokens.closed


def test_cancel_during_final_tail_is_reported_and_provider_closed():
    cancel = Event()
    tokens = Tokens("An unfinished answer")
    emitted = []
    def emit(sentence):
        emitted.append(sentence)
        cancel.set()
    text, cancelled = _stream_and_speak(tokens, cancel, emit)
    assert text == "An unfinished answer"
    assert emitted == [text]
    assert cancelled is True
    assert tokens.closed


def test_pre_cancelled_reply_emits_nothing_and_closes_source():
    cancel = Event()
    cancel.set()
    tokens = Tokens("No audio.")
    emitted = []
    assert _stream_and_speak(tokens, cancel, emitted.append) == ("", True)
    assert emitted == []
    assert tokens.closed


def test_uncancelled_multisentence_token_and_tail_remain_complete():
    tokens = Tokens("First sentence. Second sentence. Tail")
    emitted = []
    assert _stream_and_speak(tokens, Event(), emitted.append) == (
        "First sentence. Second sentence. Tail", False)
    assert emitted == ["First sentence.", "Second sentence.", "Tail"]
    assert not tokens.closed
