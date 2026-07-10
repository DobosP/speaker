"""Barge-in must STOP the compute, not just mute the audio.

When a turn is cancelled mid-stream, ``_collect`` / ``_stream_and_speak`` now
close the token generator explicitly so the underlying HTTP body / SDK stream
starts cleanup at the barge point instead of lingering until garbage collection.
These tests pin that: a cancelled stream's ``finally`` (which is where the real
clients request transport cleanup) runs DURING the drain, with the generator still
referenced so GC cannot be what closed it. Pure: a fake generator + a real Event,
no LLM, no network.
"""
from __future__ import annotations

from threading import Event

from core.capabilities import _close_token_stream, _collect, _stream_and_speak


def _make_stream(state: dict, n: int = 100):
    """A token generator whose ``finally`` records that it was torn down -- the
    seam the real OllamaLLM/OpenAI stream uses to close its HTTP body."""

    def gen():
        try:
            for i in range(n):
                yield f"tok{i} "
        finally:
            state["closed"] = True

    return gen()


def test_collect_closes_stream_on_cancel():
    state = {"closed": False}
    cancel = Event()
    cancel.set()  # barge before the first token is kept
    stream = _make_stream(state)  # keep a reference so GC can't be the closer
    text, cancelled = _collect(stream, cancel)
    assert cancelled is True
    assert state["closed"] is True  # explicitly closed by _collect, not by GC


def test_stream_and_speak_closes_stream_on_cancel():
    state = {"closed": False}
    cancel = Event()
    cancel.set()
    stream = _make_stream(state)
    spoken: list[str] = []
    text, cancelled = _stream_and_speak(stream, cancel, spoken.append)
    assert cancelled is True
    assert state["closed"] is True


def test_collect_normal_completion_is_not_cancelled():
    state = {"closed": False}
    text, cancelled = _collect(_make_stream(state, n=3), None)
    assert cancelled is False
    assert text == "tok0 tok1 tok2"


def test_close_token_stream_is_a_noop_on_a_plain_iterator():
    # A plain iterator has no .close(); closing must not raise.
    _close_token_stream(iter(["a ", "b"]))


def test_close_token_stream_swallows_close_errors():
    class _Boom:
        def close(self):
            raise RuntimeError("teardown blew up")

    _close_token_stream(_Boom())  # must not propagate
