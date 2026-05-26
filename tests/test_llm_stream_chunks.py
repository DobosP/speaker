"""Unit tests for LLM streaming chunk boundaries (no Ollama)."""
import re

import pytest

from utils.llm import LocalLLM

pytestmark = pytest.mark.dev


@pytest.fixture
def patch_ollama_connect(monkeypatch):
    monkeypatch.setattr("ollama.list", lambda: {})


def test_pop_stream_chunk_phrase_breaks_on_punctuation():
    llm = LocalLLM.__new__(LocalLLM)
    se = re.compile(r"(?<=[.!?])\s")
    c, rest = LocalLLM._pop_stream_chunk(
        llm,
        "one two three four five six, more here",
        "phrase",
        se,
        4,
        12,
    )
    assert "six" in c
    assert "more" in rest


def test_pop_stream_chunk_max_words():
    llm = LocalLLM.__new__(LocalLLM)
    se = re.compile(r"(?<=[.!?])\s")
    c, rest = LocalLLM._pop_stream_chunk(
        llm,
        "a b c d e f g h i j k l m n",
        "phrase",
        se,
        4,
        8,
    )
    assert len(c.split()) == 8
    assert rest


def test_get_streaming_response_token_coalesces_deltas(patch_ollama_connect, monkeypatch):
    monkeypatch.setattr(
        "utils.llm._ollama_chat",
        lambda **kw: iter(
            [
                {"message": {"content": "One"}},
                {"message": {"content": " two"}},
            ]
        ),
    )
    llm = LocalLLM(model="m", generation_profile="balanced")
    parts = list(
        llm.get_streaming_response("ping", stream_mode="token")
    )
    assert parts == ["One two"]


def test_get_streaming_response_token_raw_deltas_when_coalesce_off(
    patch_ollama_connect, monkeypatch
):
    monkeypatch.setattr(
        "utils.llm._ollama_chat",
        lambda **kw: iter(
            [
                {"message": {"content": "One"}},
                {"message": {"content": " two"}},
            ]
        ),
    )
    llm = LocalLLM(model="m", generation_profile="balanced")
    parts = list(
        llm.get_streaming_response(
            "ping",
            stream_mode="token",
            coalesce_tokens=False,
        )
    )
    assert parts == ["One", " two"]


def test_get_streaming_response_word_coalesces(patch_ollama_connect, monkeypatch):
    monkeypatch.setattr(
        "utils.llm._ollama_chat",
        lambda **kw: iter(
            [
                {"message": {"content": "Hello "}},
                {"message": {"content": "world"}},
            ]
        ),
    )
    llm = LocalLLM(model="m", generation_profile="balanced")
    parts = list(llm.get_streaming_response("ping", stream_mode="word"))
    assert parts == ["Hello world"]


def test_get_streaming_response_word_emits_each_word_when_coalesce_off(
    patch_ollama_connect, monkeypatch
):
    monkeypatch.setattr(
        "utils.llm._ollama_chat",
        lambda **kw: iter(
            [
                {"message": {"content": "Hello "}},
                {"message": {"content": "world"}},
            ]
        ),
    )
    llm = LocalLLM(model="m", generation_profile="balanced")
    parts = list(
        llm.get_streaming_response(
            "ping",
            stream_mode="word",
            coalesce_tokens=False,
        )
    )
    assert parts == ["Hello", "world"]
