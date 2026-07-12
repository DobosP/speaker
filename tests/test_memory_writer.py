"""Unit tests for smart Postgres memory writer (no real DB or Ollama)."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from core.llm import OllamaLLM
from utils.memory_config import MemoryWriterConfig, config_from_dict
from utils.memory_writer import (
    CleanupResult,
    LLMClientMemoryCleanup,
    MemoryWriter,
    UtteranceCandidate,
    is_junk_stt_text,
    should_persist,
    texts_near_duplicate,
)


class MockLLM:
    def cleanup(self, text: str, *, gate: bool) -> CleanupResult:
        if "skip me" in text.lower():
            return CleanupResult(False, "", "not_worthy")
        return CleanupResult(True, text.strip().title(), "ok")


class MockGenerator:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def generate(self, prompt: str, *, system=None) -> str:
        self.calls.append((prompt, system))
        return self.response


class MockStructuredGenerator(MockGenerator):
    def generate_json(self, prompt: str, *, system=None) -> str:
        self.calls.append((prompt, system, "json"))
        return self.response


class MockOllamaTransport:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "message": {
                "content": (
                    '{"worth_saving": true, "cleaned_text": "A fact.", '
                    '"reason": "fact"}'
                )
            }
        }


def test_config_from_dict_nested_memory_block():
    cfg = config_from_dict(
        {
            "memory": {
                "save_interval_sec": 120,
                "cleanup_model": "tiny",
                "min_confidence": 0.7,
            }
        }
    )
    assert cfg.save_interval_sec == 120
    assert cfg.cleanup_model == "tiny"
    assert cfg.min_confidence == 0.7


def test_is_junk_stt_text():
    assert is_junk_stt_text("")
    assert is_junk_stt_text("[blank_audio]")
    assert not is_junk_stt_text("remind me to buy milk tomorrow")


def test_should_persist_rejects_partial_and_boilerplate():
    cfg = MemoryWriterConfig(save_control_phrases=False)
    partial = UtteranceCandidate("hello there", source="user_partial", confidence=0.9)
    assert should_persist(partial, cfg)[0] is False
    stop = UtteranceCandidate("stop", source="user_final", confidence=1.0)
    assert should_persist(stop, cfg)[0] is False


def test_dedupe_near_duplicate():
    assert texts_near_duplicate("buy milk", "buy  milk", 0.9)


def test_memory_writer_flush_calls_persist_with_llm_cleanup():
    saved = []

    def persist_fn(**kwargs):
        saved.append(kwargs)

    writer = MemoryWriter(
        config=MemoryWriterConfig(
            save_interval_sec=0,
            llm_cleanup=True,
            llm_gate=True,
            min_chars=2,
        ),
        persist_fn=persist_fn,
        llm_client=MockLLM(),
    )
    writer.enqueue("i like python programming")
    count = writer.flush(force=True)
    assert count == 1
    assert saved[0]["cleaned_text"] == "I Like Python Programming"
    assert saved[0]["raw_text"] == "i like python programming"


def test_memory_writer_skips_gated_content():
    saved = []

    writer = MemoryWriter(
        config=MemoryWriterConfig(save_interval_sec=0, llm_gate=True),
        persist_fn=lambda **kw: saved.append(kw),
        llm_client=MockLLM(),
    )
    writer.enqueue("please skip me now")
    assert writer.flush(force=True) == 0
    assert saved == []


def test_memory_writer_uses_text_cleaner_hook():
    saved = []

    def cleaner(role: str, content: str):
        assert role == "user"
        return content.upper()

    writer = MemoryWriter(
        config=MemoryWriterConfig(
            save_interval_sec=0, llm_cleanup=False, llm_gate=False
        ),
        persist_fn=lambda **kw: saved.append(kw),
        text_cleaner=cleaner,
    )
    writer.enqueue("hello world")
    writer.flush(force=True)
    assert saved[0]["cleaned_text"] == "HELLO WORLD"


def test_memory_writer_rejects_assistant_echo():
    saved = []
    writer = MemoryWriter(
        config=MemoryWriterConfig(save_interval_sec=0, dedupe_similarity=0.85),
        persist_fn=lambda **kw: saved.append(kw),
    )
    ok = writer.enqueue(
        "the weather is nice today",
        last_assistant_text="the weather is nice today",
    )
    assert ok is False
    assert writer.flush(force=True) == 0


def test_memory_writer_default_cleanup_model_is_minicpm_fast_alias():
    assert MemoryWriterConfig().cleanup_model == "minicpm5-1b:q8"


def test_runtime_llm_cleanup_reuses_client_and_decodes_fenced_json():
    generator = MockGenerator(
        'Here is the result:\n```json\n'
        '{"worth_saving": false, "cleaned_text": "Stop", "reason": "control"}'
        '\n```'
    )

    result = LLMClientMemoryCleanup(generator).cleanup("stop", gate=True)

    assert result == CleanupResult(False, "Stop", "control")
    assert len(generator.calls) == 1
    assert "substantive user content" in generator.calls[0][1]


def test_runtime_llm_cleanup_fails_open_on_invalid_model_output():
    result = LLMClientMemoryCleanup(MockGenerator("not json")).cleanup(
        "keep the raw transcript", gate=True
    )

    assert result.worth_saving is True
    assert result.cleaned_text == "keep the raw transcript"
    assert result.reason.startswith("llm_error:")


def test_runtime_llm_cleanup_prefers_existing_clients_structured_mode():
    generator = MockStructuredGenerator(
        '{"worth_saving": true, "cleaned_text": "A fact.", "reason": "fact"}'
    )

    result = LLMClientMemoryCleanup(generator).cleanup("a fact", gate=True)

    assert result == CleanupResult(True, "A fact.", "fact")
    assert generator.calls[0][2] == "json"


def test_runtime_ollama_cleanup_preserves_fast_client_request_settings():
    transport = MockOllamaTransport()
    fast = OllamaLLM(
        model="minicpm5-1b:q8",
        options={"num_ctx": 4096},
        keep_alive="30m",
        think=False,
        client=transport,
    )

    result = LLMClientMemoryCleanup(fast).cleanup("a fact", gate=True)

    assert result == CleanupResult(True, "A fact.", "fact")
    call = transport.calls[0]
    assert call["model"] == "minicpm5-1b:q8"
    assert call["format"] == "json"
    assert call["options"] == {"num_ctx": 4096}
    assert call["keep_alive"] == "30m"
    assert call["think"] is False
