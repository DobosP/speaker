"""Unit tests for smart Postgres memory writer (no real DB or Ollama)."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from utils.memory_config import MemoryWriterConfig, config_from_dict
from utils.memory_writer import (
    CleanupResult,
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
