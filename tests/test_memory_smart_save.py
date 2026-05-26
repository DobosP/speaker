from __future__ import annotations

from utils.memory_config import MemoryWriterConfig
from utils.memory_writer import (
    CleanupResult,
    MemoryWriter,
    UtteranceCandidate,
    should_persist,
)


class FakeCleanup:
    def cleanup(self, text: str, *, gate: bool) -> CleanupResult:
        if "thank you thank you" in text.lower():
            return CleanupResult(False, "", "boilerplate")
        return CleanupResult(True, text.replace("teh", "the").strip(), "cleaned")


def test_should_persist_filters_control_and_partial_text():
    config = MemoryWriterConfig()

    ok, reason = should_persist(
        UtteranceCandidate("stop", source="user_final"),
        config,
    )
    assert not ok
    assert reason == "boilerplate"

    ok, reason = should_persist(
        UtteranceCandidate("remember this later", source="user_partial"),
        config,
    )
    assert not ok
    assert reason == "partial_not_allowed"


def test_memory_writer_cleans_dedupes_and_flushes_once():
    saved: list[tuple[str, str]] = []
    config = MemoryWriterConfig(
        save_interval_sec=999,
        llm_cleanup=True,
        llm_gate=True,
        max_buffer_items=8,
    )
    writer = MemoryWriter(
        config=config,
        persist_fn=lambda **kw: saved.append((kw["raw_text"], kw["cleaned_text"])),
        llm_client=FakeCleanup(),
    )

    assert writer.enqueue("I like teh local voice assistant")
    assert not writer.enqueue("I like the local voice assistant")
    assert writer.enqueue("thank you thank you")

    assert writer.flush(force=True) == 1
    assert saved == [
        ("I like teh local voice assistant", "I like the local voice assistant")
    ]


def test_memory_writer_close_flushes_pending_items():
    saved: list[str] = []
    writer = MemoryWriter(
        config=MemoryWriterConfig(save_interval_sec=999, llm_cleanup=False, llm_gate=False),
        persist_fn=lambda **kw: saved.append(kw["cleaned_text"]),
    )

    assert writer.enqueue("Save my preference for local models")
    writer.close()

    assert saved == ["Save my preference for local models"]
