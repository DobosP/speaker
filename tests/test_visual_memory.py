"""Tier-0 tests for persistent visual memory (``core/visual_memory.py``).

No display, no models, no tesseract, no DB: fake grabber/ocr/caption + SessionMemory.
Pins the contract: caption+OCR trace production, on-change + throttle gating, the
off-hot-path worker, graceful degradation, default-OFF, and -- end to end -- that a
'vision' memory is RECALLED via the shared smart-recall layer (build_block) with the
canonical 'Screen:' label while staying OUT of the recent-conversation block.
"""
from __future__ import annotations

import time

from always_on_agent.memory import SessionMemory
from always_on_agent.recall import VISION_LABEL, Candidate, RecallBudget, build_block
from core.conversation import RecentContextConfig, build_recent_context
from core.visual_memory import (
    VisualMemoryConfig,
    VisualMemorizer,
    build_visual_memorizer,
)


def _mk(ingest, *, caption="a code editor showing pgvector", ocr="def build_block(...)",
        min_interval=0.0, on_change=True, do_caption=True, do_ocr=True) -> VisualMemorizer:
    cfg = VisualMemoryConfig(
        enabled=True, min_interval_sec=min_interval, on_change=on_change,
        caption=do_caption, ocr=do_ocr,
    )
    return VisualMemorizer(
        ingest=ingest,
        caption_fn=(lambda _f: caption),
        ocr_fn=(lambda _f: ocr),
        config=cfg,
    )


# --- compose ---------------------------------------------------------------


def test_compose_caption_and_ocr():
    m = _mk(lambda _t: None)
    trace = m.compose(b"frame")
    assert "a code editor showing pgvector" in trace
    assert "on-screen text: def build_block(...)" in trace


def test_compose_caption_only_when_ocr_off():
    m = _mk(lambda _t: None, do_ocr=False)
    trace = m.compose(b"frame")
    assert trace == "a code editor showing pgvector"


def test_compose_empty_when_both_empty():
    m = _mk(lambda _t: None, caption="", ocr="")
    assert m.compose(b"frame") == ""


def test_compose_degrades_when_caption_raises():
    cfg = VisualMemoryConfig(enabled=True)
    def boom(_f):
        raise RuntimeError("no model")
    m = VisualMemorizer(ingest=lambda _t: None, caption_fn=boom, ocr_fn=lambda _f: "hello world", config=cfg)
    assert m.compose(b"frame") == "on-screen text: hello world"  # OCR still carries


def test_compose_trims_to_max_chars():
    cfg = VisualMemoryConfig(enabled=True, max_chars=20)
    m = VisualMemorizer(ingest=lambda _t: None, caption_fn=lambda _f: "x" * 100, ocr_fn=None, config=cfg)
    assert len(m.compose(b"frame")) <= 20


# --- observe gating (deterministic via the queue, no worker) ----------------


def test_observe_skips_duplicate_frame_on_change():
    m = _mk(lambda _t: None, min_interval=0.0, on_change=True)
    m.observe(b"AAAA")
    assert m._q.qsize() == 1
    m.observe(b"AAAA")  # byte-identical -> skipped
    assert m._q.qsize() == 1
    m.observe(b"BBBB")  # changed -> enqueued
    assert m._q.qsize() == 2


def test_observe_throttles_by_min_interval():
    m = _mk(lambda _t: None, min_interval=1000.0, on_change=False)
    m.observe(b"AAAA")
    assert m._q.qsize() == 1
    m.observe(b"BBBB")  # within the interval -> throttled despite being a new frame
    assert m._q.qsize() == 1


def test_observe_ignores_empty_frame():
    m = _mk(lambda _t: None)
    m.observe(None)
    m.observe(b"")
    assert m._q.qsize() == 0


# --- worker (off the hot path) ---------------------------------------------


def test_worker_ingests_the_trace():
    got: list[str] = []
    m = _mk(got.append, min_interval=0.0)
    m.start()
    try:
        m.observe(b"frame-1")
        deadline = time.monotonic() + 2.0
        while not got and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        m.stop()
    assert got and "pgvector" in got[0]


# --- build / default-off ----------------------------------------------------


def test_build_visual_memorizer_off_by_default():
    class _RT:
        memory = SessionMemory()
    assert build_visual_memorizer({"screen_capture": {"enabled": True}}, _RT(), llm=None) is None
    assert build_visual_memorizer({}, _RT(), llm=None) is None


def test_config_from_dict_reads_memorize_flags():
    cfg = VisualMemoryConfig.from_dict({"memorize": True, "memorize_min_interval_sec": 5, "memorize_ocr": False})
    assert cfg.enabled is True and cfg.min_interval_sec == 5.0 and cfg.ocr is False


# --- recall integration -----------------------------------------------------


def test_vision_candidate_renders_with_screen_label():
    block = build_block(
        [Candidate("a terminal running pytest", 0.9, kind="vision")],
        "what was on my terminal", RecallBudget(max_tokens=120),
    )
    assert block.startswith("=== Past Conversations ===")
    assert f"{VISION_LABEL} a terminal running pytest" in block


def test_end_to_end_vision_memory_recalled_via_session_memory():
    mem = SessionMemory()
    # The memorizer ingests exactly as wired in build_visual_memorizer.
    m = _mk(lambda t: mem.add(t, tags=("vision",)),
            caption="a browser on the pgvector documentation", ocr="HNSW index")
    m.observe(b"frame")  # enqueue
    m.start()
    try:
        deadline = time.monotonic() + 2.0
        while not mem.all() and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        m.stop()
    block = mem.context_for_llm("what was the pgvector documentation about")
    assert VISION_LABEL in block and "pgvector" in block


def test_vision_memory_excluded_from_recent_conversation_block():
    mem = SessionMemory()
    mem.add("a screenshot of an email client", tags=("vision",))
    mem.add("what time is my meeting", tags=("user",))
    recent = build_recent_context(mem, RecentContextConfig())
    assert "what time is my meeting" in recent
    assert "email client" not in recent  # vision never pollutes conversational context
