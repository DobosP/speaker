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


# --- §9.7: captioning must be LOCAL-ONLY (never a cloud chain) ---------------


class _Spy:
    def __init__(self, reply="local caption"):
        self.reply = reply
        self.image_calls = 0

    def generate(self, prompt, *, system=None, images=None):
        if images:
            self.image_calls += 1
        return self.reply


class HedgeLLM:  # name matters: build_visual_memorizer detects cloud-capable by type name
    def __init__(self, local=None, cloud=None):
        self.local = local
        self._cloud = cloud

    def generate(self, prompt, *, system=None, images=None):
        # The cloud path: if visual memory ever calls THIS with images, that's the leak.
        if images and self._cloud is not None:
            self._cloud.generate(prompt, system=system, images=images)
        return "cloud caption"


def test_caption_uses_local_handle_not_cloud():
    """A cloud-wrapped main (HedgeLLM with .local) must caption on the LOCAL spy;
    the cloud member must NEVER receive the frame (§9.7)."""
    local, cloud = _Spy("on-device caption"), _Spy()
    wrapped = HedgeLLM(local=local, cloud=cloud)

    class _RT:
        memory = SessionMemory()

    m = build_visual_memorizer(
        {"screen_capture": {"memorize": True, "memorize_min_interval_sec": 0, "memorize_ocr": False}},
        _RT(), wrapped,
    )
    assert m is not None
    trace = m.compose(b"frame-bytes")
    assert trace == "on-device caption"
    assert local.image_calls == 1 and cloud.image_calls == 0  # frame stayed on-device


def test_caption_disabled_when_only_cloud_handle_available():
    """If the main is cloud-capable AND no local handle exists, captioning is
    hard-disabled (OCR-only) rather than risk leaking the frame."""
    cloud_only = HedgeLLM(local=None, cloud=_Spy())

    class _RT:
        memory = SessionMemory()

    m = build_visual_memorizer(
        {"screen_capture": {"memorize": True, "memorize_min_interval_sec": 0,
                            "memorize_ocr": True}},
        _RT(), cloud_only,
    )
    assert m is not None
    # OCR (default) carries; caption is disabled -> the cloud generate is never hit.
    assert m._caption_fn is None


def test_factory_tags_local_main():
    from core.llm_factory import _tag_local_main

    class _W:
        pass
    local = _Spy()
    wrapped = _W()
    assert _tag_local_main(wrapped, local) is wrapped
    assert wrapped.local_main is local


# --- degradation of the real default factories ------------------------------


def test_default_ocr_fn_degrades_on_bad_image():
    from core.visual_memory import default_ocr_fn
    # Not a valid image (and tesseract/Pillow may be absent) -> '' , never raises.
    assert default_ocr_fn(200)(b"not-an-image") == ""


def test_llm_caption_fn_degrades_when_generate_raises():
    from core.visual_memory import llm_caption_fn

    class _Boom:
        def generate(self, *a, **k):
            raise RuntimeError("no model")
    assert llm_caption_fn(_Boom())(b"frame") == ""


def test_worker_skips_empty_trace():
    got = []
    m = _mk(got.append, caption="", ocr="", min_interval=0.0)
    m.start()
    try:
        m.observe(b"frame")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        m.stop()
    assert got == []  # an empty trace is never ingested


# --- get_context_for_llm: 3-pass shared budget, vision never starves recall --


def test_get_context_three_pass_budget_no_starvation(monkeypatch):
    import pytest
    pytest.importorskip("numpy")
    from always_on_agent.recall import RecallBudget, VISION_LABEL, estimate_tokens
    from utils.memory import MemoryManager
    from contextlib import contextmanager

    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **k): pass
        def fetchall(self): return []
        def fetchone(self): return None
        @property
        def rowcount(self): return 0

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self, **k): return _Cur()

    class _Pool:
        def __init__(self, *a, **k): pass
        @contextmanager
        def connection(self): yield _Conn()
        def close(self): pass

    mgr = MemoryManager(
        db_url="postgresql://fake", enable_embeddings=False, smart_save=False,
        pool_factory=lambda **k: _Pool(), recall_budget=RecallBudget(max_tokens=80),
    )
    mgr._embeddings_available = True
    mgr._db_available = True
    # MANY high-similarity vision rows + ONE user message: with a single shared
    # pool the vision rows would crowd out the user recall; the separate sub-pass
    # must still surface the user message.
    monkeypatch.setattr(mgr, "search_memory",
                        lambda q, limit=5: [{"type": "message", "role": "user",
                                             "content": "my flight is at 6pm on friday",
                                             "timestamp": 0.0, "similarity": 0.7}])
    monkeypatch.setattr(mgr, "_search_observations",
                        lambda q, limit=5: [{"type": "vision",
                                             "content": f"a screen showing window number {i} " * 3,
                                             "timestamp": float(i), "similarity": 0.95}
                                            for i in range(8)])
    try:
        ctx = mgr.get_context_for_llm("when is my flight and what was on screen")
    finally:
        mgr.close()
    assert estimate_tokens(ctx) <= 80                 # combined still bounded
    assert "my flight is at 6pm" in ctx               # user recall NOT starved by vision
    assert VISION_LABEL in ctx                          # vision present too
