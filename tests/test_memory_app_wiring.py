"""App-level wiring tests for the P2b memory seam (``core.app._build_memory``).

These assert the *plumbing*, not the engine: that ``_build_memory`` forwards the
P2b knobs (``summarizer`` / ``profile_enabled`` / ``episodic_ttl_days`` /
``summary_ttl_days``) from the ``memory`` config block into
:class:`always_on_agent.memory.MemoryManagerAdapter`, that the summarizer closure
wraps the fast LLM, and that ``profile_enabled`` defaults to False (recall stays
OFF by default elsewhere).

A fake adapter (monkeypatched over ``MemoryManagerAdapter``) captures the kwargs
so no real PostgreSQL/psycopg is needed -- the Postgres branch is forced by
setting a throwaway ``$DATABASE_URL`` under ``auto`` backend selection.
"""
from __future__ import annotations

import core.app as app


class _FakeAdapter:
    """Stand-in for MemoryManagerAdapter that records its kwargs."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    # Memory protocol surface the runtime might touch (unused here, kept safe).
    def add(self, text, tags=()):  # pragma: no cover - not exercised
        return None

    def close(self):  # pragma: no cover - not exercised
        return None


class _FakeFastLLM:
    """Records the prompt the summarizer closure forwards to generate()."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt, *, system=None, images=None) -> str:
        self.prompts.append(prompt)
        return f"summary::{prompt}"


def _force_postgres(monkeypatch) -> None:
    # auto backend + a DATABASE_URL => the adapter branch (Locked Decision 4).
    # _build_memory imports the adapter lazily from always_on_agent.memory, so
    # patch it there (not on core.app).
    monkeypatch.setattr(
        "always_on_agent.memory.MemoryManagerAdapter", _FakeAdapter, raising=True
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/ignored")
    _FakeAdapter.last_kwargs = {}


def test_build_memory_forwards_p2b_knobs_to_adapter(monkeypatch):
    _force_postgres(monkeypatch)
    fast = _FakeFastLLM()
    config = {
        "memory": {
            "backend": "auto",
            "profile_enabled": True,
            "episodic_ttl_days": 45,
            "summary_ttl_days": 200,
            "embeddings": False,
            "max_recent": 12,
        }
    }

    mem = app._build_memory(config, fast)

    assert isinstance(mem, _FakeAdapter)
    kw = _FakeAdapter.last_kwargs
    assert kw["profile_enabled"] is True
    assert kw["episodic_ttl_days"] == 45
    assert kw["summary_ttl_days"] == 200
    # The summarizer closure wraps fast_llm.generate.
    summarizer = kw["summarizer"]
    assert callable(summarizer)
    assert summarizer("hello") == "summary::hello"
    assert fast.prompts == ["hello"]


def test_build_memory_profile_enabled_defaults_false(monkeypatch):
    _force_postgres(monkeypatch)
    # An otherwise-empty memory block: every P2b knob falls to its default.
    mem = app._build_memory({"memory": {"backend": "auto"}}, _FakeFastLLM())

    assert isinstance(mem, _FakeAdapter)
    kw = _FakeAdapter.last_kwargs
    assert kw["profile_enabled"] is False
    assert kw["episodic_ttl_days"] == 90
    assert kw["summary_ttl_days"] == 365


def test_build_memory_summarizer_none_without_fast_llm(monkeypatch):
    _force_postgres(monkeypatch)
    # --llm echo path: no fast tier -> the manager uses its keyword fallback.
    mem = app._build_memory({"memory": {"backend": "auto"}}, None)

    assert isinstance(mem, _FakeAdapter)
    assert _FakeAdapter.last_kwargs["summarizer"] is None


def test_build_memory_inmemory_when_no_db(monkeypatch):
    # auto backend + no DATABASE_URL -> in-RAM SessionMemory, never the adapter.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from always_on_agent.memory import SessionMemory

    mem = app._build_memory({"memory": {"backend": "auto"}}, _FakeFastLLM())
    assert isinstance(mem, SessionMemory)


def test_build_memory_forwards_working_window_to_adapter(monkeypatch):
    """lm-7: the adapter's all() ring is sized off working_window (not max_recent)
    so it matches SessionMemory / SqliteVecMemory."""
    _force_postgres(monkeypatch)
    config = {"memory": {"backend": "auto", "working_window": 137, "max_recent": 20}}

    mem = app._build_memory(config, _FakeFastLLM())

    assert isinstance(mem, _FakeAdapter)
    assert _FakeAdapter.last_kwargs["working_window"] == 137


def test_build_memory_working_window_defaults_to_200(monkeypatch):
    _force_postgres(monkeypatch)
    mem = app._build_memory({"memory": {"backend": "auto"}}, _FakeFastLLM())
    assert isinstance(mem, _FakeAdapter)
    assert _FakeAdapter.last_kwargs["working_window"] == 200


def test_config_has_no_dead_memory_knobs():
    """Regression guard (lm-4/5/6/8): the zero-reader memory knobs were deleted;
    keep them gone so they don't drift back as dead config."""
    import json
    import os

    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json")))
    dead_top_level = (
        "memory_smart_save",
        "memory_flush_interval_sec",
        "memory_enable_embeddings",
        "memory_persist_assistant",
        "memory_llm_clean",
    )
    for k in dead_top_level:
        assert k not in cfg, f"dead top-level memory knob resurfaced: {k}"
    assert "meeting_persist" not in cfg.get("memory", {}), "dead meeting_persist resurfaced"
