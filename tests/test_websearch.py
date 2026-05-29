"""Tests for the pluggable ``web.search`` capability (core/websearch.py).

These pin the §9.7 egress boundary and the never-raise corpus fallback. They
need no network and no real SearXNG -- the backend is faked (or absent) so the
suite stays dependency-free (no httpx import on the corpus-only paths).

Coverage map (docs/p3_design.md §7, BR2/BR7):
- PRIVATE/PII tripwire: the gate runs FIRST, the backend is NEVER called, and
  the corpus answers with egress=False.
- PUBLIC query: hits a fake backend; citations == source urls; corpus-compatible
  result shape ({name, summary}).
- SearXNG unreachable / blocks past timeout_s => corpus fallback ok=True (BR7).
- Bogus context['mode'] => corpus fallback, still gated (BR2; never raises).
- Disabled / no base_url => corpus, dependency-free.
"""
from __future__ import annotations

import threading
import time

from always_on_agent.capabilities import create_default_capabilities
from always_on_agent.events import Mode
from always_on_agent.models import IntentKind

from core.websearch import (
    SearxngBackend,
    WebSearchConfig,
    attach_web_search_capability,
)


# --- test doubles ----------------------------------------------------------


class _Tripwire:
    """A backend that must never be called. Records if it ever was."""

    def __init__(self):
        self.calls = 0

    def search(self, query):
        self.calls += 1
        raise AssertionError("tripwire backend should not have been reached")


class _FakeSearxng:
    """A reachable backend returning canned SearXNG-shaped results."""

    def __init__(self, results):
        self._results = results
        self.queries: list[str] = []

    def search(self, query):
        self.queries.append(query)
        return self._results


class _Unreachable:
    """Mimics SearXNG being down: every call raises (a transport error)."""

    def search(self, query):
        raise ConnectionError("connection refused")


class _BlocksPastTimeout:
    """A backend that would block past timeout_s if its own bound didn't fire.

    Models the BR7 wedge: a blocking GET can't poll cancel; only the timeout
    protects. Here the bound is internal (a short wait) and it raises a
    timeout-like error, exactly as httpx would, which the provider treats as a
    network error -> corpus fallback ok=True with latency bounded."""

    def __init__(self, timeout_s):
        self._timeout_s = timeout_s

    def search(self, query):
        ev = threading.Event()
        ev.wait(timeout=self._timeout_s)  # never set -> waits the full bound
        raise TimeoutError("read timed out")


def _enabled_cfg(**kw) -> WebSearchConfig:
    base = dict(enabled=True, base_url="http://searx.local", timeout_s=0.2, max_results=5)
    base.update(kw)
    return WebSearchConfig(**base)


def _registry_with_web_search(config, backend):
    registry = create_default_capabilities()
    attach_web_search_capability(registry, config, backend=backend)
    return registry


# --- PRIVATE tripwire: gate first, never egress ----------------------------


def test_private_query_never_calls_backend_and_egress_false():
    tripwire = _Tripwire()
    registry = _registry_with_web_search(_enabled_cfg(), tripwire)

    result = registry.invoke("web.search", "my coworker John's salary", {})

    assert result.ok is True
    assert tripwire.calls == 0  # gate blocked egress BEFORE any network call
    assert result.data["egress"] is False
    assert result.data["sensitivity"] == "private"
    # Corpus-compatible shape preserved (search.local returns a results list).
    assert "results" in result.data


def test_code_with_credential_never_egresses():
    """PII precedence (BR5): a CODE query carrying a credential is blocked."""
    tripwire = _Tripwire()
    registry = _registry_with_web_search(_enabled_cfg(), tripwire)

    result = registry.invoke("web.search", "debug this, the api key is sk-abc123", {})

    assert result.ok is True
    assert tripwire.calls == 0
    assert result.data["egress"] is False


# --- PUBLIC query: hits the backend, citations == urls ---------------------


def test_public_query_hits_backend_with_citations_and_corpus_shape():
    fake = _FakeSearxng(
        [
            {"title": "Berlin weather", "content": "Sunny, 20C in Berlin.", "url": "https://a/1"},
            {"title": "Berlin forecast", "content": "Clear skies tomorrow.", "url": "https://a/2"},
        ]
    )
    registry = _registry_with_web_search(_enabled_cfg(), fake)

    result = registry.invoke("web.search", "weather in Berlin", {})

    assert result.ok is True
    assert fake.queries == ["weather in Berlin"]  # the gate permitted egress
    assert result.data["egress"] is True
    assert result.data["source"] == "web"
    # citations are the source urls, in order.
    assert result.citations == ("https://a/1", "https://a/2")
    # Result shape mirrors the corpus search(): list of {name, summary}.
    assert result.data["results"] == [
        {"name": "Berlin weather", "summary": "Sunny, 20C in Berlin."},
        {"name": "Berlin forecast", "summary": "Clear skies tomorrow."},
    ]
    assert "Sunny" in result.text


def test_max_results_caps_mapped_hits():
    fake = _FakeSearxng(
        [{"title": f"t{i}", "content": f"c{i}", "url": f"https://a/{i}"} for i in range(10)]
    )
    registry = _registry_with_web_search(_enabled_cfg(max_results=3), fake)

    result = registry.invoke("web.search", "open source voice assistants", {})

    assert len(result.data["results"]) == 3
    assert len(result.citations) == 3


# --- BR7: unreachable / blocks-past-timeout => corpus ok=True --------------


def test_unreachable_backend_falls_back_to_corpus_ok_true():
    registry = _registry_with_web_search(_enabled_cfg(), _Unreachable())

    result = registry.invoke("web.search", "what is pipecat", {})

    assert result.ok is True  # never aborts the plan (tasks.py:229-231)
    assert result.data["source"] == "corpus"
    assert result.data["error"] == "ConnectionError"
    # The corpus actually answered (pipecat is in the default corpus).
    assert "pipecat" in result.text.lower()


def test_backend_blocking_past_timeout_returns_corpus_bounded(monkeypatch):
    """BR7: a backend that blocks past timeout_s yields the corpus fallback with
    ok=True and bounded latency -- the timeout is the only wedge bound."""
    timeout_s = 0.2
    registry = _registry_with_web_search(
        _enabled_cfg(timeout_s=timeout_s), _BlocksPastTimeout(timeout_s)
    )

    t0 = time.monotonic()
    result = registry.invoke("web.search", "what is ollama", {})
    elapsed = time.monotonic() - t0

    assert result.ok is True
    assert result.data["source"] == "corpus"
    # Bounded by the backend's own timeout, never wedged indefinitely.
    assert elapsed < timeout_s + 1.0


# --- BR2: bogus context['mode'] => still gated, corpus ok=True -------------


def test_bogus_mode_does_not_abort_and_still_gates():
    """An out-of-vocab context['mode'] must not raise (which invoke() would turn
    into ok=False -> abort the plan) and must not bypass the gate (BR2). A plain
    public query under a bogus mode still egresses (mode coerced to None)."""
    fake = _FakeSearxng([{"title": "t", "content": "c", "url": "https://a/1"}])
    registry = _registry_with_web_search(_enabled_cfg(), fake)

    result = registry.invoke("web.search", "who won the 2022 world cup", {"mode": "not-a-mode"})

    assert result.ok is True  # bogus enum did not abort the plan
    assert fake.queries  # gate still ran and permitted this public query


def test_bogus_mode_with_pii_query_still_blocked():
    """BR2 fail-safe: a bogus mode is coerced to None, but a PII query is still
    blocked by the query-text branch of the gate (corpus, no egress)."""
    tripwire = _Tripwire()
    registry = _registry_with_web_search(_enabled_cfg(), tripwire)

    result = registry.invoke(
        "web.search", "what is my home address", {"mode": "{bogus}"}
    )

    assert result.ok is True
    assert tripwire.calls == 0
    assert result.data["egress"] is False


def test_real_meeting_mode_still_blocks_egress():
    """A valid blocking mode (MEETING) is honoured -- coercion preserves it."""
    tripwire = _Tripwire()
    registry = _registry_with_web_search(_enabled_cfg(), tripwire)

    result = registry.invoke(
        "web.search", "what is the agenda", {"mode": Mode.MEETING.value}
    )

    assert result.ok is True
    assert tripwire.calls == 0
    assert result.data["egress"] is False


def test_command_intent_blocks_egress():
    tripwire = _Tripwire()
    registry = _registry_with_web_search(_enabled_cfg(), tripwire)

    result = registry.invoke(
        "web.search", "what time is it", {"intent_kind": IntentKind.COMMAND.value}
    )

    assert result.ok is True
    assert tripwire.calls == 0
    assert result.data["egress"] is False


# --- disabled / no base_url => corpus, dependency-free ---------------------


def test_disabled_config_is_corpus_only_and_dependency_free():
    """No backend injected + disabled config: corpus-only, no SearxngBackend
    built, so no httpx import path is ever exercised."""
    registry = create_default_capabilities()
    attach_web_search_capability(registry, WebSearchConfig(enabled=False))

    result = registry.invoke("web.search", "what is livekit", {})

    assert result.ok is True
    assert result.data["egress"] is False
    assert result.data["source"] == "corpus"
    assert "livekit" in result.text.lower()


def test_enabled_but_no_base_url_is_corpus_only():
    registry = create_default_capabilities()
    attach_web_search_capability(
        registry, WebSearchConfig(enabled=True, base_url="")
    )

    result = registry.invoke("web.search", "what is wyoming", {})

    assert result.ok is True
    assert result.data["egress"] is False
    assert result.data["source"] == "corpus"


# --- config + backend unit shapes ------------------------------------------


def test_config_from_dict_mirrors_recall_config_shape():
    cfg = WebSearchConfig.from_dict(
        {"enabled": True, "base_url": "http://x:8888/", "timeout_s": 6, "max_results": 8}
    )
    assert cfg.enabled is True
    assert cfg.base_url == "http://x:8888/"
    assert cfg.timeout_s == 6.0
    assert cfg.max_results == 8


def test_config_from_dict_defaults():
    cfg = WebSearchConfig.from_dict(None)
    assert cfg.enabled is False
    assert cfg.base_url == ""
    assert cfg.timeout_s == 4.0
    assert cfg.max_results == 5


def test_searxng_backend_strips_trailing_slash():
    backend = SearxngBackend("http://searx.local/", timeout_s=3.0)
    assert backend._base_url == "http://searx.local"
    assert backend._timeout_s == 3.0


def test_empty_results_from_backend_falls_back_to_corpus():
    """A reachable backend returning zero usable hits => corpus fallback."""
    registry = _registry_with_web_search(_enabled_cfg(), _FakeSearxng([]))

    result = registry.invoke("web.search", "what is moonshine", {})

    assert result.ok is True
    assert result.data["source"] == "corpus"
    assert result.data["egress"] is True  # we did egress; corpus is the fallback
    assert "moonshine" in result.text.lower()
