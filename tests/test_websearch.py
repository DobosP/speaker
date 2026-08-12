"""Tests for the pluggable ``web.search`` capability (core/websearch.py).

These pin the §9.7 egress boundary and the never-raise corpus fallback. They
need no network and no real SearXNG -- the backend is faked (or absent) so the
suite stays dependency-free (no httpx import on the corpus-only paths).

Coverage map (docs/archive/p3_design.md §7, BR2/BR7):
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

import pytest

from always_on_agent.capabilities import (
    CapabilityRegistry,
    CapabilityResult,
    create_default_capabilities,
)
from always_on_agent.events import Mode
from always_on_agent.models import (
    CLOUD_EGRESS_SCOPE_CONTEXT_KEY,
    RETAINED_PROMPT_CONTEXT_METADATA_KEY,
    CloudEgressScope,
    IntentKind,
)

from core.sensitivity import CODE, PRIVATE, PUBLIC
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


class _CountingClassifier:
    """A permissive gate whose call count proves an earlier veto won."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return True


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


class _EqualityBomb:
    """A hostile context value whose comparison/coercion must never run."""

    def __eq__(self, other):
        raise AssertionError("hostile sensitivity equality hook was invoked")

    def __hash__(self):
        raise AssertionError("hostile sensitivity hash hook was invoked")

    def __str__(self):
        raise AssertionError("hostile sensitivity coercion hook was invoked")


class _FalseyPrivateContext(dict):
    """A supplied context that registry normalization must not discard."""

    def __bool__(self):
        return False


class _LyingPrivateContext(dict):
    """A dict subclass whose virtual lookup must never hide PRIVATE."""

    def get(self, key, default=None):
        if key == "sensitivity":
            return default
        return super().get(key, default)


class _SensitivityAliasKey:
    """A non-string key that must never impersonate ``sensitivity``."""

    def __init__(self):
        self.comparisons = 0

    def __hash__(self):
        return hash("sensitivity")

    def __eq__(self, other):
        self.comparisons += 1
        return other == "sensitivity"


class _RetainedMarkerAliasKey:
    """A colliding nested key whose equality hook must never run."""

    def __init__(self):
        self.comparisons = 0

    def __hash__(self):
        return hash(RETAINED_PROMPT_CONTEXT_METADATA_KEY)

    def __eq__(self, other):
        self.comparisons += 1
        raise AssertionError("nested retained-marker equality hook was invoked")


class _ExplodingHit(dict):
    def get(self, key, default=None):
        raise RuntimeError("malformed backend hit")


class _EnumBomb:
    """A malformed mode/intent value whose hooks must not be invoked."""

    def __hash__(self):
        raise AssertionError("malformed enum hash hook was invoked")

    def __eq__(self, other):
        raise AssertionError("malformed enum equality hook was invoked")

    def __str__(self):
        raise AssertionError("malformed enum coercion hook was invoked")


class _PrefixOnlyBackend:
    """A result iterator that fails if normalization exceeds its cap."""

    def __init__(self):
        self.consumed = 0

    def search(self, query):
        del query
        for index in range(2):
            self.consumed += 1
            yield {
                "title": f"result {index}",
                "content": "safe hit",
                "url": f"https://a/{index}",
            }
        raise AssertionError("web result iterator consumed beyond max_results")


def _enabled_cfg(**kw) -> WebSearchConfig:
    base = dict(enabled=True, base_url="http://searx.local", timeout_s=0.2, max_results=5)
    base.update(kw)
    return WebSearchConfig(**base)


def _registry_with_web_search(config, backend):
    registry = create_default_capabilities()
    attach_web_search_capability(registry, config, backend=backend)
    return registry


def test_registry_preserves_an_explicit_empty_context_object():
    """Only ``None`` means absent; an explicit dict keeps its identity."""
    context: dict[str, object] = {}
    registry = CapabilityRegistry()
    registry.register(
        "context.identity",
        lambda _query, supplied: CapabilityResult(
            True,
            "same" if supplied is context else "replaced",
        ),
    )

    result = registry.invoke("context.identity", "", context)

    assert result.ok is True
    assert result.text == "same"


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


def test_private_turn_context_vetoes_before_classifier_and_backend():
    """A floated PRIVATE turn dominates a harmless planner-generated query."""
    tripwire = _Tripwire()
    classifier = _CountingClassifier()

    registry = create_default_capabilities()
    attach_web_search_capability(
        registry,
        _enabled_cfg(),
        classify=classifier,
        backend=tripwire,
    )

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        {"sensitivity": PRIVATE},
    )

    assert result.ok is True
    assert classifier.calls == 0
    assert tripwire.calls == 0
    assert result.data["egress"] is False
    assert result.data["sensitivity"] == PRIVATE
    assert result.data["source"] == "corpus"


@pytest.mark.parametrize("marker_value", (True, False, "true", None))
def test_retained_prompt_marker_vetoes_before_classifier_and_backend(marker_value):
    tripwire = _Tripwire()
    classifier = _CountingClassifier()
    registry = create_default_capabilities()
    attach_web_search_capability(
        registry,
        _enabled_cfg(),
        classify=classifier,
        backend=tripwire,
    )
    context = {
        "sensitivity": PUBLIC,
        CLOUD_EGRESS_SCOPE_CONTEXT_KEY: CloudEgressScope.CURRENT_TURN_ONLY,
        "metadata": {RETAINED_PROMPT_CONTEXT_METADATA_KEY: marker_value},
    }

    result = registry.invoke("web.search", "weather in Berlin", context)

    assert result.ok is True
    assert classifier.calls == 0
    assert tripwire.calls == 0
    assert result.data["egress"] is False
    assert result.data["source"] == "corpus"


def test_hostile_nested_metadata_key_fails_closed_without_equality_hook():
    alias = _RetainedMarkerAliasKey()
    tripwire = _Tripwire()
    classifier = _CountingClassifier()
    registry = create_default_capabilities()
    attach_web_search_capability(
        registry,
        _enabled_cfg(),
        classify=classifier,
        backend=tripwire,
    )

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        {"sensitivity": PUBLIC, "metadata": {alias: True}},
    )

    assert result.ok is True
    assert alias.comparisons == 0
    assert classifier.calls == 0
    assert tripwire.calls == 0
    assert result.data["egress"] is False


@pytest.mark.parametrize(
    ("scope", "permitted"),
    (
        (CloudEgressScope.CURRENT_TURN_ONLY, True),
        (CloudEgressScope.LOCAL_ONLY, False),
        (None, False),
        ("current_turn_only", False),
        ("local_only", False),
        (_EqualityBomb(), False),
    ),
    ids=("current", "local", "none", "raw-current", "raw-local", "hostile"),
)
def test_present_cloud_scope_is_exact_and_restrictive(scope, permitted):
    backend = _FakeSearxng(
        [{"title": "result", "content": "safe hit", "url": "https://a/1"}]
    )
    classifier = _CountingClassifier()
    registry = create_default_capabilities()
    attach_web_search_capability(
        registry,
        _enabled_cfg(),
        classify=classifier,
        backend=backend,
    )

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        {
            "sensitivity": PUBLIC,
            CLOUD_EGRESS_SCOPE_CONTEXT_KEY: scope,
            "metadata": {},
        },
    )

    assert result.ok is True
    assert classifier.calls == int(permitted)
    assert backend.queries == (["weather in Berlin"] if permitted else [])
    assert result.data["egress"] is permitted


def test_absent_scope_and_empty_metadata_preserve_direct_web_compatibility():
    backend = _FakeSearxng(
        [{"title": "result", "content": "safe hit", "url": "https://a/1"}]
    )
    registry = _registry_with_web_search(_enabled_cfg(), backend)

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        {"sensitivity": PUBLIC, "metadata": {}},
    )

    assert result.ok is True
    assert backend.queries == ["weather in Berlin"]
    assert result.data["egress"] is True


def test_falsey_private_context_survives_registry_normalization_and_vetoes():
    tripwire = _Tripwire()
    classifier = _CountingClassifier()
    registry = create_default_capabilities()
    attach_web_search_capability(
        registry,
        _enabled_cfg(),
        classify=classifier,
        backend=tripwire,
    )

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        _FalseyPrivateContext(sensitivity=PRIVATE),
    )

    assert result.ok is True
    assert classifier.calls == 0
    assert tripwire.calls == 0
    assert result.data["egress"] is False
    assert result.data["sensitivity"] == PRIVATE
    assert result.data["source"] == "corpus"


def test_dict_subclass_cannot_hide_private_sensitivity_from_veto():
    tripwire = _Tripwire()
    classifier = _CountingClassifier()
    registry = create_default_capabilities()
    attach_web_search_capability(
        registry,
        _enabled_cfg(),
        classify=classifier,
        backend=tripwire,
    )

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        _LyingPrivateContext(sensitivity=PRIVATE),
    )

    assert result.ok is True
    assert classifier.calls == 0
    assert tripwire.calls == 0
    assert result.data["egress"] is False
    assert result.data["sensitivity"] == PRIVATE
    assert result.data["source"] == "corpus"


def test_non_string_context_key_fails_closed_without_equality_hook():
    tripwire = _Tripwire()
    classifier = _CountingClassifier()
    alias = _SensitivityAliasKey()
    registry = create_default_capabilities()
    attach_web_search_capability(
        registry,
        _enabled_cfg(),
        classify=classifier,
        backend=tripwire,
    )

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        {alias: PUBLIC},
    )

    assert result.ok is True
    assert alias.comparisons == 0
    assert classifier.calls == 0
    assert tripwire.calls == 0
    assert result.data["egress"] is False
    assert result.data["sensitivity"] == PRIVATE
    assert result.data["source"] == "corpus"


def test_invocation_observer_metadata_cannot_mutate_private_context_before_veto():
    holder: dict[str, dict[object, object]] = {}

    class _MutatingTaskIdKey:
        def __init__(self) -> None:
            self.comparisons = 0

        def __hash__(self):
            return hash("task_id")

        def __eq__(self, other):
            self.comparisons += 1
            holder["context"].clear()
            return other == "task_id"

    key = _MutatingTaskIdKey()
    context: dict[object, object] = {"sensitivity": PRIVATE, key: object()}
    holder["context"] = context
    events = []
    tripwire = _Tripwire()
    registry = _registry_with_web_search(_enabled_cfg(), tripwire)
    registry.observe_invocations(events.append)

    result = registry.invoke("web.search", "weather in Berlin", context)  # type: ignore[arg-type]

    assert result.ok is True
    assert key.comparisons == 0
    assert context["sensitivity"] == PRIVATE
    assert tripwire.calls == 0
    assert [event.phase for event in events] == [
        "started",
        "started",
        "finished",
        "finished",
    ]
    assert result.data["egress"] is False
    assert result.data["source"] == "corpus"


@pytest.mark.parametrize(
    "invalid",
    [None, True, 1, "", "PRIVATE", "unknown", [], {}, object(), _EqualityBomb()],
    ids=(
        "none",
        "bool",
        "int",
        "empty",
        "case-variant",
        "unknown",
        "list",
        "dict",
        "object",
        "hostile-object",
    ),
)
def test_present_invalid_turn_sensitivity_fails_closed_before_work(invalid):
    tripwire = _Tripwire()
    classifier = _CountingClassifier()

    registry = create_default_capabilities()
    attach_web_search_capability(
        registry,
        _enabled_cfg(),
        classify=classifier,
        backend=tripwire,
    )

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        {"sensitivity": invalid},
    )

    assert result.ok is True
    assert classifier.calls == 0
    assert tripwire.calls == 0
    assert result.data["egress"] is False
    assert result.data["sensitivity"] == PRIVATE
    assert result.data["source"] == "corpus"


@pytest.mark.parametrize(
    ("sensitivity", "query"),
    ((PUBLIC, "weather in Berlin"), (CODE, "debug this Python function")),
)
def test_canonical_non_private_context_still_requires_and_passes_raw_gate(
    sensitivity,
    query,
):
    fake = _FakeSearxng(
        [{"title": "result", "content": "safe hit", "url": "https://a/1"}]
    )
    registry = _registry_with_web_search(_enabled_cfg(), fake)

    result = registry.invoke(
        "web.search",
        query,
        {"sensitivity": sensitivity},
    )

    assert result.ok is True
    assert fake.queries == [query]
    assert result.data["egress"] is True
    assert result.data["source"] == "web"


@pytest.mark.parametrize("sensitivity", (PUBLIC, CODE))
def test_non_private_context_cannot_override_private_raw_query(sensitivity):
    tripwire = _Tripwire()
    registry = _registry_with_web_search(_enabled_cfg(), tripwire)

    result = registry.invoke(
        "web.search",
        "what is my home address",
        {"sensitivity": sensitivity},
    )

    assert result.ok is True
    assert tripwire.calls == 0
    assert result.data["egress"] is False
    assert result.data["sensitivity"] == PRIVATE


@pytest.mark.parametrize(
    ("sensitivity", "raw_context"),
    (
        (PUBLIC, {"mode": Mode.MEETING.value}),
        (CODE, {"intent_kind": IntentKind.COMMAND.value}),
    ),
    ids=("public-meeting", "code-command"),
)
def test_non_private_context_cannot_override_mode_or_intent_gate(
    sensitivity,
    raw_context,
):
    tripwire = _Tripwire()
    registry = _registry_with_web_search(_enabled_cfg(), tripwire)

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        {"sensitivity": sensitivity, **raw_context},
    )

    assert result.ok is True
    assert tripwire.calls == 0
    assert result.data["egress"] is False
    assert result.data["sensitivity"] == PRIVATE


def _classify_false(*_args, **_kwargs):
    return False


def _classify_raises(*_args, **_kwargs):
    raise RuntimeError("classifier unavailable")


@pytest.mark.parametrize(
    "classifier",
    (_classify_false, _classify_raises),
    ids=("denied", "raised"),
)
def test_public_context_never_overrides_raw_classifier_denial(classifier):
    tripwire = _Tripwire()
    registry = create_default_capabilities()
    attach_web_search_capability(
        registry,
        _enabled_cfg(),
        classify=classifier,
        backend=tripwire,
    )

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        {"sensitivity": PUBLIC},
    )

    assert result.ok is True
    assert tripwire.calls == 0
    assert result.data["egress"] is False
    assert result.data["sensitivity"] == PRIVATE


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


def test_max_results_bounds_backend_iterable_consumption():
    backend = _PrefixOnlyBackend()
    registry = _registry_with_web_search(
        _enabled_cfg(max_results=2),
        backend,
    )

    result = registry.invoke("web.search", "weather in Berlin", {})

    assert result.ok is True
    assert backend.consumed == 2
    assert len(result.data["results"]) == 2
    assert result.data["source"] == "web"


# --- BR7: unreachable / blocks-past-timeout => corpus ok=True --------------


def test_unreachable_backend_falls_back_to_corpus_ok_true():
    registry = _registry_with_web_search(_enabled_cfg(), _Unreachable())

    result = registry.invoke("web.search", "what is pipecat", {})

    assert result.ok is True  # never aborts the plan (tasks.py:586-588)
    assert result.data["source"] == "corpus"
    assert result.data["error"] == "ConnectionError"
    # The corpus actually answered (pipecat is in the default corpus).
    assert "pipecat" in result.text.lower()


def test_backend_with_its_own_timeout_returns_corpus_bounded():
    """A fake backend's own timeout yields a bounded, successful fallback.

    This pins provider error handling, not an HTTPX total wall-clock deadline.
    """
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


@pytest.mark.parametrize("field", ("mode", "intent_kind"))
def test_hostile_mode_or_intent_value_uses_historical_none_fallback(field):
    fake = _FakeSearxng(
        [{"title": "result", "content": "safe hit", "url": "https://a/1"}]
    )
    registry = _registry_with_web_search(_enabled_cfg(), fake)

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        {"sensitivity": PUBLIC, field: _EnumBomb()},
    )

    assert result.ok is True
    assert fake.queries == ["weather in Berlin"]
    assert result.data["egress"] is True
    assert result.data["source"] == "web"


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


def test_disabled_config_never_uses_an_injected_backend():
    tripwire = _Tripwire()
    registry = _registry_with_web_search(
        WebSearchConfig(
            enabled=False,
            base_url="http://searx.local",
        ),
        tripwire,
    )

    result = registry.invoke("web.search", "weather in Berlin", {})

    assert result.ok is True
    assert tripwire.calls == 0
    assert result.data["egress"] is False
    assert result.data["source"] == "corpus"


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


def test_malformed_backend_hits_are_skipped_without_aborting_plan():
    registry = _registry_with_web_search(
        _enabled_cfg(),
        _FakeSearxng([None, "not-a-result", 7]),
    )

    result = registry.invoke("web.search", "what is moonshine", {})

    assert result.ok is True
    assert result.data["source"] == "corpus"
    assert result.data["egress"] is True
    assert "moonshine" in result.text.lower()


def test_backend_hit_normalization_error_falls_back_with_egress_receipt():
    registry = _registry_with_web_search(
        _enabled_cfg(),
        _FakeSearxng([_ExplodingHit()]),
    )

    result = registry.invoke("web.search", "what is moonshine", {})

    assert result.ok is True
    assert result.data["source"] == "corpus"
    assert result.data["egress"] is True
    assert result.data["error"] == "RuntimeError"
    assert "moonshine" in result.text.lower()
