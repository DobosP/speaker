"""Tests for the pluggable ``web.search`` capability (core/websearch.py).

These pin the §9.7 egress boundary and the never-raise corpus fallback. They
need no network and no real SearXNG -- custom backends and HTTPX's in-process
``MockTransport`` keep every path deterministic and offline.

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

import json
import threading
import time

import httpx
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
    _BackendInputError,
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


class _EnabledAliasKey:
    """A non-string config key must not impersonate ``enabled``."""

    def __init__(self):
        self.comparisons = 0

    def __hash__(self):
        return hash("enabled")

    def __eq__(self, other):
        self.comparisons += 1
        raise AssertionError("config-key equality hook was invoked")


class _TitleAliasKey:
    """A non-string result key must not impersonate ``title``."""

    def __init__(self):
        self.comparisons = 0

    def __hash__(self):
        return hash("title")

    def __eq__(self, other):
        self.comparisons += 1
        raise AssertionError("result-key equality hook was invoked")


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


class _TrackedByteStream(httpx.SyncByteStream):
    """Deterministic HTTPX stream with observable reads and closure."""

    def __init__(
        self,
        chunks,
        *,
        before_yield=None,
        after_yield=None,
        after_exhaustion=None,
    ):
        self._chunks = tuple(chunks)
        self._before_yield = before_yield
        self._after_yield = after_yield
        self._after_exhaustion = after_exhaustion
        self.yielded = 0
        self.closed = False

    def __iter__(self):
        for index, chunk in enumerate(self._chunks):
            if self._before_yield is not None:
                self._before_yield(index)
            self.yielded += 1
            yield chunk
            if self._after_yield is not None:
                self._after_yield(index)
        if self._after_exhaustion is not None:
            self._after_exhaustion()

    def close(self):
        self.closed = True


class _RaisingByteStream(httpx.SyncByteStream):
    def __init__(self, message):
        self._message = message
        self.closed = False
        self.yielded = 0

    def __iter__(self):
        self.yielded += 1
        yield b"{"
        raise httpx.ReadError(self._message)

    def close(self):
        self.closed = True


class _FakeClock:
    def __init__(self, now=0.0):
        self.now = float(now)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


class _ExpiringClock:
    def __init__(self, expire_call, *, expired=1.0):
        self.expire_call = expire_call
        self.expired = expired
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.expired if self.calls >= self.expire_call else 0.0


class _HostileText(str):
    """A str subclass that must be rejected before string operations."""

    def __str__(self):
        raise AssertionError("hostile string coercion hook was invoked")

    def strip(self, *args, **kwargs):
        raise AssertionError("hostile string strip hook was invoked")

    def __getitem__(self, item):
        raise AssertionError("hostile string slicing hook was invoked")


class _HostileHit(dict):
    """A dict subclass that must be rejected before virtual lookup."""

    def get(self, key, default=None):
        raise AssertionError("hostile hit lookup hook was invoked")


def _json_body(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def _mock_searx_backend(
    body: bytes,
    *,
    headers=None,
    chunks=None,
    stream=None,
    clock=None,
    on_request=None,
    status_code=200,
    **backend_kwargs,
):
    """Return a shipped backend driven only by HTTPX's in-process transport."""

    requests = []
    if stream is None:
        stream = _TrackedByteStream(chunks if chunks is not None else (body,))
    response_headers = (
        [("Content-Type", "application/json")] if headers is None else headers
    )

    def handler(request):
        requests.append(request)
        if on_request is not None:
            on_request(request)
        return httpx.Response(status_code, headers=response_headers, stream=stream)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = SearxngBackend(
        "http://searx.local/",
        stream_factory=client.stream,
        clock=clock or time.monotonic,
        **backend_kwargs,
    )
    return backend, client, stream, requests


def _invoke_shipped_backend(backend, *, context=None, config=None):
    registry = _registry_with_web_search(config or _enabled_cfg(), backend)
    return registry.invoke(
        "web.search",
        "weather in Berlin",
        {} if context is None else context,
    )


def _enabled_cfg(**kw) -> WebSearchConfig:
    base = dict(
        enabled=True, base_url="http://searx.local", timeout_s=0.2, max_results=5
    )
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


def test_max_results_does_not_scan_past_a_malformed_prefix():
    class _MalformedPrefixBackend:
        def __init__(self):
            self.consumed = 0

        def search(self, query):
            del query
            self.consumed += 1
            yield {"title": 123, "content": "invalid", "url": ""}
            self.consumed += 1
            raise AssertionError("provider consumed past its one-hit prefix")

    backend = _MalformedPrefixBackend()
    result = _registry_with_web_search(
        _enabled_cfg(max_results=1),
        backend,
    ).invoke("web.search", "weather in Berlin", {})

    assert result.ok is True
    assert backend.consumed == 1
    assert result.data["source"] == "corpus"
    assert result.data["egress"] is True
    assert "error" not in result.data


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


def test_backend_dict_subclass_is_skipped_without_virtual_lookup():
    registry = _registry_with_web_search(
        _enabled_cfg(),
        _FakeSearxng([_ExplodingHit()]),
    )

    result = registry.invoke("web.search", "what is moonshine", {})

    assert result.ok is True
    assert result.data["source"] == "corpus"
    assert result.data["egress"] is True
    assert "error" not in result.data
    assert "moonshine" in result.text.lower()


# --- bounded shipped SearXNG ingestion ------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ({"enabled": True}, True),
        ({"enabled": False}, False),
        ({"enabled": 1}, False),
        ({"enabled": "true"}, False),
        ({"enabled": _EqualityBomb()}, False),
    ),
    ids=("true", "false", "int", "string", "hostile"),
)
def test_config_enabled_requires_exact_builtin_true(raw, expected):
    assert WebSearchConfig.from_dict(raw).enabled is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("  http://searx.local/  ", "http://searx.local/"),
        ("", ""),
        (None, ""),
        (1, ""),
        (True, ""),
        (_HostileText("http://hostile.invalid"), ""),
    ),
    ids=("stripped", "empty", "none", "int", "bool", "subclass"),
)
def test_config_base_url_requires_exact_builtin_string(raw, expected):
    assert WebSearchConfig.from_dict({"base_url": raw}).base_url == expected


@pytest.mark.parametrize(
    ("field", "default", "low", "high"),
    (
        ("timeout_s", 4.0, 0.05, 15.0),
        ("total_timeout_s", 8.0, 0.05, 20.0),
        ("max_response_bytes", 262_144, 1_024, 1_048_576),
        ("max_results", 5, 1, 8),
        ("max_title_chars", 256, 1, 512),
        ("max_summary_chars", 2_048, 1, 4_096),
        ("max_url_chars", 2_048, 1, 2_048),
        ("max_output_chars", 16_384, 1, 32_768),
    ),
)
def test_config_numeric_fields_default_and_clamp_in_both_construction_paths(
    field,
    default,
    low,
    high,
):
    for raw in (None, True, False, "7", float("nan"), float("inf"), object()):
        assert getattr(WebSearchConfig.from_dict({field: raw}), field) == default
        assert getattr(WebSearchConfig(**{field: raw}), field) == default

    assert getattr(WebSearchConfig.from_dict({field: -1}), field) == low
    assert getattr(WebSearchConfig(**{field: -1}), field) == low
    assert getattr(WebSearchConfig.from_dict({field: 10**9}), field) == high
    assert getattr(WebSearchConfig(**{field: 10**9}), field) == high


def test_config_direct_construction_applies_exact_bool_and_string_rules():
    cfg = WebSearchConfig(enabled="true", base_url=_HostileText("unsafe"))

    assert cfg.enabled is False
    assert cfg.base_url == ""


def test_config_from_dict_rejects_malformed_container_without_virtual_hooks():
    hostile = _HostileHit(enabled=True, base_url="http://unsafe.invalid")

    assert WebSearchConfig.from_dict(hostile) == WebSearchConfig()
    assert WebSearchConfig.from_dict([("enabled", True)]) == WebSearchConfig()


def test_config_non_string_alias_key_cannot_enable_or_run_equality_hooks():
    alias = _EnabledAliasKey()

    config = WebSearchConfig.from_dict({alias: True})

    assert config == WebSearchConfig()
    assert alias.comparisons == 0


def test_config_integer_caps_reject_float_while_timeouts_accept_builtin_int():
    cfg = WebSearchConfig(
        timeout_s=2,
        total_timeout_s=3,
        max_results=3.5,
        max_response_bytes=2_048.0,
    )

    assert cfg.timeout_s == 2.0
    assert cfg.total_timeout_s == 3.0
    assert cfg.max_results == 5
    assert cfg.max_response_bytes == 262_144


def test_config_all_explicit_values_survive_inside_bounds():
    raw = {
        "enabled": True,
        "base_url": " http://searx.local/ ",
        "timeout_s": 1.25,
        "total_timeout_s": 2.5,
        "max_response_bytes": 4_096,
        "max_results": 3,
        "max_title_chars": 12,
        "max_summary_chars": 34,
        "max_url_chars": 56,
        "max_output_chars": 78,
    }

    cfg = WebSearchConfig.from_dict(raw)

    assert cfg == WebSearchConfig(
        enabled=True,
        base_url="http://searx.local/",
        timeout_s=1.25,
        total_timeout_s=2.5,
        max_response_bytes=4_096,
        max_results=3,
        max_title_chars=12,
        max_summary_chars=34,
        max_url_chars=56,
        max_output_chars=78,
    )


def test_httpx_runtime_is_the_direct_pinned_version():
    assert httpx.__version__ == "0.28.1"


def test_shipped_backend_uses_raw_identity_stream_and_phase_timeout():
    body = _json_body(
        {
            "results": [
                {
                    "title": "Berlin weather",
                    "content": "Sunny",
                    "url": "https://weather.invalid/berlin",
                }
            ]
        }
    )
    backend, client, stream, requests = _mock_searx_backend(
        body,
        headers=[
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Encoding", "identity"),
            ("Content-Length", str(len(body))),
        ],
        timeout_s=0.25,
        total_timeout_s=1.0,
        max_response_bytes=1_024,
    )
    try:
        hits = backend.search("weather in Berlin")
    finally:
        client.close()

    assert hits == [
        {
            "title": "Berlin weather",
            "content": "Sunny",
            "url": "https://weather.invalid/berlin",
        }
    ]
    assert stream.closed is True
    assert stream.yielded == 1
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/search"
    assert request.url.params["q"] == "weather in Berlin"
    assert request.url.params["format"] == "json"
    assert request.headers["accept"] == "application/json"
    assert request.headers["accept-encoding"] == "identity"
    assert request.extensions["timeout"] == {
        "connect": 0.25,
        "read": 0.25,
        "write": 0.25,
        "pool": 0.25,
    }


def test_shipped_backend_calls_iter_raw_without_chunk_size(monkeypatch):
    calls = []
    original = httpx.Response.iter_raw

    def tracked_iter_raw(response, *args, **kwargs):
        calls.append((args, kwargs))
        return original(response, *args, **kwargs)

    monkeypatch.setattr(httpx.Response, "iter_raw", tracked_iter_raw)
    body = _json_body({"results": []})
    backend, client, stream, _requests = _mock_searx_backend(
        body,
        chunks=(body[:1], body[1:]),
    )
    try:
        assert backend.search("weather") == []
    finally:
        client.close()

    assert calls == [((), {})]
    assert stream.yielded == 2
    assert stream.closed is True


@pytest.mark.parametrize(
    ("headers", "error"),
    (
        ([], "invalid_content_type"),
        ([("Content-Type", "text/plain")], "invalid_content_type"),
        (
            [
                ("Content-Type", "application/json"),
                ("Content-Encoding", "gzip"),
            ],
            "unsupported_content_encoding",
        ),
        (
            [
                ("Content-Type", "application/json"),
                ("Content-Encoding", "identity"),
                ("Content-Encoding", "identity"),
            ],
            "unsupported_content_encoding",
        ),
        (
            [
                ("Content-Type", "application/json"),
                ("Content-Length", "not-decimal"),
            ],
            "invalid_content_length",
        ),
        (
            [
                ("Content-Type", "application/json"),
                ("Content-Length", "+10"),
            ],
            "invalid_content_length",
        ),
        (
            [
                ("Content-Type", "application/json"),
                ("Content-Length", "10"),
                ("Content-Length", "10"),
            ],
            "invalid_content_length",
        ),
        (
            [
                ("Content-Type", "application/json"),
                ("Content-Length", "1025"),
            ],
            "response_too_large",
        ),
    ),
    ids=(
        "missing-content-type",
        "wrong-content-type",
        "compressed",
        "duplicate-identity",
        "nondigit-length",
        "signed-length",
        "duplicate-length",
        "declared-too-large",
    ),
)
def test_shipped_backend_rejects_unsafe_headers_and_closes(headers, error):
    body = _json_body({"results": []})
    backend, client, stream, _requests = _mock_searx_backend(
        body,
        headers=headers,
        max_response_bytes=1_024,
    )
    try:
        result = _invoke_shipped_backend(backend)
    finally:
        client.close()

    assert result.ok is True
    assert result.data["source"] == "corpus"
    assert result.data["egress"] is True
    assert result.data["error"] == error
    assert stream.closed is True


def test_very_long_decimal_content_length_has_a_stable_bounded_error():
    backend, client, stream, _requests = _mock_searx_backend(
        _json_body({"results": []}),
        headers=[
            ("Content-Type", "application/json"),
            ("Content-Length", "9" * 5_000),
        ],
        max_response_bytes=1_024,
    )
    try:
        result = _invoke_shipped_backend(backend)
    finally:
        client.close()

    assert result.data["error"] == "response_too_large"
    assert result.data["egress"] is True
    assert stream.yielded == 0
    assert stream.closed is True


def test_shipped_backend_accepts_exact_body_cap_and_rejects_one_byte_over():
    prefix = b'{"results":[]}'
    exact = prefix + (b" " * (1_024 - len(prefix)))
    over = exact + b" "

    accepted, accepted_client, accepted_stream, _ = _mock_searx_backend(
        exact,
        chunks=(exact[:511], exact[511:]),
        max_response_bytes=1_024,
    )
    rejected, rejected_client, rejected_stream, _ = _mock_searx_backend(
        over,
        chunks=(over[:1_024], over[1_024:]),
        max_response_bytes=1_024,
    )
    try:
        assert accepted.search("weather") == []
        result = _invoke_shipped_backend(rejected)
    finally:
        accepted_client.close()
        rejected_client.close()

    assert accepted_stream.yielded == 2
    assert accepted_stream.closed is True
    assert result.data["error"] == "response_too_large"
    assert rejected_stream.yielded == 2
    assert rejected_stream.closed is True


@pytest.mark.parametrize(
    ("body", "error"),
    (
        (b"\xff", "invalid_utf8"),
        (b"\xef\xbb\xbf" + _json_body({"results": []}), "invalid_utf8"),
        (b'{"results":', "invalid_json"),
        (b'{"results":[],"results":[]}', "invalid_json"),
        (b'{"results":[],"value":NaN}', "invalid_json"),
        (b"[" * 10_000 + b"]" * 10_000, "invalid_json"),
        (b"[]", "invalid_payload"),
        (b"{}", "invalid_payload"),
        (b'{"results":{}}', "invalid_payload"),
        (b'{"results":[null]}', "invalid_payload"),
        (b'{"results":[{"title":1}]}', "invalid_payload"),
    ),
    ids=(
        "invalid-utf8",
        "bom",
        "truncated-json",
        "duplicate-key",
        "nan",
        "excessive-nesting",
        "top-level-list",
        "missing-results",
        "results-not-list",
        "hit-not-dict",
        "field-not-string",
    ),
)
def test_shipped_backend_strictly_rejects_malformed_payloads(body, error):
    backend, client, stream, _requests = _mock_searx_backend(body)
    try:
        result = _invoke_shipped_backend(backend)
    finally:
        client.close()

    assert result.ok is True
    assert result.data["source"] == "corpus"
    assert result.data["egress"] is True
    assert result.data["error"] == error
    assert stream.closed is True


def test_shipped_backend_allows_missing_and_null_hit_fields():
    body = _json_body(
        {
            "results": [
                {"title": None, "content": "summary"},
                {"title": "name", "content": None, "url": None},
                {},
            ]
        }
    )
    backend, client, stream, _requests = _mock_searx_backend(body)
    try:
        hits = backend.search("weather")
    finally:
        client.close()

    assert hits == [
        {"title": "", "content": "summary", "url": ""},
        {"title": "name", "content": "", "url": ""},
        {"title": "", "content": "", "url": ""},
    ]
    assert stream.closed is True


def test_shipped_backend_never_normalizes_or_validates_past_result_prefix():
    body = _json_body(
        {
            "results": [
                {"title": "usable", "content": "first", "url": "https://one/"},
                {"title": 7, "content": "malformed suffix"},
            ]
        }
    )
    backend, client, stream, _requests = _mock_searx_backend(body)
    try:
        result = _invoke_shipped_backend(
            backend,
            config=_enabled_cfg(max_results=1),
        )
    finally:
        client.close()

    assert result.data["source"] == "web"
    assert result.data["results"] == [{"name": "usable", "summary": "first"}]
    assert result.citations == ("https://one/",)
    assert stream.closed is True


@pytest.mark.parametrize("checkpoint", ("headers", "chunk", "eof"))
def test_shipped_backend_cooperative_total_deadline_checkpoints(checkpoint):
    clock = _FakeClock()
    body = _json_body({"results": []})

    def on_request(_request):
        if checkpoint == "headers":
            clock.advance(1.0)

    def before_yield(_index):
        if checkpoint == "chunk":
            clock.advance(1.0)

    def after_exhaustion():
        if checkpoint == "eof":
            clock.advance(1.0)

    stream = _TrackedByteStream(
        (body,),
        before_yield=before_yield,
        after_exhaustion=after_exhaustion,
    )
    backend, client, stream, requests = _mock_searx_backend(
        body,
        stream=stream,
        clock=clock,
        on_request=on_request,
        total_timeout_s=0.05,
    )
    try:
        result = _invoke_shipped_backend(backend)
    finally:
        client.close()

    assert result.data["error"] == "deadline_exceeded"
    assert result.data["egress"] is True
    assert len(requests) == 1
    assert stream.closed is True


def test_shipped_backend_deadline_at_exact_boundary_expires():
    clock = _FakeClock()

    def on_request(_request):
        clock.advance(0.05)

    backend, client, stream, requests = _mock_searx_backend(
        _json_body({"results": []}),
        clock=clock,
        on_request=on_request,
        total_timeout_s=0.05,
    )
    try:
        result = _invoke_shipped_backend(backend)
    finally:
        client.close()

    assert result.data["error"] == "deadline_exceeded"
    assert result.data["egress"] is True
    assert len(requests) == 1
    assert stream.closed is True


def test_shipped_backend_deadline_before_open_has_no_egress():
    clock = _ExpiringClock(2)
    backend, client, stream, requests = _mock_searx_backend(
        _json_body({"results": []}),
        clock=clock,
        total_timeout_s=0.05,
    )
    try:
        result = _invoke_shipped_backend(backend)
    finally:
        client.close()

    assert result.data["error"] == "deadline_exceeded"
    assert result.data["egress"] is False
    assert requests == []
    assert stream.yielded == 0


@pytest.mark.parametrize("expire_call", (9, 10), ids=("after-utf8", "after-json"))
def test_shipped_backend_checks_deadline_around_decoding_and_json(expire_call):
    clock = _ExpiringClock(expire_call)
    backend, client, stream, requests = _mock_searx_backend(
        _json_body({"results": []}),
        clock=clock,
        total_timeout_s=0.05,
    )
    try:
        result = _invoke_shipped_backend(backend)
    finally:
        client.close()

    assert result.data["error"] == "deadline_exceeded"
    assert result.data["egress"] is True
    assert len(requests) == 1
    assert stream.closed is True


@pytest.mark.parametrize(
    "cancel_check",
    (7, 8, 9),
    ids=("before-utf8", "before-json", "after-json"),
)
def test_shipped_backend_checks_cancel_around_decoding_and_json(cancel_check):
    cancel = threading.Event()
    calls = 0

    def is_set():
        nonlocal calls
        calls += 1
        return calls >= cancel_check

    cancel.is_set = is_set
    backend, client, stream, requests = _mock_searx_backend(_json_body({"results": []}))
    try:
        result = _invoke_shipped_backend(backend, context={"cancel_event": cancel})
    finally:
        client.close()

    assert result.data["error"] == "cancelled"
    assert result.data["egress"] is True
    assert len(requests) == 1
    assert stream.closed is True


def test_shipped_backend_checks_cancel_before_request_without_egress():
    cancel = threading.Event()
    cancel.set()
    backend, client, stream, requests = _mock_searx_backend(_json_body({"results": []}))
    try:
        result = _invoke_shipped_backend(backend, context={"cancel_event": cancel})
    finally:
        client.close()

    assert result.ok is True
    assert result.data["source"] == "corpus"
    assert result.data["egress"] is False
    assert result.data["error"] == "cancelled"
    assert requests == []
    assert stream.yielded == 0


def test_shipped_backend_checks_cancel_after_headers_before_first_read():
    cancel = threading.Event()

    def on_request(_request):
        cancel.set()

    backend, client, stream, requests = _mock_searx_backend(
        _json_body({"results": []}),
        on_request=on_request,
    )
    try:
        result = _invoke_shipped_backend(backend, context={"cancel_event": cancel})
    finally:
        client.close()

    assert result.data["error"] == "cancelled"
    assert result.data["egress"] is True
    assert len(requests) == 1
    assert stream.yielded == 0
    assert stream.closed is True


@pytest.mark.parametrize("checkpoint", ("chunk", "eof"))
def test_shipped_backend_checks_cancel_during_and_after_raw_stream(checkpoint):
    cancel = threading.Event()
    body = _json_body({"results": []})

    def before_yield(_index):
        if checkpoint == "chunk":
            cancel.set()

    def after_exhaustion():
        if checkpoint == "eof":
            cancel.set()

    stream = _TrackedByteStream(
        (body,),
        before_yield=before_yield,
        after_exhaustion=after_exhaustion,
    )
    backend, client, stream, requests = _mock_searx_backend(body, stream=stream)
    try:
        result = _invoke_shipped_backend(backend, context={"cancel_event": cancel})
    finally:
        client.close()

    assert result.data["error"] == "cancelled"
    assert result.data["egress"] is True
    assert len(requests) == 1
    assert stream.closed is True


def test_non_event_cancel_object_is_not_polled_or_forwarded():
    class _CancelBomb:
        def is_set(self):
            raise AssertionError("non-Event cancellation hook was invoked")

    backend, client, stream, requests = _mock_searx_backend(
        _json_body({"results": [{"title": "weather", "content": "sunny", "url": ""}]})
    )
    try:
        result = _invoke_shipped_backend(
            backend,
            context={"cancel_event": _CancelBomb()},
        )
    finally:
        client.close()

    assert result.data["source"] == "web"
    assert len(requests) == 1
    assert stream.closed is True


def test_sync_raw_read_is_only_checked_after_it_returns():
    """The cooperative budget does not claim to preempt a blocked ``next``."""
    clock = _FakeClock()
    body = _json_body({"results": []})

    def entered_sync_read(_index):
        clock.advance(1.0)

    stream = _TrackedByteStream((body,), before_yield=entered_sync_read)
    backend, client, stream, _requests = _mock_searx_backend(
        body,
        stream=stream,
        clock=clock,
        total_timeout_s=0.05,
    )
    try:
        result = _invoke_shipped_backend(backend)
    finally:
        client.close()

    assert stream.yielded == 1
    assert result.data["error"] == "deadline_exceeded"


def test_shipped_backend_error_is_stable_and_does_not_log_response_body(caplog):
    secret = "should-never-appear-in-safe-output"
    body = ("{" + secret).encode()
    backend, client, stream, _requests = _mock_searx_backend(body)
    try:
        with caplog.at_level("INFO", logger="speaker.websearch"):
            result = _invoke_shipped_backend(backend)
    finally:
        client.close()

    assert result.data["error"] == "invalid_json"
    assert secret not in caplog.text
    assert secret not in result.text
    assert stream.closed is True


def test_transport_read_failure_closes_response_and_hides_detail(caplog):
    secret = "private-transport-diagnostic"
    stream = _RaisingByteStream(secret)
    backend, client, stream, _requests = _mock_searx_backend(
        b"",
        stream=stream,
    )
    try:
        with caplog.at_level("INFO", logger="speaker.websearch"):
            result = _invoke_shipped_backend(backend)
    finally:
        client.close()

    assert result.data["error"] == "ReadError"
    assert result.data["egress"] is True
    assert stream.yielded == 1
    assert stream.closed is True
    assert secret not in caplog.text


def test_http_status_failure_closes_response_and_falls_back_truthfully():
    backend, client, stream, requests = _mock_searx_backend(
        b"unused",
        status_code=503,
    )
    try:
        result = _invoke_shipped_backend(backend)
    finally:
        client.close()

    assert result.ok is True
    assert result.data["source"] == "corpus"
    assert result.data["egress"] is True
    assert result.data["error"] == "HTTPStatusError"
    assert len(requests) == 1
    assert stream.yielded == 0
    assert stream.closed is True


def test_stream_factory_failure_counts_as_attempted_egress_without_detail(caplog):
    secret = "private-open-detail"
    calls = []

    def fail_open(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError(secret)

    backend = SearxngBackend(
        "http://searx.local",
        stream_factory=fail_open,
    )
    with caplog.at_level("INFO", logger="speaker.websearch"):
        result = _invoke_shipped_backend(backend)

    assert len(calls) == 1
    assert result.ok is True
    assert result.data["source"] == "corpus"
    assert result.data["egress"] is True
    assert result.data["error"] == "RuntimeError"
    assert secret not in caplog.text


def test_custom_backend_cannot_forge_shipped_error_or_pre_attempt_receipt():
    class _ForgingBackend:
        def search(self, _query):
            raise _BackendInputError("cancelled", egress=False)

    result = _registry_with_web_search(_enabled_cfg(), _ForgingBackend()).invoke(
        "web.search", "weather in Berlin", {}
    )

    assert result.ok is True
    assert result.data["source"] == "corpus"
    assert result.data["egress"] is True
    assert result.data["error"] == "_BackendInputError"


def test_disabled_and_private_paths_never_open_shipped_transport():
    body = _json_body({"results": []})
    backend, client, _stream, requests = _mock_searx_backend(body)
    try:
        disabled = _invoke_shipped_backend(
            backend,
            config=_enabled_cfg(enabled=False),
        )
        private = _invoke_shipped_backend(
            backend,
            context={"sensitivity": PRIVATE},
        )
    finally:
        client.close()

    assert disabled.data["egress"] is False
    assert private.data["egress"] is False
    assert requests == []


def test_legacy_one_argument_backend_keeps_compatibility_with_cancel_context():
    backend = _FakeSearxng(
        [{"title": "weather", "content": "sunny", "url": "https://a/1"}]
    )
    registry = _registry_with_web_search(_enabled_cfg(), backend)

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        {"cancel_event": threading.Event()},
    )

    assert result.data["source"] == "web"
    assert backend.queries == ["weather in Berlin"]


def test_searxng_subclass_remains_a_legacy_one_argument_backend():
    class _CustomSearxng(SearxngBackend):
        def __init__(self):
            self.queries = []

        def search(self, query):
            self.queries.append(query)
            return [{"title": "weather", "content": "sunny", "url": ""}]

    backend = _CustomSearxng()
    registry = _registry_with_web_search(_enabled_cfg(), backend)

    result = registry.invoke(
        "web.search",
        "weather in Berlin",
        {"cancel_event": threading.Event()},
    )

    assert result.data["source"] == "web"
    assert backend.queries == ["weather in Berlin"]


def test_result_fields_and_aggregate_output_are_bounded_before_normalization():
    backend = _FakeSearxng(
        [
            {
                "title": "  ABCDEFG",
                "content": "  123456789",
                "url": "https://example.invalid/very/long/path",
            },
            {
                "title": "SECOND",
                "content": "ANOTHER SUMMARY",
                "url": "https://second.invalid/path",
            },
        ]
    )
    config = _enabled_cfg(
        max_results=2,
        max_title_chars=3,
        max_summary_chars=4,
        max_url_chars=8,
        max_output_chars=5,
    )

    result = _registry_with_web_search(config, backend).invoke(
        "web.search", "weather in Berlin", {}
    )

    assert result.ok is True
    assert len(result.data["results"]) <= 2
    assert all(len(hit["name"]) <= 3 for hit in result.data["results"])
    assert all(len(hit["summary"]) <= 4 for hit in result.data["results"])
    assert all(len(url) <= 8 for url in result.citations)
    assert len(result.text) <= 5
    # Slice-before-strip: the first three title bytes are two spaces + ``A``.
    assert result.data["results"][0]["name"] == "A"


def test_exact_builtin_hit_and_field_types_avoid_hostile_hooks():
    backend = _FakeSearxng(
        [
            _HostileHit(title="bad", content="bad", url="bad"),
            {
                "title": _HostileText("bad"),
                "content": _HostileText("bad"),
                "url": _HostileText("bad"),
            },
            {"title": "safe", "content": "summary", "url": "https://a/1"},
        ]
    )

    result = _registry_with_web_search(_enabled_cfg(), backend).invoke(
        "web.search", "weather in Berlin", {}
    )

    assert result.ok is True
    assert result.data["source"] == "web"
    assert result.data["results"] == [{"name": "safe", "summary": "summary"}]
    assert result.citations == ("https://a/1",)


def test_non_string_result_key_cannot_impersonate_field_or_run_hooks():
    alias = _TitleAliasKey()
    backend = _FakeSearxng(
        [
            {alias: "aliased", "content": "unsafe"},
            {"title": "safe", "content": "summary", "url": "https://a/1"},
        ]
    )

    result = _registry_with_web_search(_enabled_cfg(), backend).invoke(
        "web.search", "weather in Berlin", {}
    )

    assert result.data["results"] == [{"name": "safe", "summary": "summary"}]
    assert alias.comparisons == 0
