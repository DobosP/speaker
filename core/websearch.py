"""Pluggable ``web.search`` capability backed by self-hosted SearXNG.

This is the P3 "real web research" surface (`docs/p3_design.md` §1, Locked
Decision 3). It lives in ``core/`` -- not ``always_on_agent/`` (which stays
``core``-free per ``always_on_agent/react.py:9-12``) -- so it can call
``core.sensitivity.may_leave_device`` directly and enforce the §9.7 data
boundary *before* any network call.

Shape:

- :class:`WebSearchConfig` mirrors ``core.capabilities.RecallConfig`` (a flat
  ``web_search`` config block with a ``from_dict`` factory).
- :class:`Backend` is a small ``Protocol`` so the SearXNG backend is pluggable
  (Decision 3); :class:`SearxngBackend` is the shipped implementation (lazy
  ``httpx`` import à la ``core/llm.py``'s lazy ``openai``).
- :func:`attach_web_search_capability` registers a ``web.search`` provider whose
  closure enforces the order **GUARD-COERCE -> gate -> SearXNG -> corpus
  fallback** and **never raises** (a non-ok step aborts the whole plan in
  ``always_on_agent/tasks.py:229-231``, so a raised exception or an ``ok=False``
  here would silently break RESEARCH/SEARCH plans).

The result mirrors the corpus ``search()`` shape
(``always_on_agent/capabilities.py:75-80``): ``data["results"]`` is a list of
``{"name", "summary"}`` and ``citations`` are the source URLs, so a downstream
synthesis step (``research.local``) consumes web hits exactly like corpus hits.
Audit stamps ``data["egress"]`` / ``data["sensitivity"]`` record whether the
query was permitted to leave the device.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult
from always_on_agent.events import Mode
from always_on_agent.models import IntentKind

from .sensitivity import may_leave_device

log = logging.getLogger("speaker.websearch")


@dataclass(frozen=True)
class WebSearchConfig:
    """Gating + connection settings for the ``web.search`` capability (the flat
    ``web_search`` config block).

    ``enabled`` defaults to **false** so the fully-local corpus path is the
    out-of-the-box behaviour -- a user opts in by setting ``enabled`` AND a
    ``base_url`` pointing at their self-hosted SearXNG. ``timeout_s`` is the
    bounded connect+read budget (BR7): a blocking ``httpx`` GET cannot poll the
    brain's ``cancel_event``, so this timeout is the only wedge bound. ``max_results``
    caps how many hits are mapped into the corpus-compatible result shape.
    """

    enabled: bool = False
    base_url: str = ""
    timeout_s: float = 4.0
    max_results: int = 5

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "WebSearchConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            base_url=str(data.get("base_url", "") or "").strip(),
            timeout_s=float(data.get("timeout_s", 4.0) or 4.0),
            max_results=int(data.get("max_results", 5) or 5),
        )


@runtime_checkable
class Backend(Protocol):
    """A pluggable web-search backend (Decision 3).

    Returns a list of ``{"title", "content", "url"}`` mappings (the SearXNG
    JSON result shape). MUST raise on a network/transport error so the provider
    closure can fall back to the corpus -- the closure, never the backend, owns
    the never-raise guarantee.
    """

    def search(self, query: str) -> Sequence[Mapping[str, object]]: ...


class SearxngBackend:
    """Query a self-hosted SearXNG instance over its JSON API.

    ``httpx`` is imported lazily (mirroring ``core/llm.py``'s lazy ``openai`` /
    ``ollama``) so the runtime and the logic test suite work without it
    installed -- it is only needed when web search is actually enabled.

    The GET is ``{base_url}/search?q=<query>&format=json`` bounded by a single
    connect+read ``timeout_s`` (BR7): a blocking GET can't poll ``cancel_event``,
    so the timeout is the wedge bound. SearXNG ``results[]`` entries
    (``{title, content, url}``) are returned raw; the provider closure maps them
    into the corpus ``search()`` shape.
    """

    def __init__(self, base_url: str, *, timeout_s: float = 4.0):
        # Strip a trailing slash so ``{base_url}/search`` never doubles it.
        self._base_url = (base_url or "").rstrip("/")
        self._timeout_s = float(timeout_s)

    def search(self, query: str) -> Sequence[Mapping[str, object]]:
        import httpx  # lazy: only needed when web search is enabled

        # One bounded connect+read budget. httpx.Timeout(x) applies x to every
        # phase (connect/read/write/pool), so a stalled SearXNG can't wedge the
        # turn past timeout_s (BR7).
        resp = httpx.get(
            f"{self._base_url}/search",
            params={"q": query, "format": "json"},
            timeout=httpx.Timeout(self._timeout_s),
        )
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results") if isinstance(payload, Mapping) else None
        return list(results or [])


def _coerce_mode(raw: object) -> Optional[Mode]:
    """Fail-safe enum coercion (BR2): never raise on an out-of-vocab value.

    The brain publishes ``context['mode']`` as a string (``task.mode.value``).
    An unguarded ``Mode(raw)`` would raise ``ValueError`` on a bad value, which
    ``CapabilityRegistry.invoke`` turns into ``ok=False`` -> aborts the plan, or
    worse skips the gate. Defaulting to ``None`` keeps the gate running (it then
    decides purely on the query text + intent, failing safe to PRIVATE on PII)."""
    if raw is None:
        return None
    if isinstance(raw, Mode):
        return raw
    try:
        return Mode(raw)
    except (ValueError, TypeError):
        return None


def _coerce_intent(raw: object) -> Optional[IntentKind]:
    """Fail-safe enum coercion (BR2) for ``context['intent_kind']`` (a string
    ``task.intent.value``). See :func:`_coerce_mode`."""
    if raw is None:
        return None
    if isinstance(raw, IntentKind):
        return raw
    try:
        return IntentKind(raw)
    except (ValueError, TypeError):
        return None


def attach_web_search_capability(
    registry: CapabilityRegistry,
    config: WebSearchConfig,
    *,
    classify=may_leave_device,
    backend: Optional[Backend] = None,
    fallback_capability: str = "search.local",
) -> CapabilityRegistry:
    """Register a ``web.search`` provider on top of the corpus ``search.local``.

    The provider closure enforces, in order:

    1. **GUARD-COERCE** ``mode`` / ``intent_kind`` from ``context`` via a
       fail-safe try/except (-> ``None`` on a bad value, BR2) so a bogus enum
       string never aborts the plan or bypasses the gate.
    2. **Gate FIRST** (``classify`` == :func:`core.sensitivity.may_leave_device`)
       on the raw ``query`` arg -- NOT a trusted ``context['sensitivity']`` tag
       (which is set only on the assistant path and may be absent for
       gather/ReAct steps). A PRIVATE/personal query is hard-blocked: SearXNG is
       never called and the corpus answers with ``data["egress"]=False``.
    3. If permitted AND web search is enabled AND a backend/``base_url`` exists,
       call the backend; map ``{title, content, url}`` -> corpus
       ``{name, summary}`` + ``citations``.
    4. On empty results / network error / import error, fall back to the corpus
       with ``ok=True`` (BR7: an unreachable or slow SearXNG must not abort the
       plan) and stamp ``data["source"]`` / ``data["error"]``.

    NEVER raises and NEVER returns ``ok=False`` from a fallback path: a non-ok
    step aborts the whole plan (``always_on_agent/tasks.py:229-231``), so the
    corpus fallback must keep the plan alive.

    ``classify`` / ``backend`` are injectable for tests. When ``backend`` is
    ``None`` and ``config.base_url`` is set, a :class:`SearxngBackend` is built.
    """
    # Build the default SearXNG backend from config only when one wasn't
    # injected AND web search is actually enabled with a base_url. Otherwise
    # the provider is corpus-only (and dependency-free -- no httpx import).
    if backend is None and config.enabled and config.base_url:
        backend = SearxngBackend(config.base_url, timeout_s=config.timeout_s)

    def _corpus(query: str, context: dict[str, object]) -> CapabilityResult:
        """Delegate to the registered corpus search (``search.local``)."""
        return registry.invoke(fallback_capability, query, context)

    def web_search(query: str, context: dict[str, object]) -> CapabilityResult:
        # 1. GUARD-COERCE mode/intent (BR2) -- fail-safe to None, never raise.
        mode = _coerce_mode(context.get("mode"))
        intent_kind = _coerce_intent(context.get("intent_kind"))

        # 2. Gate FIRST on the raw query. PRIVATE/personal/MEETING/COMMAND/...
        #    => corpus only, no network egress (§9.7).
        permitted = True
        try:
            permitted = bool(classify(query, mode=mode, intent_kind=intent_kind))
        except Exception:  # noqa: BLE001 - the gate must fail safe, never abort
            permitted = False

        # 3. Denied OR disabled OR no usable backend => corpus fallback, no egress.
        if not permitted or backend is None:
            result = _corpus(query, context)
            data = dict(result.data)
            data["egress"] = False
            data["sensitivity"] = "private" if not permitted else "local"
            data["source"] = "corpus"
            return CapabilityResult(
                True, result.text, data=data, citations=result.citations
            )

        # 4. Permitted: hit the backend. On empty/error fall back to corpus with
        #    ok=True (BR7) + a stamp so the audit trail records what happened.
        try:
            hits = list(backend.search(query))
        except Exception as exc:  # noqa: BLE001 - any backend/transport/import error
            log.info("web.search backend failed; falling back to corpus: %r", exc)
            result = _corpus(query, context)
            data = dict(result.data)
            data["egress"] = True
            data["sensitivity"] = "public"
            data["source"] = "corpus"
            data["error"] = type(exc).__name__
            return CapabilityResult(
                True, result.text, data=data, citations=result.citations
            )

        results: list[dict[str, str]] = []
        urls: list[str] = []
        for hit in hits[: config.max_results]:
            title = str(hit.get("title", "") or "").strip()
            summary = str(hit.get("content", "") or "").strip()
            url = str(hit.get("url", "") or "").strip()
            if not (title or summary):
                continue
            results.append({"name": title or url, "summary": summary})
            if url:
                urls.append(url)

        if not results:
            # Reachable but no usable hits: corpus fallback keeps the plan alive.
            log.info("web.search returned no usable results; falling back to corpus")
            result = _corpus(query, context)
            data = dict(result.data)
            data["egress"] = True
            data["sensitivity"] = "public"
            data["source"] = "corpus"
            return CapabilityResult(
                True, result.text, data=data, citations=result.citations
            )

        text = " ".join(r["summary"] or r["name"] for r in results).strip()
        return CapabilityResult(
            True,
            text,
            data={
                "results": results,
                "egress": True,
                "sensitivity": "public",
                "source": "web",
            },
            citations=tuple(urls),
        )

    registry.register("web.search", web_search)
    return registry


__all__ = [
    "WebSearchConfig",
    "Backend",
    "SearxngBackend",
    "attach_web_search_capability",
]
