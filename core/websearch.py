"""Pluggable ``web.search`` capability backed by self-hosted SearXNG.

This is the P3 "real web research" surface (`docs/archive/p3_design.md` §1, Locked
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
  closure enforces the order **turn-context veto -> GUARD-COERCE -> raw-query
  gate -> SearXNG -> corpus fallback** and **never raises** (a non-ok step aborts
  the whole plan in
  ``always_on_agent/tasks.py:586-588``, so a raised exception or an ``ok=False``
  here would silently break RESEARCH/SEARCH plans).

The result mirrors the corpus ``search()`` shape
(``always_on_agent/capabilities.py:75-80``): ``data["results"]`` is a list of
``{"name", "summary"}`` and ``citations`` are the source URLs, so a downstream
synthesis step (``research.local``) consumes web hits exactly like corpus hits.
Audit stamps ``data["egress"]`` / ``data["sensitivity"]`` record whether the
query was permitted to leave the device.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from itertools import islice
from threading import Event
from typing import (
    Callable,
    ContextManager,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult
from always_on_agent.events import Mode
from always_on_agent.models import (
    CLOUD_EGRESS_SCOPE_CONTEXT_KEY,
    RETAINED_PROMPT_CONTEXT_METADATA_KEY,
    CloudEgressScope,
    IntentKind,
)

from .sensitivity import CODE, PRIVATE, PUBLIC, may_leave_device

log = logging.getLogger("speaker.websearch")

_WEB_EGRESS_CONTEXT_SENSITIVITIES = frozenset({CODE, PUBLIC})


def _bounded_float(value: object, default: float, *, low: float, high: float) -> float:
    if type(value) is int:
        if value <= low:
            return low
        if value >= high:
            return high
        return float(value)
    if type(value) is not float:
        return default
    if not math.isfinite(value):
        return default
    return max(low, min(high, value))


def _bounded_int(value: object, default: int, *, low: int, high: int) -> int:
    if type(value) is not int:
        return default
    return max(low, min(high, value))


@dataclass(frozen=True)
class WebSearchConfig:
    """Gating + connection settings for the ``web.search`` capability (the flat
    ``web_search`` config block).

    ``enabled`` defaults to **false** so the fully-local corpus path is the
    out-of-the-box behaviour -- a user opts in by setting ``enabled`` AND a
    ``base_url`` pointing at their self-hosted SearXNG. ``timeout_s`` is the
    per-phase connect/read/write/pool inactivity budget (BR7). The separate
    ``total_timeout_s`` budget is checked cooperatively and cannot preempt a
    blocked synchronous transport operation. Body, result, field, and output
    limits are clamped here so direct construction and JSON construction have
    the same bounded behavior.
    """

    enabled: bool = False
    base_url: str = ""
    timeout_s: float = 4.0
    total_timeout_s: float = 8.0
    max_response_bytes: int = 262_144
    max_results: int = 5
    max_title_chars: int = 256
    max_summary_chars: int = 2_048
    max_url_chars: int = 2_048
    max_output_chars: int = 16_384

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", type(self.enabled) is bool and self.enabled)
        object.__setattr__(
            self,
            "base_url",
            self.base_url.strip() if type(self.base_url) is str else "",
        )
        object.__setattr__(
            self,
            "timeout_s",
            _bounded_float(self.timeout_s, 4.0, low=0.05, high=15.0),
        )
        object.__setattr__(
            self,
            "total_timeout_s",
            _bounded_float(self.total_timeout_s, 8.0, low=0.05, high=20.0),
        )
        object.__setattr__(
            self,
            "max_response_bytes",
            _bounded_int(self.max_response_bytes, 262_144, low=1_024, high=1_048_576),
        )
        object.__setattr__(
            self,
            "max_results",
            _bounded_int(self.max_results, 5, low=1, high=8),
        )
        object.__setattr__(
            self,
            "max_title_chars",
            _bounded_int(self.max_title_chars, 256, low=1, high=512),
        )
        object.__setattr__(
            self,
            "max_summary_chars",
            _bounded_int(self.max_summary_chars, 2_048, low=1, high=4_096),
        )
        object.__setattr__(
            self,
            "max_url_chars",
            _bounded_int(self.max_url_chars, 2_048, low=1, high=2_048),
        )
        object.__setattr__(
            self,
            "max_output_chars",
            _bounded_int(self.max_output_chars, 16_384, low=1, high=32_768),
        )

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "WebSearchConfig":
        if type(data) is not dict:
            data = {}
        else:
            try:
                data = dict.copy(data)
                if any(type(key) is not str for key, _value in dict.items(data)):
                    data = {}
            except Exception:
                data = {}
        return cls(
            enabled=dict.get(data, "enabled", False),
            base_url=dict.get(data, "base_url", ""),
            timeout_s=dict.get(data, "timeout_s", 4.0),
            total_timeout_s=dict.get(data, "total_timeout_s", 8.0),
            max_response_bytes=dict.get(data, "max_response_bytes", 262_144),
            max_results=dict.get(data, "max_results", 5),
            max_title_chars=dict.get(data, "max_title_chars", 256),
            max_summary_chars=dict.get(data, "max_summary_chars", 2_048),
            max_url_chars=dict.get(data, "max_url_chars", 2_048),
            max_output_chars=dict.get(data, "max_output_chars", 16_384),
        )


@runtime_checkable
class Backend(Protocol):
    """A pluggable web-search backend (Decision 3).

    Returns an iterable of ``{"title", "content", "url"}`` mappings (the
    SearXNG JSON result shape). MUST raise on a network/transport error so the
    provider closure can fall back to the corpus -- the closure, never the
    backend, owns the never-raise guarantee.
    """

    def search(self, query: str) -> Iterable[Mapping[str, object]]: ...


class _BackendInputError(Exception):
    """A stable, detail-free failure from the shipped response reader."""

    def __init__(self, code: str, *, egress: bool) -> None:
        super().__init__(code)
        self.code = code
        self.egress = egress


def _strict_json_value(text: str) -> object:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise ValueError("non-finite JSON number")

    return json.loads(
        text,
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )


class SearxngBackend:
    """Query a self-hosted SearXNG instance over its JSON API.

    ``httpx`` is imported lazily so disabled/local-only paths never initialize
    transport code. The response is consumed through a context-managed raw
    identity stream. It must be JSON with a strictly bounded body and exact
    ``results[]`` / hit / field shapes. ``timeout_s`` applies separately to
    connect/read/write/pool inactivity; ``total_timeout_s`` and cancellation
    are cooperative checkpoints and cannot interrupt a blocked synchronous
    stream enter or read.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 4.0,
        total_timeout_s: float = 8.0,
        max_response_bytes: int = 262_144,
        stream_factory: Optional[Callable[..., ContextManager[object]]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        # Strip a trailing slash so ``{base_url}/search`` never doubles it.
        config = WebSearchConfig(
            enabled=True,
            base_url=base_url,
            timeout_s=timeout_s,
            total_timeout_s=total_timeout_s,
            max_response_bytes=max_response_bytes,
        )
        self._base_url = config.base_url.rstrip("/")
        self._timeout_s = config.timeout_s
        self._total_timeout_s = config.total_timeout_s
        self._max_response_bytes = config.max_response_bytes
        self._stream_factory = stream_factory
        self._clock = clock

    @staticmethod
    def _check_cancel_or_deadline(
        cancel_event: object,
        *,
        clock: Callable[[], float],
        deadline: float,
        egress: bool,
    ) -> None:
        if type(cancel_event) is Event and cancel_event.is_set():
            raise _BackendInputError("cancelled", egress=egress)
        if clock() >= deadline:
            raise _BackendInputError("deadline_exceeded", egress=egress)

    @staticmethod
    def _raw_header_values(response: object, name: bytes) -> list[bytes]:
        raw = response.headers.raw  # type: ignore[attr-defined]
        return [value for key, value in raw if key.lower() == name]

    def search(
        self,
        query: str,
        *,
        cancel_event: object = None,
        max_results: int = 5,
    ) -> Sequence[Mapping[str, object]]:
        import httpx  # lazy: only needed when web search is enabled

        deadline = self._clock() + self._total_timeout_s
        self._check_cancel_or_deadline(
            cancel_event, clock=self._clock, deadline=deadline, egress=False
        )
        stream_factory = self._stream_factory or httpx.stream
        # Calling/entering the stream may synchronously block. The HTTPX timeout
        # bounds phase inactivity; the total budget is checked once control
        # returns and cannot preempt a blocked synchronous transport operation.
        stream_context = stream_factory(
            "GET",
            f"{self._base_url}/search",
            params={"q": query, "format": "json"},
            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
            timeout=httpx.Timeout(self._timeout_s),
        )
        with stream_context as response:
            self._check_cancel_or_deadline(
                cancel_event, clock=self._clock, deadline=deadline, egress=True
            )
            response.raise_for_status()  # type: ignore[attr-defined]

            content_types = self._raw_header_values(response, b"content-type")
            if len(content_types) != 1:
                raise _BackendInputError("invalid_content_type", egress=True)
            media_type = content_types[0].split(b";", 1)[0].strip().lower()
            if media_type != b"application/json":
                raise _BackendInputError("invalid_content_type", egress=True)

            encodings = self._raw_header_values(response, b"content-encoding")
            if len(encodings) > 1 or (
                encodings and encodings[0].strip().lower() != b"identity"
            ):
                raise _BackendInputError("unsupported_content_encoding", egress=True)

            lengths = self._raw_header_values(response, b"content-length")
            if len(lengths) > 1:
                raise _BackendInputError("invalid_content_length", egress=True)
            if lengths:
                raw_length = lengths[0]
                if not raw_length or any(byte < 48 or byte > 57 for byte in raw_length):
                    raise _BackendInputError("invalid_content_length", egress=True)
                normalized_length = raw_length.lstrip(b"0") or b"0"
                cap = str(self._max_response_bytes).encode("ascii")
                if len(normalized_length) > len(cap) or (
                    len(normalized_length) == len(cap) and normalized_length > cap
                ):
                    raise _BackendInputError("response_too_large", egress=True)

            body = bytearray()
            iterator = iter(response.iter_raw())  # type: ignore[attr-defined]
            while True:
                self._check_cancel_or_deadline(
                    cancel_event, clock=self._clock, deadline=deadline, egress=True
                )
                try:
                    chunk = next(iterator)
                except StopIteration:
                    self._check_cancel_or_deadline(
                        cancel_event,
                        clock=self._clock,
                        deadline=deadline,
                        egress=True,
                    )
                    break
                self._check_cancel_or_deadline(
                    cancel_event, clock=self._clock, deadline=deadline, egress=True
                )
                if type(chunk) is not bytes:
                    raise _BackendInputError("invalid_payload", egress=True)
                if len(chunk) > self._max_response_bytes - len(body):
                    raise _BackendInputError("response_too_large", egress=True)
                body.extend(chunk)

        self._check_cancel_or_deadline(
            cancel_event, clock=self._clock, deadline=deadline, egress=True
        )
        if body.startswith(b"\xef\xbb\xbf"):
            raise _BackendInputError("invalid_utf8", egress=True)
        try:
            decoded = bytes(body).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise _BackendInputError("invalid_utf8", egress=True) from None
        self._check_cancel_or_deadline(
            cancel_event, clock=self._clock, deadline=deadline, egress=True
        )
        try:
            payload = _strict_json_value(decoded)
        except (json.JSONDecodeError, RecursionError, ValueError):
            raise _BackendInputError("invalid_json", egress=True) from None
        self._check_cancel_or_deadline(
            cancel_event, clock=self._clock, deadline=deadline, egress=True
        )

        if type(payload) is not dict:
            raise _BackendInputError("invalid_payload", egress=True)
        if "results" not in payload or type(payload["results"]) is not list:
            raise _BackendInputError("invalid_payload", egress=True)
        result_limit = _bounded_int(max_results, 5, low=1, high=8)
        results: list[dict[str, str]] = []
        # Parse is necessarily whole-payload under the response-byte cap, but
        # validation and normalized copies stop at the configured prefix. A
        # malformed or enormous suffix cannot poison/expand the consumed hits.
        for hit in payload["results"][:result_limit]:
            self._check_cancel_or_deadline(
                cancel_event, clock=self._clock, deadline=deadline, egress=True
            )
            if type(hit) is not dict:
                raise _BackendInputError("invalid_payload", egress=True)
            normalized: dict[str, str] = {}
            for field in ("title", "content", "url"):
                value = dict.get(hit, field)
                if value is None:
                    normalized[field] = ""
                elif type(value) is str:
                    normalized[field] = value
                else:
                    raise _BackendInputError("invalid_payload", egress=True)
            results.append(normalized)
        return results


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
    if type(raw) is not str:
        return None
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
    if type(raw) is not str:
        return None
    try:
        return IntentKind(raw)
    except (ValueError, TypeError):
        return None


def _turn_context_egress_snapshot(
    context: dict[str, object],
) -> tuple[Optional[bool], dict[str, object]]:
    """Snapshot ``context`` and return its restrictive web-egress verdict.

    ``None`` means no restrictive receipt vetoed and sensitivity is absent, so
    deterministic SEARCH/RESEARCH calls keep the historical raw-query gate.
    Dict subclasses are rejected before virtual container hooks can hide or
    rewrite a field. A present LLM scope proceeds only for the exact
    ``CURRENT_TURN_ONLY`` enum; ``LOCAL_ONLY`` and malformed values fail closed.
    A present retained-prompt key also fails closed (the canonical receipt is
    exact ``True``). The sensitivity receipt remains restrictive-only: only
    exact canonical non-private strings can proceed.

    This receipt is never sufficient to authorize egress: callers must still
    pass :func:`may_leave_device` on the exact tool query, mode, and intent.
    """
    if type(context) is not dict:
        return False, {}
    snapshot = dict.copy(context)
    missing = object()
    raw: object = missing
    raw_scope: object = missing
    for key, value in dict.items(snapshot):
        if type(key) is not str:
            return False, {}
        if key == "sensitivity":
            raw = value
        elif key == CLOUD_EGRESS_SCOPE_CONTEXT_KEY:
            raw_scope = value
    if raw_scope is not missing and raw_scope is not CloudEgressScope.CURRENT_TURN_ONLY:
        return False, snapshot
    metadata_missing = object()
    metadata: object = dict.get(snapshot, "metadata", metadata_missing)
    if metadata is not metadata_missing:
        if type(metadata) is not dict:
            return False, snapshot
        metadata_snapshot = dict.copy(metadata)
        snapshot["metadata"] = metadata_snapshot
        for metadata_key, _metadata_value in dict.items(metadata_snapshot):
            if type(metadata_key) is not str:
                return False, snapshot
            if metadata_key == RETAINED_PROMPT_CONTEXT_METADATA_KEY:
                return False, snapshot
    if raw is missing:
        return None, snapshot
    return (
        type(raw) is str and raw in _WEB_EGRESS_CONTEXT_SENSITIVITIES,
        snapshot,
    )


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

    1. **Turn-context veto**: exact local-only LLM scope, malformed present
       scope, retained-prompt task metadata, exact ``private`` sensitivity, or
       malformed/unknown sensitivity fails closed to the local corpus before
       classifier or backend work. Exact current-turn-only scope and exact
       ``public``/``code`` sensitivity are restrictive only and cannot override
       the raw-query gate. Absent scope/metadata preserves direct deterministic
       SEARCH/RESEARCH compatibility.
    2. **GUARD-COERCE** ``mode`` / ``intent_kind`` from ``context`` via a
       fail-safe try/except (-> ``None`` on a bad value, BR2) so a bogus enum
       string never aborts the plan or bypasses the gate.
    3. **Raw-query gate** (``classify`` ==
       :func:`core.sensitivity.may_leave_device`) on the exact tool query.  A
       PRIVATE/personal query is hard-blocked regardless of a supplied
       ``public``/``code`` context: SearXNG is never called and the corpus
       answers with ``data["egress"]=False``.
    4. If permitted AND web search is enabled AND a backend/``base_url`` exists,
       call the backend; map ``{title, content, url}`` -> corpus
       ``{name, summary}`` + ``citations``.
    5. On empty results / transport error / bounded-reader refusal, fall back to
       the corpus with ``ok=True`` (BR7: an unreachable or slow SearXNG must not
       abort the plan) and stamp truthful egress/source/error receipts.

    NEVER raises and NEVER returns ``ok=False`` from a fallback path: a non-ok
    step aborts the whole plan (``always_on_agent/tasks.py:586-588``), so the
    corpus fallback must keep the plan alive.

    ``classify`` / ``backend`` are injectable for tests. When ``backend`` is
    ``None`` and ``config.base_url`` is set, a :class:`SearxngBackend` is built.
    """
    # Build the default SearXNG backend from config only when one wasn't
    # injected AND web search is actually enabled with a base_url. Otherwise
    # the provider is corpus-only (and dependency-free -- no httpx import).
    if backend is None and config.enabled and config.base_url:
        backend = SearxngBackend(
            config.base_url,
            timeout_s=config.timeout_s,
            total_timeout_s=config.total_timeout_s,
            max_response_bytes=config.max_response_bytes,
        )

    def _corpus(query: str, context: dict[str, object]) -> CapabilityResult:
        """Delegate to the registered corpus search (``search.local``)."""
        return registry.invoke(fallback_capability, query, context)

    def web_search(query: str, context: dict[str, object]) -> CapabilityResult:
        # 1. An exact floated turn receipt is a veto, never an authorization.
        #    Evaluate it before enum coercion, the injectable classifier, or the
        #    backend so PRIVATE/malformed context cannot trigger egress work.
        turn_context_permitted, context_snapshot = _turn_context_egress_snapshot(
            context
        )
        if turn_context_permitted is False:
            result = _corpus(query, context_snapshot)
            data = dict(result.data)
            data["egress"] = False
            data["sensitivity"] = PRIVATE
            data["source"] = "corpus"
            return CapabilityResult(
                True, result.text, data=data, citations=result.citations
            )

        # 2. GUARD-COERCE mode/intent (BR2) -- fail-safe to None, never raise.
        mode = _coerce_mode(context_snapshot.get("mode"))
        intent_kind = _coerce_intent(context_snapshot.get("intent_kind"))

        # 3. Gate the raw query. PRIVATE/personal/MEETING/COMMAND/...
        #    => corpus only, no network egress (§9.7).
        permitted = True
        try:
            permitted = bool(classify(query, mode=mode, intent_kind=intent_kind))
        except Exception:  # noqa: BLE001 - the gate must fail safe, never abort
            permitted = False

        # 4. Denied OR disabled OR no usable backend => corpus fallback, no egress.
        if not permitted or not config.enabled or backend is None:
            result = _corpus(query, context_snapshot)
            data = dict(result.data)
            data["egress"] = False
            data["sensitivity"] = "private" if not permitted else "local"
            data["source"] = "corpus"
            return CapabilityResult(
                True, result.text, data=data, citations=result.citations
            )

        # 5. Permitted: hit the backend. On empty/error fall back to corpus with
        #    ok=True (BR7) + a stamp so the audit trail records what happened.
        try:
            results: list[dict[str, str]] = []
            urls: list[str] = []
            shipped_backend = type(backend) is SearxngBackend
            if shipped_backend:
                raw_cancel_event = dict.get(context_snapshot, "cancel_event")
                cancel_event = (
                    raw_cancel_event if type(raw_cancel_event) is Event else None
                )
                hits = backend.search(
                    query,
                    cancel_event=cancel_event,
                    max_results=config.max_results,
                )
            else:
                # Preserve the original single-argument protocol for injected
                # custom backends, including subclasses of SearxngBackend.
                hits = backend.search(query)
            for hit in islice(hits, config.max_results):
                if type(hit) is not dict:
                    continue
                try:
                    if any(type(key) is not str for key, _value in dict.items(hit)):
                        continue
                except Exception:
                    continue
                raw_title = dict.get(hit, "title")
                raw_summary = dict.get(hit, "content")
                raw_url = dict.get(hit, "url")
                if any(
                    value is not None and type(value) is not str
                    for value in (raw_title, raw_summary, raw_url)
                ):
                    continue
                title = (raw_title or "")[: config.max_title_chars].strip()
                summary = (raw_summary or "")[: config.max_summary_chars].strip()
                url = (raw_url or "")[: config.max_url_chars].strip()
                if not (title or summary):
                    continue
                results.append({"name": title or url, "summary": summary})
                if url:
                    urls.append(url)
        except _BackendInputError as exc:
            if not shipped_backend:
                error_code = type(exc).__name__
                log.info(
                    "web.search backend failed; falling back to corpus: %s",
                    error_code,
                )
                result = _corpus(query, context_snapshot)
                data = dict(result.data)
                data["egress"] = True
                data["sensitivity"] = "public"
                data["source"] = "corpus"
                data["error"] = error_code
                return CapabilityResult(
                    True, result.text, data=data, citations=result.citations
                )
            log.info(
                "web.search shipped backend failed; falling back to corpus: %s",
                exc.code,
            )
            result = _corpus(query, context_snapshot)
            data = dict(result.data)
            data["egress"] = exc.egress
            data["sensitivity"] = "public"
            data["source"] = "corpus"
            data["error"] = exc.code
            return CapabilityResult(
                True, result.text, data=data, citations=result.citations
            )
        except Exception as exc:  # noqa: BLE001 - backend/payload must never abort
            error_code = type(exc).__name__
            log.info(
                "web.search backend failed; falling back to corpus: %s", error_code
            )
            result = _corpus(query, context_snapshot)
            data = dict(result.data)
            data["egress"] = True
            data["sensitivity"] = "public"
            data["source"] = "corpus"
            data["error"] = error_code
            return CapabilityResult(
                True, result.text, data=data, citations=result.citations
            )

        if not results:
            # Reachable but no usable hits: corpus fallback keeps the plan alive.
            log.info("web.search returned no usable results; falling back to corpus")
            result = _corpus(query, context_snapshot)
            data = dict(result.data)
            data["egress"] = True
            data["sensitivity"] = "public"
            data["source"] = "corpus"
            return CapabilityResult(
                True, result.text, data=data, citations=result.citations
            )

        text = " ".join(r["summary"] or r["name"] for r in results)
        text = text[: config.max_output_chars].strip()
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
