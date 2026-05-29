from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from threading import Event
from typing import Callable, Iterator, Mapping, Optional

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult
from always_on_agent.events import Mode
from always_on_agent.memory import Memory
from always_on_agent.models import IntentKind

from .contract import drain_complete_sentences
from .llm import LLMClient, capability_context
from .metrics import MetricsRecorder, mark_first_token
from .routing import LIVE_CONTEXT_KEY, HeuristicRouter, Router
from .sensitivity import classify_sensitivity

log = logging.getLogger("speaker.llm")

# Predicate deciding whether an ASSISTANT-mode query should escalate to the
# ReAct planner instead of a one-shot reply.
EscalatePredicate = Callable[[str, Mapping[str, object]], bool]

DEFAULT_SYSTEM = (
    "You are a local, on-device voice assistant. Reply in one or two short, "
    "natural spoken sentences. Do not use markdown, lists, headings, or "
    "preambles like 'Sure'. If you don't know, say so briefly."
)


@dataclass(frozen=True)
class RecallConfig:
    """Gating for memory recall injection (the flat ``memory`` config block).

    ``enabled`` defaults to **false**: the cheap config short-circuit before any
    embedding/recall work happens. ``max_chars`` bounds how much of the recall
    block is prepended so TTFT stays bounded."""

    enabled: bool = False
    max_chars: int = 600

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "RecallConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("recall_enabled", False)),
            max_chars=int(data.get("recall_max_chars", 600) or 600),
        )


def _collect(tokens: Iterator[str], cancel: Optional[Event]) -> tuple[str, bool]:
    """Drain a token stream, stopping early if ``cancel`` fires.

    Returns ``(text, cancelled)``. Streaming (rather than a blocking
    ``generate``) is what makes a slow local model interruptible: barge-in cuts
    generation off mid-stream instead of waiting for the whole answer."""
    parts: list[str] = []
    cancelled = False
    for token in tokens:
        if cancel is not None and cancel.is_set():
            cancelled = True
            break
        parts.append(token)
    return "".join(parts).strip(), cancelled


def _stream_and_speak(
    tokens: Iterator[str], cancel: Optional[Event], emit: Callable[[str], None]
) -> tuple[str, bool]:
    """Drain a token stream, speaking each complete sentence as it lands.

    This is the latency win: playback of sentence one starts while the model is
    still generating sentence two. Sentence boundaries follow the shared contract
    (:mod:`core.contract`) so the desktop and mobile shells split identically.
    Returns the full ``(text, cancelled)`` so the caller can still log/remember
    the whole answer."""
    parts: list[str] = []
    buffer = ""
    cancelled = False
    for token in tokens:
        if cancel is not None and cancel.is_set():
            cancelled = True
            break
        parts.append(token)
        buffer += token
        sentences, buffer = drain_complete_sentences(buffer)
        for sentence in sentences:
            emit(sentence)
    tail = buffer.strip()
    if tail and not cancelled:
        emit(tail)
    return "".join(parts).strip(), cancelled


def attach_llm_capabilities(
    registry: CapabilityRegistry,
    llm: LLMClient,
    *,
    fast_llm: Optional[LLMClient] = None,
    system: str = DEFAULT_SYSTEM,
    router: Optional[Router] = None,
    escalate: Optional[EscalatePredicate] = None,
    agent_capability: str = "agent.react",
    recorder: Optional[MetricsRecorder] = None,
    memory: Optional[Memory] = None,
    recall: Optional[RecallConfig] = None,
    live_routing: bool = False,
) -> CapabilityRegistry:
    """Replace the brain's stub providers with real LLM-backed ones.

    ``create_default_capabilities`` registers offline stubs (a tiny hardcoded
    corpus). Here we override the two that should reason with the model:
    ``assistant.answer`` (direct replies) and ``research.local`` (synthesis of
    the gathered search/scope steps). Local-only providers such as
    ``search.local`` and ``meeting.note`` are left untouched.

    Two-model split: ``assistant.answer`` picks its model per query via
    ``router`` (a small ``fast_llm`` for snappy replies, the larger ``llm`` for
    reasoning-heavy asks) while ``research.local`` always runs on ``llm``. When
    ``fast_llm`` is omitted both tiers collapse to ``llm`` (routing is a no-op).

    ``live_routing`` (default off; also honoured via ``SPEAKER_LIVE_ROUTING``)
    opts into headroom-aware routing: the recorder's rolling local TTFT is fed
    to the router as a live nudge toward main/cloud when the local tier is slow.
    The nudge is additive-only + clamped (see :mod:`core.routing`) and a missing
    sample is a no-op, so it can never starve the local tier.
    """

    fast = fast_llm or llm
    router = router or HeuristicRouter()
    recall_cfg = recall or RecallConfig()
    debug_routing = bool(os.environ.get("SPEAKER_DEBUG_ROUTING"))
    # Headroom-aware routing (smart-routing-2): feed the recorder's rolling
    # local TTFT into the router as a live signal that NUDGES borderline turns
    # toward main/cloud when the local tier is slow. Off by default so existing
    # behaviour is byte-for-byte unchanged; opt in via the ``live_routing`` arg
    # or ``SPEAKER_LIVE_ROUTING=1``. Even when on, the nudge is additive-only +
    # clamped in core.routing, and a missing TTFT sample (cold start, no
    # recorder) yields no nudge -- the static per-profile decision is the floor,
    # so this can never starve the local tier.
    live_routing = live_routing or bool(os.environ.get("SPEAKER_LIVE_ROUTING"))

    def _enrich_context(query: str, context: dict[str, object]) -> dict[str, object]:
        """Add ``intent_kind`` + ``sensitivity`` to the context before routing.

        Both are downstream signals for the router (tier choice) and the
        SensitivityRouterLLM (cloud-chain choice). Tasks publish ``intent_kind``
        in ``task.intent``; capabilities invoked outside the task layer get a
        default of ``IntentKind.ASSISTANT``. Sensitivity is classified here so
        a single source of truth feeds both router signals."""
        if "intent_kind" not in context:
            context["intent_kind"] = IntentKind.ASSISTANT.value
        intent_raw = context.get("intent_kind")
        try:
            intent_kind = IntentKind(intent_raw) if intent_raw else IntentKind.ASSISTANT
        except ValueError:
            intent_kind = IntentKind.ASSISTANT
        mode_raw = context.get("mode")
        try:
            mode = Mode(mode_raw) if mode_raw else None
        except ValueError:
            mode = None
        if "sensitivity" not in context:
            context["sensitivity"] = classify_sensitivity(
                query, mode=mode, intent_kind=intent_kind
            )
        return context

    def assistant(query: str, context: dict[str, object]) -> CapabilityResult:
        cancel = context.get("cancel_event")
        # Smart-mode escalation: hand reasoning/gathering queries to the ReAct
        # planner (when registered) instead of a one-shot reply.
        if (
            escalate is not None
            and agent_capability in registry.names()
            and escalate(query, context)
        ):
            return registry.invoke(agent_capability, query, context)
        _enrich_context(query, context)
        # Memory recall (the headline P2 fix), assistant-only (R5):
        #   1. Ingest the answered query first (R1) so recall has something to
        #      find on the next turn. Never mutate ``query`` itself -- the
        #      router/sensitivity inputs stay clean.
        #   2. Gated prepend: only when recall is enabled (cheap config gate,
        #      default off) AND the backend returns a non-empty (relevant)
        #      block. Wrapped so a memory error can never break a live turn.
        system_for_call = system
        if memory is not None:
            try:
                memory.add(query, tags=("user",))
                if recall_cfg.enabled:
                    recall_block = memory.context_for_llm(query)
                    if recall_block:
                        system_for_call = recall_block[: recall_cfg.max_chars] + "\n\n" + system
            except Exception:  # noqa: BLE001 - recall is best-effort, never fatal
                log.exception("memory recall failed; answering without it")
        # Live headroom hint (smart-routing-2): publish the recorder's rolling
        # local TTFT under context['live'] so HeuristicRouter can nudge a slow
        # local tier toward main/cloud. Gated + fail-safe: only when enabled and
        # a measurable sample exists; reading the EWMA is a cheap locked dict
        # read. A SystemMonitor load snapshot is a follow-up (wiring it needs the
        # monitor plumbed into this closure from runtime.py/app.py).
        if live_routing and recorder is not None and LIVE_CONTEXT_KEY not in context:
            ttft_ms = recorder.recent_ttft_ms()
            if ttft_ms is not None:
                context[LIVE_CONTEXT_KEY] = {"ttft_ms": ttft_ms}
        tier = router.choose(query, context)
        model = llm if tier == "main" else fast
        if debug_routing:
            print(
                f"[route] {tier} sens={context.get('sensitivity')} <- {query!r}",
                flush=True,
            )
        log.info(
            "answering on %s tier (sensitivity=%s, intent=%s): %r",
            tier, context.get("sensitivity"), context.get("intent_kind"), query,
        )
        emit = context.get("emit_speech")
        started = time.monotonic()
        # Publish the per-turn context so SensitivityRouterLLM can pick the
        # right cloud chain at stream() time. Reset on the way out so the
        # ContextVar doesn't leak between unrelated turns.
        ctx_token = capability_context.set(context)
        try:
            tokens = mark_first_token(model.stream(query, system=system_for_call), recorder)
            if callable(emit):
                text, cancelled = _stream_and_speak(tokens, cancel, emit)  # type: ignore[arg-type]
                log.info(
                    "%s tier %s in %.2fs (%d chars, streamed)",
                    tier, "cancelled" if cancelled else "done", time.monotonic() - started, len(text),
                )
                return CapabilityResult(
                    True,
                    text or "Sorry, I don't have an answer for that.",
                    data={
                        "route": tier,
                        "streamed": True,
                        "cancelled": cancelled,
                        "sensitivity": context.get("sensitivity"),
                    },
                )
            text, cancelled = _collect(tokens, cancel)  # type: ignore[arg-type]
            log.info(
                "%s tier %s in %.2fs (%d chars)",
                tier, "cancelled" if cancelled else "done", time.monotonic() - started, len(text),
            )
            if cancelled:
                return CapabilityResult(
                    True, text,
                    data={"cancelled": True, "route": tier, "sensitivity": context.get("sensitivity")},
                )
            return CapabilityResult(
                True,
                text or "Sorry, I don't have an answer for that.",
                data={"route": tier, "sensitivity": context.get("sensitivity")},
            )
        finally:
            capability_context.reset(ctx_token)

    def research_synth(query: str, context: dict[str, object]) -> CapabilityResult:
        previous = context.get("previous_steps", [])
        gathered = " ".join(
            str(step.get("text", ""))
            for step in previous
            if isinstance(step, dict) and step.get("text")
        )
        prompt = (
            f"Question: {query}\n"
            f"Local findings: {gathered or '(none)'}\n"
            "Give a brief spoken-style summary and one concrete recommendation."
        )
        cancel = context.get("cancel_event")
        emit = context.get("emit_speech")
        tokens = mark_first_token(llm.stream(prompt, system=system), recorder)
        if callable(emit):
            text, cancelled = _stream_and_speak(tokens, cancel, emit)  # type: ignore[arg-type]
            return CapabilityResult(True, text, data={"cancelled": cancelled, "streamed": True})
        text, cancelled = _collect(tokens, cancel)  # type: ignore[arg-type]
        return CapabilityResult(True, text, data={"cancelled": cancelled})

    registry.register("assistant.answer", assistant)
    registry.register("research.local", research_synth)
    return registry
