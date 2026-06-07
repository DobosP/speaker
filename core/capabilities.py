from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from threading import Event
from typing import Callable, Iterator, Mapping, Optional, Sequence

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult
from always_on_agent.events import Mode
from always_on_agent.memory import Memory
from always_on_agent.models import IntentKind

from .contract import drain_complete_sentences
from .conversation import RecentContextConfig, collect_recent_turns, format_recent_block
from .llm import HedgeLLM, LLMClient, SensitivityRouterLLM, capability_context
from .metrics import MetricsRecorder, mark_first_token
from .persona import DEFAULT_SYSTEM
from .routing import (
    LIVE_CONTEXT_KEY,
    HeuristicRouter,
    Router,
    dynamic_hedge_delay_ms,
)
from .sensitivity import PRIVATE, classify_sensitivity, most_sensitive

log = logging.getLogger("speaker.llm")


def _answers_locally(model: LLMClient) -> bool:
    """True when ``model`` can only answer from the on-device tier.

    Gates the LOCAL TTFT EWMA fold (P4 low): a turn's ASR_FINAL -> first-token
    delta is a *local* headroom sample only when the model has no cloud that
    could have won the race. A :class:`HedgeLLM` with a non-empty cloud chain --
    or a :class:`SensitivityRouterLLM`, whose chosen chain may be cloud-backed --
    might have answered from cloud, so we conservatively do NOT fold those (a
    fast cloud answer must not be mislabeled as a fast local tier). Anything
    else (plain ``OllamaLLM`` / ``LlamaCppLLM`` / ``EchoLLM`` / a HedgeLLM with
    no clouds) is purely local and folds. Best-effort: any introspection error
    falls back to ``True`` (the historical fold-always behaviour)."""
    try:
        if isinstance(model, SensitivityRouterLLM):
            return False
        if isinstance(model, HedgeLLM):
            return not model.clouds
        return True
    except Exception:  # noqa: BLE001 - never let a fold gate break a turn
        return True


def _base_hedge_delay_ms(model: LLMClient) -> Optional[float]:
    """The static hedge delay (ms) of a cloud-racing model, or ``None``.

    Used to derive the dynamic per-call hedge override (Task 4): only a
    :class:`HedgeLLM` with a cloud chain -- or a :class:`SensitivityRouterLLM`
    whose chains are such HedgeLLMs -- has a meaningful hedge delay to shorten.
    The factory builds every chain with the same ``hedge_delay_ms``, so any
    cloud-backed chain's delay is representative. Returns ``None`` for a
    pure-local model (no race to start sooner) or on any introspection error."""
    try:
        if isinstance(model, HedgeLLM):
            return model.hedge_delay * 1000.0 if model.clouds else None
        if isinstance(model, SensitivityRouterLLM):
            for chain in model.chains.values():
                if isinstance(chain, HedgeLLM) and chain.clouds:
                    return chain.hedge_delay * 1000.0
            return None
        return None
    except Exception:  # noqa: BLE001 - hedge timing is best-effort, never fatal
        return None


# Predicate deciding whether an ASSISTANT-mode query should escalate to the
# ReAct planner instead of a one-shot reply.
EscalatePredicate = Callable[[str, Mapping[str, object]], bool]

# The assistant's identity + skills now live in core/persona.py (so the prompt
# can enumerate the live capability manifest and reflect §9.7 web state).
# DEFAULT_SYSTEM (imported at the top) stays the byte-identical legacy prompt and
# the default for direct callers/tests; the runtime feeds the answering model the
# richer build_system_prompt() instead.


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


def _close_token_stream(tokens: Iterator[str]) -> None:
    """Best-effort close of a token generator after a barge-in cut.

    The token chain is ``mark_first_token(model.stream(...))`` -- generators whose
    ``finally`` blocks tear down the underlying HTTP body / SDK stream (Ollama,
    OpenAI-compat) so the SERVER stops generating. On a plain ``break`` those
    ``finally`` blocks only run when the generator is garbage-collected, which is
    not deterministic (and the ``mark_first_token`` wrapper adds a layer that must
    ALSO be collected) -- so an interrupted turn could keep the model generating
    (burning compute) until GC catches up. Calling ``close()`` here propagates
    ``GeneratorExit`` down the chain immediately at the barge point. A plain
    iterator with no ``close()`` is a no-op; any teardown error is swallowed (the
    turn is already being abandoned)."""
    close = getattr(tokens, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - teardown of an abandoned stream is best-effort
            pass


def _collect(tokens: Iterator[str], cancel: Optional[Event]) -> tuple[str, bool]:
    """Drain a token stream, stopping early if ``cancel`` fires.

    Returns ``(text, cancelled)``. Streaming (rather than a blocking
    ``generate``) is what makes a slow local model interruptible: barge-in cuts
    generation off mid-stream instead of waiting for the whole answer. On a cut
    the stream is closed explicitly so the model server stops generating at once
    (see :func:`_close_token_stream`)."""
    parts: list[str] = []
    cancelled = False
    try:
        for token in tokens:
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            parts.append(token)
    finally:
        if cancelled:
            _close_token_stream(tokens)
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
    try:
        for token in tokens:
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            parts.append(token)
            buffer += token
            sentences, buffer = drain_complete_sentences(buffer)
            for sentence in sentences:
                emit(sentence)
    finally:
        # Barge-in cut: close the stream so the model server stops generating
        # immediately rather than at GC time (see :func:`_close_token_stream`).
        if cancelled:
            _close_token_stream(tokens)
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
    recent_context: Optional[RecentContextConfig] = None,
    live_routing: bool = False,
    load_snapshot: Optional[Callable[[], Optional[float]]] = None,
    image_provider: Optional[Callable[[], Optional[Sequence[object]]]] = None,
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

    ``load_snapshot`` (optional) is a cheap ``() -> Optional[float]`` callable --
    the :class:`core.sysinfo.SystemMonitor` ``load_fraction`` reader -- whose
    0..1 value is added to the live context alongside the TTFT sample. Also
    gated behind ``live_routing`` (default off) so default behaviour stays
    byte-identical; ``live_nudge`` clamps the load and treats ``None`` as no
    nudge, so a loaded device offloads while a missing reading never starves
    local (the live-routing follow-up).
    """

    fast = fast_llm or llm
    router = router or HeuristicRouter()
    recall_cfg = recall or RecallConfig()
    recent_cfg = recent_context or RecentContextConfig()
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
        # A continuation turn (ADD-ON) carries a synthetic, folded prompt and the
        # supervisor has already ingested the raw user utterance, so skip the
        # query ingest here -- otherwise memory would hold both the prior turn and
        # the merged "prev + addon" prompt (the double-ingest the design flagged).
        meta = context.get("metadata")
        skip_user_memory = bool(meta.get("skip_user_memory")) if isinstance(meta, dict) else False
        system_for_call = system
        if memory is not None:
            try:
                # Recent-conversation context (short-term memory): the PRIOR turns
                # (collected BEFORE ingesting the current query) so the model can
                # resolve "its"/"the second one". Suppressed on a continuation turn
                # -- its merge/continue prompt already embeds the prior context.
                recent_turns = (
                    [] if skip_user_memory else collect_recent_turns(memory, recent_cfg)
                )
                if not skip_user_memory:
                    memory.add(query, tags=("user",))
                recall_block = memory.context_for_llm(query) if recall_cfg.enabled else ""
                recent_block = format_recent_block(recent_turns, recent_cfg)
                # §9.7: a private prior turn in the recent block must not ride a
                # public current query to a public/cloud chain. Float the prompt's
                # sensitivity to the most-private over the current turn AND every
                # included prior turn.
                if recent_turns:
                    sens = context.get("sensitivity") or PRIVATE
                    for _role, turn_text in recent_turns:
                        sens = most_sensitive(str(sens), classify_sensitivity(turn_text))
                    context["sensitivity"] = sens
                # Compose: stable system FIRST (the pre-warmed, cacheable prefix),
                # then the volatile recent block AFTER, so the prefix cache is
                # reused turn to turn. Recall (default-off, past sessions) keeps its
                # historical position ahead of system so its contract is unchanged.
                prefix = (recall_block[: recall_cfg.max_chars] + "\n\n") if recall_block else ""
                suffix = ("\n\n" + recent_block) if recent_block else ""
                system_for_call = prefix + system + suffix
            except Exception:  # noqa: BLE001 - context is best-effort, never fatal
                log.exception("conversation context / recall failed; answering without it")
        # Live headroom hint (smart-routing-2 + load follow-up): publish the
        # recorder's rolling local TTFT and an optional SystemMonitor load
        # snapshot under context['live'] so HeuristicRouter can nudge a slow or
        # loaded local tier toward main/cloud. Gated + fail-safe: only when
        # live_routing is on and at least one signal is present; reading the
        # EWMA is a cheap locked dict read and load_snapshot reads the last
        # background sample (no hot-path sampling). A missing/garbage value is a
        # no-op nudge in core.routing, so this can never starve the local tier.
        if live_routing and LIVE_CONTEXT_KEY not in context:
            live: dict[str, object] = {}
            if recorder is not None:
                ttft_ms = recorder.recent_ttft_ms()
                if ttft_ms is not None:
                    live["ttft_ms"] = ttft_ms
            if load_snapshot is not None:
                try:
                    load = load_snapshot()
                except Exception:  # noqa: BLE001 - load is best-effort, never fatal
                    load = None
                if load is not None:
                    live["load"] = load
            if live:
                context[LIVE_CONTEXT_KEY] = live
        # Visual context: an image attached to THIS turn (context['images'], set
        # explicitly per turn) takes precedence over the AMBIENT frame a machine
        # feeds continuously via image_provider (e.g. the current screen). Only the
        # main/multimodal model can SEE an image -- the fast tier (e.g. gemma3:1b)
        # ignores or chokes on it -- so an image-bearing turn is forced to main,
        # and its sensitivity floats to PRIVATE so a screen capture never rides a
        # public cloud chain (docs/target_architecture.md §9.7).
        images = context.get("images")
        if not images and image_provider is not None:
            try:
                images = image_provider()
            except Exception:  # noqa: BLE001 - the frame feed is best-effort, never fatal
                images = None
        if images:
            context["sensitivity"] = most_sensitive(
                str(context.get("sensitivity") or PRIVATE), PRIVATE
            )
        tier = router.choose(query, context)
        if images:
            tier = "main"
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
            # Only fold this turn's first-token latency into the LOCAL TTFT EWMA
            # when the answering model is purely local; a cloud-hedge win would
            # otherwise mislabel the headroom signal (P4 low).
            fold_local = _answers_locally(model)
            # Dynamic hedge timing (Task 4): when live routing is on and the
            # local tier looks slow/loaded, shorten the per-call hedge delay so
            # the cloud race starts sooner. Computed from the same live signal as
            # the nudge and clamped to a floor (core.routing) so it can never
            # reach zero/negative. ``None`` -> the constructor's static delay
            # stands, so default behaviour (live_routing off, good/missing
            # signal, or a pure-local model) is byte-identical. Forwarded only to
            # a backend that accepts the PINNED-CONTRACT keyword.
            stream_kwargs: dict[str, object] = {"system": system_for_call}
            if images:
                stream_kwargs["images"] = list(images)
            if live_routing:
                base_ms = _base_hedge_delay_ms(model)
                if base_ms is not None:
                    hd = dynamic_hedge_delay_ms(context.get(LIVE_CONTEXT_KEY), base_ms)
                    if hd is not None:
                        stream_kwargs["hedge_delay_ms"] = hd
            tokens = mark_first_token(
                model.stream(query, **stream_kwargs), recorder,  # type: ignore[arg-type]
                fold_local_ttft=fold_local,
            )
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
        tokens = mark_first_token(
            llm.stream(prompt, system=system), recorder,
            fold_local_ttft=_answers_locally(llm),
        )
        if callable(emit):
            text, cancelled = _stream_and_speak(tokens, cancel, emit)  # type: ignore[arg-type]
            return CapabilityResult(True, text, data={"cancelled": cancelled, "streamed": True})
        text, cancelled = _collect(tokens, cancel)  # type: ignore[arg-type]
        return CapabilityResult(True, text, data={"cancelled": cancelled})

    registry.register("assistant.answer", assistant)
    registry.register("research.local", research_synth)
    return registry
