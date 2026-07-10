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
from always_on_agent.procedural import extract_rule, render_rules
from always_on_agent.recall import VISION_LABEL, compress, trim_block_to_tokens
from always_on_agent.untrusted import detect_injection, wrap_untrusted
from always_on_agent.models import IntentKind

from .contract import drain_complete_sentences
from .conversation import (
    RecentContextConfig,
    collect_recent_turns,
    format_recent_block,
    history_messages,
)
from .llm import HedgeLLM, LLMClient, SensitivityRouterLLM, capability_context
from .metrics import LLM_FIRST_TOKEN, MetricsRecorder, mark_first_token
from .persona import DEFAULT_SYSTEM
from .routing import (
    LIVE_CONTEXT_KEY,
    HeuristicRouter,
    Router,
    dynamic_hedge_delay_ms,
)
from .sensitivity import PRIVATE, classify_sensitivity, most_sensitive

log = logging.getLogger("speaker.llm")

# Context-dict key under which assistant() publishes the bounded recent-conversation
# block so the escalated ReAct planner sees "what was just said" too (not only the
# one-shot answer path). Read by literal in always_on_agent/react.py (the brain
# avoids importing core), so keep the two in sync.
RECENT_CONVERSATION_KEY = "recent_conversation"


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
    embedding/recall work happens. ``max_tokens`` bounds how much of the recall
    block is prepended so TTFT stays bounded -- a whole-line token cap at the
    injection site, a secondary safety net on top of the memory's own
    :class:`~always_on_agent.recall.RecallBudget`. ``max_chars`` is a DEPRECATED
    alias kept for back-compat: when set it derives ``max_tokens = max_chars//4``
    (the same ratio the rest of the recall path uses)."""

    enabled: bool = False
    max_tokens: int = 150
    max_chars: Optional[int] = None  # deprecated; derives max_tokens when set
    # Procedural memory (user-taught behavior rules): capture explicit teach
    # directives + inject the rule block on every turn. Default OFF (opt-in), so the
    # capture + injection are skipped and the prompt is byte-identical until enabled.
    procedural_enabled: bool = False

    def __post_init__(self) -> None:
        if self.max_chars is not None:
            object.__setattr__(self, "max_tokens", max(1, int(self.max_chars) // 4))

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "RecallConfig":
        data = data or {}
        tok = data.get("recall_max_tokens")
        if tok is None:  # legacy: derive from the deprecated char cap, else default
            ch = data.get("recall_max_chars")
            tok = (int(ch) // 4) if ch else 150
        return cls(
            enabled=bool(data.get("recall_enabled", False)),
            max_tokens=max(1, int(tok)),
            procedural_enabled=bool(data.get("procedural_enabled", False)),
        )


def _close_token_stream(tokens: Iterator[str]) -> None:
    """Best-effort close of a token generator after a barge-in cut.

    The token chain is ``mark_first_token(model.stream(...))`` -- generators whose
    ``finally`` blocks tear down the underlying HTTP body / SDK stream (Ollama,
    OpenAI-compat) so provider cancellation starts promptly. On a plain ``break``,
    those ``finally`` blocks only run when the generator is garbage-collected,
    which is not deterministic (and the ``mark_first_token`` wrapper adds a layer
    that must ALSO be collected) -- so an interrupted provider request could
    linger until GC catches up. Calling ``close()`` here propagates
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
    the local consumer off instead of waiting for the whole answer. On a cut
    the stream is closed explicitly so SDK/provider cleanup begins immediately
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
        # Barge-in cut: start provider cancellation/cleanup immediately rather
        # than at GC time (see :func:`_close_token_stream`).
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
    # lm-2 Wire 3: a process-start latch so the one-shot "Last session" recap is
    # prepended on the FIRST one-shot answer turn only, then never again this run.
    # Mutable dict so the per-turn closure can flip it without ``nonlocal``.
    memory_state = {"last_session_injected": False}
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

    def _metric_turn_token(context: Mapping[str, object]) -> object:
        """Original ASR turn identity carried through queued/multistep work."""
        meta = context.get("metadata")
        if isinstance(meta, Mapping) and "metrics_turn_token" in meta:
            return meta["metrics_turn_token"]
        return recorder.current_turn_token() if recorder is not None else None

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
        metric_turn = _metric_turn_token(context)
        # Classify intent + sensitivity once, up front, for BOTH the one-shot and
        # the escalated path (idempotent: only fills what's absent).
        _enrich_context(query, context)
        # A continuation turn (ADD-ON) carries a synthetic, folded prompt and the
        # supervisor has already ingested the raw user utterance, so skip the query
        # ingest + recent block here -- otherwise memory would hold both the prior
        # turn and the merged "prev + addon" prompt (the double-ingest the design
        # flagged), and its folded prompt already embeds the prior context.
        meta = context.get("metadata")
        skip_user_memory = bool(meta.get("skip_user_memory")) if isinstance(meta, dict) else False
        # Procedural capture (default-OFF): detect an explicit teach directive in the
        # LIVE user query and store it as a durable behavior rule -- done BEFORE the
        # escalation branch so an escalated teach utterance still registers. The rule
        # rides every later turn. detect_injection() guards against a garbled /
        # bystander / self-transcribed directive becoming a TRUSTED rule. Best-effort.
        if memory is not None and not skip_user_memory and recall_cfg.procedural_enabled:
            try:
                rule = extract_rule(query)
                if rule and not detect_injection(rule):
                    memory.add(rule, tags=("procedural",))
            except Exception:  # noqa: BLE001 - capture is best-effort, never fatal
                log.exception("procedural rule capture failed; continuing")
        # Recent-conversation context (short-term memory): the PRIOR turns, built
        # ONCE here -- BEFORE the escalation branch and BEFORE ingesting the current
        # query -- so the model can resolve "its"/"the second one". Built up front
        # (not just on the one-shot path) so an ESCALATED "explain that step by
        # step" reaches the ReAct planner WITH the conversation thread instead of a
        # contextless query. §9.7: a private prior turn must float the turn's
        # sensitivity to the most-private over EVERY included prior turn before
        # EITHER path's LLM calls run. Best-effort: never break a turn.
        recent_block = ""
        recent_history: list[dict] = []
        try:
            # current_query=query: a reset utterance ("start again") answers
            # FRESH (no block), and a reset turn in memory cuts the thread for
            # later turns -- see core/conversation.is_topic_reset.
            collect_cfg = recent_cfg
            if recent_cfg.as_messages:
                # R11 messages mode: collect MORE turns at a FULLER per-turn cap --
                # the volume rides the chat-messages budget, not the text max_chars.
                from dataclasses import replace

                collect_cfg = replace(
                    recent_cfg,
                    max_turns=recent_cfg.messages_max_turns,
                    per_turn_chars=recent_cfg.messages_per_turn_chars,
                )
            recent_turns = (
                collect_recent_turns(memory, collect_cfg, current_query=query)
                if (memory is not None and not skip_user_memory)
                else []
            )
            if recent_turns:
                sens = context.get("sensitivity") or PRIVATE
                for _role, turn_text in recent_turns:
                    sens = most_sensitive(str(sens), classify_sensitivity(turn_text))
                context["sensitivity"] = sens
                # Messages mode -> role-structured history for the chat API; text
                # mode -> the historical pasted-into-system block (byte-identical).
                if recent_cfg.as_messages:
                    recent_history = history_messages(recent_turns)
                else:
                    recent_block = format_recent_block(recent_turns, recent_cfg)
        except Exception:  # noqa: BLE001 - context is best-effort, never fatal
            log.exception("recent-conversation context failed; continuing without it")
        if recent_block:
            context[RECENT_CONVERSATION_KEY] = recent_block
        # Smart-mode escalation: hand reasoning/gathering queries to the ReAct
        # planner (when registered) instead of a one-shot reply.
        if (
            escalate is not None
            and agent_capability in registry.names()
            and escalate(query, context)
        ):
            # An escalated (ReAct planner) turn must STILL publish capability_context
            # so SensitivityRouterLLM can pick the right cloud chain for the nested
            # LLM calls the planner makes (sensitivity already classified + floated
            # over the recent block above). Reset via finally so the ContextVar is
            # restored on EVERY return/raise out of the planner (a missed reset is a
            # cross-turn sensitivity LEAK -- the §9.7 risk the isolation tests guard).
            if recorder is not None:
                context["first_token_hook"] = lambda: recorder.mark(
                    LLM_FIRST_TOKEN,
                    turn_token=metric_turn,
                )
            ctx_token = capability_context.set(context)
            try:
                return registry.invoke(agent_capability, query, context)
            finally:
                capability_context.reset(ctx_token)
        # One-shot answer path. Memory recall (the headline P2 fix), assistant-only
        # (R5): ingest the answered query first (R1) so recall has something to find
        # next turn (never mutate ``query`` itself -- the router/sensitivity inputs
        # stay clean), then a gated prepend only when recall is enabled AND the
        # backend returns a non-empty block. Wrapped so a memory error can't break
        # a live turn.
        system_for_call = system
        if memory is not None:
            try:
                if not skip_user_memory:
                    memory.add(query, tags=("user",))
                # Episodic recall (default-OFF) brings the FULL memory context
                # (recall + vision + profile, sharing one budget). When recall is
                # OFF we still surface durable PROFILE facts alone (Recall-B:
                # decoupled from recall_enabled) so the model knows WHO it's talking
                # to without paying for episodic recall. Both return '' at their
                # defaults (recall off + profiles off), so the opening turn stays
                # byte-identical. ``profile_block`` is getattr-guarded so a minimal
                # duck-typed Memory double still works.
                if recall_cfg.enabled:
                    recall_block = memory.context_for_llm(query)
                else:
                    profile_fn = getattr(memory, "profile_block", None)
                    recall_block = profile_fn() if callable(profile_fn) else ""
                # lm-2 Wire 3: a one-shot "Last session" recap, prepended on the
                # FIRST one-shot answer turn since process start only (a
                # process-start latch), ahead of recall. Empty unless
                # cross_session_continuity seeded a prior-session head
                # (Postgres-only) -- '' otherwise keeps the default-OFF opening turn
                # byte-identical. The head is compressed (whole-word, never
                # mid-word) to the recall token budget so it can't blow up TTFT.
                last_session_block = ""
                if not memory_state["last_session_injected"]:
                    head_fn = getattr(memory, "last_session_summary", None)
                    head = head_fn() if callable(head_fn) else ""
                    if head:
                        head = compress(head, query, recall_cfg.max_tokens)
                        last_session_block = f"=== Last Session ===\n{head}"
                        # Burn the one-shot latch ONLY once the recap is actually
                        # built: a transient failure while fetching/compressing the
                        # head then stays retryable next turn instead of silently
                        # suppressing the recap for the whole process, and an empty
                        # head (continuity off / no prior summary) never burns it
                        # (every turn just recomputes head="" -> no block, so the
                        # default-OFF opening turn stays byte-identical regardless).
                        memory_state["last_session_injected"] = True
                # §9.7: ALL remembered text (last-session head, recall, profile) is
                # PRIVATE-floatable post-ASR content -- float the turn's sensitivity
                # over the COMBINED block before any LLM call, exactly as the
                # recent-turns block does, so a private remembered fact ("my salary
                # is...") can never ride the cheapest/public chain (review finding
                # lm-3; must hold before recall/continuity is enabled by default).
                memory_blocks = [b for b in (last_session_block, recall_block) if b]
                if memory_blocks:
                    combined = "\n\n".join(memory_blocks)
                    context["sensitivity"] = most_sensitive(
                        str(context.get("sensitivity") or PRIVATE),
                        classify_sensitivity(combined),
                    )
                    # A recalled VISUAL (screen) observation is PRIVATE by policy
                    # (§9.7) regardless of what its caption/OCR text classifies as,
                    # so a screen memory surfacing into a turn pins it to the
                    # private chain even when the words look benign.
                    if VISION_LABEL in combined:
                        context["sensitivity"] = most_sensitive(
                            str(context.get("sensitivity") or PRIVATE), PRIVATE
                        )
                # Compose: stable system FIRST (the pre-warmed, cacheable prefix),
                # then the volatile recent block AFTER, so the prefix cache is
                # reused turn to turn. Remembered blocks (last-session recap, then
                # recall; default-off, past sessions) keep their historical position
                # ahead of system. Secondary, whole-LINE token cap (never a mid-word
                # cut) on the recall block: it already arrived budget-bounded from
                # the memory's own RecallBudget, so this only bites if max_tokens is
                # set tighter here. Replaces the old blunt recall_block[:max_chars].
                recall_block = trim_block_to_tokens(recall_block, recall_cfg.max_tokens)
                prefix_parts = [b for b in (last_session_block, recall_block) if b]
                # Prompt-injection hardening (OWASP LLM01): the recalled/last-session
                # blocks carry content the assistant did NOT author -- prior-session
                # messages AND 'Screen:' OCR/caption lines (a prime indirect-injection
                # vector). Spotlight them as UNTRUSTED data with a never-obey directive
                # so an instruction smuggled into a recalled line / on-screen text
                # can't hijack the turn. No-op + byte-identical when nothing recalls.
                memory_block = (
                    wrap_untrusted("\n\n".join(prefix_parts), source="memory")
                    if prefix_parts else ""
                )
                # Procedural memory: standing user-taught behavior rules, injected on
                # EVERY turn as a small TRUSTED instruction block (the user authored
                # them, so -- unlike recalled/screen/web content -- they are NOT
                # spotlighted as untrusted). Placed adjacent to the system prompt so
                # they read as authoritative instructions. Empty (-> byte-identical)
                # when procedural memory is off or no rules are stored.
                procedural_block = ""
                if recall_cfg.procedural_enabled:
                    rules_fn = getattr(memory, "procedural_rules", None)
                    rules = list(rules_fn()) if callable(rules_fn) else []
                    block = render_rules(rules)
                    if block:
                        procedural_block = block
                        # §9.7: a rule may carry private info ("address me as <X>"),
                        # so float the turn's sensitivity over it before any LLM call
                        # -- a private rule must not pin the turn to a public chain.
                        context["sensitivity"] = most_sensitive(
                            str(context.get("sensitivity") or PRIVATE),
                            classify_sensitivity(block),
                        )
                # R06b: the STABLE system is FIRST (the pre-warmed, cacheable
                # prefix), then the session-stable procedural rules (trusted +
                # authoritative), then the VOLATILE blocks -- recall/last-session
                # (untrusted-wrapped) and the recent-turns text. Keeping every
                # per-turn-varying block AFTER system means Ollama's
                # longest-common-prefix KV cache is reused turn to turn instead of
                # being busted by a changed recall block (llm-inference-1). Default
                # config (recall + procedural off, text recent block) reduces to
                # `system` + the recent suffix -> byte-identical to before.
                system_for_call = "\n\n".join(
                    p for p in (system, procedural_block, memory_block, recent_block) if p
                )
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
        primary = llm if tier == "main" else fast
        other = fast if tier == "main" else llm
        other_name = "fast" if tier == "main" else "main"
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
        # Track whether any audio was emitted so a retry never double-speaks
        # (sr-2): if the primary already streamed a sentence before erroring, we
        # do NOT retry on the other tier.
        emitted_any = [False]

        def _emit_tracked(sentence: str) -> None:
            emitted_any[0] = True
            emit(sentence)  # type: ignore[misc]

        def _attempt(attempt_model: LLMClient, attempt_tier: str) -> CapabilityResult:
            # A retry can be entered after the primary failed but while barge-in
            # races the exception handler.  Check again at attempt entry; the
            # token wrapper below performs the definitive pre-yield gate.
            if cancel is not None and cancel.is_set():  # type: ignore[union-attr]
                raise RuntimeError("task cancelled before model attempt")
            # Only fold this turn's first-token latency into the LOCAL TTFT EWMA
            # when the answering model is purely local; a cloud-hedge win would
            # otherwise mislabel the headroom signal (P4 low).
            fold_local = _answers_locally(attempt_model)
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
            # R11: prior turns as role-structured chat messages (messages mode
            # only; empty otherwise, so the default single-turn call is unchanged).
            if recent_history:
                stream_kwargs["history"] = recent_history
            if live_routing:
                base_ms = _base_hedge_delay_ms(attempt_model)
                if base_ms is not None:
                    hd = dynamic_hedge_delay_ms(context.get(LIVE_CONTEXT_KEY), base_ms)
                    if hd is not None:
                        stream_kwargs["hedge_delay_ms"] = hd
            tokens = mark_first_token(
                attempt_model.stream(query, **stream_kwargs), recorder,  # type: ignore[arg-type]
                fold_local_ttft=fold_local,
                cancel=cancel if isinstance(cancel, Event) else None,
                turn_token=metric_turn,
            )
            if callable(emit):
                text, cancelled = _stream_and_speak(tokens, cancel, _emit_tracked)  # type: ignore[arg-type]
                log.info(
                    "%s tier %s in %.2fs (%d chars, streamed)",
                    attempt_tier, "cancelled" if cancelled else "done",
                    time.monotonic() - started, len(text),
                )
                return CapabilityResult(
                    True,
                    text or "Sorry, I don't have an answer for that.",
                    data={
                        "route": attempt_tier,
                        "streamed": True,
                        "cancelled": cancelled,
                        "sensitivity": context.get("sensitivity"),
                    },
                )
            text, cancelled = _collect(tokens, cancel)  # type: ignore[arg-type]
            log.info(
                "%s tier %s in %.2fs (%d chars)",
                attempt_tier, "cancelled" if cancelled else "done",
                time.monotonic() - started, len(text),
            )
            if cancelled:
                return CapabilityResult(
                    True, text,
                    data={"cancelled": True, "route": attempt_tier, "sensitivity": context.get("sensitivity")},
                )
            return CapabilityResult(
                True,
                text or "Sorry, I don't have an answer for that.",
                data={"route": attempt_tier, "sensitivity": context.get("sensitivity")},
            )

        # Publish the per-turn context so SensitivityRouterLLM can pick the
        # right cloud chain at stream() time. Set ONCE around both attempts so the
        # chain choice is consistent across a retry. Reset on the way out so the
        # ContextVar doesn't leak between unrelated turns.
        ctx_token = capability_context.set(context)
        try:
            try:
                return _attempt(primary, tier)
            except Exception:  # noqa: BLE001 - cross-tier fallback (sr-2)
                # The task coordinator may already have detached this provider
                # after a barge-in.  If its blocked stream later wakes by
                # raising, do not turn that stale failure into a brand-new call
                # on the other tier.  The cancelled task has no consumer left.
                if cancel is not None and cancel.is_set():  # type: ignore[union-attr]
                    raise
                # If the chosen tier's backend errors (e.g. Ollama unreachable,
                # fast-tier GGUF load failure), retry once on the OTHER tier
                # before failing the turn -- but only when (a) the other tier is
                # a DISTINCT model (fast = fast_llm or llm, so an un-split config
                # has one object and a retry would just fail again), and (b) no
                # audio was emitted yet (else the retry would double-speak).
                if other is primary or emitted_any[0]:
                    raise
                log.warning(
                    "%s tier failed; retrying on %s tier", tier, other_name, exc_info=True
                )
                claim_start = context.get("claim_provider_start")
                if callable(claim_start) and not claim_start():
                    raise
                return _attempt(other, other_name)
        finally:
            capability_context.reset(ctx_token)

    def research_synth(query: str, context: dict[str, object]) -> CapabilityResult:
        metric_turn = _metric_turn_token(context)
        previous = context.get("previous_steps", [])
        gathered_parts: list[str] = []
        for step in previous:
            if not (isinstance(step, dict) and step.get("text")):
                continue
            text = str(step["text"])
            # Prompt-injection hardening (OWASP LLM01), RESEARCH plan path: a gathered
            # step that egressed (web.search) carries attacker-controllable webpage
            # text -- fence it as UNTRUSTED before folding it into the synthesis
            # prompt (same defense as the ReAct path) so an injected page can't steer
            # the summary. Local steps are left as-is, byte-identical.
            data = step.get("data")
            if isinstance(data, dict) and data.get("egress"):
                text = wrap_untrusted(text, source="web")
            gathered_parts.append(text)
        gathered = " ".join(gathered_parts)
        prompt = (
            f"Question: {query}\n"
            f"Local findings: {gathered or '(none)'}\n"
            "Give a brief spoken-style summary and one concrete recommendation."
        )
        cancel = context.get("cancel_event")
        emit = context.get("emit_speech")
        # Classify + publish sensitivity for the synthesis call too: research.local
        # streams through the same SensitivityRouterLLM, so without this its turns
        # always fell back to the default cloud chain (smart-routing-2 / backlog P2
        # (d)). Enrich first (no-op if sensitivity already present), then set the
        # ContextVar and reset it in a finally so it is restored on EVERY exit
        # path -- a missed reset would leak this turn's tag into the next, unrelated
        # turn (the §9.7 cross-turn leak the isolation tests guard).
        _enrich_context(query, context)
        ctx_token = capability_context.set(context)
        try:
            tokens = mark_first_token(
                llm.stream(prompt, system=system), recorder,
                fold_local_ttft=_answers_locally(llm),
                cancel=cancel if isinstance(cancel, Event) else None,
                turn_token=metric_turn,
            )
            if callable(emit):
                text, cancelled = _stream_and_speak(tokens, cancel, emit)  # type: ignore[arg-type]
                return CapabilityResult(True, text, data={"cancelled": cancelled, "streamed": True})
            text, cancelled = _collect(tokens, cancel)  # type: ignore[arg-type]
            return CapabilityResult(True, text, data={"cancelled": cancelled})
        finally:
            capability_context.reset(ctx_token)

    registry.register("assistant.answer", assistant)
    registry.register("research.local", research_synth)
    return registry
