"""Unified capability router -- the "middle layer" that decides, once per turn,
WHAT a turn needs before the brain answers it.

Today the decision is spread across separate gates: the addressing gate (act vs
ingest), the tier ``Router`` (fast vs main, ``core/routing.py``), and the ReAct
``escalate`` predicate (one-shot vs gather-with-tools, ``always_on_agent/react``).
This module fuses the *action* decision into one coherent, testable place that
picks among four capabilities:

* :data:`CONTROL`  -- stop / cancel / mode switch (instant, no LLM in the loop);
* :data:`SIMPLE`   -- a direct one-shot reply on the fast/main tier;
* :data:`RESEARCH` -- gather / multi-step / compare before answering (the planner);
* :data:`ACT`      -- the user wants an ACTION performed (set a timer, open, play,
  control a device) before/instead of an answer.

Design (matches the repo's router/endpoint pattern): a phone-safe deterministic
floor (:class:`HeuristicCapabilityRouter`) that *reuses* the existing signals --
the tier ``HeuristicRouter`` for fast/main, :func:`should_escalate` for the
gather markers, and :func:`is_stop_command` for control -- with an optional
fast-LLM disambiguator (:class:`LLMCapabilityRouter`) that only fires on a
LOW-confidence decision, so capable devices get smarter routing while weak ones
pay nothing. Device-adaptivity is therefore just config: ``llm_assist`` on/off
per ``device_profiles`` entry.

It does NOT re-implement execution: the runtime wires this one router to BACK
both the tier ``Router`` and the ``escalate`` predicate that
``attach_llm_capabilities`` already accepts (see :class:`CapabilityTierRouter`
and :func:`escalate_predicate`), so a single decision drives the existing brain
with no ``always_on_agent`` changes. When no router is configured, behaviour is
byte-identical to before.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Protocol, Sequence, runtime_checkable

from always_on_agent.react import should_escalate

from .contract import is_stop_command, normalize_command
from .llm import LLMClient
from .routing import (
    FAST,
    MAIN,
    HeuristicRouter,
    LatencyPolicy,
    Router,
    build_router,
    classify_latency_policy,
)

log = logging.getLogger("speaker.capability_router")

# The four capabilities a turn can route to.
CONTROL = "control"
SIMPLE = "simple"
RESEARCH = "research"
ACT = "act"

_ACTIONS = (CONTROL, SIMPLE, RESEARCH, ACT)

# Imperative phrases that signal the user wants an ACTION taken (a tool/function
# run) rather than just an answer. Deliberately distinct from research's gather
# markers (``react._ESCALATE_MARKERS``): those mean "look things up first", these
# mean "go do this". Conservative -- a miss just falls back to SIMPLE/RESEARCH.
_ACT_MARKERS: tuple[str, ...] = (
    "set a ", "set an ", "set the ", "set my ",
    "turn on", "turn off", "switch on", "switch off", "turn up", "turn down",
    "open ", "play ", "pause ", "resume ", "skip ", "mute", "unmute",
    "send ", "email ", "text ", "call ", "message ",
    "remind me", "reminder", "timer", "alarm", "schedule ",
    "add to", "create a ", "delete ", "remove ", "rename ",
    "volume", "brightness",
)


@dataclass(frozen=True)
class RouteDecision:
    """One turn's routing decision: which capability + which model tier."""

    action: str            # one of :data:`_ACTIONS`
    tier: str              # FAST or MAIN (core.routing)
    confidence: float      # 0..1; below the router's threshold -> LLM disambiguates
    reason: str            # short human-readable why (lands in logs / run summary)
    source: str = "heuristic"  # "heuristic" | "llm"
    latency_policy: str = LatencyPolicy.SNAPPY_ANSWER.value

    @property
    def escalates(self) -> bool:
        """Whether this turn should hand off to the ReAct planner (gather/act
        with tools) rather than answer one-shot. RESEARCH and ACT both gather
        with the planner's read-only tools in v1; ACT additionally gates any
        state-changing tool behind confirmation once such tools are registered."""
        return self.action in (RESEARCH, ACT)


@runtime_checkable
class CapabilityRouter(Protocol):
    """Maps an utterance + context to a :class:`RouteDecision`."""

    def route(self, text: str, context: Mapping[str, object]) -> RouteDecision: ...


class HeuristicCapabilityRouter:
    """Dependency-light unified router. Runs anywhere (phone included), no model.

    Order matters: control phrases are checked first (a "stop" must never be
    mistaken for a query), then explicit gather/action markers, then the
    fast/main tier for an ordinary reply."""

    def __init__(
        self,
        *,
        tier_router: Optional[Router] = None,
        command_phrases: Iterable[str] = (),
        act_markers: Sequence[str] = _ACT_MARKERS,
    ) -> None:
        self._tier = tier_router or HeuristicRouter()
        self._commands = {normalize_command(p) for p in command_phrases if p}
        self._act_markers = tuple(act_markers)

    def route(self, text: str, context: Mapping[str, object]) -> RouteDecision:
        q = (text or "").strip()
        ql = q.lower()
        ctx = context or {}

        # 1. CONTROL: a stop/cancel phrase or a configured command phrase. Instant
        #    and unambiguous -- never spend an LLM on it.
        if q and (is_stop_command(q) or normalize_command(q) in self._commands):
            return self._decision(q, ctx, CONTROL, FAST, 0.97, "control phrase")

        # 2. RESEARCH: reuse the exact predicate the planner already escalates on,
        #    so routing here never diverges from what the brain would have done.
        research = should_escalate(q, ctx)

        # 3. ACT: an imperative that wants something DONE (not just answered).
        act = any(m in ql for m in self._act_markers)
        if act and not research:
            return self._decision(q, ctx, ACT, MAIN, 0.7, "action marker")
        if research:
            return self._decision(q, ctx, RESEARCH, MAIN, 0.7, "gather/multi-step marker")

        # 4. SIMPLE: ordinary reply. Tier from the existing heuristic; confidence
        #    reflects how sure we are it ISN'T really a gather/act turn -- a long
        #    marker-free utterance is the ambiguous case the LLM should reconsider.
        tier = self._tier.choose(q, ctx)
        n = len(ql.split())
        if n <= 4:
            conf = 0.85
        elif n >= 12:
            conf = 0.5
        else:
            conf = 0.7
        return self._decision(q, ctx, SIMPLE, tier, conf, "no gather/action markers")

    @staticmethod
    def _decision(
        text: str,
        context: Mapping[str, object],
        action: str,
        tier: str,
        confidence: float,
        reason: str,
    ) -> RouteDecision:
        policy = classify_latency_policy(
            text,
            {**dict(context), "route_action": action, "tier": tier},
        )
        return RouteDecision(action, tier, confidence, reason, latency_policy=policy.value)


_LLM_SYSTEM = (
    "You route ONE turn of a voice assistant to a single capability.\n"
    "Reply with EXACTLY one word, no punctuation, no explanation:\n"
    "- SIMPLE   -- a direct question or chat you answer in one shot.\n"
    "- RESEARCH -- needs gathering, comparison, or multiple steps before answering.\n"
    "- ACT      -- the user wants you to DO something (set a timer, send a message,\n"
    "              open/play something, control a device).\n"
    "One word."
)

_LLM_ACTIONS = {SIMPLE.upper(): SIMPLE, RESEARCH.upper(): RESEARCH, ACT.upper(): ACT}


def _parse_action(reply: str) -> Optional[str]:
    """First word of an LLM reply -> a route action, or ``None`` if unparseable
    (so an LLM hiccup falls back to the heuristic, never a wrong confident route)."""
    stripped = (reply or "").strip()
    if not stripped:
        return None
    first = stripped.split()[0].strip(".,!?:;\"'").upper()
    return _LLM_ACTIONS.get(first)


class LLMCapabilityRouter:
    """Wraps a base router; on a LOW-confidence decision, asks the fast LLM to
    classify the *action*. The tier always comes from the (cheap, context-aware)
    base, so only the action is LLM-decided.

    The LLM action is memoized by utterance text: the runtime consults the router
    more than once per turn (for the escalate predicate AND the tier choice), and
    this keeps that to a single fast-LLM call per turn. CONTROL and confident
    decisions skip the LLM entirely."""

    def __init__(
        self,
        base: CapabilityRouter,
        llm: LLMClient,
        *,
        confidence_threshold: float = 0.65,
    ) -> None:
        self._base = base
        self._llm = llm
        self._threshold = float(confidence_threshold)
        self._cache_key: Optional[str] = None
        self._cache_action: Optional[str] = None

    def route(self, text: str, context: Mapping[str, object]) -> RouteDecision:
        base = self._base.route(text, context)
        if base.action == CONTROL or base.confidence >= self._threshold:
            return base
        action = self._action_for(text)
        if action is None or action == base.action:
            return base
        tier = MAIN if action in (RESEARCH, ACT) else base.tier
        policy = classify_latency_policy(
            text,
            {**dict(context), "route_action": action, "tier": tier},
        )
        return RouteDecision(
            action, tier, 0.9, "llm disambiguation", source="llm",
            latency_policy=policy.value,
        )

    def _action_for(self, text: str) -> Optional[str]:
        key = (text or "").strip().lower()
        if key and key == self._cache_key:
            return self._cache_action
        action: Optional[str]
        try:
            reply = self._llm.generate(self._prompt(text), system=_LLM_SYSTEM)
            action = _parse_action(reply)
        except Exception:  # noqa: BLE001 - routing must never break a turn
            log.exception("capability-router LLM failed; keeping the heuristic action")
            action = None
        if key:
            self._cache_key, self._cache_action = key, action
        return action

    @staticmethod
    def _prompt(text: str) -> str:
        return f'Turn: "{(text or "").strip()}"'


class CapabilityTierRouter:
    """Adapts a :class:`CapabilityRouter` to the tier ``Router`` Protocol so the
    one unified decision also drives the fast/main model choice in
    ``attach_llm_capabilities``."""

    def __init__(self, router: CapabilityRouter):
        self._router = router

    def score(self, query: str, context: Mapping[str, object]) -> float:
        return 1.0 if self._router.route(query, context).tier == MAIN else 0.0

    def choose(self, query: str, context: Mapping[str, object]) -> str:
        return self._router.route(query, context).tier


def escalate_predicate(router: CapabilityRouter):
    """A ``should_escalate``-shaped predicate backed by the unified router: a turn
    escalates to the planner exactly when the router routes it to RESEARCH/ACT."""

    def _escalate(query: str, context: Optional[Mapping[str, object]] = None) -> bool:
        return router.route(query, context or {}).escalates

    return _escalate


def build_capability_router(
    config: Optional[Mapping[str, object]],
    *,
    tier_router: Optional[Router] = None,
    fast_llm: Optional[LLMClient] = None,
    command_phrases: Iterable[str] = (),
) -> Optional[CapabilityRouter]:
    """Build the unified router from the ``capability_router`` config block.

    Returns ``None`` when the block is absent or ``enabled`` is false -- the
    caller then keeps the existing per-gate routing (byte-identical behaviour).
    ``llm_assist`` wraps the heuristic with the fast-LLM disambiguator only when a
    ``fast_llm`` is available, so a profile can turn it off (or simply omit the
    fast model) on weak devices."""
    cfg = (config or {}).get("capability_router", {}) or {}
    if not cfg.get("enabled", False):
        return None
    base = HeuristicCapabilityRouter(
        tier_router=tier_router or build_router(config),
        command_phrases=command_phrases,
    )
    if cfg.get("llm_assist", False) and fast_llm is not None:
        return LLMCapabilityRouter(
            base, fast_llm,
            confidence_threshold=float(cfg.get("confidence_threshold", 0.65) or 0.65),
        )
    return base
