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
import threading
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Protocol, Sequence, runtime_checkable

from always_on_agent.react import should_escalate

from .contract import is_stop_command, normalize_command
from .llm import LLMCallCancelled, LLMClient, collect_llm_text
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

# Phrases that may introduce a genuine request before its action verb.  Strip
# only complete leading phrases: an informational question such as "can you
# explain how email works" then starts with ``explain`` and cannot be promoted
# merely because ``email`` appears later.
_EXPLICIT_ACTION_REQUEST_PREFIXES = frozenset({
    "i would like you to", "i would like to", "i want you to", "i need you to",
    "id like you to", "i want to", "i need to", "id like to",
    "could you", "would you", "can you", "will you", "would you mind",
    "do you mind", "help me", "please", "kindly",
})
_ACTION_REQUEST_PREFIXES: tuple[str, ...] = tuple(sorted(
    (*_EXPLICIT_ACTION_REQUEST_PREFIXES, "just", "quickly"),
    key=len,
    reverse=True,
))

# A request prefix commonly selects the gerund ("would you mind turning the
# lights off"). Rewrite only supported action verbs after such a prefix; this
# must not become a general stemmer for informational prose.
_REQUEST_GERUNDS = {
    "turning": "turn", "switching": "switch", "setting": "set",
    "opening": "open", "playing": "play", "pausing": "pause",
    "resuming": "resume", "skipping": "skip", "muting": "mute",
    "unmuting": "unmute", "sending": "send", "emailing": "email",
    "texting": "text", "calling": "call", "scheduling": "schedule",
    "creating": "create", "deleting": "delete", "removing": "remove",
    "renaming": "rename",
}

# English phrasal verbs can place their particle after the object ("turn the
# lights off").  These are the only action markers for which that separated
# shape is accepted; keeping the set explicit avoids turning arbitrary words in
# the middle of an informational question into an action.
_SEPARABLE_ACT_MARKERS = frozenset({
    "turn on", "turn off", "turn up", "turn down",
    "switch on", "switch off",
})

# These leading terms can be terse commands ("volume down", "open calendar")
# or topic fragments ("alarm fatigue", "open source"). Keep conservative ACT
# fallback, but make it low-confidence so the configured fast model gets the
# chance to disambiguate it.
_AMBIGUOUS_ACT_MARKERS = frozenset({
    "open", "play", "pause", "resume", "skip", "mute", "unmute",
    "send", "email", "text", "call", "message", "schedule",
    "delete", "remove", "rename",
    "reminder", "timer", "alarm", "volume", "brightness",
})

# Explicit verb + device/object grammars that the old substring matcher caught
# but a leading-marker matcher otherwise loses ("set timer", "lower volume").
# The target must immediately follow bounded determiners, so informational text
# such as "show me what alarm fatigue means" cannot match.
_TARGETED_ACTION_VERBS = {
    "timer": frozenset(
        {"set", "start", "stop", "cancel", "reset", "create", "delete"}
    ),
    "alarm": frozenset(
        {
            "set", "start", "stop", "cancel", "reset", "create", "delete",
            "snooze",
        }
    ),
    "reminder": frozenset(
        {"set", "add", "cancel", "create", "delete", "remove", "show", "list"}
    ),
    "volume": frozenset(
        {"set", "change", "adjust", "raise", "lower", "increase", "decrease"}
    ),
    "brightness": frozenset(
        {"set", "change", "adjust", "raise", "lower", "increase", "decrease"}
    ),
}
_TARGET_ALIASES = {
    "timers": "timer", "alarms": "alarm", "reminders": "reminder",
}
_TARGET_DETERMINERS = frozenset({"a", "an", "the", "my", "that", "this", "me"})


def _normalize_action_text(text: str) -> str:
    """Lowercase/collapse speech punctuation while retaining marker identity.

    Control-command normalization deliberately removes digits and punctuation;
    action routing cannot use it because ``call 911`` and custom markers such as
    ``c++`` or ``open:`` need those characters to remain meaningful.
    """
    normalized = (text or "").casefold().replace("'", "").replace("’", "")
    for separator in ",.?!;":
        normalized = normalized.replace(separator, " ")
    return " ".join(normalized.split())


def _action_request_body(text: str) -> tuple[str, bool]:
    """Normalized body plus whether an explicit request prefix was present."""
    body = _normalize_action_text(text)
    explicit_request = False
    while body:
        prefix = next(
            (p for p in _ACTION_REQUEST_PREFIXES if body.startswith(f"{p} ")),
            None,
        )
        if prefix is None:
            break
        body = body[len(prefix):].lstrip()
        explicit_request = bool(
            explicit_request or prefix in _EXPLICIT_ACTION_REQUEST_PREFIXES
        )
    if explicit_request and body:
        first, separator, tail = body.partition(" ")
        replacement = _REQUEST_GERUNDS.get(first)
        if replacement is not None:
            body = replacement + (separator + tail if separator else "")
    return body, explicit_request


def _command_action_marker(
    body: str,
    markers: Sequence[tuple[str, bool]],
) -> Optional[str]:
    """Return the leading command marker, never a mid-question substring.

    ``bool`` records whether the configured marker ended in a space and
    therefore historically required an argument (``"open "`` must not make a
    bare ``"open"`` actionable).
    """
    words = body.split()
    marker_requirements = dict(markers)
    if len(words) >= 2:
        verb = words[0]
        index = 1
        while index < len(words) and words[index] in _TARGET_DETERMINERS:
            index += 1
        if index < len(words):
            target = _TARGET_ALIASES.get(words[index], words[index])
            if (
                target in marker_requirements
                and verb in _TARGETED_ACTION_VERBS.get(target, ())
                and (
                    not marker_requirements[target]
                    or index + 1 < len(words)
                )
            ):
                return f"{verb} {target}"
    for marker, requires_argument in markers:
        if body.startswith(f"{marker} "):
            return marker
        if body == marker and not requires_argument:
            return marker
        if marker in _SEPARABLE_ACT_MARKERS and len(words) > 1:
            verb, particle = marker.split()
            if words[0] == verb and particle in words[1:]:
                return marker
    return None


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
        prepared = (
            (_normalize_action_text(marker), marker.endswith(" "))
            for marker in act_markers
            if _normalize_action_text(marker)
        )
        self._act_markers = tuple(
            sorted(prepared, key=lambda item: len(item[0]), reverse=True)
        )

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

        # 3. ACT: a command-shaped imperative that wants something DONE (not an
        #    informational question which merely contains an action noun/verb).
        action_body, explicit_request = _action_request_body(q)
        act_marker = _command_action_marker(
            action_body,
            self._act_markers,
        )
        if act_marker is not None and not research:
            ambiguous = bool(
                act_marker in _AMBIGUOUS_ACT_MARKERS and not explicit_request
            )
            return self._decision(
                q,
                ctx,
                ACT,
                MAIN,
                0.5 if ambiguous else 0.7,
                (
                    "ambiguous leading action term"
                    if ambiguous
                    else "command-shaped action marker"
                ),
            )
        if research:
            return self._decision(q, ctx, RESEARCH, MAIN, 0.7, "gather/multi-step marker")

        # 4. SIMPLE: ordinary reply. Tier from the existing heuristic; confidence
        #    reflects how sure we are it ISN'T really a gather/act turn -- a long
        #    marker-free utterance is the ambiguous case the LLM should reconsider.
        tier = self.answer_tier(q, ctx)
        n = len(ql.split())
        if n <= 4:
            conf = 0.85
        elif n >= 12:
            conf = 0.5
        else:
            conf = 0.7
        return self._decision(q, ctx, SIMPLE, tier, conf, "no gather/action markers")

    def answer_tier(self, text: str, context: Mapping[str, object]) -> str:
        """Tier an ordinary answer independently of an ACT/RESEARCH decision.

        The LLM disambiguator uses this when it demotes a low-confidence action
        noun back to SIMPLE; otherwise the provisional ACT decision's MAIN tier
        would leak into the corrected answer and still bypass the fast model.
        """
        return self._tier.choose(text, context)

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
        self._cache_lock = threading.Lock()

    def route(self, text: str, context: Mapping[str, object]) -> RouteDecision:
        base = self._base.route(text, context)
        if base.action == CONTROL or base.confidence >= self._threshold:
            return base
        action = self._action_for(
            text,
            cancel_event=context.get("cancel_event"),
        )
        if action is None or action == base.action:
            return base
        if action in (RESEARCH, ACT):
            tier = MAIN
        elif action == SIMPLE:
            answer_tier = getattr(self._base, "answer_tier", None)
            tier = (
                answer_tier(text, context)
                if callable(answer_tier)
                else base.tier
            )
        else:
            tier = base.tier
        policy = classify_latency_policy(
            text,
            {**dict(context), "route_action": action, "tier": tier},
        )
        return RouteDecision(
            action, tier, 0.9, "llm disambiguation", source="llm",
            latency_policy=policy.value,
        )

    def _action_for(
        self,
        text: str,
        *,
        cancel_event: object | None = None,
    ) -> Optional[str]:
        key = (text or "").strip().lower()
        if key:
            with self._cache_lock:
                if key == self._cache_key:
                    return self._cache_action
        action: Optional[str]
        try:
            reply = collect_llm_text(
                self._llm,
                self._prompt(text),
                system=_LLM_SYSTEM,
                cancel_event=cancel_event,
            )
            action = _parse_action(reply)
        except LLMCallCancelled:
            raise
        except Exception:  # noqa: BLE001 - routing must never break a turn
            log.exception("capability-router LLM failed; keeping the heuristic action")
            action = None
        if key:
            with self._cache_lock:
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
