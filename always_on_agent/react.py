"""A bounded ReAct planner capability (LLM plans, capabilities are the tools).

Reimplements the design of Leon's agent mode (MIT, github.com/leon-ai/leon) in
Python: a plan -> execute -> recover -> final loop where each step the model
either calls one of the registered capabilities (the "tools") or emits a final
answer. The existing ``CapabilityRegistry`` *is* the toolset, so no new tool
contract is introduced; cancellation flows through ``context['cancel_event']``
exactly like every other capability, keeping the loop barge-in responsive.

This module stays free of any ``core`` import (``core`` depends on
``always_on_agent``, not the other way round), so the LLM is accepted via a
small structural protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import Event
from typing import Callable, Iterator, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .capabilities import CapabilityRegistry, CapabilityResult, CapabilitySpec
from .untrusted import wrap_untrusted


@runtime_checkable
class SupportsLLM(Protocol):
    def generate(self, prompt: str, *, system: Optional[str] = None) -> str: ...

    def stream(self, prompt: str, *, system: Optional[str] = None) -> Iterator[str]: ...


# Capabilities the planner may call by default: read-only gather/synthesis tools.
# ``assistant.answer`` is excluded so the planner never recurses into itself.
# ``web.search`` leads ``search.local``: it queries real web search (self-hosted
# SearXNG) when permitted by the §9.7 egress gate, and falls back to the local
# corpus otherwise -- so it's the right default gather tool. This tuple matches
# the ``planner_tool=True`` capabilities in ``create_default_capabilities``; the
# startup reconciliation check (core/runtime.py) warns if they drift.
DEFAULT_TOOLS: tuple[str, ...] = (
    "web.search",
    "search.local",
    "research.scope",
    "research.local",
)

PLANNER_SYSTEM = (
    "You are the planning loop of a local voice assistant. Decide the next step "
    "to answer the user. Respond with EXACTLY one line, no extra text:\n"
    "  TOOL <tool_name>: <input>   to call a tool, or\n"
    "  FINAL: <answer>             when you can answer directly.\n"
    "Prefer FINAL as soon as you have enough. Never invent tools."
)

FINAL_SYSTEM = (
    "You are a local, on-device voice assistant. Answer the user's request in "
    "one or two short, natural spoken sentences. Use the findings when they are "
    "relevant; if they are empty or unhelpful, answer from your own knowledge "
    "instead -- never reply that you cannot answer a general-knowledge question. "
    "No markdown or preambles."
)

_LINE = re.compile(r"^\s*(TOOL|FINAL)\b", re.IGNORECASE)
_TOOL_LINE = re.compile(r"^\s*TOOL\s+([\w.]+)\s*:?\s*(.*)$", re.IGNORECASE | re.DOTALL)
_FINAL_LINE = re.compile(r"^\s*FINAL\s*:?\s*(.*)$", re.IGNORECASE | re.DOTALL)

# Heuristic markers that an ASSISTANT-mode query wants tool use / gathering
# rather than a one-shot reply (used by the smart-mode escalation policy).
_ESCALATE_MARKERS = (
    "search",
    "look up",
    "look for",
    "find",
    "research",
    "compare",
    "latest",
    "options",
    "list of",
    "and then",
)
# NB: "step by step" was DROPPED as an escalation marker. It asks for answer
# *structure* ("explain how rainbows form step by step"), not for tool use, so it
# was sending pure general-knowledge questions into the ReAct tool planner -- which
# then called the local/web search stubs, got an irrelevant hit, and answered with
# that junk instead of from the model's own knowledge. Genuine gathering turns
# still escalate via "search"/"find"/"compare"/"latest"/etc.; every routing test
# that paired "step by step" with one of those (e.g. "compare a and b step by
# step") still escalates on the surviving marker.


@dataclass
class PlannerConfig:
    enabled: bool = False
    max_steps: int = 4
    tools: tuple[str, ...] = DEFAULT_TOOLS
    escalate: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> "PlannerConfig":
        data = data or {}
        tools = data.get("tools")
        return cls(
            enabled=bool(data.get("enabled", False)),
            max_steps=int(data.get("max_steps", 4)),
            tools=tuple(tools) if tools else DEFAULT_TOOLS,
            escalate=bool(data.get("escalate", True)),
        )


def should_escalate(query: str, context: Mapping[str, object] | None = None) -> bool:
    """True when an ASSISTANT-mode query looks like it needs tools/gathering."""
    q = (query or "").lower()
    if sum(1 for m in _ESCALATE_MARKERS if m in q) >= 1 and len(q.split()) >= 4:
        return True
    return False


def _cancelled(cancel: Optional[Event]) -> bool:
    return cancel is not None and cancel.is_set()


def _parse_step(text: str) -> tuple[Optional[str], str]:
    """Return ``(action, argument)``.

    ``action`` is a tool name, the literal ``"FINAL"``, or ``None`` when the
    model produced something unparseable (treated as "wrap up").
    """
    first = text.strip().splitlines()[0] if text.strip() else ""
    if not _LINE.match(first):
        return None, text.strip()
    final = _FINAL_LINE.match(first)
    if final:
        return "FINAL", final.group(1).strip()
    tool = _TOOL_LINE.match(first)
    if tool:
        return tool.group(1).strip(), tool.group(2).strip()
    return None, text.strip()


class ReactPlanner:
    """Bounded LLM plan/execute loop over the capability registry."""

    def __init__(
        self,
        llm: SupportsLLM,
        registry: CapabilityRegistry,
        *,
        max_steps: int = 4,
        tools: Sequence[str] = DEFAULT_TOOLS,
        persona_name: str = "",
        first_token_hook: Optional[Callable[[], None]] = None,
    ):
        self._llm = llm
        self._registry = registry
        self._max_steps = max(1, int(max_steps))
        self._tools = tuple(tools)
        self._persona_name = (persona_name or "").strip()
        # Fired on the first token this planner streams from the model. The
        # runtime wires it to stamp LLM_FIRST_TOKEN, so an escalated (ReAct) turn
        # -- whose LLM work happens here, not in the one-shot assistant path --
        # isn't false-flagged "llm stuck" by the watchdog (B3). Idempotent on the
        # recorder side, so firing once per drain is harmless.
        self._first_token_hook = first_token_hook

    def _final_system(self) -> str:
        # Keep an escalated turn's final answer in the configured persona's voice
        # (the one-shot path already does via build_system_prompt). A plain
        # string keeps the brain free of any ``core`` import.
        if self._persona_name:
            return (
                f"You are {self._persona_name}, a local, on-device voice assistant. "
                "Answer the user's request in one or two short, natural spoken "
                "sentences. Use the findings when relevant; if they are empty or "
                "unhelpful, answer from your own knowledge instead -- never reply "
                "that you cannot answer a general-knowledge question. No markdown "
                "or preambles."
            )
        return FINAL_SYSTEM

    def _catalog(self) -> str:
        # Descriptions come straight from the capability manifest (planner-facing
        # ``when_to_use``), so the planner's tool catalog can never drift from
        # the actually-registered capabilities.
        return self._registry.describe(self._tools, planner=True)

    def _plan_prompt(self, query: str, observations: list[str], recent: str = "") -> str:
        gathered = "\n".join(observations) if observations else "(none yet)"
        # The recent-conversation block (when the runtime supplies it) lets the
        # planner resolve references in the request -- "explain THAT" / "the second
        # one" -- against what was just said, instead of planning on a bare query.
        convo = f"{recent}\n\n" if recent else ""
        return (
            f"{convo}"
            f"User request: {query}\n\n"
            f"Available tools:\n{self._catalog()}\n\n"
            f"Findings so far:\n{gathered}\n\n"
            "Next step:"
        )

    def _drain(self, tokens: Iterator[str], cancel: Optional[Event]) -> str:
        parts: list[str] = []
        for token in tokens:
            if _cancelled(cancel):
                break
            if not parts and self._first_token_hook is not None:
                # First model token of this drain -> mark the turn alive.
                try:
                    self._first_token_hook()
                except Exception:  # noqa: BLE001 - metrics stamping is best-effort
                    pass
            parts.append(token)
        return "".join(parts).strip()

    def _final(
        self, query: str, observations: list[str], cancel: Optional[Event], recent: str = ""
    ) -> str:
        gathered = "\n".join(observations) if observations else "(no findings)"
        # Prepend the recent conversation (when supplied) so the spoken answer keeps
        # the thread -- an escalated "tell me more about it" resolves "it".
        convo = f"{recent}\n\n" if recent else ""
        prompt = (
            f"{convo}"
            f"User request: {query}\n"
            f"Findings:\n{gathered}\n\n"
            "Give the final spoken answer. If the findings do not actually help, "
            "ignore them and answer from your own knowledge."
        )
        return self._drain(self._llm.stream(prompt, system=self._final_system()), cancel)

    def run(self, query: str, context: Mapping[str, object]) -> CapabilityResult:
        cancel = context.get("cancel_event")  # type: ignore[assignment]
        # Bounded recent-conversation block the runtime publishes (core.capabilities
        # RECENT_CONVERSATION_KEY) so an escalated turn keeps the conversation
        # thread; absent on a bare/test invocation -> "" -> prompts unchanged.
        recent = str(context.get("recent_conversation", "") or "")
        # This runs under the short ASSISTANT mode budget, but a multi-step plan
        # (up to max_steps LLM calls + tool calls) legitimately takes longer.
        # Push the reap deadline out so a real agent turn isn't killed mid-plan.
        renew = context.get("renew_deadline")
        if callable(renew):
            try:
                renew(max(60.0, self._max_steps * 30.0))
            except Exception:  # noqa: BLE001 - deadline renewal is best-effort
                pass
        observations: list[str] = []
        steps_taken: list[str] = []

        for _ in range(self._max_steps):
            if _cancelled(cancel):
                return CapabilityResult(
                    True, "", data={"cancelled": True, "agent": True}
                )
            # Drain the plan step as a cancel-aware stream (like ``_final``) so a
            # barge-in/STOP between chunks aborts mid-planning instead of blocking
            # on ``generate()``. ``_drain`` strips, which is parse-neutral:
            # ``_parse_step`` strips again before matching TOOL/FINAL.
            raw = self._drain(
                self._llm.stream(
                    self._plan_prompt(query, observations, recent), system=PLANNER_SYSTEM
                ),
                cancel,
            )
            if _cancelled(cancel):
                return CapabilityResult(
                    True, "", data={"cancelled": True, "agent": True}
                )
            action, arg = _parse_step(raw)

            if action is None or action.upper() == "FINAL":
                text = arg if (action and arg) else self._final(query, observations, cancel, recent)
                return CapabilityResult(
                    True,
                    text or "Sorry, I couldn't work that out.",
                    data={"agent": True, "steps": steps_taken},
                )

            if action not in self._tools:
                # Recover: tell the next planning turn the tool is unavailable.
                observations.append(f"(tool '{action}' is unavailable)")
                continue

            result = self._registry.invoke(action, arg or query, dict(context))
            steps_taken.append(action)
            if result.ok:
                # Prompt-injection hardening (OWASP LLM01): web/external (egress)
                # tool output is attacker-controllable webpage text -- fence it as
                # UNTRUSTED data so an instruction smuggled into a fetched page
                # can't steer the planner's next TOOL/FINAL step. Local tool output
                # (no egress) is left as-is, byte-identical.
                obs_text = result.text
                if (result.data or {}).get("egress"):
                    obs_text = wrap_untrusted(obs_text, source="web")
                observations.append(f"{action}: {obs_text}".strip())
            else:
                # Recovery phase: surface the failure so the model can replan.
                observations.append(f"{action} failed: {result.error}")

        # Step budget exhausted -> synthesize whatever we have.
        text = self._final(query, observations, cancel, recent)
        return CapabilityResult(
            True,
            text or "Sorry, I couldn't work that out.",
            data={"agent": True, "steps": steps_taken, "exhausted": True},
        )


def attach_react_capability(
    registry: CapabilityRegistry,
    llm: SupportsLLM,
    *,
    config: PlannerConfig | None = None,
    planner: ReactPlanner | None = None,
    capability_name: str = "agent.react",
    persona_name: str = "",
    first_token_hook: Optional[Callable[[], None]] = None,
) -> CapabilityRegistry:
    """Register the ReAct planner as a capability provider.

    Parallels ``core.agent.attach_agent_capability``: a self-contained provider
    that honours ``context['cancel_event']``. ``planner`` may be injected for
    testing with a scripted LLM. ``persona_name`` (a plain string -- the brain
    takes no ``core`` import) keeps an escalated turn's final answer in voice.
    ``first_token_hook`` lets the runtime stamp LLM_FIRST_TOKEN for an escalated
    turn (B3 -- so the watchdog doesn't false-flag it as stuck).
    """
    config = config or PlannerConfig()
    planner = planner or ReactPlanner(
        llm, registry, max_steps=config.max_steps, tools=config.tools,
        persona_name=persona_name, first_token_hook=first_token_hook,
    )

    def provider(query: str, context: dict[str, object]) -> CapabilityResult:
        instruction = (query or "").strip()
        if not instruction:
            return CapabilityResult(False, "", error="empty instruction")
        return planner.run(instruction, context)

    registry.register(
        capability_name, provider,
        spec=CapabilitySpec(
            capability_name,
            summary="work through a multi-step question using your tools",
            when_to_use="multi-step reasoning or gathering before answering",
            egress="local", speaks=True, planner_tool=False, user_facing=False,
        ),
    )
    return registry
