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
from .planner_steps import (
    PlannerCall,
    PlannerExchange,
    PlannerStepBackend,
    PlannerTool,
)
from .speech_analyzer import (
    is_vault_lookup_request,
    is_vault_public_source_request,
    is_vault_scoped_request,
)
from .text import normalize_text
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
    "Prefer FINAL as soon as you have enough. After a tool fails, never retry "
    "the same failed tool; choose a different relevant tool or FINAL. Never "
    "invent tools."
)

# Re-prompt appended once when a small model's step is unparseable (P3 reliability).
_FORMAT_REMINDER = (
    "Your previous reply was NOT in the required format. Reply with EXACTLY one "
    "line and nothing else: either\n  TOOL <tool_name>: <input>\nor\n  FINAL: <answer>"
)

FINAL_SYSTEM = (
    "You are a local, on-device voice assistant. Answer the user's request in "
    "one or two short, natural spoken sentences. Use the findings when they are "
    "relevant; if they are empty or unhelpful, answer from your own knowledge "
    "instead -- never reply that you cannot answer a general-knowledge question. "
    "No markdown or preambles."
)

# STRICT directives, applied to EVERY scanned line (line 1 included): a COLON is
# REQUIRED (right after FINAL, or after the tool name) -- exactly the format the
# model is told to emit (PLANNER_SYSTEM: "TOOL <name>: <input>" / "FINAL: <answer>").
# Requiring the colon everywhere is what stops an ordinary prose line that merely
# starts with "Tool ..." / "Final ..." (e.g. "Tool web.search is great for code.")
# from being misread as a directive -- including when it is the model's only line.
# Only LEADING bullet/markdown decoration is stripped (never interior chars), so a
# payload like "C# tutorials" or a query containing * / # is preserved verbatim.
_LEAD_NOISE = re.compile(r"^[\s>*#`\-]+")
_TOOL_STRICT = re.compile(r"^TOOL\s+([\w.]+)\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)
_FINAL_STRICT = re.compile(r"^FINAL\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"^\s*```")

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
    q = normalize_text(query)
    if is_vault_public_source_request(q):
        return True
    if is_vault_scoped_request(q):
        # A request naming private local notes must never fall through to a web
        # gather merely because the optional machine-local vault is absent.
        return (
            (context or {}).get("vault_available") is True
            and is_vault_lookup_request(q)
            and len(q.split()) >= 4
        )
    if sum(1 for m in _ESCALATE_MARKERS if m in q) >= 1 and len(q.split()) >= 4:
        return True
    return False


def _cancelled(cancel: Optional[Event]) -> bool:
    return cancel is not None and cancel.is_set()


def _parse_step(text: str) -> tuple[Optional[str], str]:
    """Return ``(action, argument)``.

    ``action`` is a tool name, the literal ``"FINAL"``, or ``None`` when the
    model produced something unparseable (treated as "wrap up").

    A single STRICT (colon-required) scan over every line (P3 reliability for small
    models that decorate / preface a step). Only LEADING bullet/markdown decoration
    + a trailing markdown fence are removed (never interior chars), so a directive
    emitted after a preamble line or wrapped in a bullet still parses while a payload
    like "C# tutorials" is preserved verbatim. The colon requirement applies to the
    FIRST line too, so an ordinary prose line that merely starts with "Tool ..." /
    "Final ..." is never misread as a directive -- even as the model's only line.
    Lines inside a fenced ``` code block are skipped, so an *example* directive the
    model quotes (rather than intends to run) cannot fire a real tool call / egress.
    """
    stripped = text.strip()
    if not stripped:
        return None, ""

    in_fence = False
    for raw in stripped.splitlines():
        if _FENCE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = _LEAD_NOISE.sub("", raw)
        if line.endswith("`"):
            line = line[:-1]
        final = _FINAL_STRICT.match(line)
        if final:
            return "FINAL", final.group(1).strip()
        tool = _TOOL_STRICT.match(line)
        if tool:
            return tool.group(1).strip(), tool.group(2).strip()
    return None, stripped


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
        step_backend: PlannerStepBackend | None = None,
    ):
        self._llm = llm
        self._registry = registry
        self._max_steps = max(1, int(max_steps))
        self._tools = tuple(tools)
        self._persona_name = (persona_name or "").strip()
        # None is the historical textual ReAct protocol.  A native backend is
        # injected only for a provider/profile whose exact grammar was verified;
        # it never owns execution or the allowlist.
        self._step_backend = step_backend
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

    def _catalog(self, tools: Sequence[str] | None = None) -> str:
        # Descriptions come straight from the capability manifest (planner-facing
        # ``when_to_use``), so the planner's tool catalog can never drift from
        # the actually-registered capabilities.
        return self._registry.describe(
            self._tools if tools is None else tools,
            planner=True,
        )

    def _native_tools(self) -> tuple[PlannerTool, ...]:
        """Return the safe configured/registered intersection for native calls.

        Text ReAct stays byte-identical.  The new native seam is deliberately
        narrower: an untyped/missing spec, a capability not marked as a planner
        tool, or any side effect keeps that capability out of the model schema.
        The controller still checks the parsed name again before invocation.
        """

        registered = set(self._registry.names())
        offered: list[PlannerTool] = []
        for name in self._tools:
            spec = self._registry.spec(name)
            if (
                name not in registered
                or spec is None
                or not spec.planner_tool
                or spec.side_effecting
                or spec.authority != "none"
            ):
                continue
            offered.append(
                PlannerTool(
                    name=name,
                    description=spec.when_to_use or spec.summary,
                )
            )
        return tuple(offered)

    def _plan_prompt(
        self,
        query: str,
        observations: list[str],
        recent: str = "",
        tools: Sequence[str] | None = None,
    ) -> str:
        gathered = "\n".join(observations) if observations else "(none yet)"
        # The recent-conversation block (when the runtime supplies it) lets the
        # planner resolve references in the request -- "explain THAT" / "the second
        # one" -- against what was just said, instead of planning on a bare query.
        convo = f"{recent}\n\n" if recent else ""
        return (
            f"{convo}"
            f"User request: {query}\n\n"
            f"Available tools:\n{self._catalog(tools)}\n\n"
            f"Findings so far:\n{gathered}\n\n"
            "Next step:"
        )

    def _drain(
        self,
        tokens: Iterator[str],
        cancel: Optional[Event],
        first_token_hook: Optional[Callable[[], None]] = None,
    ) -> str:
        parts: list[str] = []
        hook = first_token_hook or self._first_token_hook
        cancelled = False
        try:
            for token in tokens:
                if _cancelled(cancel):
                    cancelled = True
                    break
                if not parts and hook is not None:
                    # First model token of this drain -> mark the turn alive.
                    try:
                        hook()
                    except Exception:  # noqa: BLE001 - metrics stamping is best-effort
                        pass
                parts.append(token)
        finally:
            if cancelled:
                # Propagate a between-token barge to the actual provider; for
                # Ollama this waits for cooperative request/client cleanup.
                closer = getattr(tokens, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:  # noqa: BLE001 - abandoned stream teardown
                        pass
        return "".join(parts).strip()

    def _final(
        self,
        query: str,
        observations: list[str],
        cancel: Optional[Event],
        recent: str = "",
        first_token_hook: Optional[Callable[[], None]] = None,
        claim_start: Optional[Callable[[], bool]] = None,
    ) -> str:
        if _cancelled(cancel):
            return ""
        if claim_start is not None and not claim_start():
            return ""
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
        return self._drain(
            self._llm.stream(prompt, system=self._final_system()),
            cancel,
            first_token_hook,
        )

    def run(self, query: str, context: Mapping[str, object]) -> CapabilityResult:
        cancel = context.get("cancel_event")  # type: ignore[assignment]
        scoped_hook = context.get("first_token_hook")
        first_token_hook = scoped_hook if callable(scoped_hook) else self._first_token_hook
        scoped_claim = context.get("claim_provider_start")

        def claim_start() -> bool:
            if _cancelled(cancel):
                return False
            if not callable(scoped_claim):
                return True
            try:
                return bool(scoped_claim())
            except Exception:  # noqa: BLE001 - never start after a broken guard
                return False
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
        native_backend = self._step_backend
        vault_scoped = is_vault_scoped_request(query)
        vault_lookup = is_vault_lookup_request(query)
        # Private vault reads require an explicit local lookup. Never advertise
        # vault.search on generic or explicitly-public turns and rely on model
        # obedience to protect the private source.
        safe_tools = tuple(
            name
            for name in self._tools
            if (
                (spec := self._registry.spec(name)) is not None
                and spec.planner_tool
                and not spec.side_effecting
                and spec.authority == "none"
            )
        )
        run_tools = tuple(name for name in safe_tools if name != "vault.search")
        if vault_scoped:
            # Controller-enforced source scope: a private local-vault request
            # cannot select web/search.local even if a model ignores guidance.
            # Only a real read lookup may see vault.search; declarations and
            # mutation requests get no read tool at all.
            run_tools = (
                tuple(
                    name
                    for name in safe_tools
                    if name == "vault.search" and name in self._registry.names()
                )
                if vault_lookup
                else ()
            )
        if native_backend is not None and vault_scoped:
            # ADR-0033/0073: vault.search is appended after MiniCPM's verified
            # first four schemas. Do not filter/reorder it into native eligibility;
            # deterministic speech analysis handles supported vault turns before
            # ReAct, while a direct native planner call fails closed with no tools.
            native_tools: tuple[PlannerTool, ...] = ()
        else:
            native_tools = (
                tuple(
                    tool for tool in self._native_tools() if tool.name in run_tools
                )
                if native_backend is not None
                else ()
            )
        native_allowed = {tool.name for tool in native_tools}
        native_exchanges: list[PlannerExchange] = []
        reprompt_budget = 1  # one strict-format retry across the whole plan (P3)
        web_local_fallback_used = False
        failed_tools: set[str] = set()

        def record_tool_result(
            action: str,
            tool_query: str,
            result: CapabilityResult,
        ) -> None:
            """Record one actual controller-owned capability invocation."""

            # A tool can discover data more sensitive than the original query
            # (notably a local vault lookup after a public-looking question).
            # Float PRIVATE onto the same dict held by capability_context before
            # the planner's next LLM call; private is the dominant class, so no
            # core import or duplicate rank table is needed in this core-free
            # controller.
            if (
                (result.data or {}).get("sensitivity") == "private"
                and isinstance(context, dict)
            ):
                context["sensitivity"] = "private"
            steps_taken.append(action)
            exchange_text = ""
            exchange_untrusted = False
            if result.ok:
                # Prompt-injection hardening (OWASP LLM01): web/external (egress)
                # tool output is attacker-controllable webpage text -- fence it as
                # UNTRUSTED data so an instruction smuggled into a fetched page
                # can't steer the planner's next TOOL/FINAL step. Local tool output
                # (no egress) is left as-is, byte-identical.
                obs_text = result.text
                if (result.data or {}).get("egress"):
                    exchange_untrusted = True
                    obs_text = wrap_untrusted(obs_text, source="web")
                observations.append(f"{action}: {obs_text}".strip())
                # The native backend applies a compact, bounded envelope using
                # this explicit trust bit. Keep raw bytes here so clipping never
                # cuts the full spotlight directive/fences before the finding.
                exchange_text = result.text
            else:
                observations.append(f"{action} failed: {result.error}")
                exchange_text = f"Tool failed: {result.error}"
                failed_tools.add(action)
            if native_backend is not None:
                native_exchanges.append(
                    PlannerExchange(
                        call=PlannerCall(action, tool_query),
                        result=exchange_text,
                        ok=result.ok,
                        untrusted=exchange_untrusted,
                    )
                )

        def native_action(reminder: bool) -> tuple[Optional[str], str]:
            """Ask the optional backend, retaining controller-side authority."""

            assert native_backend is not None
            try:
                step = native_backend.next_step(
                    query=query,
                    recent=recent,
                    tools=native_tools,
                    exchanges=tuple(native_exchanges),
                    reminder=reminder,
                    cancel=cancel,
                    first_token_hook=first_token_hook,
                )
            except Exception:
                # Provider-native cancellation commonly arrives as its own
                # exception.  The task event is the authority for whether this
                # is cancellation or an ordinary provider failure.
                if _cancelled(cancel):
                    return None, ""
                raise
            if step.malformed:
                return None, ""
            if step.call is not None:
                # Redundant with the backend parser by design: only the
                # controller's safe configured/registered intersection can
                # become executable, even if a backend regresses.
                if step.call.name not in native_allowed:
                    return None, ""
                return step.call.name, step.call.query
            return "FINAL", step.final or ""

        for _ in range(self._max_steps):
            if _cancelled(cancel):
                return CapabilityResult(
                    True, "", data={"cancelled": True, "agent": True}
                )
            if not claim_start():
                return CapabilityResult(True, "", data={"cancelled": True, "agent": True})
            if native_backend is None:
                # Drain the plan step as a cancel-aware stream (like ``_final``)
                # so barge-in between chunks closes the provider. ``_drain``
                # strips, which is parse-neutral for the text protocol.
                raw = self._drain(
                    self._llm.stream(
                        self._plan_prompt(query, observations, recent, run_tools),
                        system=PLANNER_SYSTEM,
                    ),
                    cancel,
                    first_token_hook,
                )
                action, arg = _parse_step(raw)
            else:
                action, arg = native_action(False)
            if _cancelled(cancel):
                return CapabilityResult(
                    True, "", data={"cancelled": True, "agent": True}
                )

            # P3 reliability: an unparseable step (no TOOL/FINAL directive at all)
            # gets ONE strict-format re-prompt before we give up and synthesize a
            # FINAL -- a small model that rambled instead of emitting a directive
            # often complies on the reminder. Bounded to once per plan (latency).
            if action is None and reprompt_budget > 0:
                reprompt_budget -= 1
                if not claim_start():
                    return CapabilityResult(True, "", data={"cancelled": True, "agent": True})
                if native_backend is None:
                    raw = self._drain(
                        self._llm.stream(
                            self._plan_prompt(query, observations, recent, run_tools)
                            + "\n\n"
                            + _FORMAT_REMINDER,
                            system=PLANNER_SYSTEM,
                        ),
                        cancel,
                        first_token_hook,
                    )
                    action, arg = _parse_step(raw)
                else:
                    action, arg = native_action(True)
                if _cancelled(cancel):
                    return CapabilityResult(True, "", data={"cancelled": True, "agent": True})

            if action is None or action.upper() == "FINAL":
                text = (
                    arg
                    if (action and arg)
                    else self._final(
                        query,
                        observations,
                        cancel,
                        recent,
                        first_token_hook,
                        claim_start,
                    )
                )
                if native_backend is not None:
                    text = native_backend.validate_final(text) or ""
                return CapabilityResult(
                    True,
                    text or "Sorry, I couldn't work that out.",
                    data={"agent": True, "steps": steps_taken},
                )

            allowed = native_allowed if native_backend is not None else set(run_tools)
            if action not in allowed:
                # Recover: tell the next planning turn the tool is unavailable.
                observations.append(f"(tool '{action}' is unavailable)")
                continue

            if action in failed_tools:
                # The model may ignore the planner prompt and request the same
                # failed provider again. Keep that failure non-repeatable at the
                # controller boundary; a denied retry is not an invocation.
                observations.append(
                    f"({action} was not retried because it already failed)"
                )
                if native_backend is not None:
                    native_exchanges.append(
                        PlannerExchange(
                            call=PlannerCall(action, arg or query),
                            result=(
                                "Tool call denied: this provider already failed "
                                "during the current plan."
                            ),
                            ok=False,
                            untrusted=False,
                        )
                    )
                continue

            if not claim_start():
                return CapabilityResult(True, "", data={"cancelled": True, "agent": True})
            tool_query = arg or query
            result = self._registry.invoke(action, tool_query, dict(context))
            if _cancelled(cancel):
                return CapabilityResult(True, "", data={"cancelled": True, "agent": True})
            record_tool_result(action, tool_query, result)

            # Small planners commonly retry the just-failed web tool or stop
            # early. Keep the recovery deterministic and bounded: the first
            # failed web.search may make exactly one controller-owned attempt
            # against the configured local corpus with the identical query.
            # Every execution fence is re-checked because barge-in can arrive
            # between the two capability calls.
            if (
                not result.ok
                and action == "web.search"
                and not web_local_fallback_used
                and "search.local" in allowed
            ):
                web_local_fallback_used = True
                if not claim_start():
                    return CapabilityResult(
                        True, "", data={"cancelled": True, "agent": True}
                    )
                fallback = self._registry.invoke(
                    "search.local", tool_query, dict(context)
                )
                if _cancelled(cancel):
                    return CapabilityResult(
                        True, "", data={"cancelled": True, "agent": True}
                    )
                record_tool_result("search.local", tool_query, fallback)

        # Step budget exhausted -> synthesize whatever we have.
        text = self._final(
            query,
            observations,
            cancel,
            recent,
            first_token_hook,
            claim_start,
        )
        if native_backend is not None:
            text = native_backend.validate_final(text) or ""
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
    step_backend: PlannerStepBackend | None = None,
) -> CapabilityRegistry:
    """Register the ReAct planner as a capability provider.

    Parallels ``core.agent.attach_agent_capability``: a self-contained provider
    that honours ``context['cancel_event']``. ``planner`` may be injected for
    testing with a scripted LLM. ``persona_name`` (a plain string -- the brain
    takes no ``core`` import) keeps an escalated turn's final answer in voice.
    ``first_token_hook`` lets the runtime stamp LLM_FIRST_TOKEN for an escalated
    turn (B3 -- so the watchdog doesn't false-flag it as stuck).
    ``step_backend`` is an optional verified native planning grammar; execution
    remains in :class:`ReactPlanner`.
    """
    config = config or PlannerConfig()
    planner = planner or ReactPlanner(
        llm, registry, max_steps=config.max_steps, tools=config.tools,
        persona_name=persona_name, first_token_hook=first_token_hook,
        step_backend=step_backend,
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
