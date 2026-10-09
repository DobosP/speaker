"""Small local classification requests, separate from answer generation.

Native output/grammar limits are implemented by Ollama and llama.cpp. The
monotonic deadline uses their existing cancellation seam; it cannot preempt an
injected blocking iterator or a native backend that ignores cancellation.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import json
import math
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient

MAX_DECISION_TOKENS = 16
MAX_DECISION_SECONDS = 3.0
MAX_DECISION_CHARS = 256


@dataclass(frozen=True)
class DecisionRequest:
    choices: tuple[str, ...]
    max_tokens: int = MAX_DECISION_TOKENS
    timeout_sec: float = MAX_DECISION_SECONDS

    def __post_init__(self) -> None:
        if (type(self.choices) is not tuple or not 1 <= len(self.choices) <= 8
                or any(type(x) is not str or re.fullmatch(r"[A-Z][A-Z_]{0,31}", x) is None
                       for x in self.choices)
                or len(set(self.choices)) != len(self.choices)):
            raise ValueError("decision choices must be distinct bounded labels")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= MAX_DECISION_TOKENS:
            raise ValueError("decision token budget is out of bounds")
        if (type(self.timeout_sec) not in {int, float}
                or not math.isfinite(self.timeout_sec)
                or not 0 < self.timeout_sec <= MAX_DECISION_SECONDS):
            raise ValueError("decision deadline is out of bounds")

    def options(self, original: dict, token_key: str) -> dict:
        options = dict(original)
        previous = options.get(token_key)
        options[token_key] = min(self.max_tokens, previous) if (
            type(previous) is int and previous > 0
        ) else self.max_tokens
        options["temperature"] = 0.0
        return options

    def schema(self) -> dict:
        return {"type": "string", "enum": list(self.choices)}

    def grammar(self):
        # Imported only after the native client is selected, never on the
        # orchestration/module-import path. Labels have a closed ASCII grammar.
        from llama_cpp import LlamaGrammar
        return LlamaGrammar.from_string(
            "root ::= " + " | ".join(json.dumps(x) for x in self.choices),
            verbose=False,
        )


decision_request: ContextVar[DecisionRequest | None] = ContextVar(
    "speaker_decision_request", default=None,
)


def current_decision_request() -> DecisionRequest | None:
    request = decision_request.get()
    if request is None:
        # Factory source owners copy capability_context into their worker. Carry
        # this immutable typed request through that already-audited snapshot.
        from .llm import capability_context
        context = capability_context.get()
        request = dict.get(context, "local_llm_decision_request") if type(context) is dict else None
    return request if type(request) is DecisionRequest else None


class _DecisionCancel:
    def __init__(self, external, timeout_sec: float) -> None:
        self.external = external
        self.deadline = time.monotonic() + timeout_sec
        self.stopped = False

    def expired(self) -> bool:
        return time.monotonic() >= self.deadline

    def is_set(self) -> bool:
        return self.stopped or self.external.is_set() or self.expired()


def collect_llm_decision(
    llm: LLMClient,
    prompt: str,
    *,
    system: str | None,
    choices: tuple[str, ...],
    max_tokens: int = MAX_DECISION_TOKENS,
    timeout_sec: float = MAX_DECISION_SECONDS,
    cancel_event: object | None = None,
) -> str | None:
    """Return a complete selected label; refuse overflow/deadline/malformed text.

    This request narrows egress to LOCAL_ONLY; it grants no input, owner or tool
    authority. External cancellation wins even if deadline/completion coincides.
    Unknown injected clients retain their protocol, with cooperative deadline and
    bounded collection only. No fallback call or background request is spawned.
    """
    from always_on_agent.models import CLOUD_EGRESS_SCOPE_CONTEXT_KEY, CloudEgressScope
    from .llm import (
        LLMCallCancelled, OpenAICompatLLM, _CombinedCancelEvent, capability_context,
        snapshot_capability_context,
    )

    request = DecisionRequest(choices, max_tokens, timeout_sec)
    context = snapshot_capability_context()
    external = _CombinedCancelEvent(context.get("cancel_event"), cancel_event)
    if external.is_set():
        raise LLMCallCancelled("decision cancelled before request")
    # Direct compatible-API clients do not enforce capability egress scope.
    # This helper cannot attest their locality/cancellation contract, so do not
    # start them. Factory hybrids below still reduce to their audited local leg.
    if isinstance(llm, OpenAICompatLLM):
        return None
    budget = _DecisionCancel(external, timeout_sec)
    # LOCAL_ONLY is the strictest egress scope; all other context/authority
    # carriers are preserved. Classification cannot expand a retained-data scope.
    context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] = CloudEgressScope.LOCAL_ONLY
    context["cancel_event"] = budget
    context["local_llm_decision_request"] = request
    context_token = capability_context.set(context)
    request_token = decision_request.set(request)
    stream = None
    try:
        if external.is_set():
            raise LLMCallCancelled("decision cancelled before request")
        stream = llm.stream(prompt, system=system)
        parts: list[str] = []
        chars = 0
        for piece in stream:
            if external.is_set():
                raise LLMCallCancelled("decision cancelled during request")
            if budget.expired():
                return None
            if type(piece) is not str:
                return None
            if not piece:
                continue
            chars += len(piece)
            if chars > MAX_DECISION_CHARS:
                return None
            parts.append(piece)
        if external.is_set():
            raise LLMCallCancelled("decision cancelled at completion")
        if budget.expired():
            return None
        raw = "".join(parts).strip()
        if raw.startswith('"'):
            try:
                raw = json.loads(raw)
            except (ValueError, TypeError):
                return None
        if type(raw) is not str:
            return None
        # Preserve the historical bounded single-token punctuation/alias seam.
        label = raw.strip().strip(".,!?:;\"'").upper()
        return label if label in choices else None
    except (Exception, asyncio.CancelledError) as exc:
        if external.is_set():
            if isinstance(exc, LLMCallCancelled):
                raise
            raise LLMCallCancelled("decision cancelled during provider failure") from exc
        if budget.expired():
            return None
        raise
    finally:
        budget.stopped = True
        try:
            closer = getattr(stream, "close", None)
            if callable(closer):
                try:
                    closer()
                except (Exception, asyncio.CancelledError):
                    pass
        finally:
            decision_request.reset(request_token)
            capability_context.reset(context_token)
