"""sr-2: cross-tier fast<->main retry in core.capabilities.assistant().

When the chosen tier's LLM errors BEFORE emitting any audio, the turn is retried
once on the OTHER tier (when it's a distinct model). The retry must not
double-speak (no retry once a sentence was emitted) and must not loop on a
single-tier config.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence

from always_on_agent.capabilities import CapabilityRegistry
from core.capabilities import attach_llm_capabilities
from core.routing import FAST, MAIN


class _HealthyLLM:
    def __init__(self, tag: str):
        self.tag = tag
        self.stream_calls = 0

    def generate(self, prompt: str, *, system: Optional[str] = None,
                 images: Optional[Sequence[object]] = None) -> str:
        return f"[{self.tag}] answer"

    def stream(self, prompt: str, *, system: Optional[str] = None,
               images: Optional[Sequence[object]] = None) -> Iterator[str]:
        self.stream_calls += 1
        yield f"[{self.tag}] answer"


class _RaisingLLM:
    def __init__(self, tag: str):
        self.tag = tag
        self.stream_calls = 0

    def generate(self, prompt: str, *, system: Optional[str] = None,
                 images: Optional[Sequence[object]] = None) -> str:
        raise RuntimeError(f"{self.tag} generate down")

    def stream(self, prompt: str, *, system: Optional[str] = None,
               images: Optional[Sequence[object]] = None) -> Iterator[str]:
        self.stream_calls += 1
        raise RuntimeError(f"{self.tag} backend down")
        yield ""  # pragma: no cover - marks this a generator


class _EmitThenRaiseLLM:
    def __init__(self, tag: str):
        self.tag = tag
        self.stream_calls = 0

    def generate(self, prompt: str, *, system: Optional[str] = None,
                 images: Optional[Sequence[object]] = None) -> str:  # pragma: no cover
        raise RuntimeError(f"{self.tag} down")

    def stream(self, prompt: str, *, system: Optional[str] = None,
               images: Optional[Sequence[object]] = None) -> Iterator[str]:
        self.stream_calls += 1
        yield "Partial sentence. "  # a complete sentence -> emitted
        raise RuntimeError(f"{self.tag} died mid-stream")


# HeuristicRouter routes these deterministically (see tests/test_core_routing.py).
_SIMPLE = "what time is it"  # -> FAST
_COMPLEX = "explain how the event bus works and why barge-in cancels tasks"  # -> MAIN


def test_fast_failure_retries_on_main():
    main = _HealthyLLM("main")
    fast = _RaisingLLM("fast")
    registry = attach_llm_capabilities(CapabilityRegistry(), main, fast_llm=fast)

    result = registry.invoke("assistant.answer", _SIMPLE)

    assert result.ok
    assert result.text == "[main] answer"
    assert result.data["route"] == MAIN  # answered on the fallback tier
    assert fast.stream_calls == 1  # tried once, then fell back
    assert main.stream_calls == 1


def test_main_failure_retries_on_fast():
    main = _RaisingLLM("main")
    fast = _HealthyLLM("fast")
    registry = attach_llm_capabilities(CapabilityRegistry(), main, fast_llm=fast)

    result = registry.invoke("assistant.answer", _COMPLEX)

    assert result.ok
    assert result.text == "[fast] answer"
    assert result.data["route"] == FAST
    assert main.stream_calls == 1
    assert fast.stream_calls == 1


def test_no_retry_when_single_tier():
    # fast_llm omitted -> both tiers are the SAME object; a retry would just
    # re-run the failing model, so the turn must fail without retrying.
    only = _RaisingLLM("only")
    registry = attach_llm_capabilities(CapabilityRegistry(), only)

    result = registry.invoke("assistant.answer", _SIMPLE)

    assert not result.ok
    assert only.stream_calls == 1  # exactly one attempt, no loop


def test_no_retry_after_audio_emitted():
    # The fast tier emits a sentence (audio goes out) THEN dies mid-stream.
    # Retrying would double-speak, so the turn must fail instead.
    main = _HealthyLLM("main")
    fast = _EmitThenRaiseLLM("fast")
    registry = attach_llm_capabilities(CapabilityRegistry(), main, fast_llm=fast)

    spoken: list[str] = []
    result = registry.invoke("assistant.answer", _SIMPLE, {"emit_speech": spoken.append})

    assert not result.ok  # propagated -> the turn fails (no double-speak retry)
    assert spoken == ["Partial sentence."]
    assert fast.stream_calls == 1
    assert main.stream_calls == 0  # never retried after audio was emitted
