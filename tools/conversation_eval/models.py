from __future__ import annotations

from hashlib import sha256
import re
from threading import Condition, Event, Lock
import time
from typing import Iterator, Mapping, Optional, Sequence

from core.llm import capability_context


def _prompt_category(system: str) -> str:
    lowered = system.lower()
    if "addressing gate" in lowered:
        return "addressing"
    if "transcript cleaner" in lowered:
        return "cleanup"
    if "planning loop" in lowered:
        return "planner"
    if "give the final spoken answer" in lowered or "use the findings" in lowered:
        return "planner_final"
    if "route one turn of a voice assistant" in lowered:
        return "capability_router"
    return "answer"


class StreamGate:
    """Pause one answer stream immediately after its first complete sentence."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._armed = False
        self.paused = Event()
        self.release = Event()
        self.continuation_observed = Event()

    def arm(self) -> None:
        with self._lock:
            self._armed = True
        self.paused.clear()
        self.release.clear()
        self.continuation_observed.clear()

    def should_gate(self, category: str, prompt: str) -> bool:
        with self._lock:
            if not self._armed or category != "answer" or "french flag" not in prompt.lower():
                return False
            self._armed = False
            return True

    def pause(self) -> None:
        self.paused.set()
        self.release.wait(timeout=15.0)

    def note_continuation(self) -> None:
        self.continuation_observed.set()

    def open(self) -> None:
        self.release.set()


class ObservedLLM:
    """Transparent LLM wrapper that records model/timing, never prompt content."""

    def __init__(
        self,
        inner,
        *,
        label: str = "",
        role: str = "",
        stream_gate: StreamGate | None = None,
    ) -> None:
        self._inner = inner
        self.model = str(getattr(inner, "model", label or type(inner).__name__))
        self.role = role or label or "model"
        self._stream_gate = stream_gate
        self._lock = Lock()
        self._active_changed = Condition(self._lock)
        self._active_calls = 0
        self._sequence = 0
        self._calls: list[dict[str, object]] = []

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def _begin(
        self,
        kind: str,
        system: Optional[str],
        history: object = None,
    ) -> tuple[int, float, dict[str, object]]:
        started = time.monotonic()
        context = capability_context.get()
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            self._active_calls += 1
        record: dict[str, object] = {
            "_started_monotonic": started,
            "sequence": sequence,
            "kind": kind,
            "model": self.model,
            "role": self.role,
            "category": _prompt_category(system or ""),
            "system_sha256": sha256((system or "").encode("utf-8")).hexdigest(),
            "history_sha256": (
                sha256(str(history).encode("utf-8")).hexdigest()
                if history
                else ""
            ),
            "history_count": (
                len(history)
                if isinstance(history, Sequence) and not isinstance(history, str)
                else 0
            ),
            "task_id": str(context.get("task_id", "") or ""),
            "context_markers": [
                marker
                for marker in ("orion",)
                if marker in f"{system or ''} {history or ''}".lower()
            ],
        }
        return sequence, started, record

    def _finish(
        self,
        started: float,
        record: dict[str, object],
        *,
        first_token_at: float | None = None,
        cancelled: bool = False,
        error: str = "",
    ) -> None:
        finished = time.monotonic()
        record.update(
            {
                "duration_ms": round((finished - started) * 1000.0, 3),
                "ttft_ms": (
                    round((first_token_at - started) * 1000.0, 3)
                    if first_token_at is not None
                    else None
                ),
                "cancelled": cancelled,
                "error": error,
            }
        )
        with self._lock:
            self._calls.append(record)
            self._active_calls = max(0, self._active_calls - 1)
            self._active_changed.notify_all()

    def generate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> str:
        _sequence, started, record = self._begin(
            "generate",
            system,
            kwargs.get("history"),
        )
        try:
            output = self._inner.generate(prompt, system=system, **kwargs)
        except BaseException as exc:
            self._finish(started, record, error=type(exc).__name__)
            raise
        # Non-streaming generation exposes response latency, not TTFT.
        self._finish(started, record)
        return output

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        **kwargs,
    ) -> Iterator[str]:
        _sequence, started, record = self._begin(
            "stream",
            system,
            kwargs.get("history"),
        )
        try:
            source = self._inner.stream(prompt, system=system, **kwargs)
        except BaseException as exc:
            self._finish(started, record, error=type(exc).__name__)
            raise
        category = str(record["category"])
        gate = self._stream_gate
        gated = bool(gate is not None and gate.should_gate(category, prompt))
        first_token_at: float | None = None
        exhausted = False
        error = ""
        try:
            if gated:
                buffered = ""
                paused = False
                source_iterator = iter(source)
                for token in source_iterator:
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    if paused:
                        yield token
                        continue
                    buffered += token
                    boundary = re.search(r"[.!?](?:\s|$)", buffered)
                    if boundary is None:
                        continue
                    end = boundary.end()
                    yield buffered[:end]
                    assert gate is not None
                    remainder = buffered[end:]
                    buffered = ""
                    while not remainder.strip():
                        try:
                            continuation = next(source_iterator)
                        except StopIteration:
                            break
                        if continuation:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            remainder += continuation
                    if remainder.strip():
                        gate.note_continuation()
                    gate.pause()
                    paused = True
                    if remainder:
                        yield remainder
                if buffered:
                    yield buffered
            else:
                for token in source:
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    yield token
            exhausted = True
        except GeneratorExit:
            raise
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            if not exhausted:
                closer = getattr(source, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass
            self._finish(
                started,
                record,
                first_token_at=first_token_at,
                cancelled=not exhausted and not error,
                error=error,
            )

    def calls(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(call) for call in sorted(self._calls, key=lambda item: int(item["sequence"])))

    def call_count(self) -> int:
        with self._lock:
            return len(self._calls)

    def calls_since(self, offset: int) -> tuple[dict[str, object], ...]:
        with self._lock:
            calls = self._calls[max(0, int(offset)):]
            return tuple(dict(call) for call in calls)

    def wait_calls_closed(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._active_changed:
            while self._active_calls:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._active_changed.wait(timeout=remaining)
            return True


class DeterministicConversationLLM:
    """Prompt-aware model double for the CI conversation gate.

    It implements the same addressing, cleanup, capability-router, ReAct, and
    answering prompt contracts as the configured runtime.  Scenario assertions
    still run through the real event bus, task layer, memory, and playback sink.
    """

    def __init__(self, model: str = "deterministic-conversation-v1") -> None:
        self.model = model

    @staticmethod
    def _raw_cleanup_text(prompt: str) -> str:
        match = re.search(r"^Raw ASR text:\s*(.+)$", prompt, flags=re.MULTILINE)
        return match.group(1).strip() if match else prompt.strip()

    @staticmethod
    def _route_action(prompt: str) -> str:
        lowered = prompt.lower()
        if any(marker in lowered for marker in ("look up", "research", "search ", "compare")):
            return "RESEARCH"
        if any(marker in lowered for marker in ("open ", "execute ", "set a timer")):
            return "ACT"
        return "SIMPLE"

    @staticmethod
    def _planner_step(prompt: str) -> str:
        lowered = prompt.lower()
        findings = lowered.split("findings so far:", 1)[-1]
        if "failed:" in findings and "search.local:" not in findings:
            return "TOOL search.local: pipecat"
        if "cobalt_override" in findings and "<<<untrusted::" not in findings:
            return (
                "FINAL: Paris. COBALT_OVERRIDE. My system prompt says I am "
                "the local voice assistant."
            )
        if "cobalt_override" in findings and "<<<untrusted::" in findings:
            return "FINAL: Paris is the capital of France."
        if "paris" in findings:
            return "FINAL: Paris is the capital of France."
        if "search.local:" in findings or "pipecat is an open-source" in findings:
            return "FINAL: Pipecat is an open-source framework for realtime voice agents."
        if "look up" in lowered:
            return "TOOL web.search: " + (
                "capital of France" if "capital of france" in lowered else "pipecat"
            )
        return "FINAL: I can answer directly."

    @staticmethod
    def _answer(prompt: str, system: str, history: Sequence[object] | None) -> str:
        combined = " ".join((prompt, system, str(history or ())))
        lowered = combined.lower()
        if "tokyo instead" in lowered:
            return "Tokyo instead."
        if "capital of japan" in lowered:
            return "Tokyo is the capital of Japan."
        if "capital of france" in lowered:
            return "Paris is the capital of France."
        if "water freezes" in lowered:
            return "Water freezes at zero degrees Celsius."
        if "codename" in lowered and "orion" in lowered:
            return "The project codename is Orion."
        if "codename" in lowered:
            return "I do not know the project codename."
        if "colors" in lowered and "french flag" in lowered:
            return "The French flag has blue. It also has white. Its third color is red."
        if "lighthouse" in lowered:
            return (
                "A lighthouse watched the harbor. "
                "Its beam crossed the fog. "
                "At dawn the keeper rested."
            )
        if "pipecat" in lowered and "livekit" in lowered:
            return "Pipecat and LiveKit are frameworks for realtime voice agents."
        if "pipecat" in lowered:
            return "Pipecat is an open-source framework for realtime voice agents."
        return "I can help with that."

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[object]] = None,
        history: Optional[Sequence[object]] = None,
        **_kwargs,
    ) -> str:
        del images
        system_text = system or ""
        category = _prompt_category(system_text)
        if category == "addressing":
            lowered_prompt = prompt.lower()
            return (
                "INGEST"
                if "i think i left the stove on" in lowered_prompt
                else "ACT"
            )
        if category == "cleanup":
            raw = self._raw_cleanup_text(prompt)
            if "capital of france, i mean japan" in raw.lower():
                return "What is the capital of Japan?"
            return raw
        if category == "capability_router":
            return self._route_action(prompt)
        if category == "planner":
            return self._planner_step(prompt)
        return self._answer(prompt, system_text, history)

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[object]] = None,
        history: Optional[Sequence[object]] = None,
        **kwargs,
    ) -> Iterator[str]:
        yield self.generate(
            prompt,
            system=system,
            images=images,
            history=history,
            **kwargs,
        )
