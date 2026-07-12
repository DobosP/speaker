from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from threading import Condition, Lock
import time
from typing import Mapping

from always_on_agent.capabilities import (
    CapabilityInvocation,
    CapabilityInvocationResult,
)
from always_on_agent.events import AgentEvent

from .schema import TraceEvent


def json_value(value: object) -> object:
    """Convert runtime payloads to stable JSON values without arbitrary reprs."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=str)
    if is_dataclass(value):
        return json_value(asdict(value))
    return f"<{type(value).__name__}>"


def result_payload(result: CapabilityInvocationResult | None) -> dict[str, object]:
    if result is None:
        return {}
    return {
        "ok": result.ok,
        "text": result.text,
        "data": json_value(result.data),
        "citations": list(result.citations),
        "error": result.error,
    }


class TraceRecorder:
    """Thread-safe merger of bus, capability, and evaluator observations."""

    def __init__(self) -> None:
        self._started = time.monotonic()
        self._lock = Lock()
        self._invocation_changed = Condition(self._lock)
        self._sequence = 0
        self._events: list[TraceEvent] = []
        self._task_ids: dict[str, str] = {}
        self._open_invocations: set[int] = set()
        self._unattributed_playback: list[tuple[str, str, int]] = []

    def _append(
        self,
        kind: str,
        payload: Mapping[str, object] | None = None,
        *,
        task_id: str = "",
        turn_id: int | None = None,
        at: float | None = None,
    ) -> int:
        observed = time.monotonic() if at is None else at
        converted = dict(json_value(dict(payload or {})))
        with self._lock:
            def canonical(raw: str) -> str:
                return self._task_ids.setdefault(
                    raw,
                    f"T{len(self._task_ids) + 1}",
                )

            def rewrite_ids(value: object, key: str = "") -> object:
                if isinstance(value, dict):
                    return {
                        item_key: rewrite_ids(item, item_key)
                        for item_key, item in value.items()
                    }
                if isinstance(value, list):
                    return [rewrite_ids(item, key) for item in value]
                if isinstance(value, str):
                    if key in {
                        "task_id",
                        "continuation_of",
                        "continue_after",
                        "victim_task_id",
                    } and value:
                        return canonical(value)
                    if key == "aux_tts_id" and ":" in value:
                        prefix, suffix = value.split(":", 1)
                        return f"{canonical(prefix)}:{suffix}"
                return value

            canonical_task = ""
            if task_id:
                canonical_task = canonical(task_id)
            converted = dict(rewrite_ids(converted))
            self._sequence += 1
            self._events.append(
                TraceEvent(
                    sequence=self._sequence,
                    elapsed_ms=round((observed - self._started) * 1000.0, 3),
                    kind=kind,
                    task_id=canonical_task,
                    turn_id=turn_id,
                    payload=converted,
                )
            )
            return self._sequence

    @staticmethod
    def _turn_id(payload: Mapping[str, object]) -> int | None:
        value = payload.get("input_generation")
        metadata = payload.get("metadata")
        if value is None and isinstance(metadata, Mapping):
            value = metadata.get("input_generation")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def on_agent_event(self, event: AgentEvent) -> None:
        payload = event.payload
        sequence = self._append(
            event.kind.value,
            payload,
            task_id=str(payload.get("task_id", "") or ""),
            turn_id=self._turn_id(payload),
        )
        if event.kind.value == "tts.request":
            self._attribute_playback(payload, tts_sequence=sequence)

    def playback_requested(
        self,
        *,
        fragment_id: str,
        text: str,
        style: object,
    ) -> None:
        """Record a sink request before the later bus observation attributes it.

        ``VoiceRuntime`` invokes the engine while handling ``tts.request``; the
        trace subscriber sees that bus event only after the engine call returns.
        Keeping the request sequence separately lets the evaluator prove whether
        a fragment crossed the barge cut without relying on callback scheduling.
        """

        sequence = self._append(
            "playback.requested",
            {
                "fragment_id": fragment_id,
                "text": text,
                "style": style,
            },
        )
        with self._lock:
            self._unattributed_playback.append((fragment_id, text, sequence))

    def _attribute_playback(
        self,
        payload: Mapping[str, object],
        *,
        tts_sequence: int,
    ) -> None:
        text = str(payload.get("text", "") or "").strip()
        matched: tuple[str, str, int] | None = None
        with self._lock:
            for index, pending in enumerate(self._unattributed_playback):
                if pending[1].strip() == text:
                    matched = self._unattributed_playback.pop(index)
                    break
        if matched is None:
            return
        fragment_id, _requested_text, requested_sequence = matched
        self._append(
            "playback.attributed",
            {
                "fragment_id": fragment_id,
                "requested_sequence": requested_sequence,
                "tts_sequence": tts_sequence,
                "epoch": payload.get("epoch"),
                "input_generation": payload.get("input_generation"),
                "auxiliary_tts": bool(payload.get("auxiliary_tts", False)),
            },
            task_id=str(payload.get("task_id", "") or ""),
            turn_id=self._turn_id(payload),
        )

    def on_capability_invocation(self, event: CapabilityInvocation) -> None:
        if event.phase == "started":
            # Mark live before publishing the trace row so finalization can
            # never observe an empty set in the append->mark gap.
            with self._invocation_changed:
                self._open_invocations.add(event.invocation_id)
        payload: dict[str, object] = {
            "invocation_id": event.invocation_id,
            "name": event.name,
            "query": event.query,
            "planner_tool": event.planner_tool,
        }
        if event.phase == "finished":
            payload["result"] = result_payload(event.result)
        self._append(
            f"capability.{event.phase}",
            payload,
            task_id=event.task_id,
            at=event.monotonic,
        )
        if event.phase == "finished":
            with self._invocation_changed:
                self._open_invocations.discard(event.invocation_id)
                self._invocation_changed.notify_all()

    def mark(
        self,
        kind: str,
        payload: Mapping[str, object] | None = None,
        *,
        task_id: str = "",
        turn_id: int | None = None,
    ) -> None:
        self._append(kind, payload, task_id=task_id, turn_id=turn_id)

    def events(self) -> tuple[TraceEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def canonical_task_id(self, task_id: str) -> str:
        if not task_id:
            return ""
        with self._lock:
            return self._task_ids.setdefault(
                task_id,
                f"T{len(self._task_ids) + 1}",
            )

    def wait_invocations_closed(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._invocation_changed:
            while self._open_invocations:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._invocation_changed.wait(timeout=remaining)
            return True

    def event_kinds(self) -> tuple[str, ...]:
        return tuple(event.kind for event in self.events())

    def tool_names(self) -> tuple[str, ...]:
        return tuple(
            str(event.payload.get("name", ""))
            for event in self.events()
            if event.kind == "capability.started"
            and bool(event.payload.get("planner_tool"))
        )

    def sink_attested_texts(self) -> tuple[str, ...]:
        return tuple(
            str(event.payload.get("text", ""))
            for event in self.events()
            if event.kind == "memory.commit"
            and event.payload.get("source") == "playback_receipt"
            and str(event.payload.get("text", "")).strip()
        )

    def cancellation_reasons(self) -> tuple[str, ...]:
        return tuple(
            str(event.payload.get("reason", ""))
            for event in self.events()
            if event.kind == "control.stop"
        )
