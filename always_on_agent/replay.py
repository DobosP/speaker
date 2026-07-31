from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from .events import AgentEvent, EventKind
from .runtime import AlwaysOnAgentRuntime
from .supervisor import AgentSupervisor


def event_from_record(record: dict[str, object]) -> AgentEvent:
    kind = str(record.get("kind", "stt.final"))
    text = str(record.get("text", ""))
    delay = float(record.get("delay", 0.0))
    if delay > 0:
        time.sleep(delay)
    if kind == EventKind.STT_PARTIAL.value:
        return AgentEvent.partial(text)
    if kind == EventKind.CONTROL_STOP.value:
        return AgentEvent.stop("replay")
    return AgentEvent.final(text)


def replay_records(records: Iterable[dict[str, object]]) -> AgentSupervisor:
    supervisor = AgentSupervisor()
    runtime = AlwaysOnAgentRuntime(supervisor)
    for record in records:
        event = event_from_record(record)
        text = str(event.payload.get("text", ""))
        if event.kind == EventKind.STT_PARTIAL:
            runtime.ingest_partial(text)
        elif event.kind == EventKind.STT_FINAL:
            runtime.ingest_final(text)
        elif event.kind == EventKind.CONTROL_STOP:
            runtime.stop("replay")
        else:
            supervisor.publish(event)
            supervisor.drain()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        supervisor.drain()
        if not supervisor.state.active_tasks:
            supervisor.drain()
            break
        time.sleep(0.01)
    return supervisor


def replay_jsonl(path: str | Path) -> AgentSupervisor:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return replay_records(records)
