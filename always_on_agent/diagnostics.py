from __future__ import annotations

from collections import Counter

from .events import EventKind
from .supervisor import AgentSupervisor


def summarize(supervisor: AgentSupervisor) -> dict[str, object]:
    counts = Counter(event.kind.value for event in supervisor.state.event_log)
    return {
        "mode": supervisor.state.mode.value,
        "events": dict(sorted(counts.items())),
        "transcripts": len(supervisor.state.transcript_log),
        "outputs": list(supervisor.state.spoken_outputs),
        "active_tasks": sorted(supervisor.state.active_tasks),
        "queued_tasks": [task.task_id for task in supervisor.state.queued_tasks],
        "pending_confirmations": sorted(supervisor.state.pending_confirmations),
        "tts_requests": counts[EventKind.TTS_REQUEST.value],
        "decisions": [
            {
                "kind": decision.kind.value,
                "confidence": round(decision.confidence, 3),
                "reason": decision.reason,
                "text": decision.text,
            }
            for decision in supervisor.state.decisions
        ],
    }
