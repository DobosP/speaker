from __future__ import annotations

from dataclasses import asdict

from .runtime import AlwaysOnAgentRuntime


def snapshot_dict(runtime: AlwaysOnAgentRuntime) -> dict[str, object]:
    return asdict(runtime.snapshot())


def readable_snapshot(runtime: AlwaysOnAgentRuntime) -> str:
    snap = runtime.snapshot()
    return "\n".join(
        [
            f"mode: {snap.mode}",
            f"last_partial: {snap.last_partial}",
            f"transcripts: {len(snap.transcripts)}",
            f"outputs: {len(snap.outputs)}",
            f"active_tasks: {len(snap.active_tasks)}",
            f"queued_tasks: {len(snap.queued_tasks)}",
            f"pending_confirmations: {len(snap.pending_confirmations)}",
            f"failures: {len(snap.failures)}",
            f"events: {snap.event_count}",
        ]
    )
