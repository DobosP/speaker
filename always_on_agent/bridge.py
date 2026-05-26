from __future__ import annotations

from .runtime import AlwaysOnAgentRuntime


class TranscriptBridge:
    """
    Minimal callback adapter for existing STT code.

    The current `main.py` can instantiate this and call `on_partial_text` from
    the partial STT worker and `on_final_text` after final STT.
    """

    def __init__(self, runtime: AlwaysOnAgentRuntime | None = None):
        self.runtime = runtime or AlwaysOnAgentRuntime()

    def on_partial_text(self, text: str) -> None:
        self.runtime.ingest_partial(text)

    def on_final_text(self, text: str) -> None:
        self.runtime.ingest_final(text)

    def on_stop_requested(self, reason: str = "bridge") -> None:
        self.runtime.stop(reason)
