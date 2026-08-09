"""Bounded, model-isolated streaming and final-only STT benchmark support.

The package deliberately imports no optional recognizer runtime. Candidate
models run in separately provisioned Python processes behind the JSONL contract
implemented here. Public compatibility exports are resolved lazily so importing
one bounded helper does not initialize the supervisor or manifest stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manifest import WorkerManifest
    from .protocol import CaseTrace, PcmInput, StreamConfig, TranscribeRequest
    from .supervisor import StreamingWorker

__all__ = [
    "CaseTrace",
    "PcmInput",
    "StreamConfig",
    "StreamingWorker",
    "TranscribeRequest",
    "WorkerManifest",
    "load_worker_manifest",
]


def __getattr__(name: str) -> object:
    if name in {"WorkerManifest", "load_worker_manifest"}:
        from .manifest import WorkerManifest, load_worker_manifest

        value = {
            "WorkerManifest": WorkerManifest,
            "load_worker_manifest": load_worker_manifest,
        }[name]
    elif name in {"CaseTrace", "PcmInput", "StreamConfig", "TranscribeRequest"}:
        from .protocol import CaseTrace, PcmInput, StreamConfig, TranscribeRequest

        value = {
            "CaseTrace": CaseTrace,
            "PcmInput": PcmInput,
            "StreamConfig": StreamConfig,
            "TranscribeRequest": TranscribeRequest,
        }[name]
    elif name == "StreamingWorker":
        from .supervisor import StreamingWorker

        value = StreamingWorker
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
