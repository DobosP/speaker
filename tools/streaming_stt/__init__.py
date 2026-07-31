"""Bounded, model-isolated streaming-STT benchmark support.

The package deliberately imports no optional recognizer runtime. Candidate
models run in separately provisioned Python processes behind the JSONL contract
implemented here.
"""

from .manifest import WorkerManifest, load_worker_manifest
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
