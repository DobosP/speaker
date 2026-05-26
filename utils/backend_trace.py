"""
Thread-safe counters and optional JSONL/stderr tracing for Ollama and STT calls.

Used for diagnosing concurrent faster-whisper vs whisper.cpp partial load.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

_lock = threading.Lock()
_enabled = False
_jsonl_path: str | None = None
_counters: dict[str, int] = {
    "ollama_chat_batch": 0,
    "ollama_chat_stream": 0,
    "stt_final": 0,
    "stt_partial": 0,
}


def configure(*, enabled: bool = False, diagnostics_log_path: str | None = None) -> None:
    """Enable tracing; JSONL lines append to ``diagnostics_log_path`` when both are set."""
    global _enabled, _jsonl_path
    with _lock:
        _enabled = bool(enabled)
        _jsonl_path = diagnostics_log_path if (_enabled and diagnostics_log_path) else None


def reset() -> None:
    """Clear counters (for tests)."""
    with _lock:
        for k in _counters:
            _counters[k] = 0


def snapshot() -> dict[str, int]:
    with _lock:
        return dict(_counters)


def _bump(key: str) -> None:
    with _lock:
        _counters[key] = _counters.get(key, 0) + 1


def _emit_stderr(msg: str) -> None:
    with _lock:
        if not _enabled:
            return
    print(msg, file=sys.stderr, flush=True)


def _emit_jsonl(payload: dict[str, Any]) -> None:
    with _lock:
        if not _enabled or not _jsonl_path:
            return
        path = _jsonl_path
    try:
        line = json.dumps({"timestamp": time.time(), **payload}, ensure_ascii=True) + "\n"
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def record_ollama(trace_kind: str, model: str, purpose: str = "") -> None:
    """One logical ``ollama.chat`` call (streaming counts once)."""
    key = "ollama_chat_stream" if trace_kind == "stream" else "ollama_chat_batch"
    _bump(key)
    _emit_stderr(
        f"(trace) ollama.chat kind={trace_kind} model={model} purpose={purpose or 'unknown'}"
    )
    _emit_jsonl(
        {
            "event": "backend_trace",
            "subsystem": "ollama",
            "chat_kind": trace_kind,
            "model": model,
            "purpose": purpose or None,
        }
    )


def record_stt_final(model_id: str, model_type: str, n_samples: int) -> None:
    _bump("stt_final")
    _emit_stderr(
        f"(trace) stt_final model={model_id} type={model_type} samples={n_samples}"
    )
    _emit_jsonl(
        {
            "event": "backend_trace",
            "subsystem": "stt_final",
            "model_id": model_id,
            "model_type": model_type,
            "n_samples": n_samples,
        }
    )


def record_stt_partial(model_id: str, n_samples: int) -> None:
    _bump("stt_partial")
    _emit_stderr(
        f"(trace) stt_partial model={model_id} samples={n_samples}"
    )
    _emit_jsonl(
        {
            "event": "backend_trace",
            "subsystem": "stt_partial",
            "model_id": model_id,
            "n_samples": n_samples,
        }
    )
