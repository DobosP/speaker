"""
Structured pipeline diagnostics for JSONL analysis (LLM-friendly).

When disabled: near-zero cost (module-level flag check only).

When enabled (``diagnostics_log_path`` set on VoiceAssistant): events go through a
bounded queue and a daemon writer thread so the realtime audio path avoids
blocking on disk IO.

Each line is compact JSON: ``{"event":"...","subsystem":"pipeline",...}``.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

_enabled = False
_path: str | None = None
_session_id = "local"
_queue: queue.Queue[dict[str, Any] | None] | None = None
_writer: threading.Thread | None = None
_lock = threading.Lock()
_shutdown_sentinel = None  # stop writer when put on queue


def configure(
    *,
    enabled: bool,
    path: str | None,
    session_id: str | None = None,
) -> None:
    """Enable queued JSONL pipeline logging to ``path``."""
    global _enabled, _path, _session_id, _queue, _writer
    with _lock:
        want = bool(enabled and path)
        if not want:
            old_q = _queue
            old_w = _writer
            _enabled = False
            _path = None
            _session_id = session_id or "local"
            if old_q is not None:
                try:
                    old_q.put_nowait(None)
                except queue.Full:
                    pass
                if old_w is not None:
                    old_w.join(timeout=3.0)
            _queue = None
            _writer = None
            return
        _enabled = True
        _path = path
        _session_id = session_id or "local"
        if _queue is None:
            _queue = queue.Queue(maxsize=4096)
            _writer = threading.Thread(target=_writer_loop, name="pipeline_log", daemon=True)
            _writer.start()


def _writer_loop() -> None:
    buf: list[str] = []
    while True:
        try:
            item = _queue.get(timeout=0.15)
        except queue.Empty:
            if buf:
                _flush_buf(buf)
                buf.clear()
            continue
        if item is None:
            if buf:
                _flush_buf(buf)
            break
        if isinstance(item, dict):
            try:
                buf.append(
                    json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n"
                )
            except Exception:
                pass
            if len(buf) >= 32:
                _flush_buf(buf)
                buf.clear()


def _flush_buf(lines: list[str]) -> None:
    if not _path or not lines:
        return
    try:
        parent = os.path.dirname(_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        pass


def emit(event: str, **fields: Any) -> None:
    """Record one structured event (no-op when disabled)."""
    if not _enabled or not _path:
        return
    q = _queue
    if q is None:
        return
    payload = {
        "event": event,
        "subsystem": "pipeline",
        "session_id": _session_id,
        "timestamp": time.time(),
        **fields,
    }
    try:
        q.put_nowait(payload)
    except queue.Full:
        pass


@contextmanager
def span(component: str, name: str, **extra: Any) -> Iterator[None]:
    """Time a block; emits ``pipeline_span`` with ``duration_ms``."""
    if not _enabled:
        yield
        return
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        emit(
            "pipeline_span",
            component=component,
            span=name,
            duration_ms=round(dt_ms, 3),
            **extra,
        )


def flush_sync() -> None:
    """Drain queue for tests / clean shutdown (best-effort)."""
    if not _queue:
        return
    time.sleep(0.35)
